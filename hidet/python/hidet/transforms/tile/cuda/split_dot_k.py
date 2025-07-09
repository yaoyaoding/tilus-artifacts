"""
d = dot(a, b, c)    # a: [m, ks], b: [ks, n], c: [m, n] where ks = s * k

=>

a = convert_layout(a, smem)
b = convert_layout(b, smem)
a0 = convert_layout(extract_slice(a, start=0, extent=k, axis=1), dot_operand_layout)
b0 = convert_layout(extract_slice(b, start=0, extent=k, axis=0), dot_operand_layout)
d0 = dot(a0, b0, c)
a1 = convert_layout(extract_slice(a, start=k, extent=k, axis=1), dot_operand_layout)
b1 = convert_layout(extract_slice(b, start=k, extent=k, axis=0), dot_operand_layout)
d = dot(a1, b1, d0)
"""
from typing import List, Dict, Optional, Set, Tuple
from collections import deque
from hidet.ir.func import Function
from hidet.ir.functors import IRRewriter
from hidet.ir.dtypes import int32
from hidet.ir.expr import Expr, Var, Let
from hidet.ir.tile.expr import CallTileOp, TileOp
from hidet.ir.tile.ops.dot import Dot, SimtDot, MmaDot
from hidet.ir.tile.ops.convert_layout import ConvertLayout
from hidet.ir.tile.ops.transform import Broadcast, broadcast
from hidet.ir.tile.ops.smem import extract_slice, ExtractSlice
from hidet.ir.tile.ops.arthimatic import BinaryTileOp, UnaryTileOp
from hidet.ir.tile.ops import convert_layout, slice
from hidet.ir.tile.type import TileType, TileScope
from hidet.ir.tile.layout import repeat
from hidet.ir.tile.layout import MmaOutputLayout, MmaDotOperandLayout, SliceLayout, BlockLayout, BlockDotOperandLayout
from hidet.ir.tools import TypeInfer
from hidet.transforms.base import TileFunctionPass
from hidet.transforms.tile.generic.canonicalize_to_ssa import canonicalize_to_ssa
from hidet.transforms.tile.generic.dead_code_elimination import DeadCodeEliminationRewriter
from hidet.transforms.tile.generic.loop_invariant_code_motion import LoopInvariantCodeMotionRewriter
from hidet.transforms.tile.cuda.remove_layout_convert import remove_layout_convert
from hidet.transforms.tile.analyzers import DefinitionAnalyzer, VarDefinition, LetDefinition, ForArgDefinition
from hidet.transforms.tile.utils.pattern import PatternTransform, TilePattern, Pattern, MatchedResult, MatchedTileGraph


class LetChain:
    def __init__(self):
        self.lets: List[Let] = []
        self.type_infer = TypeInfer()

    def let(self, hint, value: Expr) -> Var:
        tp = self.type_infer(value)
        v = Var(hint, tp)
        self.lets.append(Let(v, value, None))
        return v

    def make_expr(self, body: Expr):
        for let in reversed(self.lets):
            body = Let(let.var, let.value, body)
        return body


class SplitDotKRewriter(IRRewriter):
    def __init__(self):
        super().__init__()
        self.type_infer = TypeInfer()
        self.definitions: Dict[Var, VarDefinition] = {}

    def visit_Function(self, func: Function):
        definition_analyzer = DefinitionAnalyzer()
        definition_analyzer(func)
        self.definitions = definition_analyzer.definitions

        return super().visit_Function(func)

    def analyze_prologue(self, a: Var, b: Var) -> Tuple[List[Var], List[Var], List[TileOp], Dict[Var, str]]:
        inputs: List[Var] = []
        let_vars: List[Var] = []
        nodes: List[TileOp] = []
        involved: Dict[Var, str] = {a: 'a', b: 'b'}

        queue = deque([a, b])
        visited: Set[Var] = set(queue)

        while queue:
            u = queue.pop()

            assert isinstance(u, Var) and u in self.definitions

            definition = self.definitions[u]
            if isinstance(definition, LetDefinition):
                rhs = definition.bind_value
                if isinstance(rhs, CallTileOp):
                    op: TileOp = rhs.op
                    if isinstance(op, (UnaryTileOp, BinaryTileOp, ConvertLayout)):
                        let_vars.append(u)
                        nodes.append(op)
                        for arg in op.args:
                            if arg not in visited:
                                queue.appendleft(arg)
                                involved[arg] = involved[u]
                                visited.add(arg)

                            if involved[arg] != involved[u]:
                                raise NotImplementedError(
                                    'We do not support the case where a tile is used in multiple dot operand computation'
                                )

            if u not in inputs:
                inputs.append(u)

        let_vars = list(reversed(let_vars))
        nodes = list(reversed(nodes))

        return inputs, let_vars, nodes, involved

    def visit_Dot(self, e: Dot):
        a = self.visit(e.a)
        b = self.visit(e.b)
        c = self.visit(e.c)
        a_type: TileType = self.type_infer(a)
        b_type: TileType = self.type_infer(b)
        c_type: TileType = self.type_infer(c)
        a_layout = a_type.layout
        b_layout = b_type.layout
        c_layout = c_type.layout
        m, n, k = a_type.shape[0], b_type.shape[1], a_type.shape[1]

        # analyze the subgraph with dot as final op
        inputs: List[Var]
        let_vars: List[Var]
        nodes: List[TileOp]
        involved: Dict[Var, str]
        inputs, let_vars, nodes, involved = self.analyze_prologue(a, b)

        cb = LetChain()  # chain builder

        if isinstance(e, SimtDot):
            inst_k = 8
            if k <= inst_k:
                return super().visit_Dot(e)

            # prepare layouts
            assert (
                isinstance(a_layout, BlockDotOperandLayout)
                and isinstance(b_layout, BlockDotOperandLayout)
                and isinstance(c_layout, BlockLayout)
                and k % inst_k == 0
            )
            s = k // inst_k
            aa_smem_layout = SliceLayout(repeat(m, k), dim=1, extent=inst_k)
            bb_smem_layout = SliceLayout(repeat(k, n), dim=0, extent=inst_k)
            aa_layout = BlockDotOperandLayout(parent=a_layout.parent, k_size=inst_k, op_idx=a_layout.op_idx)
            bb_layout = BlockDotOperandLayout(parent=b_layout.parent, k_size=inst_k, op_idx=b_layout.op_idx)

            remap: Dict[Var, Var] = {}
            smem_inputs: List[Var] = []
            # convert inputs to smem
            for inp in inputs:
                smem_inp = cb.let(inp.name, convert_layout(inp, repeat(*inp.type.shape), scope=TileScope.Shared))
                smem_inputs.append(smem_inp)

            for i in range(s):
                remap.clear()

                # extract a slice
                for inp, smem_inp in zip(inputs, smem_inputs):
                    if involved[inp] == 'a':
                        ext = cb.let(
                            inp.name,
                            extract_slice(smem_inp, axis=1, start=i * inst_k, extent=inst_k, layout=aa_smem_layout),
                        )
                        cvt = cb.let(inp.name, convert_layout(ext, layout=aa_layout))
                        remap[inp] = cvt
                    else:
                        ext = cb.let(
                            inp.name,
                            extract_slice(smem_inp, axis=0, start=i * inst_k, extent=inst_k, layout=bb_smem_layout),
                        )
                        cvt = cb.let(inp.name, convert_layout(ext, layout=bb_layout))
                        remap[inp] = cvt

                # apply the nodes
                for let_var, node in zip(let_vars, nodes):
                    args = [remap[arg] for arg in node.args]
                    remap[let_var] = cb.let(let_var.name, node.reforward(args=args).make_call())

                # mma
                c = cb.let('c', SimtDot(remap[a], remap[b], c).make_call())
            return cb.make_expr(c)
        elif isinstance(e, MmaDot):
            assert (
                isinstance(a_layout, MmaDotOperandLayout)
                and isinstance(b_layout, MmaDotOperandLayout)
                and isinstance(c_layout, MmaOutputLayout)
                and a_layout.mma == b_layout.mma == c_layout
            )
            mma: MmaOutputLayout = c_layout
            cc_layout = MmaOutputLayout(
                num_warps=mma.num_warps, m_size=mma.m_size, n_size=mma.n_size, k_size=mma.inst_k, config=mma.config
            )
            aa_layout = MmaDotOperandLayout(cc_layout, op_idx=0)
            bb_layout = MmaDotOperandLayout(cc_layout, op_idx=1)
            inst_k = mma.inst_k
            s: int = k // mma.inst_k
            if s == 1:
                return super().visit_Dot(e)

            remap: Dict[Var, Var] = {}
            smem_inputs: List[Var] = []
            # convert inputs to smem
            for inp in inputs:
                layout = a_layout.shared_layout() if involved[inp] == 'a' else b_layout.shared_layout()
                smem_inp = cb.let(inp.name, convert_layout(inp, layout, scope=TileScope.Shared))
                smem_inputs.append(smem_inp)

            c = cb.let('c', convert_layout(c, cc_layout))
            for i in range(s):
                remap.clear()

                # extract a slice
                for inp, smem_inp in zip(inputs, smem_inputs):
                    if involved[inp] == 'a':
                        axis = 1
                        layout = aa_layout
                    else:
                        axis = 0
                        layout = bb_layout
                    remap[inp] = cb.let(
                        inp.name, extract_slice(smem_inp, axis=axis, start=i * inst_k, extent=inst_k, layout=layout)
                    )

                # apply the nodes
                for let_var, node in zip(let_vars, nodes):
                    args = [remap[arg] for arg in node.args]
                    if isinstance(node, ConvertLayout):
                        # ignore convert_layout
                        remap[let_var] = args[0]
                        continue
                    remap[let_var] = cb.let(let_var.name, node.reforward(args=args).make_call())

                # mma
                ai = remap[a]
                bi = remap[b]
                c = cb.let('c', MmaDot(ai, bi, c).make_call())
            c = cb.let('c', convert_layout(c, c_layout))
            return cb.make_expr(c)
        else:
            raise NotImplementedError()


class FuseBroadcastConvertLayoutExtractSliceTransform(PatternTransform):
    """
    extract_slice(convert_layout(broadcast(x, shape)), axis, layout) -> convert_layout(broadcast(x, layout))
    """

    def __init__(self):
        super().__init__()
        self.x = self.any_tile()
        self.start = self.any_scalar_expr()
        self.brdcst = self.broadcast(self.x)
        self.cvt = self.convert_layout(self.brdcst)
        self.out = self.extract_slice(self.cvt, self.start)

    def source(self) -> TilePattern:
        return self.out

    def target(self, matched: Dict[Pattern, MatchedResult], var2call: Dict[Var, CallTileOp]) -> Optional[Expr]:
        x = matched[self.x]
        broadcast_op: Broadcast = self.get_tile_op(self.brdcst, matched, var2call)
        convert_layout_op: ConvertLayout = self.get_tile_op(self.cvt, matched, var2call)
        extract_slice_op: ExtractSlice = self.get_tile_op(self.out, matched, var2call)

        assert isinstance(x, Var) and isinstance(x.type, TileType)

        broadcast_dims = [i for i, (a, b) in enumerate(zip(x.type.shape, broadcast_op.shape)) if a != b]
        if not (len(broadcast_dims) == 1 and extract_slice_op.axis == broadcast_dims[0]):
            return None

        shape = broadcast_op.shape.copy()
        shape[extract_slice_op.axis] = extract_slice_op.extent
        scope = TileScope.Shared if extract_slice_op.layout.num_workers() == 1 else TileScope.Register
        brdcst_layout = repeat(*[b // a for a, b in zip(x.type.shape, shape)], flatten_local=False) * x.type.layout
        brdcst = broadcast_op.reforward([x], attr_update={'shape': shape, 'layout': brdcst_layout}).make_call()
        return convert_layout(brdcst, layout=extract_slice_op.layout, scope=scope)


def split_k_related_transforms(node):
    transforms = [FuseBroadcastConvertLayoutExtractSliceTransform()]
    for transform in transforms:
        node = transform(node)
    return node


class SplitDotKPass(TileFunctionPass):
    def process_tile_func(self, func: Function) -> Function:
        return self.apply_transforms(
            func,
            [
                SplitDotKRewriter(),
                canonicalize_to_ssa,
                remove_layout_convert,
                split_k_related_transforms,
                canonicalize_to_ssa,
                DeadCodeEliminationRewriter(),
                LoopInvariantCodeMotionRewriter(allow=[ConvertLayout, Broadcast]),
            ],
            repeat_limit=2,
        )


def split_dot_k_pass() -> TileFunctionPass:
    return SplitDotKPass()
