from typing import List

from hidet.ir.expr import Var, Expr, var
from hidet.ir.func import Function
from hidet.ir.functors import IRRewriter
from hidet.ir.stmt import LetStmt
from hidet.ir.tile.cuda.mma_configs import TileMmaConfig
from hidet.ir.tile.expr import CallTileOp
from hidet.ir.tile.layout import BlockLayout, FlattenBlockLayout, BlockDotOperandLayout, TileLayout
from hidet.ir.tile.layout import MmaOutputLayout, MmaDotOperandLayout, ReduceLayout
from hidet.ir.tile.ops import Broadcast, BinaryTileOp, ReduceOp, Dot, ExpandDims, SimtDot, StoreBaseOp, MmaDot
from hidet.ir.tile.ops import Create, Assign, convert_layout
from hidet.ir.tile.stmt import PureForStmt, YieldStmt
from hidet.ir.tile.type import TileType
from hidet.ir.tile.layout import repeat
from hidet.ir.tools import TypeInfer
from hidet.ir.type import DataType
from hidet.transforms.base import TileFunctionPass
from hidet.transforms.tile.generic.canonicalize_to_ssa import canonicalize_to_ssa
from hidet.utils import same_list


class InstantiateLayoutRewriter(IRRewriter):
    def __init__(self):
        super().__init__(use_memo=True)
        self.num_warps: int = 0
        self.type_infer = TypeInfer()

    def layout2priority(self, layout: TileLayout):
        priority_map = {
            MmaOutputLayout: 100,
            MmaDotOperandLayout: 90,
            BlockDotOperandLayout: 80,
            ReduceLayout: 70,
            FlattenBlockLayout: 60,
            BlockLayout: 0,
        }
        others = 50
        for cls in priority_map:
            if isinstance(layout, cls):
                return priority_map[cls]
        return others

    def visit_Function(self, func: Function):
        if 'cuda.block_dim' not in func.attrs:
            raise ValueError("cuda.block_dim must be specified for 'cuda_tile' function")
        block_dim = func.attrs['cuda.block_dim']
        try:
            block_dim = int(block_dim)
        except ValueError as e:
            raise ValueError(f"cuda.block_dim must be a constant integer, got {block_dim}") from e
        self.num_warps = block_dim // 32
        if block_dim % 32 != 0:
            raise ValueError(f"cuda.block_dim must be a multiple of 32, got {block_dim}")
        return super().visit_Function(func)

    def visit_CallTileOp(self, call: CallTileOp):
        op = self.visit(call.op)
        if op is call.op:
            ret = call
        else:
            ret = op.make_call()
        ttype = self.type_infer.visit(ret)
        if isinstance(ttype, TileType) and ttype.layout is None:
            raise NotImplementedError(
                'The layout of the following tile op has not been instantiated:\n'
                + '  {}\n'.format(type(call.op).__name__)
            )
        return ret

    def visit_LetStmt(self, stmt: LetStmt):
        bind_vars = []
        bind_values = []
        for orig_var, orig_value in zip(stmt.bind_vars, stmt.bind_values):
            bind_value = self.visit(orig_value)
            bind_var = var(orig_var.hint, self.type_infer(bind_value))
            self.memo[orig_var] = bind_var
            bind_vars.append(bind_var)
            bind_values.append(bind_value)
        body = self.visit(stmt.body)
        return LetStmt(bind_vars, bind_values, body)

    def visit_PureForStmt(self, stmt: PureForStmt):
        self.pure_for_stmts.append(stmt)
        extent: Expr = self.visit(stmt.extent)
        values: List[Expr] = self.visit(stmt.values)
        args: List[Var] = [Var(arg.hint, self.type_infer(value)) for arg, value in zip(stmt.args, values)]
        for orig_arg, arg in zip(stmt.args, args):
            self.memo[orig_arg] = arg
        body = self.visit(stmt.body)
        self.pure_for_stmts.pop()
        let_vars = [Var(let_var.hint, arg.type) for let_var, arg in zip(stmt.let_vars, args)]
        for orig_let_var, let_var in zip(stmt.let_vars, let_vars):
            self.memo[orig_let_var] = let_var
        let_body = self.visit(stmt.let_body)
        return PureForStmt(args, values, stmt.loop_var, extent, body, let_vars, let_body)

    def visit_YieldStmt(self, stmt: YieldStmt):
        for_stmt = self.pure_for_stmts[-1]
        args: List[Var] = self.visit(for_stmt.args)
        yields: List[Expr] = self.visit(stmt.values)
        updated_values = []
        for arg, yie in zip(args, yields):
            yield_type = self.type_infer(yie)
            if isinstance(arg.type, TileType):
                assert isinstance(yield_type, TileType)
                if arg.type.layout != yield_type.layout:
                    updated_values.append(convert_layout(yie, arg.type.layout))
                else:
                    updated_values.append(yie)
            else:
                updated_values.append(yie)
        return YieldStmt(updated_values)

    def visit_Create(self, e: Create):
        if e.layout:
            layout = e.layout
        else:
            layout = BlockLayout.from_shape(e.shape, self.num_warps)
        value = self.visit(e.value)
        return Create(value, e.shape, e.axes, layout)

    def visit_BinaryTileOp(self, e: BinaryTileOp):
        x = self.visit(e.x)
        y = self.visit(e.y)
        x_type = self.type_infer.visit(x)
        y_type = self.type_infer.visit(y)
        assert isinstance(x_type, TileType) and isinstance(y_type, TileType)
        assert same_list(x_type.shape, y_type.shape)

        if x_type.layout != y_type.layout:
            layout = max(x_type.layout, y_type.layout, key=self.layout2priority)
            if x_type.layout != layout:
                x = convert_layout(x, layout)
            if y_type.layout != layout:
                y = convert_layout(y, layout)
            return e.reforward([x, y])
        else:
            return super().visit_BinaryTileOp(e)

    def visit_Broadcast(self, e: Broadcast):
        x = self.visit(e.x)
        x_type = self.type_infer.visit(x)
        x_layout = x_type.layout
        assert isinstance(x_type, TileType)

        if isinstance(x_layout, BlockLayout):
            y_layout = BlockLayout(
                shape=e.shape,
                size_per_thread=x_layout.size_per_thread,
                thread_per_warp=x_layout.thread_per_warp,
                warps_per_block=x_layout.warps_per_block,
            )
            return Broadcast(x, e.shape, y_layout)

        if isinstance(x_layout, ReduceLayout) and same_list(x_layout.parent.logical_shape(), e.shape):
            y_layout = x_layout.parent
            return Broadcast(x, e.shape, y_layout)

        repeat_shape = []
        for x_dim, y_dim in zip(x_type.shape, e.shape):
            if x_dim == y_dim:
                repeat_shape.append(1)
            elif x_dim == 1:
                repeat_shape.append(y_dim)
            else:
                raise ValueError('Can not broadcast tensor of shape {} to shape {}'.format(x_type.shape, e.shape))
        y_layout = repeat(*repeat_shape) * x_type.layout
        return Broadcast(x, e.shape, y_layout)

    def visit_ExpandDims(self, e: ExpandDims):
        x = self.visit(e.x)
        x_type = self.type_infer.visit(x)
        assert isinstance(x_type, TileType)
        if isinstance(x_type.layout, BlockLayout):
            y_shape = x_type.shape[: e.axis] + [1] + x_type.shape[e.axis :]
            y_layout = BlockLayout.from_shape(y_shape, self.num_warps)
            return ExpandDims(
                x=convert_layout(x, layout=FlattenBlockLayout(y_layout, axis=e.axis)), axis=e.axis, layout=y_layout
            )
        else:
            raise NotImplementedError()

    def visit_ReduceOp(self, e: ReduceOp):
        x = self.visit(e.x)
        x_type = self.type_infer.visit(x)
        assert isinstance(x_type, TileType)

        return ReduceOp(
            x=x,
            axis=e.axis,
            keepdims=e.keepdims,
            kind=e.kind,
            layout=ReduceLayout(x_type.layout, dim=e.axis, keep_dim=e.keepdims),
        )

    def visit_Dot(self, e: Dot):
        a = self.visit(e.a)
        b = self.visit(e.b)
        c = self.visit(e.c)
        a_type: TileType = self.type_infer.visit(a)
        b_type: TileType = self.type_infer.visit(b)
        c_type: TileType = self.type_infer.visit(c)
        m, n = c_type.shape
        k = a_type.shape[1]

        assert a_type.type == b_type.type
        in_dtype: DataType = a_type.type
        out_dtype: DataType = c_type.type

        if isinstance(e, SimtDot):
            num_threads = self.num_warps * 32
            if m >= 4 and n >= 4 and m * n >= num_threads * 16:
                size_per_thread = [4, 4]
            elif m >= 2 and n >= 4 and m * n >= num_threads * 8:
                size_per_thread = [2, 4]
            elif m >= 1 and n >= 4 and m * n >= num_threads * 4:
                size_per_thread = [1, 4]
            elif m >= 2 and n >= 2 and m * n >= num_threads * 4:
                size_per_thread = [2, 2]
            else:
                size_per_thread = [1, 1]
            c_layout = BlockLayout.from_shape([m, n], num_warps=self.num_warps, size_per_thread=size_per_thread)
            a_layout = BlockDotOperandLayout(parent=c_layout, k_size=k, op_idx=0)
            b_layout = BlockDotOperandLayout(parent=c_layout, k_size=k, op_idx=1)
            if a_type.layout != a_layout:
                a = convert_layout(a, a_layout)
            if b_type.layout != b_layout:
                b = convert_layout(b, b_layout)
            if c_type.layout != c_layout:
                c = convert_layout(c, c_layout)
            return SimtDot(a, b, c)
        elif isinstance(e, MmaDot):
            # find available mma config for the given shape and dtypes
            configs: List[TileMmaConfig] = []
            for config in TileMmaConfig.all():
                if config.in_dtype != in_dtype or config.out_dtype != out_dtype:
                    continue
                if m % config.m != 0 or n % config.n != 0 or k % config.k != 0:
                    continue
                configs.append(config)
            if len(configs) == 0:
                raise RuntimeError(
                    'Can not find a suitable MMA config for the given shape and dtypes: \n'
                    + '  m x n x k: {} x {} x {}\n'.format(m, n, k)
                    + '  input dtype: {}\n'.format(in_dtype)
                    + '  output dtype: {}\n'.format(out_dtype)
                )

            # pick the largest one
            config: TileMmaConfig = max(configs, key=lambda cfg: cfg.m * cfg.n * cfg.k)
            c_layout = MmaOutputLayout(num_warps=self.num_warps, m_size=m, n_size=n, k_size=k, config=config)
            a_layout = MmaDotOperandLayout(mma=c_layout, op_idx=0)
            b_layout = MmaDotOperandLayout(mma=c_layout, op_idx=1)
            if a_type.layout != a_layout:
                a = convert_layout(a, a_layout)
            if b_type.layout != b_layout:
                b = convert_layout(b, b_layout)
            if c_type.layout != c_layout:
                c = convert_layout(c, c_layout)
            return MmaDot(a, b, c)
        else:
            raise NotImplementedError()

    def visit_StoreBaseOp(self, e: StoreBaseOp):
        ptr = self.visit(e.ptr)
        value = self.visit(e.value)
        mask = self.visit(e.mask) if e.mask is not None else None

        ptr_type: TileType = self.type_infer.visit(ptr)
        value_type: TileType = self.type_infer.visit(value)
        mask_type: TileType = self.type_infer.visit(mask) if mask is not None else None

        # we use the layout of the value to determine the layout of the ptr and mask
        layout = value_type.layout

        if ptr_type.layout != layout:
            ptr = convert_layout(ptr, layout)

        if mask is not None and mask_type.layout != layout:
            mask = convert_layout(mask, layout)

        return e.reforward(args=[ptr, value, mask])

    def visit_Assign(self, e: Assign):
        dst = self.visit(e.dst)
        src = self.visit(e.src)
        dst_type = self.type_infer.visit(dst)
        src_type = self.type_infer.visit(src)
        assert isinstance(dst_type, TileType) and isinstance(src_type, TileType)
        if dst_type.layout != src_type.layout:
            src = convert_layout(src, dst_type.layout)
        return Assign(dst, src)


class InstantiateLayoutPass(TileFunctionPass):
    def process_tile_func(self, func: Function) -> Function:
        rewriter = InstantiateLayoutRewriter()
        func = rewriter(func)
        func = canonicalize_to_ssa(func)
        return func


def instantiate_layout(func: Function) -> Function:
    rewriter = InstantiateLayoutRewriter()
    return canonicalize_to_ssa(rewriter.visit(func))


def instantiate_layout_pass() -> TileFunctionPass:
    return InstantiateLayoutPass()
