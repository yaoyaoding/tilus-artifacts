from typing import Type, Dict, Union, List, Optional, Sequence, Set, Tuple
from hidet.ir.expr import Expr, Var, Constant
from hidet.ir.stmt import LetStmt
from hidet.ir.func import Function
from hidet.ir.module import IRModule
from hidet.ir.functors import IRRewriter, IRVisitor
from hidet.ir.tools import TypeInfer
from hidet.ir.tile.type import TileType
from hidet.ir.tile.expr import TileOp, CallTileOp
from hidet.ir.tile.stmt import PureForStmt
from hidet.ir.tile.ops import Create, BinaryTileOp, Dot, ConvertLayout, CastOp, UnaryTileOp, Broadcast, ExtractSlice
from hidet.ir.tile.ops.arthimatic import Add, Multiply, Equal, NotEqual
from hidet.utils import same_list


class Pattern:
    pass


class ScalarPlaceholder(Pattern):
    pass


class TilePattern(Pattern):
    pass


class TilePlaceholder(TilePattern):
    pass


class TileOpPattern(TilePattern):
    def __init__(self, op_cls: Type[TileOp], *args: Pattern):
        self.op_cls: Type[TileOp] = op_cls
        self.args: List[Pattern] = list(args)


class TileGraphPattern(TilePattern):
    def __init__(self, allowed_classes: Sequence[Type[TileOp]], final_op_cls: Type[TileOp]):
        self.allowed_classes: Tuple[Type[TileOp], ...] = tuple(allowed_classes)
        self.final_op_cls: Type[TileOp] = final_op_cls


class MatchedTileGraph:
    def __init__(self, graph_inputs: List[Expr], target: Expr):
        self.graph_inputs: List[Expr] = graph_inputs
        self.target: Expr = target


MatchedResult = Union[Expr, MatchedTileGraph]


class PatternBuilder:
    @staticmethod
    def any_scalar_expr() -> ScalarPlaceholder:
        return ScalarPlaceholder()

    @staticmethod
    def any_tile() -> TilePattern:
        return TilePlaceholder()

    @staticmethod
    def tile_graph(allowed_classes: Sequence[Type[TileOp]], final_op_cls: Type[TileOp]):
        return TileGraphPattern(list(allowed_classes), final_op_cls)

    @staticmethod
    def construct(value=None):
        if value is None:
            value = PatternBuilder.any_scalar_expr()
        return TileOpPattern(Create, value)

    @staticmethod
    def unary(x: TilePattern):
        return TileOpPattern(UnaryTileOp, x)

    @staticmethod
    def binary(x: TilePattern, y: TilePattern):
        return TileOpPattern(BinaryTileOp, x, y)

    @staticmethod
    def convert_layout(x: TilePattern):
        return TileOpPattern(ConvertLayout, x)

    @staticmethod
    def cast(x: TilePattern):
        return TileOpPattern(CastOp, x)

    @staticmethod
    def add(lhs: TilePattern, rhs: TilePattern):
        return TileOpPattern(Add, lhs, rhs)

    @staticmethod
    def broadcast(x: TilePattern):
        return TileOpPattern(Broadcast, x)

    @staticmethod
    def extract_slice(x: TilePattern, start: ScalarPlaceholder):
        return TileOpPattern(ExtractSlice, x, start)

    @staticmethod
    def dot(a: TilePattern, b: TilePattern, c: TilePattern):
        return TileOpPattern(Dot, a, b, c)

    @staticmethod
    def is_zero(e: Expr, var2call: Dict[Var, CallTileOp]) -> bool:
        if isinstance(e, Var) and e in var2call:
            e = var2call[e]
        if isinstance(e, CallTileOp):
            if isinstance(e.op, Create):
                if isinstance(e.op.value, Constant):
                    v = e.op.value
                else:
                    return False
            else:
                return False
            return isinstance(v, Constant) and v.value == v.type.zero.value
        else:
            return False

    @staticmethod
    def get_tile_op(e: Pattern, matched: Dict[Pattern, MatchedResult], var2call: Dict[Var, CallTileOp]):
        match = matched[e]
        if isinstance(match, Var) and match in var2call:
            return var2call[match].op
        elif isinstance(match, CallTileOp):
            return match.op
        else:
            raise RuntimeError()


class PatternTransform(PatternBuilder):
    def __call__(self, node):
        while True:
            orig_node = node
            node = apply_transforms(node, [self])
            if orig_node is node:
                return node

    def source(self) -> TilePattern:
        raise NotImplementedError()

    def target(self, matched: Dict[Pattern, MatchedResult], var2call: Dict[Var, CallTileOp]) -> Optional[Expr]:
        raise NotImplementedError()


class PatternAnalyzer(PatternBuilder):
    def pattern(self) -> TilePattern:
        raise NotImplementedError()

    def analyze(self, matched: Dict[Pattern, MatchedResult], var2call: Dict[Var, CallTileOp]) -> None:
        raise NotImplementedError()


class Matcher:
    def __init__(self, var2value: Dict[Var, Union[CallTileOp, Var]]):
        self.var2call: Dict[Var, Union[Var, CallTileOp]] = var2value
        self.match_dict: Dict[Pattern, MatchedResult] = {}
        self.type_infer = TypeInfer()

    def match(self, pattern: Pattern, target: Expr) -> bool:
        if pattern in self.match_dict:
            return target is self.match_dict[pattern]
        if isinstance(pattern, ScalarPlaceholder):
            return self.match_ScalarPlaceholder(pattern, target)
        elif isinstance(pattern, TilePlaceholder):
            return self.match_TilePlaceholder(pattern, target)
        elif isinstance(pattern, TileOpPattern):
            return self.match_TileOpPattern(pattern, target)
        elif isinstance(pattern, TileGraphPattern):
            return self.match_TileGraphPattern(pattern, target)
        else:
            raise NotImplementedError()

    def match_ScalarPlaceholder(self, pattern: ScalarPlaceholder, target: Expr) -> bool:
        tp = self.type_infer(target)
        if isinstance(tp, TileType):
            return False
        else:
            self.match_dict[pattern] = target
            return True

    def match_TilePlaceholder(self, pattern: TilePlaceholder, target: Expr) -> bool:
        if isinstance(target, Var) and isinstance(target.type, TileType) or isinstance(target, CallTileOp):
            self.match_dict[pattern] = target
            return True
        else:
            return False

    def match_TileOpPattern(self, pattern: TileOpPattern, target: Expr) -> bool:
        if isinstance(target, Var) and target in self.var2call:
            call_op = self.var2call[target]
        elif isinstance(target, CallTileOp):
            call_op = target
        else:
            return False

        op_cls = type(call_op.op)
        if not issubclass(op_cls, pattern.op_cls):
            return False
        if len(call_op.op.args) != len(pattern.args):
            return False

        is_commutative_binary_op = issubclass(op_cls, BinaryTileOp) and op_cls in [Add, Multiply, Equal, NotEqual]
        if is_commutative_binary_op:
            saved_match_dict = self.match_dict.copy()
            pa, pb = pattern.args
            ta, tb = call_op.op.args
            if self.match(pa, ta) and self.match(pb, tb):
                self.match_dict[pattern] = target
                return True
            else:
                self.match_dict = saved_match_dict.copy()
                if self.match(pa, tb) and self.match(pb, ta):
                    self.match_dict[pattern] = target
                    return True
                else:
                    return False
        else:
            for pattern_arg, target_arg in zip(pattern.args, call_op.op.args):
                if not self.match(pattern_arg, target_arg):
                    return False
        self.match_dict[pattern] = target
        return True

    def match_TileGraphPattern(self, pattern: TileGraphPattern, target: Expr) -> bool:
        if isinstance(target, Var) and target in self.var2call:
            call_op = self.var2call[target]
        elif isinstance(target, CallTileOp):
            call_op = target
        else:
            return False

        op_cls = type(call_op.op)
        if not issubclass(op_cls, pattern.final_op_cls):
            return False

        graph_inputs: List[Expr] = []
        queue: List[Expr] = list(call_op.op.args)
        visited: Set[Expr] = set(queue)
        while queue:
            u = queue.pop()

            if isinstance(u, Var) and u in self.var2call and isinstance(self.var2call[u].op, pattern.allowed_classes):
                op = self.var2call[u].op
            elif isinstance(u, CallTileOp) and isinstance(u.op, pattern.allowed_classes):
                op = u.op
            else:
                if u not in graph_inputs:
                    graph_inputs.append(u)
                continue

            for v in op.args:
                if v not in visited:
                    visited.add(v)
                    queue.append(v)
        self.match_dict[pattern] = MatchedTileGraph(graph_inputs=graph_inputs, target=target)
        return True


class ApplyTransformRewriter(IRRewriter):
    def __init__(self, transforms: List[PatternTransform]):
        super().__init__()
        self.var2value: Dict[Var, Union[CallTileOp, Var]] = {}
        self.transforms: List[PatternTransform] = transforms
        self.type_infer = TypeInfer()

    def visit_Function(self, func: Function):
        self.var2value.clear()
        return super().visit_Function(func)

    def visit_LetStmt(self, stmt: LetStmt):
        bind_vars = []
        bind_values = []
        for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
            bind_value = self.visit(bind_value)
            if isinstance(bind_value, Var):
                # remove the binding entry
                self.memo[bind_var] = bind_value
            else:
                bind_vars.append(bind_var)
                bind_values.append(bind_value)
                self.var2value[bind_var] = bind_value
        body = self.visit(stmt.body)
        if len(bind_vars) == 0:
            return body
        elif same_list(bind_vars, stmt.bind_vars) and same_list(bind_values, stmt.bind_values) and body is stmt.body:
            return stmt
        else:
            return LetStmt(bind_vars, bind_values, body)

    def visit_CallTileOp(self, call: CallTileOp):
        call = super().visit_CallTileOp(call)

        for transform in self.transforms:
            matcher = Matcher(self.var2value)
            if matcher.match(transform.source(), call):
                updated_call = transform.target(matcher.match_dict, self.var2value)
                if updated_call is not None:
                    return updated_call
        return call


class ApplyPatternAnalyzerVisitor(IRVisitor):
    def __init__(self, analyzers: List[PatternAnalyzer]):
        super().__init__()
        self.var2call: Dict[Var, CallTileOp] = {}
        self.analyzers: List[PatternAnalyzer] = analyzers

    def bind(self, var_list: List[Var], value_list: List[Expr]):
        for var, value in zip(var_list, value_list):
            if isinstance(value, CallTileOp):
                self.var2call[var] = value

    def visit_LetStmt(self, stmt: LetStmt):
        self.bind(stmt.bind_vars, stmt.bind_values)
        for _, bind_value in zip(stmt.bind_vars, stmt.bind_values):
            self.visit(bind_value)
        self.visit(stmt.body)

    def visit_PureForStmt(self, stmt: PureForStmt):
        self.bind(stmt.args, stmt.values)
        self.visit(stmt.args)
        self.visit(stmt.values)
        self.visit(stmt.body)
        self.visit(stmt.let_body)

    def visit_CallTileOp(self, call: CallTileOp):
        super().visit_CallTileOp(call)
        for analyzer in self.analyzers:
            matcher = Matcher(self.var2call)
            if matcher.match(analyzer.pattern(), call):
                analyzer.analyze(matcher.match_dict, self.var2call)


def apply_transforms(node: Union[IRModule, Function], transforms: List[PatternTransform]):
    rewriter = ApplyTransformRewriter(transforms)
    return rewriter.visit(node)


def apply_analyzers(node: Union[IRModule, Function], analyzers: List[PatternAnalyzer]):
    visitor = ApplyPatternAnalyzerVisitor(analyzers)
    visitor.visit(node)
