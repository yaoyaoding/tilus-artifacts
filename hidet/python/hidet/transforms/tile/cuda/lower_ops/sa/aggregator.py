from typing import List, Tuple, Any, Sequence
import os
from hidet.ir.mapping import RepeatTaskMapping
from hidet.ir.expr import Expr, Var, Constant, Add, Sub, Multiply, Div, Mod, TensorElement, Call, cast
from hidet.ir.stmt import Stmt, SeqStmt, ForMappingStmt, BufferStoreStmt, IfStmt, EvaluateStmt, ForStmt
from hidet.ir.tools import rewrite
from hidet.transforms.tile.cuda.lower_ops.sa.solver import solve


class BatchIRFunctor:
    def aggregate(self, nodes: List[Any]) -> Any:
        return self.visit(nodes)

    def visit(self, nodes: List[Any]):
        cls = type(nodes[0])
        if not all(isinstance(node, cls) for node in nodes):
            buf = ['All nodes in a batch must have the same type, got:']
            for node in nodes:
                buf.append(f'  {node}')
            raise TypeError('\n'.join(buf))

        method = getattr(self, f'visit_{cls.__name__}')
        return method(nodes)

    # python builtins
    def visit_tuple(self, nodes: List[Tuple]):
        raise NotImplementedError()

    def visit_list(self, nodes: List[List]):
        raise NotImplementedError()

    def visit_int(self, nodes: List[int]):
        raise NotImplementedError()

    def visit_NoneType(self, nodes: List[None]):
        raise NotImplementedError()

    # task mapping
    def visit_RepeatTaskMapping(self, nodes: List[RepeatTaskMapping]):
        raise NotImplementedError()

    # expressions
    def visit_Constant(self, nodes: List[Constant]):
        raise NotImplementedError()

    def visit_Var(self, nodes: List[Var]):
        raise NotImplementedError()

    def visit_Add(self, nodes: List[Add]):
        raise NotImplementedError()

    def visit_Sub(self, nodes: List[Sub]):
        raise NotImplementedError()

    def visit_Multiply(self, nodes: List[Multiply]):
        raise NotImplementedError()

    def visit_Div(self, nodes: List[Div]):
        raise NotImplementedError()

    def visit_Mod(self, nodes: List[Mod]):
        raise NotImplementedError()

    def visit_TensorElement(self, nodes: List[TensorElement]):
        raise NotImplementedError()

    def visit_Call(self, nodes: List[Call]):
        raise NotImplementedError()

    # statements
    def visit_SeqStmt(self, nodes: List[SeqStmt]):
        raise NotImplementedError()

    def visit_ForMappingStmt(self, nodes: List[ForMappingStmt]):
        raise NotImplementedError()

    def visit_BufferStoreStmt(self, nodes: List[BufferStoreStmt]):
        raise NotImplementedError()

    def visit_IfStmt(self, nodes: List[IfStmt]):
        raise NotImplementedError()

    def visit_EvaluateStmt(self, nodes: List[EvaluateStmt]):
        raise NotImplementedError()

    def visit_ForStmt(self, nodes: List[ForStmt]):
        raise NotImplementedError()


class Aggregator(BatchIRFunctor):
    def __init__(self, program_ids: List[int], worker: Expr):
        super().__init__()
        self.program_ids: List[int] = program_ids
        self.worker: Expr = worker
        self.memo = {}

    def check(self, msg, cond: bool):
        if not cond:
            raise ValueError(msg)

    # python builtins
    def visit_seq(self, nodes: List[Sequence]):
        seq_length = len(nodes[0])
        self.check(msg='all sequences should have the same length', cond=all(len(node) == seq_length for node in nodes))
        updated_seq = []
        for i in range(seq_length):
            nodes_i = [node[i] for node in nodes]
            updated_seq.append(self.visit(nodes_i))
        return updated_seq

    # python builtins
    def visit_tuple(self, nodes: List[Tuple]):
        return tuple(self.visit_seq(nodes))

    def visit_list(self, nodes: List[List]):
        return self.visit_seq(nodes)

    def visit_int(self, nodes: List[int]):
        if not all(node == nodes[0] for node in nodes):
            raise ValueError('Python int constant in a batch do not match')
        return nodes[0]

    def visit_NoneType(self, nodes: List[None]):
        return None

    # task mapping
    def visit_RepeatTaskMapping(self, nodes: List[RepeatTaskMapping]):
        ranks = self.visit([node.ranks for node in nodes])
        task_shape = self.visit([node.task_shape for node in nodes])
        return RepeatTaskMapping(task_shape, ranks, attrs=nodes[0].attrs)

    # expressions
    def visit_Constant(self, nodes: List[Constant]):
        value = nodes[0]
        if all(node == value for node in nodes):
            return value
        else:
            if not value.type.is_integer():
                raise ValueError('Cannot aggregate different non-integer constants')
            x = self.program_ids
            y = [int(node) for node in nodes]
            x_var, y_expr = solve(x, y)
            result = rewrite(y_expr, {x_var: self.worker})
            result = cast(result, value.type)
            return result

    def visit_Var(self, nodes: List[Var]):
        nodes = [self.memo.get(node, node) for node in nodes]
        if not all(node is nodes[0] for node in nodes):
            raise ValueError('Variables in a batch do not match')
        return nodes[0]

    def visit_Add(self, nodes: List[Add]):
        a = self.visit([node.a for node in nodes])
        b = self.visit([node.b for node in nodes])
        return Add(a, b)

    def visit_Sub(self, nodes: List[Sub]):
        a = self.visit([node.a for node in nodes])
        b = self.visit([node.b for node in nodes])
        return Sub(a, b)

    def visit_Multiply(self, nodes: List[Multiply]):
        a = self.visit([node.a for node in nodes])
        b = self.visit([node.b for node in nodes])
        return Multiply(a, b)

    def visit_Div(self, nodes: List[Div]):
        a = self.visit([node.a for node in nodes])
        b = self.visit([node.b for node in nodes])
        return Div(a, b)

    def visit_Mod(self, nodes: List[Mod]):
        a = self.visit([node.a for node in nodes])
        b = self.visit([node.b for node in nodes])
        return Mod(a, b)

    def visit_TensorElement(self, nodes: List[TensorElement]):
        base = self.visit([node.base for node in nodes])
        indices = self.visit([node.indices for node in nodes])
        return TensorElement(base, indices)

    def visit_Call(self, nodes: List[Call]):
        func_var = self.visit([node.func_var for node in nodes])
        args = self.visit([node.args for node in nodes])
        return Call(func_var, args)

    # statements
    def visit_SeqStmt(self, nodes: List[SeqStmt]):
        num_stats = len(nodes[0].seq)
        self.check(
            msg="all SeqStmt should have the same length", cond=all(len(node.seq) == num_stats for node in nodes)
        )
        seq = []
        for i in range(num_stats):
            nodes_i = [node.seq[i] for node in nodes]
            seq.append(self.visit(nodes_i))
        return SeqStmt(seq)

    def visit_ForMappingStmt(self, nodes: List[ForMappingStmt]):
        mapping = self.visit([node.mapping for node in nodes])
        worker = self.visit([node.worker for node in nodes])
        chief = nodes[0]
        for vice in nodes:
            for chief_loop_var, vice_loop_var in zip(chief.loop_vars, vice.loop_vars):
                self.memo[vice_loop_var] = chief_loop_var
        loop_vars = chief.loop_vars
        body = self.visit([node.body for node in nodes])
        return ForMappingStmt(loop_vars, mapping, worker, body)

    def visit_BufferStoreStmt(self, nodes: List[BufferStoreStmt]):
        buf = self.visit([node.buf for node in nodes])
        indices = self.visit([node.indices for node in nodes])
        value = self.visit([node.value for node in nodes])
        return BufferStoreStmt(buf, indices, value, nodes[0].protected)

    def visit_IfStmt(self, nodes: List[IfStmt]):
        cond = self.visit([node.cond for node in nodes])
        then_body = self.visit([node.then_body for node in nodes])
        else_body = self.visit([node.else_body for node in nodes])
        return IfStmt(cond, then_body, else_body)

    def visit_EvaluateStmt(self, nodes: List[EvaluateStmt]):
        expr = self.visit([node.expr for node in nodes])
        return EvaluateStmt(expr)

    def visit_ForStmt(self, nodes: List[ForStmt]):
        chief = nodes[0]
        for vice in nodes:
            self.memo[vice.loop_var] = chief.loop_var
        extent = self.visit([node.extent for node in nodes])
        body = self.visit([node.body for node in nodes])
        return ForStmt(chief.loop_var, extent, body)


def aggregate(programs: List[Stmt], program_ids: List[int], worker: Expr) -> Stmt:
    # dump programs
    os.makedirs('./outs/programs', exist_ok=True)
    for i, program in enumerate(programs):
        with open(f'./outs/programs/{i}.txt', 'w') as f:
            f.write(str(program))

    # aggregate programs
    aggregator = Aggregator(program_ids=program_ids, worker=worker)
    return aggregator.aggregate(programs)


if __name__ == '__main__':
    # a = BatchNode([1, 2, 3])
    # print(isinstance(a, int))
    pass
