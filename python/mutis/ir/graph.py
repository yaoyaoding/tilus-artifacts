from __future__ import annotations as _

from typing import List, Union, Optional, Dict, Any, Tuple
import os
import shutil


from hidet.ir.expr import Expr, Var
from hidet.ir.type import DataType, PointerType
from mutis.ir.bound import Bound


class Tensor:
    def __init__(self, elem_type: Union[DataType, PointerType], shape: List[Expr], producer=None):
        self.elem_type: Union[DataType, PointerType] = elem_type
        self.shape: List[Expr] = shape

        self.producer: Operator = producer

    def __str__(self):
        return '{}[{}]'.format(self.elem_type.name, ', '.join(str(dim) for dim in self.shape))

    def __add__(self, other):
        from mutis.ops import add

        return add(self, other)

    def __sub__(self, other):
        from mutis.ops import sub

        return sub(self, other)

    def __mul__(self, other):
        from mutis.ops import multiply

        return multiply(self, other)

    def __truediv__(self, other):
        from mutis.ops import divide

        return divide(self, other)


class Operator:

    NO_OUTPUT = False  # whether this operator has output, update this in subclass

    def __init__(self, inputs: List[Tensor], attrs: Dict[str, Any]):
        self.inputs: List[Tensor] = inputs
        self.attrs: Dict[str, Any] = attrs
        self.output: Optional[Tensor] = self._create_output()

    def __str__(self):
        from hidet.ir.tools import IRPrinter

        items = []
        if self.output is not None:
            items.append(str(self.output))
        for x in self.inputs:
            items.append(str(x))
        printer = IRPrinter()
        for k, v in self.attrs.items():
            if isinstance(v, tuple):
                v_str = '({})'.format(printer(v))
            elif isinstance(v, list):
                v_str = '[{}]'.format(printer(v))
            elif isinstance(v, dict):
                v_str = '{{}}'.format(printer(v))
            else:
                v_str = printer(v)
            items.append('{}={}'.format(k, v_str))
        return '{}({})'.format(self.__class__.__name__, ', '.join(items))

    def _create_output(self) -> Optional[Tensor]:
        if self.NO_OUTPUT:
            return None
        else:
            elem_type, shape = self.infer_type()
            return Tensor(elem_type, shape, producer=self)

    def get_input(self, idx: int) -> Tensor:
        if idx < 0:
            idx += len(self.inputs)
        if idx >= len(self.inputs) or idx < 0:
            raise ValueError('idx out of bound: {}'.format(idx))
        return self.inputs[idx]

    def get_output(self) -> Tensor:
        if self.output is None:
            raise ValueError('The operator does not have output, please implement `infer_type` method to provide.')
        return self.output

    def reforward(self, inputs: List[Tensor], update_attrs: Dict[str, Any]) -> Operator:
        attrs = self.attrs.copy()
        attrs.update(update_attrs)
        return self.__class__(*inputs, **attrs)

    def infer_type(self) -> Tuple[Union[DataType, PointerType], List[Expr]]:
        raise NotImplementedError()

    def tile_propagation_sets(self) -> List[List[Tuple[int, int]]]:
        raise NotImplementedError(self.__class__.__name__)


class ParamAttrs:
    def __init__(
        self,
        lower: Optional[int] = None,
        upper: Optional[int] = None,
        divisibility: Optional[int] = None,
        is_weight: bool = False,
        weight_nbytes: Optional[int] = None,
    ):
        self.lower: Optional[int] = lower
        self.upper: Optional[int] = upper
        self.divisibility: Optional[int] = divisibility
        self.is_weight: bool = is_weight
        self.weight_nbytes: Optional[int] = weight_nbytes

    def __str__(self):
        items = []
        if self.lower is not None:
            items.append('lower={}'.format(self.lower))
        if self.upper is not None:
            items.append('upper={}'.format(self.upper))
        if self.divisibility is not None:
            items.append('divisibility={}'.format(self.divisibility))
        if self.is_weight:
            items.append('is_weight=True')
            if self.weight_nbytes is not None:
                items.append('weight_nbytes={}'.format(self.weight_nbytes))
        return ', '.join(items)

    def is_nontrivial(self):
        return self.upper is not None or self.lower is not None or self.divisibility is not None or self.is_weight


class Graph:
    def __init__(self, name: str, params: List[Var], param2attrs: Dict[Var, ParamAttrs], nodes: List[Operator]):
        self.name: str = name
        self.params: List[Var] = params
        self.param2attrs: Dict[Var, ParamAttrs] = param2attrs
        self.nodes: List[Operator] = nodes

    def __str__(self):
        return self.astext()

    def astext(self, schedule=None) -> str:
        from mutis.ir.impl.graph import Graph_astext

        return Graph_astext(self, schedule)

    def dump_schedules(self, schedules, out_dir='./outs'):
        from mutis.ir.impl.graph import Graph_dump_schedules

        return Graph_dump_schedules(self, schedules, out_dir)

    def dump_schedule_summary(self, schedules, summary_path='./outs/schedules.txt'):
        from mutis.ir.impl.graph import Graph_dump_schedules_summary

        return Graph_dump_schedules_summary(self, schedules, summary_path)


class GraphContext:

    _current: Optional[GraphContext] = None

    def __init__(self, name: str, params: List[Var], param2attrs: Dict[Var, ParamAttrs]):
        self.prev: Optional[GraphContext] = None
        self.name: str = name
        self.params: List[Var] = params
        self.param2attrs: Dict[Var, ParamAttrs] = param2attrs
        self.nodes: List[Operator] = []

    def __enter__(self):
        self.prev = GraphContext._current
        GraphContext._current = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        GraphContext._current = self.prev
        self.prev = None

    @staticmethod
    def current() -> GraphContext:
        if GraphContext._current is None:
            raise ValueError('Not in a graph context')
        return GraphContext._current

    def append(self, op: Operator):
        self.nodes.append(op)

    def graph(self) -> Graph:
        return Graph(self.name, self.params, self.param2attrs, self.nodes)
