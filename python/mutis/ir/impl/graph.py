from typing import Optional, Dict, List
import shutil
import os
import json
import numpy as np

import tabulate

from hidet.ir.type import DataType
from hidet.ir.expr import Var
from hidet.ir.tools import IRPrinter
from mutis.ir.graph import Graph, Operator, Tensor
from mutis.ir.schedule import Schedule


def Graph_astext(graph: Graph, schedule: Optional[Schedule]) -> str:
    import tabulate
    from hidet.utils.doc import Doc, NewLine, Text, doc_join
    from hidet.ir.tools import IRPrinter
    from mutis.ir.tile import TensorTile
    from mutis.ir.schedule import Schedule
    from mutis.ir.utils import table_prune_empty_column

    printer = IRPrinter()
    tensor_name: Dict[Tensor, str] = {}
    sch: Schedule = schedule

    def get_tensor_name(t: Tensor) -> str:
        if t not in tensor_name:
            tensor_name[t] = '%{}'.format(len(tensor_name))
        return tensor_name[t]

    def get_attr_doc(v) -> str:
        if isinstance(v, (tuple, list)):
            return '[' + printer(v) + ']'
        elif isinstance(v, str):
            return repr(v)
        elif isinstance(v, DataType):
            return v.short_name
        else:
            return printer(v)

    def get_tensor_type_doc(t: Tensor) -> Doc:
        if isinstance(t.elem_type, DataType):
            elem_type = t.elem_type.short_name
        else:
            elem_type = t.elem_type
        return Text('{}[{}]'.format(printer(elem_type), printer(t.shape)))

    def get_op_type_doc(inputs: List[Tensor], output: Optional[Tensor]) -> Doc:
        doc = Doc()
        if len(inputs) == 0 and output is None:
            return doc
        elif len(inputs) == 0:
            doc += get_tensor_type_doc(output)
        elif output is None:
            doc += doc_join([get_tensor_type_doc(t) for t in inputs], ', ')
        else:
            doc += doc_join([get_tensor_type_doc(t) for t in inputs], ', ') + ' -> ' + get_tensor_type_doc(output)
        return doc

    def get_op_tile_size_doc(op: Operator) -> Doc:
        if op.output is None or sch is None or sch.graph_tile is None or op.output not in sch.graph_tile.tensor2tile:
            return Doc()
        tensor: Tensor = op.output
        tile: TensorTile = sch.graph_tile.tensor2tile[op.output]
        doc = Doc()
        tiled_shape = []
        for dim in range(len(tensor.shape)):
            if len(tile.tile_axes[dim]) == 0:
                tiled_shape.append(printer(tensor.shape[dim]))
            else:
                tiled_shape.append(str(tile.tile_sizes[dim][-1]))
        doc += '[' + doc_join(tiled_shape, ', ') + ']'
        return doc

    def get_op_tile_index_doc(op: Operator) -> Doc:
        if op.output is None or sch is None or sch.graph_tile is None or op.output not in sch.graph_tile.tensor2tile:
            return Doc()
        tensor: Tensor = op.output
        tile: TensorTile = sch.graph_tile.tensor2tile[op.output]
        doc = Doc()
        tile_indices = []
        for dim in range(len(tensor.shape)):
            if len(tile.tile_axes[dim]) == 0:
                tile_indices.append('*')
            else:
                e = Doc()
                for i, (axis, tile_size) in enumerate(zip(tile.tile_axes[dim], tile.tile_sizes[dim])):
                    if i != 0:
                        e += '+'
                    if tile_size == 1:
                        e += printer(axis)
                    else:
                        e += '{}*{}'.format(printer(axis), tile_size)
                tile_indices.append(e)
        doc += '[' + doc_join(tile_indices, ', ') + ']'
        return doc

    def get_op_doc(op: Operator) -> Doc:
        doc = Doc()
        if op.output:
            doc += get_tensor_name(op.output) + ' = '
        # from CamelCase to snake_case
        camel_name: str = op.__class__.__name__  # e.g., BinaryElementwise
        snake_name = "".join(["_" + c.lower() if c.isupper() else c for c in camel_name]).lstrip(
            "_"
        )  # e.g., binary_elementwise
        doc += snake_name
        doc += '('
        doc += doc_join([get_tensor_name(t) for t in op.inputs], ', ')
        if op.attrs:
            if op.inputs:
                doc += ', '
            doc += doc_join(['{}={}'.format(k, get_attr_doc(v)) for k, v in op.attrs.items()], ', ')
        doc += ')'
        return doc

    def get_op_variant(op: Operator) -> Doc:
        if sch is None or sch.variants is None or op not in sch.variants.op2variant:
            return Doc()
        return Text(str({key: str(value) for key, value in sch.variants.op2variant[op].items()}))

    def get_op_output_layout(op: Operator) -> Doc:
        if sch is None or sch.layouts is None or op.output is None or op.output not in sch.layouts:
            return Doc()
        layout = sch.layouts[op.output]
        return Text('local_size={} definition={}'.format(layout.local_size, str(layout)))

    def get_tile_axes_doc() -> Doc:
        if sch is None or sch.graph_tile is None:
            return Doc()

        for axis in sch.graph_tile.tile_size_map:
            printer(axis)

        headers = ['Kind', 'Axis', 'Tile Size', 'Num Tiles']
        rows = []

        doc = Doc()
        for kind in ['block_axes', 'inter_block_reduce_axes', 'reduce_axes', 'unroll_axes']:
            if len(getattr(sch.graph_tile, kind)) == 0:
                continue
            for axis in getattr(sch.graph_tile, kind):
                rows.append(
                    [
                        kind,
                        str(printer(axis)),
                        sch.graph_tile.tile_size_map[axis],
                        printer(sch.graph_tile.num_tiles_map[axis]),
                    ]
                )
        headers, rows = table_prune_empty_column(headers, rows)
        doc += tabulate.tabulate(rows, headers=headers).replace('\n', '\n    ')
        return doc

    def get_params_doc() -> Doc:
        doc = Doc()
        headers = ['Name', 'Type', 'Attrs']
        rows = []
        for param in graph.params:
            rows.append([param.hint, printer(param.type), str(graph.param2attrs[param])])
        headers, rows = table_prune_empty_column(headers, rows)
        doc += NewLine() + 'Params:'
        doc += NewLine(4) + tabulate.tabulate(rows, headers=headers).replace('\n', '\n    ')
        return doc

    def get_signature_doc() -> Doc:
        doc = Doc()
        doc += NewLine() + 'Graph Name:'
        doc += NewLine(4) + graph.name
        if sch and sch.num_warps:
            doc += NewLine() + 'Attributes:'
            doc += NewLine(4) + 'num_warps: {}'.format(sch.num_warps)
        # doc += NewLine() + 'Params:'
        # doc += NewLine(4) + doc_join([f'{param.hint}: {printer(param.type)} {graph.param2attrs[param]}' for param in graph.params], NewLine(4))
        doc += get_params_doc()
        if sch and sch.graph_tile:
            doc += NewLine() + 'Tiles:'
            doc += NewLine(indent=4) + get_tile_axes_doc()
        if sch and sch.partition:
            doc += NewLine() + 'Partition:'
            doc += NewLine(indent=4) + str(
                sch.partition.astext(op2name={op: str(idx) for idx, op in enumerate(graph.nodes)})
            )
        return doc

    def get_graph_doc() -> Doc:
        doc = Doc()
        doc += get_signature_doc()
        doc += NewLine() + 'Operator Table:'
        headers = ['Index', 'Operator', 'Type', 'Variant', 'Tile Size', 'Tile Index', 'Layout', 'Type']
        rows = []
        for idx, op in enumerate(graph.nodes):
            row = [
                str(idx),
                str(get_op_doc(op)),
                str(op.output.elem_type.short_name) if op.output else '',
                str(get_op_variant(op)),
                str(get_op_tile_size_doc(op)),
                str(get_op_tile_index_doc(op)),
                str(get_op_output_layout(op)),
                str(get_op_type_doc(op.inputs, op.output)),
            ]
            rows.append(row)
        headers, rows = table_prune_empty_column(headers, rows)
        doc += NewLine(indent=4) + tabulate.tabulate(rows, headers=headers).replace('\n', '\n    ')
        doc += NewLine()
        return doc

    return str(get_graph_doc())


def schedule_summary_txt(graph: Graph, schedules: List[Schedule]) -> str:
    headers = []  # idx, num_warps, tiling axes, op's tilings and variant
    rows = []
    headers.append('Index')
    headers.append('Warps')
    headers.append('Tile Axes')
    for idx, op in enumerate(graph.nodes):
        headers.append('[{}] {}'.format(idx, op.__class__.__name__))
    for idx, sch in enumerate(schedules):
        row = [idx, sch.num_warps]

        # tile axes
        printer = IRPrinter()
        tile_axes: List[Var] = []
        tile_axes.extend(sch.graph_tile.inter_block_reduce_axes)
        tile_axes.extend(sch.graph_tile.block_axes)
        tile_axes.extend(sch.graph_tile.reduce_axes)
        tile_axes.extend(sch.graph_tile.unroll_axes)
        items = ['{}: {}'.format(printer(axis), sch.graph_tile.tile_size_map[axis]) for axis in tile_axes]
        row.append(', '.join(items))

        # op's tilings and variant
        for op in graph.nodes:
            items = []
            if op.output is not None:
                tile = sch.graph_tile.tensor2tile[op.output]
                items.append('tile=[{}]'.format(printer(tile.tiled_shape())))
            if sch.variants and op in sch.variants.op2variant:
                items.append(
                    'variant={{{}}}'.format(
                        printer({key: str(value) for key, value in sch.variants.op2variant[op].items()})
                    )
                )
            if sch.layouts and op.output is not None:
                items.append('layout={}'.format(sch.layouts[op.output]))
            row.append(', '.join(items))

        rows.append(row)
    return tabulate.tabulate(rows, headers=headers)


def schedule_summary_json(graph: Graph, schedules: List[Schedule]) -> str:
    data = {}
    for idx, sch in enumerate(schedules):
        sch_data = {}
        sch_data['idx'] = idx
        sch_data['warps'] = sch.num_warps

        # tile axes
        printer = IRPrinter()
        tile_axes: List[Var] = []
        tile_axes.extend(sch.graph_tile.inter_block_reduce_axes)
        tile_axes.extend(sch.graph_tile.block_axes)
        tile_axes.extend(sch.graph_tile.reduce_axes)
        tile_axes.extend(sch.graph_tile.unroll_axes)
        sch_data['tile_axes'] = {str(printer(axis)): sch.graph_tile.tile_size_map[axis] for axis in tile_axes}

        # op's tilings and variant
        sch_data['ops'] = []
        for op in graph.nodes:
            op_data = {'op': str(op), 'kind': op.__class__.__name__.lower()}
            if op.output is not None:
                tile = sch.graph_tile.tensor2tile[op.output]
                op_data['tiled_shape'] = tile.tiled_shape()
            if sch.variants and op in sch.variants.op2variant:
                op_data['variant'] = str(
                    printer({key: str(value) for key, value in sch.variants.op2variant[op].items()})
                )
            if sch.layouts and op.output is not None:
                op_data['layout'] = str(sch.layouts[op.output])
            sch_data['ops'].append(op_data)

        data[idx] = sch_data

    return json.dumps(data, indent=2)


def Graph_dump_schedules(graph: Graph, schedules, out_dir='./outs'):
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    for idx, sch in enumerate(schedules):
        with open(os.path.join(out_dir, f'{idx}.txt'), 'w') as f:
            f.write(graph.astext(sch))


def Graph_dump_schedules_summary(graph: Graph, schedules, summary_path='./outs/schedule_summary.txt'):
    assert summary_path.endswith('.txt')
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w') as f:
        f.write(schedule_summary_txt(graph, schedules))
    json_path = summary_path.replace('.txt', '.json')
    with open(json_path, 'w') as f:
        f.write(schedule_summary_json(graph, schedules))
