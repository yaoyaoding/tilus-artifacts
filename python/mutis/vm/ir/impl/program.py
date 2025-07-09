from typing import Dict, List, Optional
from hidet.ir.expr import Var, Expr
from hidet.ir.dtypes import boolean, int32
from hidet.ir.primitives.cuda.vars import blockIdx
from mutis.vm.ir.program import VirtualMachineProgram, BlockMapping, ParamAttrs


def ThreadBlockMapping_default_mapping(graph_tile):
    # hardware_axes_bound = {
    #     blockIdx.x: 2147483647,  # 2^31 - 1
    #     blockIdx.y: 65535,  # 2^16 - 1
    #     blockIdx.z: 65535,  # 2^16 - 1
    # }
    from mutis.ir.tile import GraphTile

    assert isinstance(graph_tile, GraphTile)

    axis2size: Dict[Var, Expr] = graph_tile.num_tiles_map
    block_axes = graph_tile.block_axes
    block_reduce_axes = graph_tile.inter_block_reduce_axes
    all_axes = block_reduce_axes + block_axes

    # perform mapping
    logical2hardware: Dict[Var, Var] = {}
    hardware_axes = [blockIdx.x, blockIdx.y, blockIdx.z]

    if len(all_axes) == 1:
        logical2hardware[all_axes[0]] = blockIdx.x
    elif len(all_axes) == 2:
        logical2hardware[all_axes[0]] = blockIdx.y
        logical2hardware[all_axes[1]] = blockIdx.x
    elif len(all_axes) == 3:
        logical2hardware[all_axes[0]] = blockIdx.z
        logical2hardware[all_axes[1]] = blockIdx.y
        logical2hardware[all_axes[2]] = blockIdx.x
    else:
        for axis in all_axes[:-3]:
            logical2hardware[axis] = blockIdx.z
        logical2hardware[all_axes[-3]] = blockIdx.z
        logical2hardware[all_axes[-2]] = blockIdx.y
        logical2hardware[all_axes[-1]] = blockIdx.x

    # get the logical axis expressions of hardware axes
    hardware_axes_size = {axis: int32.one for axis in hardware_axes}
    for axis in all_axes:
        hardware_axis = logical2hardware[axis]
        hardware_axes_size[hardware_axis] = hardware_axes_size[hardware_axis] * axis2size[axis]

    # get the mapping from logical axis to the expression of hardware axes
    virtual_axes_values = {}
    hardware_axis_divisor = {blockIdx.x: 1, blockIdx.y: 1, blockIdx.z: 1}

    last_virtual_axis: Dict[Var, Var] = {}
    for axis, hardware_axis in logical2hardware.items():
        last_virtual_axis[hardware_axis] = axis

    for axis, hardware_axis in logical2hardware.items():
        virtual_axes_values[axis] = hardware_axis // hardware_axis_divisor[hardware_axis]
        if axis is not last_virtual_axis[hardware_axis]:
            virtual_axes_values[axis] = virtual_axes_values[axis] % axis2size[axis]
        hardware_axis_divisor[hardware_axis] = hardware_axis_divisor[hardware_axis] * axis2size[axis]

    return BlockMapping(
        hardware_axes=hardware_axes,
        hardware_num_blocks=[hardware_axes_size[axis] for axis in hardware_axes],
        predicate=boolean.true,
        virtual_axes_values=virtual_axes_values,
    )
