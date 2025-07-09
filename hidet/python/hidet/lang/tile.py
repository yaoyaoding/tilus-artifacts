# pylint: disable=unused-import
from typing import Sequence

from hidet.ir.tile.ops.creation import arange, full, ones, zeros, grid, compute
from hidet.ir.tile.ops.activations import silu, exp
from hidet.ir.tile.ops.memory import load, store, atomic_add
from hidet.ir.tile.ops.system import num_programs, program_id
from hidet.ir.tile.ops.transform import broadcast, reshape, expand_dims, cast
from hidet.ir.tile.ops.debug import debug_print, debug_sync
from hidet.ir.tile.ops.reduce import sum, max, min
from hidet.ir.tile.ops.dot import dot, simt_dot
from hidet.ir.tile.ops.arthimatic import maximum, minimum


def cdiv(a, b):
    return (a + b - 1) // b


def deserialize(serialized_index, shape: Sequence[int]):
    index = []
    for i in range(len(shape) - 1, -1, -1):
        index.append(serialized_index % shape[i])
        serialized_index //= shape[i]
    return index[::-1]
