# pylint: disable=useless-super-delegation
from __future__ import annotations

from typing import List, Tuple, Optional, Dict
from collections import defaultdict

import io
import itertools
from hidet.ir.node import Node
from hidet.ir.dtypes import boolean, int32
from hidet.ir.expr import Expr, logical_and, equal
from hidet.utils import same_list, prod, is_power_of_two, argmin, argmax, iter_grid
from hidet.utils.vector import Vector


class TileLayout(Node):
    def __str__(self):
        from hidet.ir.tools import astext

        return astext(self, inline_attrs=True)

    def __eq__(self, other):
        raise NotImplementedError('{}.__eq__'.format(type(self)))

    def __hash__(self):
        return hash(str(self))

    def __mul__(self, other):
        assert isinstance(other, TileLayout)
        return ComposedLayout(outer=self, inner=other)

    def num_workers(self) -> int:
        raise NotImplementedError()

    def local_shape(self) -> List[int]:
        raise NotImplementedError()

    def logical_shape(self) -> List[int]:
        raise NotImplementedError()

    def local2logical(self, local_indices: List[Expr], worker: Optional[Expr] = None) -> Tuple[List[Expr], Expr]:
        """
        ret: logical_indices, not_duplicated
        """
        raise NotImplementedError()

    def logical2local(self, logical_indices: List[Expr], worker: Optional[Expr] = None) -> Tuple[List[Expr], Expr]:
        """
        ret: local_index, is_valid
        """
        raise NotImplementedError()

    def representation(self) -> str:
        raise NotImplementedError(self.__class__.__name__)

    def atom(
        self,
        *shape,
        workers: List[int],
        ranks: Optional[List[int]] = None,
        worker_ranks: Optional[List[int]] = None,
        flatten_local=True,
        squeeze_dims: Optional[List[int]] = None,
    ):
        return ComposedLayout(
            outer=self,
            inner=atom(
                *shape,
                workers=workers,
                ranks=ranks,
                worker_ranks=worker_ranks,
                flatten_local=flatten_local,
                squeeze_dims=squeeze_dims,
            ),
        )

    def spatial(self, *shape: int, ranks: Optional[List[int]] = None, flatten_local=True):
        return ComposedLayout(outer=self, inner=spatial(*shape, ranks=ranks, flatten_local=flatten_local))

    def repeat(self, *shape: int, ranks: Optional[List[int]] = None, flatten_local=True):
        return ComposedLayout(outer=self, inner=repeat(*shape, ranks=ranks, flatten_local=flatten_local))

    def expand(self, dim: int):
        return ExpandLayout(base=self, dim=dim)

    def swizzle(self, dim: int, regards_dim: Optional[int] = None, log_step: int = 0):
        return SwizzleLayout(base=self, dim=dim, regards_dim=regards_dim, log_step=log_step)

    def visualize(self, verbose: bool = False) -> str:
        shape = self.logical_shape()
        if len(shape) not in [1, 2]:
            raise ValueError('Cannot visualize layout with rank {} (shape={})'.format(len(shape), shape))
        grid: Dict[Tuple[int, ...], List[Tuple[int, List[int], bool]]] = defaultdict(list)

        for worker in range(self.num_workers()):
            for local_indices in iter_grid(self.local_shape()):
                logical_indices, not_duplicated = self.local2logical(local_indices, int32(worker))
                grid[tuple(int(v) for v in logical_indices)].append(
                    (int(worker), [int(v) for v in local_indices], bool(not_duplicated))
                )

        str_grid: Dict[Tuple[int, ...], str] = {}
        for indices in grid:
            items = []
            for worker, local_indices, not_duplicated in grid[indices]:
                if verbose:
                    items.append('{}:{}'.format(worker, local_indices))
                else:
                    items.append(str(worker))
            if len(items) == 0:
                str_grid[indices] = '.'
            elif len(items) == 1:
                str_grid[indices] = items[0]
            else:
                str_grid[indices] = '{' + ', '.join(items) + '}'

        width = max(len(str(a)) for a in str_grid.values())
        width = max(width, max(len(str(d)) for d in self.logical_shape())) + 1
        fmt = '{:>' + str(width) + '}'
        f = io.StringIO()

        idx_width = max(len(str(d)) for d in self.logical_shape()) + 1
        idx_fmt = '{:>' + str(idx_width) + '}'

        # print the logical shape, num of workers, and local extent
        print('  logical shape: {}'.format(self.logical_shape()), file=f)
        print('    local shape: {}'.format(self.local_shape()), file=f)
        print(' num of workers: {}'.format(self.num_workers()), file=f)
        # print the first row of indices
        for j in range(shape[-1]):
            if j == 0:
                print(' ' * idx_width + ' |', file=f, end='')
            print(fmt.format(j), file=f, end='')
        print(file=f)
        # print the second row of separator
        for j in range(shape[-1]):
            if j == 0:
                sep = ' ' + '-' * (idx_width - 1) + ' +'
                print(sep, file=f, end='')
            sep = ' ' + '-' * (width - 1)
            print(sep, file=f, end='')
        print(file=f)
        # print each row of the layout
        for logical_indices in itertools.product(*map(range, shape)):
            if logical_indices[-1] == 0:
                print(idx_fmt.format(logical_indices[0]) + ' |', file=f, end='')
            print(fmt.format(str_grid[logical_indices]), file=f, end='')
            if logical_indices[-1] == shape[-1] - 1:
                print(file=f)
        ret = f.getvalue()
        f.close()
        return ret


class AtomLayout(TileLayout):
    def __init__(
        self,
        shape: List[int],
        worker_shape: List[int],
        ranks: Optional[List[int]] = None,
        worker_ranks: Optional[List[int]] = None,
        flatten_local=False,
        squeeze_dims: Optional[List[int]] = None,
    ):
        self.shape: List[int] = shape
        self.worker_shape: List[int] = worker_shape
        self.ranks: Optional[List[int]] = ranks
        self.worker_ranks: List[int] = worker_ranks if worker_ranks is not None else list(range(len(worker_shape)))
        self.flatten_local: bool = flatten_local
        self.squeeze_dims: Optional[List[int]] = squeeze_dims if squeeze_dims is not None else []

        # if squeeze_dims is not None:
        #     raise NotImplementedError()

        if flatten_local and self.ranks is None:
            self.ranks = list(range(len(shape)))

        self.origin_local_shape: List[int] = [max(a // b, 1) for a, b in zip(shape, worker_shape)]
        self.flatten_local_shape: List[int] = [prod(self.origin_local_shape)]

        if not flatten_local and self.ranks is not None:
            raise ValueError('When special ranks are given, flatten_local must be True')

        assert all(a % b == 0 or b % a == 0 for a, b in zip(self.shape, self.worker_shape))

    def __eq__(self, other):
        assert isinstance(other, TileLayout)
        if not isinstance(other, AtomLayout):
            return False

        if (self.ranks is None) ^ (other.ranks is None):
            return False

        if self.ranks and not same_list(self.ranks, other.ranks):
            return False

        return isinstance(other, AtomLayout) and (
            same_list(self.shape, other.shape)
            and same_list(self.worker_shape, other.worker_shape)
            and same_list(self.worker_ranks, other.worker_ranks)
            and self.flatten_local == other.flatten_local
            and same_list(self.squeeze_dims, other.squeeze_dims)
        )

    def __hash__(self):
        return super().__hash__()

    def num_workers(self) -> int:
        return prod(self.worker_shape)

    def local_shape(self) -> List[int]:
        if self.flatten_local:
            return self.flatten_local_shape
        else:
            return self.origin_local_shape

    def logical_shape(self) -> List[int]:
        if self.squeeze_dims:
            return [d for i, d in enumerate(self.shape) if i not in self.squeeze_dims]
        return self.shape

    def local2logical(self, local_indices: List[Expr], worker: Optional[Expr] = None) -> Tuple[List[Expr], Expr]:
        from hidet.ir.utils.index_transform import index_deserialize

        if worker is None:
            from hidet.ir.primitives.cuda import threadIdx

            worker = threadIdx.x

        if self.flatten_local:
            assert len(local_indices) == 1
            local_indices = index_deserialize(local_indices[0], self.origin_local_shape, ranks=self.ranks)

        worker_indices = index_deserialize(worker, self.worker_shape, ranks=self.worker_ranks)
        logical_indices = []
        not_duplicated = boolean.true
        for i, (a, b) in enumerate(zip(self.shape, self.worker_shape)):
            if a < b:
                c = worker_indices[i] % a
                not_duplicated = logical_and(not_duplicated, worker_indices[i] < a)
            elif a > b:
                c = worker_indices[i] + local_indices[i] * b
            else:
                c = worker_indices[i]
            logical_indices.append(c)
        if self.squeeze_dims:
            logical_indices = [logical_indices[i] for i in range(len(self.shape)) if i not in self.squeeze_dims]

        return logical_indices, not_duplicated

    def logical2local(self, logical_indices: List[Expr], worker: Optional[Expr] = None) -> Tuple[List[Expr], Expr]:
        from hidet.ir.utils.index_transform import index_deserialize, index_serialize

        if worker is None:
            from hidet.ir.primitives.cuda import threadIdx

            worker = threadIdx.x

        if self.squeeze_dims:
            expanded_indices = []
            j = 0
            for i in range(len(self.shape)):
                if i not in self.squeeze_dims:
                    expanded_indices.append(logical_indices[j])
                    j += 1
                else:
                    expanded_indices.append(0)
            logical_indices = expanded_indices

        worker_indices = index_deserialize(worker, self.worker_shape, ranks=self.worker_ranks)
        local_indices = []
        is_valid = boolean.true
        for i, (a, b) in enumerate(zip(self.shape, self.worker_shape)):
            if a < b:
                # logical extent: ----
                #  worker extent: --------
                local_indices.append(int32.zero)
                is_valid = logical_and(is_valid, equal(worker_indices[i] % a, logical_indices[i]))
            elif a > b:
                # logical extent: --------
                #  worker extent: ----
                local_indices.append(logical_indices[i] // b)
                is_valid = logical_and(is_valid, equal(logical_indices[i] % b, worker_indices[i]))
            else:
                # logical extent: --------
                #  worker extent: --------
                local_indices.append(int32.zero)
                is_valid = logical_and(is_valid, equal(logical_indices[i], worker_indices[i]))
        if self.flatten_local:
            local_indices = [index_serialize(indices=local_indices, shape=self.origin_local_shape, ranks=self.ranks)]
        return local_indices, is_valid

    def representation(self) -> str:
        items = ['{}'.format(self.shape), 'worker={}'.format(self.worker_shape)]
        if self.ranks is not None and not same_list(self.ranks, list(range(len(self.shape)))):
            items.append('ranks={}'.format(self.ranks))
        if self.worker_ranks is not None and not same_list(self.worker_ranks, list(range(len(self.worker_shape)))):
            items.append('worker_ranks={}'.format(self.worker_ranks))
        if not self.flatten_local:
            items.append('flatten={}'.format(self.flatten_local))
        if not self.squeeze_dims:
            items.append('squeeze={}'.format(self.squeeze_dims))
        return 'atom({})'.format(', '.join(items))


class RepeatLayout(AtomLayout):
    def __init__(self, shape: List[int], ranks: Optional[List[int]] = None, flatten_local=True):
        super().__init__(
            shape=shape, worker_shape=[1 for _ in range(len(shape))], ranks=ranks, flatten_local=flatten_local
        )

    def representation(self) -> str:
        items = ['{}'.format(self.shape)]
        if self.ranks is not None and not same_list(self.ranks, list(range(len(self.shape)))):
            items.append('ranks={}'.format(self.ranks))
        if not self.flatten_local:
            items.append('flatten={}'.format(self.flatten_local))
        return 'repeat({})'.format(', '.join(items))


class SpatialLayout(AtomLayout):
    def __init__(self, shape: List[int], ranks: Optional[List[int]] = None, flatten_local=True):
        super().__init__(shape=shape, worker_shape=shape, worker_ranks=ranks, flatten_local=flatten_local)

    def representation(self) -> str:
        items = ['{}'.format(self.shape)]
        if self.worker_ranks is not None and not same_list(self.worker_ranks, list(range(len(self.shape)))):
            items.append('ranks={}'.format(self.worker_ranks))
        if not self.flatten_local:
            items.append('flatten={}'.format(self.flatten_local))
        return 'spatial({})'.format(', '.join(items))


class LocalFlatLayout(TileLayout):
    def __init__(self, layout: TileLayout, ranks: Optional[List[int]] = None):
        self.layout: TileLayout = layout
        self.ranks: List[int] = ranks if ranks is not None else list(range(len(layout.local_shape())))

    def __eq__(self, other):
        assert isinstance(other, TileLayout)
        return isinstance(other, LocalFlatLayout) and self.layout == other.layout and same_list(self.ranks, other.ranks)

    def __hash__(self):
        return super().__hash__()

    def num_workers(self) -> int:
        return self.layout.num_workers()

    def local_shape(self) -> List[int]:
        return [prod(self.layout.local_shape())]

    def logical_shape(self) -> List[int]:
        return self.layout.logical_shape()

    def local2logical(self, local_indices: List[Expr], worker: Optional[Expr] = None) -> Tuple[List[Expr], Expr]:
        from hidet.ir.utils.index_transform import index_deserialize

        local_shape: List[int] = self.layout.local_shape()
        local_indices = index_deserialize(local_indices[0], local_shape, ranks=self.ranks)
        logical_indices, not_duplicated = self.layout.local2logical(local_indices, worker)
        return logical_indices, not_duplicated

    def logical2local(self, logical_indices: List[Expr], worker: Optional[Expr] = None) -> Tuple[List[Expr], Expr]:
        from hidet.ir.utils.index_transform import index_serialize

        local_shape: List[int] = self.layout.local_shape()
        local_indices, is_valid = self.layout.logical2local(logical_indices, worker)
        serialized_index = index_serialize(local_indices, local_shape, ranks=self.ranks)
        return [serialized_index], is_valid


class ComposedLayout(TileLayout):
    def __init__(self, outer: TileLayout, inner: TileLayout):
        self.outer: TileLayout = outer
        self.inner: TileLayout = inner

        if len(self.outer.logical_shape()) != len(self.inner.logical_shape()):
            raise ValueError('The logical shape of the outer layout and the inner layout must have the same rank')
        if len(outer.local_shape()) != len(inner.local_shape()):
            raise ValueError('The local shape of the outer layout and the inner layout must have the same rank')

    def __eq__(self, other):
        assert isinstance(other, TileLayout)
        return isinstance(other, ComposedLayout) and self.outer == other.outer and self.inner == other.inner

    def __hash__(self):
        return super().__hash__()

    def num_workers(self) -> int:
        return self.outer.num_workers() * self.inner.num_workers()

    def local_shape(self) -> List[int]:
        return [a * b for a, b in zip(self.outer.local_shape(), self.inner.local_shape())]

    def logical_shape(self) -> List[int]:
        return [a * b for a, b in zip(self.outer.logical_shape(), self.inner.logical_shape())]

    def local2logical(self, local_indices: List[Expr], worker: Optional[Expr] = None) -> Tuple[List[Expr], Expr]:
        if worker is None:
            from hidet.ir.primitives.cuda import threadIdx

            worker = threadIdx.x
        outer_worker = worker // self.inner.num_workers()
        inner_worker = worker % self.inner.num_workers()
        outer_local_indices = [a // b for a, b in zip(local_indices, self.inner.local_shape())]
        inner_local_indices = [a % b for a, b in zip(local_indices, self.inner.local_shape())]
        outer_logical, outer_not_duplicated = self.outer.local2logical(outer_local_indices, outer_worker)
        inner_logical, inner_not_duplicated = self.inner.local2logical(inner_local_indices, inner_worker)
        logical_indices = [a * s + b for a, s, b in zip(outer_logical, self.inner.logical_shape(), inner_logical)]
        not_duplicated = logical_and(outer_not_duplicated, inner_not_duplicated)
        return logical_indices, not_duplicated

    def logical2local(self, logical_indices: List[Expr], worker: Optional[Expr] = None) -> Tuple[List[Expr], Expr]:
        if worker is None:
            from hidet.ir.primitives.cuda import threadIdx

            worker = threadIdx.x
        outer_logical = [a // b for a, b in zip(logical_indices, self.inner.logical_shape())]
        inner_logical = [a % b for a, b in zip(logical_indices, self.inner.logical_shape())]
        outer_worker = worker // self.inner.num_workers()
        inner_worker = worker % self.inner.num_workers()
        outer_local, outer_is_valid = self.outer.logical2local(outer_logical, outer_worker)
        inner_local, inner_is_valid = self.inner.logical2local(inner_logical, inner_worker)
        local_indices = [a * b + c for a, b, c in zip(outer_local, self.inner.local_shape(), inner_local)]
        is_valid = logical_and(outer_is_valid, inner_is_valid)
        return local_indices, is_valid

    def representation(self) -> str:
        return '{}.{}'.format(self.outer.representation(), self.inner.representation())


class ExpandLayout(TileLayout):
    def __init__(self, base: TileLayout, dim: int):
        self.base: TileLayout = base
        self.dim: int = dim

        if not 0 <= dim <= len(base.logical_shape()):
            raise ValueError('dim must be in [0, {}], got {}'.format(len(base.logical_shape()), dim))

    def __eq__(self, other):
        assert isinstance(other, TileLayout)
        return isinstance(other, ExpandLayout) and self.base == other.base and self.dim == other.dim

    def __hash__(self):
        return super().__hash__()

    def num_workers(self) -> int:
        return self.base.num_workers()

    def local_shape(self) -> List[int]:
        return self.base.local_shape()

    def logical_shape(self) -> List[int]:
        shape: List[int] = self.base.logical_shape().copy()
        shape.insert(self.dim, 1)
        return shape

    def local2logical(self, local_indices: List[Expr], worker: Optional[Expr] = None) -> Tuple[List[Expr], Expr]:
        logical_indices, not_duplicated = self.base.local2logical(local_indices, worker)
        logical_indices.insert(self.dim, int32.zero)
        return logical_indices, not_duplicated

    def logical2local(self, logical_indices: List[Expr], worker: Optional[Expr] = None) -> Tuple[List[Expr], Expr]:
        logical_indices.pop(self.dim)
        return self.base.logical2local(logical_indices, worker)


class SliceLayout(TileLayout):
    def __init__(self, base: TileLayout, dim: int, extent: int):
        self.base: TileLayout = base
        self.dim: int = dim
        self.extent: int = extent

        # todo: check we can slice on the given dim
        # base layout can be sliced only when the given dimension is independent (or separable) like
        # dim=1: layout(a, b, c) = layout(a, 0, c) + layout(0, b, 0)
        # dim=0: layout(a, b, c) = layout(a, 0, 0) + layout(0, b, c)

    def __eq__(self, other):
        return (
            isinstance(other, SliceLayout)
            and self.base == other.base
            and self.dim == other.dim
            and self.extent == other.extent
        )

    def num_workers(self) -> int:
        return self.base.num_workers()

    def logical_shape(self) -> List[int]:
        base_shape: List[int] = self.base.logical_shape()
        base_shape[self.dim] = self.extent
        return base_shape

    def local_shape(self) -> List[int]:
        return self.base.local_shape()

    def local2logical(self, local_indices: List[Expr], worker: Optional[Expr] = None) -> Tuple[List[Expr], Expr]:
        logical, not_duplicated = self.base.local2logical(local_indices, worker)
        return logical, not_duplicated

    def logical2local(self, logical_indices: List[Expr], worker: Optional[Expr] = None) -> Tuple[List[Expr], Expr]:
        logical_indices[self.dim] = logical_indices[self.dim]
        return self.base.logical2local(logical_indices, worker)


class SwizzleLayout(TileLayout):
    def __init__(self, base: TileLayout, dim: int, regards_dim: Optional[int] = None, log_step: int = 0):
        self.base: TileLayout = base
        self.dim: int = dim
        if regards_dim is None:
            if len(base.logical_shape()) != 2:
                raise ValueError(
                    'Optional regards_dim is only available for 2-rank layout, '
                    'got layout with shape {}.'.format(base.logical_shape())
                )
            self.regards_dim = 1 - dim
        else:
            self.regards_dim = regards_dim
        self.log_step = log_step

        if self.dim == self.regards_dim:
            raise ValueError(
                'The swizzle dim and regards dim can not be the same, got {} and {}'.format(self.dim, self.regards_dim)
            )
        if not is_power_of_two(self.base.logical_shape()[self.dim]):
            raise ValueError(
                'The swizzled dim {} must be a power of 2, got length {}'.format(
                    self.dim, self.logical_shape()[self.dim]
                )
            )

    def __eq__(self, other):
        assert isinstance(other, TileLayout)
        return (
            isinstance(other, SwizzleLayout)
            and self.base == other.base
            and self.dim == other.dim
            and self.regards_dim == other.regards_dim
            and self.log_step == other.log_step
        )

    def __hash__(self):
        return super().__hash__()

    def representation(self) -> str:
        return 'swizzle({}, dim={}, regards_dim={}, log_step={})'.format(
            self.base.representation(), self.dim, self.regards_dim, self.log_step
        )

    def _get_xor_index(self, indices: List[Expr]) -> Expr:
        indices = list(indices)  # copy
        shape: List[int] = self.base.logical_shape()
        regards_index = indices[self.regards_dim] // (2**self.log_step)
        regards_extent = shape[self.regards_dim] // (2**self.log_step)
        if regards_extent > shape[self.dim]:
            regards_index = regards_index % shape[self.dim]
        return regards_index

    def _swizzle(self, indices: List[Expr]) -> List[Expr]:
        indices[self.dim] = indices[self.dim] ^ self._get_xor_index(indices)
        return indices

    def num_workers(self) -> int:
        return self.base.num_workers()

    def local_shape(self) -> List[int]:
        return self.base.local_shape()

    def logical_shape(self) -> List[int]:
        return self.base.logical_shape()

    def local2logical(self, local_indices: List[Expr], worker: Optional[Expr] = None) -> Tuple[List[Expr], Expr]:
        logical_indices, not_duplicated = self.base.local2logical(local_indices, worker)
        logical_indices = self._swizzle(logical_indices)
        return logical_indices, not_duplicated

    def logical2local(self, logical_indices: List[Expr], worker: Optional[Expr] = None) -> Tuple[List[Expr], Expr]:
        logical_indices = self._swizzle(logical_indices)
        return self.base.logical2local(logical_indices, worker)


class ParameterizedTileLayout(TileLayout):
    def __init__(self, layout: TileLayout = None):
        self.layout: TileLayout = layout

    def representation(self) -> str:
        return self.layout.representation()

    def num_workers(self) -> int:
        return self.layout.num_workers()

    def local_shape(self) -> List[int]:
        return self.layout.local_shape()

    def logical_shape(self) -> List[int]:
        return self.layout.logical_shape()

    def local2logical(self, local_indices: List[Expr], worker: Optional[Expr] = None) -> Tuple[List[Expr], Expr]:
        return self.layout.local2logical(local_indices, worker)

    def logical2local(self, logical_indices: List[Expr], worker: Optional[Expr] = None) -> Tuple[List[Expr], Expr]:
        return self.layout.logical2local(logical_indices, worker)


class ReduceLayout(ParameterizedTileLayout):
    def __init__(self, parent: TileLayout, dim: int, keep_dim: bool):
        from hidet.ir.tile.tools.layout_reducer import reduce_layout

        self.parent: TileLayout = parent
        self.dim: int = dim
        self.keep_dim: bool = keep_dim
        super().__init__(layout=reduce_layout(parent, dim=dim, keep_dim=keep_dim))

    def __eq__(self, other):
        assert isinstance(other, TileLayout)
        return (
            isinstance(other, ReduceLayout)
            and self.parent == other.parent
            and self.dim == other.dim
            and self.keep_dim == other.keep_dim
        )

    def __hash__(self):
        return super().__hash__()

    def representation(self) -> str:
        return 'reduce({}, dim={}, keep={})'.format(self.parent.representation(), self.dim, self.keep_dim)


def repeat(*shape: int, ranks: Optional[List[int]] = None, flatten_local=True) -> TileLayout:
    return RepeatLayout(list(shape), ranks=ranks, flatten_local=flatten_local)


def spatial(*shape: int, ranks: Optional[List[int]] = None, flatten_local=True) -> TileLayout:
    return SpatialLayout(list(shape), ranks=ranks, flatten_local=flatten_local)


def atom(
    *shape,
    workers: List[int],
    ranks: Optional[List[int]] = None,
    worker_ranks: Optional[List[int]] = None,
    flatten_local=True,
    squeeze_dims: Optional[List[int]] = None,
):
    if not squeeze_dims and (worker_ranks is None or same_list(worker_ranks, list(range(len(workers))))):
        if all(w == 1 for w in workers):
            return RepeatLayout(shape=list(shape), ranks=ranks, flatten_local=flatten_local)
        if same_list(shape, workers):
            return SpatialLayout(shape=list(shape), ranks=ranks, flatten_local=flatten_local)
    return AtomLayout(
        shape=list(shape),
        worker_shape=workers,
        ranks=ranks,
        worker_ranks=worker_ranks,
        flatten_local=flatten_local,
        squeeze_dims=squeeze_dims,
    )


class BlockLayout(ParameterizedTileLayout):
    def __init__(
        self, shape: List[int], warps_per_block: List[int], thread_per_warp: List[int], size_per_thread: List[int]
    ):
        self.shape: List[int] = shape
        self.warps_per_block: List[int] = warps_per_block
        self.thread_per_warp: List[int] = thread_per_warp
        self.size_per_thread: List[int] = size_per_thread
        self.thread_shape: List[int] = [min(a, b) for a, b in zip(self.shape, self.size_per_thread)]
        self.warp_shape: List[int] = [
            min(a, b) for a, b in zip(self.shape, Vector(self.size_per_thread) * self.thread_per_warp)
        ]
        self.block_shape: List[int] = [
            min(a, b)
            for a, b in zip(self.shape, Vector(self.size_per_thread) * self.thread_per_warp * self.warps_per_block)
        ]
        self.layout_shape: List[int] = list(Vector(self.warps_per_block) * self.thread_per_warp * self.size_per_thread)
        super().__init__(
            layout=(
                atom(
                    *Vector(self.shape) // self.block_shape,
                    workers=[1 for _ in range(len(self.shape))],
                    flatten_local=False,
                )
                .atom(*Vector(self.block_shape) // self.warp_shape, workers=self.warps_per_block, flatten_local=False)
                .atom(*Vector(self.warp_shape) // self.thread_shape, workers=self.thread_per_warp, flatten_local=False)
                .atom(*self.thread_shape, workers=[1 for _ in range(len(self.shape))], flatten_local=False)
            )
        )

    def __eq__(self, other):
        assert isinstance(other, TileLayout)
        return (
            isinstance(other, BlockLayout)
            and same_list(self.shape, other.shape)
            and same_list(self.size_per_thread, other.size_per_thread)
            and same_list(self.thread_per_warp, other.thread_per_warp)
            and same_list(self.warps_per_block, other.warps_per_block)
        )

    def __hash__(self):
        return super().__hash__()

    @staticmethod
    def from_shape(shape: List[int], num_warps: int, size_per_thread: Optional[List[int]] = None):
        if not is_power_of_two(prod(shape)):
            raise ValueError(f"The tensor must have a power of 2 number of elements, got {prod(shape)}")
        if size_per_thread is not None and not is_power_of_two(prod(size_per_thread)):
            raise ValueError(f"size_per_thread must have a power of 2 number of elements, got {prod(size_per_thread)}")
        if size_per_thread is None:
            size_per_thread = [1] * len(shape)
        if len(size_per_thread) != len(shape):
            raise ValueError(f"size_per_thread must have the same length as shape, got {size_per_thread}")
        orig_shape = shape
        shape = [max(extent // size, 1) for extent, size in zip(shape, size_per_thread)]
        thread_per_warp = []
        warps_per_block = []
        remaining_threads = 32
        remaining_warps = num_warps
        for extent in reversed(shape):  # from innermost to outermost
            if extent <= remaining_threads:
                assert remaining_threads % extent == 0
                thread_per_warp.append(extent)
                warps_per_block.append(1)
                remaining_threads //= extent
            elif extent <= remaining_threads * remaining_warps:
                assert extent % remaining_threads == 0
                assert remaining_warps % (extent // remaining_threads) == 0
                allocated_warps = extent // remaining_threads
                thread_per_warp.append(remaining_threads)
                warps_per_block.append(allocated_warps)
                remaining_threads = 1
                remaining_warps //= allocated_warps
            else:
                thread_per_warp.append(remaining_threads)
                warps_per_block.append(remaining_warps)
                remaining_threads = 1
                remaining_warps = 1

        while remaining_threads > 1:
            assert remaining_threads % 2 == 0
            thread_per_warp[argmin(thread_per_warp)] *= 2
            remaining_threads //= 2

        while remaining_warps > 1:
            assert remaining_warps % 2 == 0
            warps_per_block[argmin(warps_per_block)] *= 2
            remaining_warps //= 2

        thread_per_warp = list(reversed(thread_per_warp))
        warps_per_block = list(reversed(warps_per_block))

        assert prod(warps_per_block) == num_warps
        assert prod(thread_per_warp) == 32

        return BlockLayout(orig_shape, warps_per_block, thread_per_warp, size_per_thread)


class FlattenBlockLayout(TileLayout):
    """
    todo: replace this with a more general layout
    """

    def __init__(self, parent: BlockLayout, axis: int):
        self.parent: BlockLayout = parent
        self.axis: int = axis

        shape = list(self.parent.shape)
        shape[axis] = 1
        self.flat_layout = BlockLayout(shape, parent.warps_per_block, parent.thread_per_warp, parent.size_per_thread)

    def __eq__(self, other):
        assert isinstance(other, TileLayout)
        return isinstance(other, FlattenBlockLayout) and self.parent == other.parent and self.axis == other.axis

    def __hash__(self):
        return super().__hash__()

    def num_workers(self) -> int:
        return self.flat_layout.num_workers()

    def local_shape(self) -> List[int]:
        return self.flat_layout.local_shape()

    def logical_shape(self) -> List[int]:
        shape = self.flat_layout.logical_shape()
        assert shape[self.axis] == 1
        return shape[: self.axis] + shape[self.axis + 1 :]

    def local2logical(self, local_indices: List[Expr], worker: Optional[Expr] = None) -> Tuple[List[Expr], Expr]:
        logical_indices, not_duplicated = self.flat_layout.local2logical(local_indices, worker)
        logical_indices = logical_indices[: self.axis] + logical_indices[self.axis + 1 :]
        return logical_indices, not_duplicated

    def logical2local(self, logical_indices: List[Expr], worker: Optional[Expr] = None) -> Tuple[List[Expr], Expr]:
        logical_indices = logical_indices[: self.axis] + [int32.zero] + logical_indices[self.axis :]
        return self.flat_layout.logical2local(logical_indices, worker)


class DotOperandLayout(ParameterizedTileLayout):
    pass


class BlockDotOperandLayout(DotOperandLayout):
    def __init__(self, parent: BlockLayout, k_size: int, op_idx: int):
        self.parent: BlockLayout = parent
        self.op_idx: int = op_idx

        shape: List[int] = parent.shape

        if op_idx == 0:
            layout = repeat(1, k_size, flatten_local=False) * BlockLayout(
                shape=[shape[0], 1],
                warps_per_block=parent.warps_per_block,
                thread_per_warp=parent.thread_per_warp,
                size_per_thread=[parent.size_per_thread[0], 1],
            )
        else:
            layout = repeat(k_size, 1, flatten_local=False) * BlockLayout(
                shape=[1, shape[1]],
                warps_per_block=parent.warps_per_block,
                thread_per_warp=parent.thread_per_warp,
                size_per_thread=[1, parent.size_per_thread[1]],
            )
        super().__init__(layout)

    def __eq__(self, other):
        assert isinstance(other, TileLayout)
        return isinstance(other, BlockDotOperandLayout) and self.parent == other.parent and self.op_idx == other.op_idx

    def __hash__(self):
        return super().__hash__()


class MmaOutputLayout(ParameterizedTileLayout):
    def __init__(self, num_warps: int, m_size: int, n_size: int, k_size: int, config):
        from hidet.ir.tile.cuda.mma_configs import TileMmaConfig

        self.num_warps: int = num_warps
        self.m_size: int = m_size  # the size of the tile
        self.n_size: int = n_size
        self.k_size: int = k_size
        self.config: TileMmaConfig = config
        assert isinstance(config, TileMmaConfig)

        a_shape = config.a_layout.logical_shape()
        c_shape = config.c_layout.logical_shape()
        self.inst_m: int = c_shape[0]  # the size of each mma instruction
        self.inst_n: int = c_shape[1]
        self.inst_k: int = a_shape[1]

        self.count_m: int = m_size // self.inst_m  # the number of instructions for each dimension
        self.count_n: int = n_size // self.inst_n
        self.count_k: int = k_size // self.inst_k

        warps_m, warps_n = self._distribute(num_warps, [self.count_m, self.count_n])
        self.warps_m = warps_m  # the number of warps for each dimension
        self.warps_n = warps_n
        self.warps_k = 1
        self.repeat_m = self.count_m // warps_m
        self.repeat_n = self.count_n // warps_n
        self.repeat_k = self.count_k // 1

        assert m_size % self.inst_m == 0 and n_size % self.inst_n == 0 and k_size % self.inst_k == 0

        super().__init__(
            layout=(spatial(self.warps_m, self.warps_n) * repeat(self.repeat_m, self.repeat_n) * config.c_layout)
        )

    def __eq__(self, other):
        assert isinstance(other, TileLayout)
        return (
            isinstance(other, MmaOutputLayout)
            and self.m_size == other.m_size
            and self.n_size == other.n_size
            and self.config == other.config
        )

    def __hash__(self):
        return super().__hash__()

    def _distribute(self, num_warps: int, counts: List[int]) -> List[int]:
        assert is_power_of_two(num_warps)
        if num_warps > prod(counts):
            raise ValueError('Can not distribute {} warps to work on {} mma'.format(num_warps, prod(counts)))
        warps = [1 for _ in range(len(counts))]
        while num_warps > 1:
            idx = argmax([a // b for a, b in zip(counts, warps)])
            warps[idx] *= 2
            num_warps //= 2

        return warps


class MmaDotOperandLayout(DotOperandLayout):
    def __init__(self, mma: MmaOutputLayout, op_idx: int):
        self.mma: MmaOutputLayout = mma
        self.op_idx: int = op_idx

        if op_idx == 0:
            layout = (
                atom(mma.warps_m, 1, workers=[mma.warps_m, mma.warps_n])
                * repeat(mma.repeat_m, mma.repeat_k)
                * mma.config.a_layout
            )
        else:
            layout = (
                atom(1, mma.warps_n, workers=[mma.warps_m, mma.warps_n])
                * repeat(mma.repeat_k, mma.repeat_n)
                * mma.config.b_layout
            )
        super().__init__(layout)

    def __eq__(self, other):
        assert isinstance(other, TileLayout)
        return (
            isinstance(other, MmaDotOperandLayout)
            and self.mma == other.mma
            and self.op_idx == other.op_idx
            and self.mma.k_size == other.mma.k_size
        )

    def __hash__(self):
        return super().__hash__()

    def shared_layout(self):
        return MmaDotOperandSharedLayout(self)


class MmaDotOperandSharedLayout(ParameterizedTileLayout):
    def __init__(self, mma_operand: MmaDotOperandLayout):
        self.mma_operand: MmaDotOperandLayout = mma_operand
        super().__init__(layout=self.generate_composed_layout(mma_operand))

    def __eq__(self, other):
        assert isinstance(other, TileLayout)
        return isinstance(other, MmaDotOperandSharedLayout) and self.mma_operand == other.mma_operand

    def __hash__(self):
        return super().__hash__()

    def generate_composed_layout(self, mma_operand: MmaDotOperandLayout):
        from hidet.ir.type import sizeof

        mma = mma_operand.mma
        op_idx = mma_operand.op_idx

        dtype = mma.config.in_dtype
        dtype_bytes: int = sizeof(dtype)
        vec_size: int = 16 // dtype_bytes
        if op_idx == 0:  # A
            assert mma.m_size % 8 == 0 and mma.k_size % vec_size == 0
            rows = mma.m_size // 8
            cols = mma.k_size // vec_size
        else:  # B
            assert mma.k_size % 8 == 0 and mma.n_size % vec_size == 0
            rows = mma.k_size // 8
            cols = mma.n_size // vec_size

        # pylint: disable=pointless-string-statement
        """
        repeat(rows, cols).swizzle(dim=1, log_step=*)  (all position are mod by 8)
        0   0 1   0 1 2 3
        1   2 3   4 5 6 7
        2   4 5   1 0 3 2
        3   6 7   5 4 7 6
        4   1 0   2 3 0 1
        5   3 2   6 7 4 5
        6   5 4   3 2 1 0
        7   7 6   7 6 5 4

        each block (a 1x8 block of vector) is bank-conflict free
        """
        if cols == 1:
            return repeat(rows, 1) * repeat(8, cols).swizzle(dim=1, log_step=4) * repeat(1, vec_size)
        elif cols == 2:
            return repeat(rows, 1) * repeat(8, cols).swizzle(dim=1, log_step=2) * repeat(1, vec_size)
        elif cols == 4:
            return repeat(rows, 1) * repeat(8, cols).swizzle(dim=1, log_step=1) * repeat(1, vec_size)
        else:
            return repeat(rows, 1) * repeat(8, cols).swizzle(dim=1, log_step=0) * repeat(1, vec_size)
