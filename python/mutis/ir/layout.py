from typing import List, Dict, Tuple, Optional, Union, Set
import io
import itertools
from collections import defaultdict
from hidet.ir.expr import Expr, logical_and
from hidet.ir.dtypes import int32, boolean
from hidet.ir.utils.index_transform import index_add, index_multiply, index_mod, index_divide
from hidet.ir.utils.index_transform import index_deserialize, index_serialize
from hidet.utils import prod, same_list, gcd, is_power_of_two
from mutis.utils.py import ranked_product, factorize_decomposition


class LayoutOperationFailed(Exception):
    pass


class Layout:
    def __init__(self, shape: List[int], local_size: int, num_workers: int):
        self.shape: List[int] = shape
        self.local_size: int = local_size
        self.num_workers: int = num_workers
        self.size: int = prod(shape)

    def __repr__(self):
        return str(self)

    def __mul__(self, other):
        assert isinstance(other, Layout)
        return compose(self, other)

    def __truediv__(self, other):
        assert isinstance(other, Layout)
        return divide(self, other)

    def __floordiv__(self, other):
        assert isinstance(other, Layout)
        return divide(self, other)

    def summary(self):
        return '{} size={} workers={}'.format(self.shape, self.local_size, self.num_workers)

    def quick_equal(self, other):
        return (
            same_list(self.shape, other.shape)
            and self.local_size == other.local_size
            and self.num_workers == other.num_workers
        )

    def semantics_equal(self, other):
        assert isinstance(other, Layout)
        lhs, rhs = self, other

        # quick check
        if (
            not same_list(lhs.shape, rhs.shape)
            or lhs.local_size != rhs.local_size
            or lhs.num_workers != rhs.num_workers
        ):
            return False

        # deep check
        for worker in range(self.num_workers):
            for local in range(self.local_size):
                lhs_global = [int(v) for v in self.local2global(local, int32(worker))]
                rhs_global = [int(v) for v in other.local2global(local, int32(worker))]
                if not same_list(lhs_global, rhs_global):
                    return False
        return True

    def astext(self, verbose=False):
        shape = self.shape
        if len(shape) not in [1, 2]:
            raise ValueError('Cannot visualize layout with rank {} (shape={})'.format(len(shape), shape))
        grid: Dict[Tuple[int, ...], List[Tuple[int, int, bool]]] = defaultdict(list)

        for worker in range(self.num_workers):
            for local_indices in range(self.local_size):
                logical_indices = self.local2global(local_indices, int32(worker))
                grid[tuple(int(v) for v in logical_indices)].append((int(worker), int(local_indices), True))

        str_grid: Dict[Tuple[int, ...], str] = {}
        for indices in grid:
            items = []
            for worker, local_indices, _ in grid[indices]:
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
        width = max(width, max(len(str(d)) for d in self.shape)) + 1
        fmt = '{:>' + str(width) + '}'
        f = io.StringIO()

        idx_width = max(len(str(d)) for d in self.shape) + 1
        idx_fmt = '{:>' + str(idx_width) + '}'

        # print the logical shape, num of workers, and local extent
        print('          shape: {}'.format(self.shape), file=f)
        print('     local size: {}'.format(self.local_size), file=f)
        print(' num of workers: {}'.format(self.num_workers), file=f)
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

    def atom(self, *shape, workers: List[int], ranks=None, worker_ranks=None):
        return compose(self, atom(*shape, workers=workers, ranks=ranks, worker_ranks=worker_ranks))

    def spatial(self, *shape, ranks=None):
        return compose(self, spatial(*shape))

    def column_spatial(self, *shape):
        return compose(self, column_spatial(*shape))

    def repeat(self, *shape, ranks=None):
        return compose(self, repeat(*shape, ranks=ranks))

    def column_repeat(self, *shape):
        return compose(self, column_repeat(*shape))

    def is_simple(self) -> bool:
        """
        a layout is simple iff. each element is store in a single worker by one time.
        """
        raise NotImplementedError()

    def global2local(self, global_indices: List[Expr], worker: Expr) -> Expr:
        """
        Get the local index of the element stored in the given worker.
        """
        raise NotImplementedError()

    def global2worker(self, global_indices: List[Expr]) -> List[Expr]:
        """
        Get the workers that are storing the given elements.
        """
        raise NotImplementedError()

    def local2global(self, local_index: Union[Expr, int], worker: Expr) -> List[Expr]:
        """
        Get the global indices the element corresponds to (worker, local_index).
        """
        raise NotImplementedError()

    def is_first_occurrence(self, local_index: Union[Expr, int], worker: Expr) -> Expr:
        """
        Whether the global element stored in (worker, local_index) is the first occurrence in the layout
        (e.g., either no other workers are storing the same element, or this worker is the smallest worker among
        all the workers).
        """
        raise NotImplementedError()

    def is_valid(self, global_indices: List[Expr], worker: Expr) -> Expr:
        """
        Whether the global element is stored in the given worker.
        """
        raise NotImplementedError()


class AtomLayout(Layout):
    def __init__(self, shape: List[int], workers: List[int], ranks: List[int], worker_ranks: List[int]):
        super().__init__(
            shape, local_size=prod([a // b if a >= b else 1 for a, b in zip(shape, workers)]), num_workers=prod(workers)
        )
        self.workers: List[int] = workers
        self.ranks: List[int] = ranks
        self.worker_ranks: List[int] = worker_ranks

    def __str__(self):
        num_dims = len(self.shape)
        shape_str = ', '.join(str(v) for v in self.shape)
        if self.is_repeat():
            if same_list(self.ranks, range(num_dims)):
                return 'repeat({})'.format(shape_str)
            if same_list(self.ranks, list(reversed(range(num_dims)))):
                return 'column_repeat({})'.format(shape_str)
            return 'repeat({}, ranks=[{}])'.format(shape_str, ', '.join(str(v) for v in self.ranks))
        elif self.is_spatial():
            if same_list(self.worker_ranks, range(num_dims)):
                return 'spatial({})'.format(shape_str)
            if same_list(self.worker_ranks, list(reversed(range(num_dims)))):
                return 'column_spatial({})'.format(shape_str)
            return 'spatial({}, ranks=[{}])'.format(shape_str, ', '.join(str(v) for v in self.worker_ranks))
        else:
            items = [
                ', '.join(str(v) for v in self.shape),
                'workers=[{}]'.format(', '.join(str(v) for v in self.workers)),
            ]
            if not same_list(self.ranks, range(num_dims)):
                items.append('ranks=[{}]'.format(', '.join(str(v) for v in self.ranks)))
            if not same_list(self.worker_ranks, range(num_dims)):
                items.append('worker_ranks=[{}]'.format(', '.join(str(v) for v in self.worker_ranks)))
            return 'atom({})'.format(', '.join(items))

    def is_repeat(self) -> bool:
        return all(a == 1 for a in self.workers)

    def is_spatial(self) -> bool:
        return all(a == b for a, b in zip(self.shape, self.workers))

    def is_simple(self) -> bool:
        return all(a % b == 0 for a, b in zip(self.shape, self.workers))

    def global2local(self, global_indices: List[Expr], worker: Expr) -> Expr:
        local_indices = []
        local_shape = [max(a // b, 1) for a, b in zip(self.shape, self.workers)]
        for i, global_index in enumerate(global_indices):
            local_indices.append(global_index % local_shape[i])
        return index_serialize(local_indices, local_shape, self.ranks)

    def global2worker(self, global_indices: List[Expr]) -> List[Expr]:
        local_shape = [max(a // b, 1) for a, b in zip(self.shape, self.workers)]
        worker_indices: List[List[Expr]] = []
        for i, global_index in enumerate(global_indices):
            if self.shape[i] >= self.workers[i]:
                # one worker on this dimension
                worker_indices.append([global_index // local_shape[i]])
            else:
                # multiple workers on this dimension
                num_segments = self.workers[i] // self.shape[i]
                segment_size = self.shape[i]
                worker_indices.append([global_index + j * segment_size for j in range(num_segments)])

        ret = []
        for indices in ranked_product(*worker_indices, ranks=self.worker_ranks):
            ret.append(index_serialize(indices, self.workers, self.worker_ranks))
        return ret

    def local2global(self, local_index: Expr, worker: Expr) -> List[Expr]:
        local_shape = [max(a // b, 1) for a, b in zip(self.shape, self.workers)]
        local_indices = index_deserialize(local_index, shape=local_shape, ranks=self.ranks)
        worker_indices = index_deserialize(worker, shape=self.workers, ranks=self.worker_ranks)
        global_indices = []
        for i, (local_index, worker_index) in enumerate(zip(local_indices, worker_indices)):
            if self.shape[i] < self.workers[i]:
                global_indices.append(worker_index % self.shape[i])
            elif self.shape[i] == self.workers[i]:
                global_indices.append(worker_index)
            else:
                global_indices.append(worker_index * local_shape[i] + local_index)
        return global_indices

    def is_first_occurrence(self, local_index: Expr, worker: Expr) -> Expr:
        local_shape = [max(a // b, 1) for a, b in zip(self.shape, self.workers)]
        local_indices = index_deserialize(local_index, shape=local_shape, ranks=self.ranks)
        worker_indices = index_deserialize(worker, shape=self.workers, ranks=self.worker_ranks)
        cond = boolean.true
        for i, (local_index, worker_index) in enumerate(zip(local_indices, worker_indices)):
            if self.shape[i] < self.workers[i]:
                cond = logical_and(worker_index < self.shape[i])
        return cond

    def is_valid(self, global_indices: List[Expr], worker: Expr) -> Expr:
        local_shape = [max(a // b, 1) for a, b in zip(self.shape, self.workers)]
        worker_indices = index_deserialize(worker, shape=self.workers, ranks=self.worker_ranks)
        is_valid = boolean.true
        for i, (global_index, worker_index) in enumerate(zip(global_indices, worker_indices)):
            if self.shape[i] > self.workers[i]:
                is_valid = logical_and(
                    is_valid,
                    worker_index * local_shape[i] <= global_index,
                    global_index < (worker_index + 1) * local_shape[i],
                )
            elif self.shape[i] == self.workers[i]:
                is_valid = logical_and(is_valid, global_index == worker_index)
            else:
                is_valid = logical_and(is_valid, global_index == worker_index % self.shape[i])
        return is_valid


class SqueezeLayout(Layout):
    def __init__(self, base: Layout, dims: List[int]):
        super().__init__(
            shape=[d for i, d in enumerate(base.shape) if i not in dims],
            local_size=base.local_size,
            num_workers=base.num_workers,
        )
        self.base: Layout = base
        self.dims: List[int] = list(dims)
        assert all(base.shape[dim] == 1 for dim in dims), 'Can only squeeze the dims with size != 1'

    def __str__(self):
        return 'squeeze({}, dims={})'.format(str(self.base), self.dims)

    def _get_base_global_indices(self, global_indices: List[Expr]) -> List[Expr]:
        base_global_indices = []
        global_indices = list(global_indices)
        for dim in range(len(self.base.shape)):
            if dim in self.dims:
                base_global_indices.append(0)
            else:
                base_global_indices.append(global_indices.pop(0))
        return base_global_indices

    def is_simple(self) -> bool:
        return self.base.is_simple()

    def global2local(self, global_indices: List[Expr], worker: Expr) -> Expr:
        return self.base.global2local(self._get_base_global_indices(global_indices), worker)

    def global2worker(self, global_indices: List[Expr]) -> List[Expr]:
        return self.base.global2worker(self._get_base_global_indices(global_indices))

    def local2global(self, local_index: Expr, worker: Expr) -> List[Expr]:
        base_global_indices = self.base.local2global(local_index, worker)
        return [base_global_indices[i] for i in range(len(self.base.shape)) if i not in self.dims]

    def is_first_occurrence(self, local_index: Expr, worker: Expr) -> Expr:
        return self.base.is_first_occurrence(local_index, worker=worker)

    def is_valid(self, global_indices: List[Expr], worker: Expr) -> Expr:
        return self.base.is_valid(self._get_base_global_indices(global_indices), worker=worker)


class ExpandLayout(Layout):
    def __init__(self, base: Layout, dims: List[int]):
        shape = []
        cur = 0
        for dim in range(len(base.shape) + len(dims)):
            if dim in dims:
                shape.append(1)
            else:
                shape.append(base.shape[cur])
                cur += 1
        super().__init__(shape=shape, local_size=base.local_size, num_workers=base.num_workers)
        self.base: Layout = base
        self.dims: List[int] = dims

    def __str__(self):
        return 'expand({}, dims={})'.format(self.base, self.dims)

    def is_simple(self) -> bool:
        return self.base.is_simple()

    def global2local(self, global_indices: List[Expr], worker: Expr) -> Expr:
        base_global_indices = [index for dim, index in enumerate(global_indices) if dim not in self.dims]
        return self.base.global2local(base_global_indices, worker)

    def global2worker(self, global_indices: List[Expr]) -> List[Expr]:
        base_global_indices = [index for dim, index in enumerate(global_indices) if dim not in self.dims]
        return self.base.global2worker(base_global_indices)

    def local2global(self, local_index: Expr, worker: Expr) -> List[Expr]:
        base_local_indices = self.base.local2global(local_index, worker)
        global_indices = []
        cur = 0
        for dim in range(len(self.shape)):
            if dim in self.dims:
                global_indices.append(int32.zero)
            else:
                global_indices.append(base_local_indices[cur])
                cur += 1
        return global_indices

    def is_first_occurrence(self, local_index: Expr, worker: Expr) -> Expr:
        return self.base.is_first_occurrence(local_index, worker=worker)

    def is_valid(self, global_indices: List[Expr], worker: Expr) -> Expr:
        base_global_indices = [index for dim, index in enumerate(global_indices) if dim not in self.dims]
        return self.base.is_valid(base_global_indices, worker=worker)


class FlattenLayout(Layout):
    def __init__(self, base: Layout):
        super().__init__(shape=[prod(base.shape)], local_size=base.local_size, num_workers=base.num_workers)
        self.base: Layout = base

    def __str__(self):
        return 'flatten({})'.format(self.base)

    def global2local(self, global_indices: List[Expr], worker: Expr) -> Expr:
        global_indices = index_deserialize(global_indices[0], self.base.shape)
        return self.base.global2local(global_indices, worker)

    def global2worker(self, global_indices: List[Expr]) -> List[Expr]:
        global_indices = index_deserialize(global_indices[0], self.base.shape)
        return self.base.global2worker(global_indices)

    def local2global(self, local_index: Expr, worker: Expr) -> List[Expr]:
        global_indices = self.base.local2global(local_index=local_index, worker=worker)
        return [index_serialize(global_indices, self.base.shape)]

    def is_simple(self) -> bool:
        return self.base.is_simple()

    def is_first_occurrence(self, local_index: Expr, worker: Expr) -> Expr:
        return self.base.is_first_occurrence(local_index, worker)

    def is_valid(self, global_indices: List[Expr], worker: Expr) -> Expr:
        global_indices = index_deserialize(global_indices[0], self.base.shape)
        return self.base.is_valid(global_indices, worker)


class ComposedLayout(Layout):
    def __init__(self, outer: Layout, inner: Layout):
        super().__init__(
            shape=index_multiply(outer.shape, inner.shape),
            local_size=outer.local_size * inner.local_size,
            num_workers=outer.num_workers * inner.num_workers,
        )
        self.outer: Layout = outer
        self.inner: Layout = inner

    def __str__(self):
        def get_compose_list(layout) -> List[Layout]:
            if isinstance(layout, ComposedLayout):
                return get_compose_list(layout.outer) + get_compose_list(layout.inner)
            else:
                return [layout]

        items = [str(layout) for layout in get_compose_list(self)]
        return 'compose({})'.format(', '.join(items))

    def is_simple(self) -> bool:
        return self.outer.is_simple() and self.inner.is_simple()

    def global2local(self, global_indices: List[Expr], worker: Expr) -> Expr:
        outer_global = index_divide(global_indices, self.inner.shape)
        inner_global = index_mod(global_indices, self.inner.shape)
        outer_local = self.outer.global2local(outer_global, worker // self.inner.num_workers)
        inner_local = self.inner.global2local(inner_global, worker % self.inner.num_workers)
        return outer_local * self.inner.local_size + inner_local

    def global2worker(self, global_indices: List[Expr]) -> List[Expr]:
        outer_global = index_divide(global_indices, self.inner.shape)
        inner_global = index_mod(global_indices, self.inner.shape)
        outer_workers = self.outer.global2worker(outer_global)
        inner_workers = self.inner.global2worker(inner_global)
        return [
            outer_worker * self.inner.num_workers + inner_worker
            for outer_worker in outer_workers
            for inner_worker in inner_workers
        ]

    def local2global(self, local_index: Expr, worker: Expr) -> List[Expr]:
        outer_local = local_index // self.inner.local_size
        inner_local = local_index % self.inner.local_size
        outer_global = self.outer.local2global(outer_local, worker // self.inner.num_workers)
        inner_global = self.inner.local2global(inner_local, worker % self.inner.num_workers)
        return index_add(index_multiply(outer_global, self.inner.shape), inner_global)

    def is_first_occurrence(self, local_index: Expr, worker: Expr) -> Expr:
        outer_local = local_index // self.inner.local_size
        inner_local = local_index % self.inner.local_size
        outer_worker = worker // self.inner.num_workers
        inner_worker = worker % self.inner.num_workers
        return logical_and(
            self.outer.is_first_occurrence(outer_local, worker=outer_worker),
            self.inner.is_first_occurrence(inner_local, worker=inner_worker),
        )

    def is_valid(self, global_indices: List[Expr], worker: Expr) -> Expr:
        outer_global = index_divide(global_indices, self.inner.shape)
        inner_global = index_mod(global_indices, self.inner.shape)
        outer_worker = worker // self.inner.num_workers
        inner_worker = worker % self.inner.num_workers
        return logical_and(
            self.outer.is_valid(outer_global, worker=outer_worker),
            self.inner.is_valid(inner_global, worker=inner_worker),
        )


def atom(
    *shape: int, workers: List[int], ranks: Optional[List[int]] = None, worker_ranks: Optional[List[int]] = None
) -> Layout:
    assert all(
        a % b == 0 or b % a == 0 for a, b in zip(shape, workers)
    ), "invalid atom layout with shape {} workers {}".format(shape, workers)

    if ranks is None:
        ranks = list(range(len(shape)))
    if worker_ranks is None:
        worker_ranks = list(range(len(workers)))
    assert set(ranks) == set(list(range(len(shape))))
    assert set(worker_ranks) == set(list(range(len(worker_ranks))))

    return AtomLayout(list(shape), workers=workers, ranks=ranks, worker_ranks=worker_ranks)


def spatial(*shape: int, ranks: Optional[List[int]] = None) -> Layout:
    return atom(*shape, workers=list(shape), ranks=None, worker_ranks=ranks)


def column_spatial(*shape: int) -> Layout:
    return spatial(*shape, ranks=list(reversed(range(len(shape)))))


def repeat(*shape: int, ranks: Optional[List[int]] = None) -> Layout:
    return atom(*shape, workers=[1 for _ in range(len(shape))], ranks=ranks, worker_ranks=None)


def column_repeat(*shape: int) -> Layout:
    return repeat(*shape, ranks=list(reversed(range(len(shape)))))


def compose(outer: Layout, inner: Optional[Layout]) -> Layout:
    if inner is None:
        return outer

    def is_identity(layout):
        if isinstance(layout, AtomLayout) and layout.num_workers == 1 and layout.local_size == 1:
            return True
        return False

    if is_identity(outer):
        return inner
    if is_identity(inner):
        return outer
    if len(outer.shape) < len(inner.shape):
        outer = expand(outer, list(range(len(inner.shape) - len(outer.shape))))
    if len(outer.shape) > len(inner.shape):
        inner = expand(inner, list(range(len(outer.shape) - len(inner.shape))))
    return ComposedLayout(outer, inner)


def squeeze(layout: Layout, dims: List[int]) -> Layout:
    if isinstance(layout, AtomLayout) and all(layout.workers[dim] == 1 for dim in dims):
        ranks = [layout.ranks[i] for i in range(len(layout.shape)) if i not in dims]
        worker_ranks = [layout.worker_ranks[i] for i in range(len(layout.shape)) if i not in dims]
        ranks = [sum(v < ranks[i] for v in ranks) for i in range(len(ranks))]
        worker_ranks = [sum(v < worker_ranks[i] for v in worker_ranks) for i in range(len(worker_ranks))]
        return AtomLayout(
            shape=[d for i, d in enumerate(layout.shape) if i not in dims],
            workers=[d for i, d in enumerate(layout.workers) if i not in dims],
            ranks=ranks,
            worker_ranks=worker_ranks,
        )

    return SqueezeLayout(layout, dims)


def _reduce(layout: Layout, dims: List[int]) -> Layout:
    if isinstance(layout, AtomLayout):
        return AtomLayout(
            shape=[d if i not in dims else 1 for i, d in enumerate(layout.shape)],
            workers=list(layout.workers),
            ranks=layout.ranks,
            worker_ranks=layout.worker_ranks,
        )
    elif isinstance(layout, ComposedLayout):
        return ComposedLayout(outer=_reduce(layout.outer, dims), inner=_reduce(layout.inner, dims))
    elif isinstance(layout, SqueezeLayout):
        base_dims = list(range(len(layout.base.shape)))
        squeezed_dims = [dim for dim in base_dims if dim not in layout.dims]
        new_reduce_dims = [squeezed_dims[dim] for dim in dims]
        return SqueezeLayout(base=_reduce(layout.base, new_reduce_dims), dims=layout.dims)
    elif isinstance(layout, ExpandLayout):
        base_dims_after_expanded = [dim for dim in range(len(layout.shape)) if dim not in layout.dims]
        base_dims_before_expanded = [dim for dim in range(len(layout.base.shape))]
        new_reduce_dims = [a for a, b in zip(base_dims_before_expanded, base_dims_after_expanded) if b in dims]
        return ExpandLayout(base=_reduce(layout.base, new_reduce_dims), dims=layout.dims)
    else:
        raise NotImplementedError()


def reduce(layout: Layout, dims: List[int], squeeze_dims=True) -> Layout:
    if squeeze_dims:
        return squeeze(_reduce(layout, dims), dims)
    else:
        return _reduce(layout, dims)


def expand(layout: Layout, dims: List[int]) -> Layout:
    if len(dims) == 0:
        return layout
    return ExpandLayout(layout, dims=dims)


def flatten(layout: Layout) -> Layout:
    return FlattenLayout(layout)


def get_composition_chain(layout: Layout, fine_grained=False) -> List[Layout]:
    if isinstance(layout, ComposedLayout):
        return get_composition_chain(layout.outer, fine_grained) + get_composition_chain(layout.inner, fine_grained)
    elif isinstance(layout, AtomLayout):
        if not fine_grained:
            return [layout]
        if layout.is_repeat():
            chain: List[Layout] = []
            shape = layout.shape.copy()
            ranks = layout.ranks
            ordered_dims = list(sorted(range(len(shape)), key=lambda dim: ranks[dim]))
            for dim in ordered_dims:
                factors = factorize_decomposition(shape[dim])
                for factor in factors:
                    chain.append(repeat(*[factor if i == dim else 1 for i in range(len(shape))]))
            return chain
        elif layout.is_spatial():
            chain: List[Layout] = []
            shape = layout.shape.copy()
            ranks = layout.worker_ranks
            ordered_dims = list(sorted(range(len(shape)), key=lambda dim: ranks[dim]))
            for dim in ordered_dims:
                factors = factorize_decomposition(shape[dim])
                for factor in factors:
                    chain.append(spatial(*[factor if i == dim else 1 for i in range(len(shape))]))
            return chain
        else:
            return [layout]
    else:
        return [layout]


def identity(rank: int) -> Layout:
    return repeat(*[1 for _ in range(rank)])


def compose_chain(layouts: List[Layout], init_rank: Optional[int] = None) -> Layout:
    if len(layouts) == 0:
        if init_rank is not None:
            return identity(init_rank)
        else:
            raise ValueError()
    elif len(layouts) == 1:
        return layouts[0]
    else:
        return compose(layouts[0], compose_chain(layouts[1:]))


def divide(layout: Layout, layout_rhs: Layout) -> Optional[Layout]:
    """
    Find a layout layout_lhs such that layout = layout_lhs * layout_rhs

    If no such layout exists, return None.
    """
    chain = get_composition_chain(layout, fine_grained=True)
    rhs_chain = get_composition_chain(layout_rhs, fine_grained=True)

    while rhs_chain:
        if len(chain) == 0:
            return None
        tail = chain.pop()
        rhs_tail = rhs_chain.pop()
        if not tail.semantics_equal(rhs_tail):
            return None
    return simplify(compose_chain(chain, init_rank=len(layout.shape)))


def greedy_decompose(
    layout: Layout,
    *,
    rhs_max_local_size: Optional[int] = None,
    rhs_max_workers: Optional[int] = None,
    rhs_max_elements: Optional[int] = None
) -> Tuple[Layout, Layout]:
    """
    Decompose layout into

        layout = layout_lhs * layout_rhs

    where layout_rhs has at most rhs_max_size local size and rhs_max_workers workers.

    There may be multiple solutions (like layout = layout * identity is also a valid decomposition solution).

    This function returns the solution that maximize layout_rhs.local_size * layout_rhs.num_workers.

    Parameters
    ----------
    layout : Layout
        The layout to be decomposed.

    rhs_max_local_size : Optional[int] = None
        The maximum local size of layout_rhs.

    rhs_max_workers : Optional[int] = None
        The maximum number of workers of layout_rhs.

    rhs_max_elements : Optional[int] = None
        The maximum number of elements of layout_rhs (i.e., prod(layout_rhs.shape)).

    Returns
    -------
    layout_lhs, layout_rhs: Tuple[Layout, Layout]
        The decomposed layouts such that layout = layout_lhs * layout_rhs.
    """
    chain = get_composition_chain(layout, fine_grained=True)
    rhs_chain = []

    local_size = 1
    num_workers = 1
    num_elements = 1

    while chain:
        last = chain[-1]

        if rhs_max_local_size is not None and local_size * last.local_size > rhs_max_local_size:
            break
        if rhs_max_workers is not None and num_workers * last.num_workers > rhs_max_workers:
            break
        if rhs_max_elements is not None and num_elements * prod(last.shape) > rhs_max_elements:
            break

        rhs_chain.append(chain.pop())
        local_size *= last.local_size
        num_workers *= last.num_workers
        num_elements *= prod(last.shape)

    layout_lhs = simplify(compose_chain(chain, init_rank=len(layout.shape)))
    layout_rhs = simplify(compose_chain(list(reversed(rhs_chain)), init_rank=len(layout.shape)))
    return layout_lhs, layout_rhs


def _try_merge_shape_ranks(
    lhs_shape: List[int], lhs_ranks: List[int], rhs_shape: List[int], rhs_ranks: List[int]
) -> Optional[Tuple[List[int], List[int]]]:
    assert all(a >= 1 for a in lhs_shape) and all(a >= 1 for a in rhs_shape)
    # try to unify ranks
    dims = list(range(len(lhs_shape)))
    rhs_dims: List[int] = list(sorted([dim for dim in dims if rhs_shape[dim] > 1], key=lambda dim: rhs_ranks[dim]))
    lhs_dims: List[int] = list(sorted([dim for dim in dims if lhs_shape[dim] > 1], key=lambda dim: lhs_ranks[dim]))
    shared_dims: List[int] = list(set(lhs_dims) & set(rhs_dims))

    ordered_dims: List[int] = []
    if len(shared_dims) > 1:
        return None
    elif len(shared_dims) == 1:
        shared_dim = shared_dims.pop()
        if not lhs_dims[-1] == rhs_dims[0] == shared_dim:
            return None
        ordered_dims = lhs_dims + rhs_dims[1:]
    else:
        ordered_dims = lhs_dims + rhs_dims

    for dim in dims:
        if dim not in ordered_dims:
            ordered_dims.insert(0, dim)

    assert set(ordered_dims) == set(dims)
    shape = [a * b for a, b in zip(lhs_shape, rhs_shape)]
    ranks = [ordered_dims.index(dim) for dim in dims]
    return shape, ranks


def try_merge_atom_layouts(lhs: Layout, rhs: Layout) -> Optional[Layout]:
    if not isinstance(lhs, AtomLayout) or not isinstance(rhs, AtomLayout):
        return None
    if lhs.is_spatial() and rhs.is_spatial():
        shape_ranks = _try_merge_shape_ranks(lhs.shape, lhs.worker_ranks, rhs.shape, rhs.worker_ranks)
        if shape_ranks is not None:
            shape, ranks = shape_ranks
            return spatial(*shape, ranks=ranks)
    if lhs.is_repeat() and rhs.is_repeat():
        shape_ranks = _try_merge_shape_ranks(lhs.shape, lhs.ranks, rhs.shape, rhs.ranks)
        if shape_ranks is not None:
            shape, ranks = shape_ranks
            return repeat(*shape, ranks=ranks)
    return None


def simplify(layout: Layout) -> Layout:
    if isinstance(layout, SqueezeLayout):
        if isinstance(layout.base, ExpandLayout):
            squeezed_dims: Set[int] = set(layout.dims)
            expanded_dims: Set[int] = set(layout.base.dims)
            common_dims = squeezed_dims & expanded_dims
            while common_dims:
                dim = common_dims.pop()
                squeezed_dims = set(d if d < dim else d - 1 for d in squeezed_dims if d != dim)
                expanded_dims = set(d if d < dim else d - 1 for d in expanded_dims if d != dim)
            ret = layout.base.base
            if squeezed_dims:
                ret = squeeze(ret, list(squeezed_dims))
            if expanded_dims:
                ret = expand(ret, list(expanded_dims))
            return ret
        else:
            return squeeze(simplify(layout.base), dims=layout.dims)
    elif isinstance(layout, ExpandLayout):
        if isinstance(layout.base, SqueezeLayout):
            raise NotImplementedError()
        else:
            return expand(simplify(layout.base), dims=layout.dims)
    elif isinstance(layout, ComposedLayout):
        layout = compose(simplify(layout.outer), simplify(layout.inner))
        chain: List[Layout] = get_composition_chain(layout, fine_grained=False)

        while True:
            merged = False
            for i in range(len(chain) - 1):
                if isinstance(chain[i], AtomLayout) and isinstance(chain[i + 1], AtomLayout):
                    merged_layout = try_merge_atom_layouts(chain[i], chain[i + 1])
                    if merged_layout is not None:
                        chain[i] = merged_layout
                        chain.pop(i + 1)
                        merged = True
                        break
            if not merged:
                break
        return compose_chain(chain)
    elif isinstance(layout, AtomLayout):
        return layout
    elif isinstance(layout, FlattenLayout):
        return flatten(simplify(layout.base))
    else:
        raise NotImplementedError(layout)


def is_compatible(lhs: Layout, rhs: Layout) -> bool:
    return lhs.local_size == rhs.local_size and lhs.num_workers == rhs.num_workers


def auto_repeat_spatial(num_threads: int, shape: List[int]):
    assert prod(shape) % num_threads == 0
    remain_shape = shape.copy()
    remain_threads = num_threads
    spatial_shape = [1 for i in range(len(shape))]

    for i in reversed(range(len(shape))):
        spatial_shape[i] = gcd(remain_threads, remain_shape[i])
        remain_threads //= spatial_shape[i]
        remain_shape[i] //= spatial_shape[i]

    assert remain_threads == 1

    repeat_shape = remain_shape
    return repeat(*repeat_shape).spatial(*spatial_shape)


def auto_layout_for_efficient_ldst(num_threads: int, shape: List[int], dtype_nbits: int) -> Layout:
    """
    Given the number of threads and shape, and the dtype number of bits, generate a layout that is hardware-friendly
    to load and write
    """
    assert is_power_of_two(dtype_nbits)
    assert prod(shape) % num_threads == 0
    elements = prod(shape)

    def is_valid_vec_size(vec):
        return (
            vec * dtype_nbits <= 128
            and vec % elements == 0
            and elements // vec % num_threads == 0
            and vec % shape[-1] == 0
        )

    vec_size = 1
    while is_valid_vec_size(vec_size * 2):
        vec_size *= 2

    shape = shape.copy()
    shape[-1] //= vec_size
    inner_repeat_shape = [1 for i in range(len(shape) - 1)] + [vec_size]
    return auto_repeat_spatial(num_threads, shape) * repeat(*inner_repeat_shape)


def demo_flatten():
    a = atom(2, 1, workers=[2, 3])
    b = flatten(a)

    print(a)
    print(b)

    assert a.num_workers == b.num_workers
    assert a.local_size == b.local_size
    assert len(b.shape) == 1 and prod(a.shape) == b.shape[0]
    for local in range(a.local_size):
        for worker in range(a.num_workers):
            a_global = a.local2global(local, int32(worker))
            b_global = b.local2global(local, int32(worker))
            assert index_serialize(a_global, a.shape) == b_global[0]


def main():
    demo_flatten()
    pass


if __name__ == '__main__':
    main()
