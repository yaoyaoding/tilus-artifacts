import pytest
from hidet.ir.utils.index_transform import index_serialize, index_deserialize
from hidet.ir.expr import convert
from hidet.utils import same_list


@pytest.mark.parametrize(
    "indices, shape, ranks, expected",
    [
        ([0, 0], [3, 4], None, 0),
        ([0, 1], [3, 4], None, 1),
        ([1, 0], [3, 4], None, 4),
        ([1, 1], [3, 4], None, 5),
        ([0, 0], [3, 4], [0, 1], 0),
        ([0, 1], [3, 4], [0, 1], 1),
        ([1, 0], [3, 4], [0, 1], 4),
        ([1, 1], [3, 4], [0, 1], 5),
        ([0, 0], [3, 4], [1, 0], 0),
        ([0, 1], [3, 4], [1, 0], 3),
        ([1, 0], [3, 4], [1, 0], 1),
        ([2, 1], [3, 4], [1, 0], 5),
        ([0, 0, 0], [3, 4, 5], [1, 0, 2], 0),
        ([0, 1, 0], [3, 4, 5], [1, 0, 2], 15),
        ([0, 0, 1], [3, 4, 5], [1, 0, 2], 1),
        ([1, 0, 0], [3, 4, 5], [1, 0, 2], 5),
        ([1, 2, 3], [3, 4, 5], [2, 0, 1], 1 + 3 * 3 + 2 * 3 * 5),
    ],
)
def test_index_serialize(indices, shape, ranks, expected):
    result = index_serialize([convert(v) for v in indices], shape, ranks)
    assert result == expected


@pytest.mark.parametrize(
    "scalar_index, shape, ranks, expected",
    [
        (0, [3, 4], None, [0, 0]),
        (1, [3, 4], None, [0, 1]),
        (4, [3, 4], None, [1, 0]),
        (5, [3, 4], None, [1, 1]),
        (0, [3, 4], [0, 1], [0, 0]),
        (1, [3, 4], [0, 1], [0, 1]),
        (4, [3, 4], [0, 1], [1, 0]),
        (5, [3, 4], [0, 1], [1, 1]),
        (0, [3, 4], [1, 0], [0, 0]),
        (3, [3, 4], [1, 0], [0, 1]),
        (1, [3, 4], [1, 0], [1, 0]),
        (5, [3, 4], [1, 0], [2, 1]),
        (0, [3, 4, 5], [1, 0, 2], [0, 0, 0]),
        (15, [3, 4, 5], [1, 0, 2], [0, 1, 0]),
        (1, [3, 4, 5], [1, 0, 2], [0, 0, 1]),
        (5, [3, 4, 5], [1, 0, 2], [1, 0, 0]),
        (1 + 3 * 3 + 2 * 3 * 5, [3, 4, 5], [2, 0, 1], [1, 2, 3]),
    ],
)
def test_index_deserialize(scalar_index, shape, ranks, expected):
    result = index_deserialize(convert(scalar_index), shape, ranks)
    assert same_list(result, expected)
