from mutis.ir.layout import Layout, spatial, repeat
from mutis.utils import prod


def access_efficient_layout(layout: Layout, element_nbytes: int) -> Layout:
    for nbytes in [16, 8, 4, 2, 1]:
        if layout.local_size * element_nbytes % nbytes == 0 and nbytes % element_nbytes == 0:
            num_vectors = layout.local_size * element_nbytes // nbytes
            vector_elements = nbytes // element_nbytes
            break
    else:
        num_vectors = layout.local_size
        vector_elements = 1

    ret = spatial(layout.num_workers).repeat(vector_elements)
    if num_vectors > 1:
        ret = repeat(num_vectors) * ret
    return ret
