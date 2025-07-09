def program_id(idx: int = 0):
    from hidet.ir.primitives.cuda import blockIdx

    if idx == 0:
        return blockIdx.x
    elif idx == 1:
        return blockIdx.y
    elif idx == 2:
        return blockIdx.z
    else:
        raise ValueError(f"Invalid idx: {idx}")


def num_programs():
    from hidet.ir.primitives.cuda import blockDim

    return blockDim.x
