# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=cell-var-from-loop
from typing import Optional, List

from hidet.ir.type import PointerType, DataType, TensorPointerType, data_type
from hidet.ir.expr import Expr
from hidet.ir.tools import infer_type
from hidet.ir.primitives.func import register_primitive_function, call_primitive_func
from hidet.utils import initialize


def resolve_load_inst_name(bits: int, space: str, sync: Optional[str], nc_cache=False, vec=1, scope: str = None) -> str:
    inst = 'ld'
    if sync:
        if scope is None:
            scope = 'gpu'
        inst += f'.{sync}.{scope}'
    if space != 'generic':
        inst += f'.{space}'
    if nc_cache:
        assert space == 'global'
        inst += '.nc'
    if nc_cache and vec > 1:
        if vec * bits >= 128:
            inst += '.L2::128B'
    if vec > 1:
        inst += f'.v{vec}'
    inst += f'.b{bits}'

    return inst


def resolve_store_inst_name(bits: int, space: str, sync: Optional[str], vec=1, scope: str = None) -> str:
    inst = 'st'
    if sync:
        if scope is None:
            scope = 'gpu'
        inst += f'.{sync}.{scope}'
    if space != 'generic':
        inst += f'.{space}'
    if vec > 1:
        inst += f'.v{vec}'
    inst += f'.b{bits}'

    return inst


def normalize_func_name(inst_name):
    func_name = 'cuda_' + inst_name.replace('.', '_')
    func_name = func_name.replace('::', '_')
    return func_name


@initialize()
def register_functions():
    # pylint: disable=function-redefined
    from hidet.lang import attrs, script, asm, deref, cast  # pylint: disable=import-outside-toplevel
    from hidet.lang.types import uint8, uint16, uint32, uint64, void_p

    dtype: Optional[DataType] = None

    def as_dtype(x):
        nonlocal dtype  # by changing dtype, we can change the type of the output
        return deref(cast(x, ~dtype))

    registered = set()
    for space in ['generic', 'shared', 'global']:
        for sync in ['acquire', None]:
            for vec in [1, 2, 4]:
                for dtype_ in [uint8, uint16, uint32, uint64]:
                    for nc in [True, False]:
                        dtype = dtype_
                        if nc and space != 'global':
                            continue
                        if sync is not None and space != 'generic':
                            continue
                        inst_name = resolve_load_inst_name(dtype.nbytes * 8, space, sync, nc, vec)
                        func_name = normalize_func_name(inst_name)
                        if func_name in registered:
                            continue
                        registered.add(func_name)
                        addr_type = {'generic': void_p, 'shared': uint32, 'global': void_p}[space]

                        if vec == 1:

                            @script
                            def cuda_load(addr: addr_type, v0: void_p):
                                attrs.func_kind = 'cuda_internal'
                                attrs.func_name = func_name
                                template = inst_name + ' %0, [%1];'
                                outputs = [as_dtype(v0)]
                                asm(template, outputs=outputs, inputs=[addr], is_volatile=True)

                            register_primitive_function(name=cuda_load.name, func_or_type=cuda_load)
                        if vec == 2:

                            @script
                            def cuda_load(addr: addr_type, v0: void_p, v1: void_p):
                                attrs.func_kind = 'cuda_internal'
                                attrs.func_name = func_name
                                template = inst_name + ' {%0, %1}, [%2];'
                                outputs = [as_dtype(v0), as_dtype(v1)]
                                asm(template, outputs=outputs, inputs=[addr], is_volatile=True)

                            register_primitive_function(name=cuda_load.name, func_or_type=cuda_load)
                        if vec == 4:

                            @script
                            def cuda_load(addr: addr_type, v0: void_p, v1: void_p, v2: void_p, v3: void_p):
                                attrs.func_kind = 'cuda_internal'
                                attrs.func_name = func_name
                                template = inst_name + ' {%0, %1, %2, %3}, [%4];'
                                outputs = [as_dtype(v0), as_dtype(v1), as_dtype(v2), as_dtype(v3)]
                                asm(template, outputs=outputs, inputs=[addr], is_volatile=True)

                            register_primitive_function(name=cuda_load.name, func_or_type=cuda_load)

    for space in ['generic', 'global', 'shared']:
        for sync in ['release', None]:
            for dtype_ in [uint8, uint16, uint32, uint64]:
                for vec in [1, 2, 4]:
                    dtype = dtype_
                    inst_name = resolve_store_inst_name(dtype.nbytes * 8, space, sync, vec)
                    func_name = normalize_func_name(inst_name)
                    if func_name in registered:
                        continue
                    registered.add(func_name)

                    addr_type = {'generic': void_p, 'shared': uint32, 'global': void_p}[space]

                    if vec == 1:

                        @script
                        def cuda_store(addr: addr_type, v0: void_p):
                            attrs.func_kind = 'cuda_internal'
                            attrs.func_name = func_name
                            template = inst_name + ' [%0], %1;'
                            inputs = [addr, as_dtype(v0)]
                            asm(template, inputs=inputs, is_volatile=True)

                        register_primitive_function(name=cuda_store.name, func_or_type=cuda_store)
                    if vec == 2:

                        @script
                        def cuda_store(addr: addr_type, v0: void_p, v1: void_p):
                            attrs.func_kind = 'cuda_internal'
                            attrs.func_name = func_name
                            template = inst_name + ' [%0], {%1, %2};'
                            inputs = [addr, as_dtype(v0), as_dtype(v1)]
                            asm(template, inputs=inputs, is_volatile=True)

                        register_primitive_function(name=cuda_store.name, func_or_type=cuda_store)
                    if vec == 4:

                        @script
                        def cuda_store(addr: addr_type, v0: void_p, v1: void_p, v2: void_p, v3: void_p):
                            attrs.func_kind = 'cuda_internal'
                            attrs.func_name = func_name
                            template = inst_name + ' [%0], {%1, %2, %3, %4};'
                            inputs = [addr, as_dtype(v0), as_dtype(v1), as_dtype(v2), as_dtype(v3)]
                            asm(template, inputs=inputs, is_volatile=True)

                        register_primitive_function(name=cuda_store.name, func_or_type=cuda_store)


@initialize()
def register_primitive_functions_with_body():
    # pylint: disable=import-outside-toplevel
    from hidet.ir.type import ReferenceType
    from hidet.ir.expr import Var
    from hidet.ir.stmt import AsmStmt
    from hidet.ir.builders import FunctionBuilder

    # lds128
    with FunctionBuilder('cuda_lds128', kind='cuda_internal') as fb:
        # params
        regs_vars = [Var(f'reg{i}', ReferenceType(data_type('float32'))) for i in range(4)]
        smem_addr_var = Var('smem_addr', PointerType(data_type('float32')))
        fb.extend_params(regs_vars + [smem_addr_var])
        # body
        body = AsmStmt(
            r"{"
            r"  .reg.u64 u64addr;"
            r"  cvta.to.shared.u64 u64addr, %4;"
            r"  ld.shared.v4.f32 {%0, %1, %2, %3}, [u64addr];"
            r"}",
            outputs=[('=f', reg) for reg in regs_vars],
            inputs=[('l', smem_addr_var)],
            is_volatile=True,
        )
        fb.set_body(body)
    register_primitive_function(name='cuda_lds128', func_or_type=fb.get())

    # sts128
    with FunctionBuilder('cuda_sts128', kind='cuda_internal') as fb:
        # params
        regs_vars = [Var(f'reg{i}', ReferenceType(data_type('float32'))) for i in range(4)]
        smem_addr_var = Var('smem_addr', PointerType(data_type('float32')))
        fb.extend_params(regs_vars + [smem_addr_var])
        # body
        body = AsmStmt(
            r"{"
            r"  .reg.u64 u64addr;"
            r"  cvta.to.shared.u64 u64addr, %0;"
            r"  st.shared.v4.f32 [u64addr], {%1, %2, %3, %4};"
            r"}",
            outputs=[],
            inputs=[('l', smem_addr_var)] + [('f', reg) for reg in regs_vars],
            is_volatile=True,
        )
        fb.set_body(body)
    register_primitive_function(name='cuda_sts128', func_or_type=fb.get())


def resolve_pointed_dtype(addr: Expr) -> str:
    ptr_type = infer_type(addr)
    if not isinstance(ptr_type, (PointerType, TensorPointerType)):
        raise ValueError('Expect a pointer type, got {}'.format(ptr_type))
    if isinstance(ptr_type, PointerType):
        dtype = ptr_type.base_type
    else:
        dtype = ptr_type.tensor_type.dtype
    if not isinstance(dtype, DataType):
        raise ValueError('Expect a pointer to a scalar type, got {}'.format(ptr_type))
    return dtype.name


def load(dtype, addr: Expr, dst_addrs: List[Expr], space: str = 'generic', sync=None, nc_cache=False, scope=None):
    """
    Load data from memory.

    Parameters
    ----------
    dtype: str or DataType
        The data type of the data to be loaded.

    addr: Expr
        The address of the data, in a type of pointer.

    dst_addrs: List[Expr]
        The address of the registers to store the loaded data. The length of the list must be 1, 2, or 4. The load
        instruction will load data to the registers in order, each register will be feed with 32bit data.

    space: str
        The memory space of the address. Candidates: 'generic', 'global', 'shared', 'local'

    sync: str, optional
        The synchronization behavior. Candidates: None, 'acquire', and 'relaxed'.

    nc_cache: bool
        Whether to use non-coherent cache. This parameter is only valid when space is 'global'.

    scope: str
        The scope of the synchronization. Candidates: None, 'cta', 'gpu', 'sys'.

    Returns
    -------
    ret: Expr
        The loaded data.
    """
    dtype = data_type(dtype)
    func_name = normalize_func_name(
        resolve_load_inst_name(dtype.nbytes * 8, space, sync, nc_cache, len(dst_addrs), scope)
    )
    return call_primitive_func(func_name, [addr, *dst_addrs])


def store(
    dtype, addr: Expr, src_addrs: List[Expr], space: str = 'generic', sync: Optional[str] = None, scope: str = 'gpu'
):
    """
    Store data to memory.

    Parameters
    ----------
    dtype: str or DataType
        The data type of the data to be stored.
    addr: Expr
        The address to store the data.

    src_addrs: List[Expr]
        The value to store.

    space: str
        The memory space of the address. Candidates: 'generic', 'global', 'shared', 'local'

    sync: Optional[str]
        The synchronization behavior. Candidates: 'release', and 'relaxed'.

    scope: str
        The scope of the synchronization. Candidates: 'cta', 'gpu', 'sys'.
    """
    dtype = data_type(dtype)
    func_name = normalize_func_name(resolve_store_inst_name(dtype.nbytes * 8, space, sync, len(src_addrs), scope))
    return call_primitive_func(func_name, [addr, *src_addrs])


def lds128(reg0, reg1, reg2, reg3, smem_addr):
    return call_primitive_func('cuda_lds128', [reg0, reg1, reg2, reg3, smem_addr])


def sts128(reg0, reg1, reg2, reg3, smem_addr):
    return call_primitive_func('cuda_sts128', [reg0, reg1, reg2, reg3, smem_addr])
