from mutis.utils.pkg import check_package_installed


if check_package_installed('vllm', '>=0.5.3'):
    # import mutis.extension.vllm.quantization.fp8_mutis
    # import mutis.extension.vllm.quantization.mutis_quant
    import mutis.extension.vllm.quantization.mutis_quant_new as mutis_quant
    from mutis.extension.vllm.quantization.mutis_quant_new import set_mutis_config
else:
    pass
