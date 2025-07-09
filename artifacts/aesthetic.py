ranked_executors = ['torch-f16', 'triton', 'quant-llm', 'bitblas', 'marlin', 'mutis']

colors = [
    ['#fbb4ae'],
    ['#b3cde3'],
    ['#ccebc5'],
    ['#decbe4'],
    ['#fed9a6'],
    ['#4c535d']
]

executor2color = {
    'torch-f16': 3,
    'triton': 1,
    'bitblas': 4,
    'quant-llm': 3,
    'marlin': 0,
    'mutis': 2,
}
executor2label = {
    'torch-f16': 'cuBLAS',
    'triton': 'Triton',
    'quant-llm': 'QuantLLM',
    'bitblas': 'Ladder',
    'marlin': 'Marlin',
    'mutis': 'Tilus (Ours)'
}

