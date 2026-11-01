import os
if os.environ.get('CES_CUDA_MEMORY_FRACTION'):
    import torch
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(os.environ['CES_CUDA_MEMORY_FRACTION']),0)
