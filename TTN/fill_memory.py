import torch
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

available = torch.cuda.mem_get_info()[0]
to_occupy = available - 2*1024**2 - 800*1024
x = torch.empty((to_occupy,), dtype=torch.uint8, device='cuda')

# This will block until the memory is filled


if __name__ == "__main__":
    print("Memory filled")
    print("Press Ctrl+C to exit")
    while True:
        pass