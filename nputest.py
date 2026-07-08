import torch
import time

threads = int (input("How many threads?"))
torch.set_num_threads(threads)

x = torch.randn(10000, 10000)
y = torch.randn(10000, 10000)

# Warm-up
for _ in range(3):
    torch.matmul(x, y)

# Measure
start = time.perf_counter()
torch.matmul(x, y)
end = time.perf_counter()

print(f"Time: {end - start:.3f} s")
