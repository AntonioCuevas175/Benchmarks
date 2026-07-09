import numpy as np
import time

def FFT(x):
    N = len(x)

    if N == 1:
        return x

    X_even = FFT(x[::2])
    X_odd = FFT(x[1::2])

    factor = np.exp(-2j * np.pi * np.arange(N) / N)

    return np.concatenate([
        X_even + factor[:N//2] * X_odd,
        X_even + factor[N//2:] * X_odd
    ])

N = 16384
ITERATIONS = 500

x = np.random.random(N)

start = time.perf_counter()

for _ in range(ITERATIONS):
    FFT(x)

elapsed = time.perf_counter() - start

print(f"Completed {ITERATIONS} FFTs")
print(f"Total time: {elapsed:.2f} s")
print(f"FFTs/sec: {ITERATIONS/elapsed:.2f}")