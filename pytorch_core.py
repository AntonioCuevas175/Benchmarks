import time
import multiprocessing as mp

import cpuinfo
from numba import njit, prange


# =====================================================
# Workloads
# =====================================================

def scalar_workload(n):
    total = 0

    for i in range(n):
        total += i * i

    return total


@njit
def jit_workload(n):
    total = 0

    for i in range(n):
        total += i * i

    return total


@njit(fastmath=True)
def simd_workload(n):
    total = 0

    for i in range(n):
        total += i * i

    return total


@njit(parallel=True, fastmath=True)
def parallel_simd_workload(n):
    total = 0

    for i in prange(n):
        total += i * i

    return total


# =====================================================
# Helpers
# =====================================================

def benchmark(func, n):
    start = time.perf_counter()
    func(n)
    return time.perf_counter() - start


def detect_cpu_features():
    info = cpuinfo.get_cpu_info()

    flags = set(info.get("flags", []))

    return {
        "name": info.get("brand_raw", "Unknown"),
        "cores": mp.cpu_count(),
        "SSE4.2": "sse4_2" in flags,
        "AVX": "avx" in flags,
        "AVX2": "avx2" in flags,
        "FMA": "fma" in flags,
        "AVX512": any(flag.startswith("avx512") for flag in flags),
    }


def detect_vector_width():
    try:
        sig = simd_workload.signatures[0]
        asm = simd_workload.inspect_asm(sig)

        if "zmm" in asm:
            return "AVX-512 (512-bit)"

        if "ymm" in asm:
            return "AVX/AVX2 (256-bit)"

        if "xmm" in asm:
            return "SSE (128-bit)"

        return "Scalar"

    except Exception:
        return "Unknown"


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":

    N = 1_000_000_000

    print("Compiling Numba kernels...")

    jit_workload(10)
    simd_workload(10)
    parallel_simd_workload(10)

    cpu = detect_cpu_features()

    print("\nCPU Information")
    print("---------------")
    print(f"Name: {cpu['name']}")
    print(f"Cores: {cpu['cores']}")

    print("\nInstruction Support")
    print("-------------------")
    print(f"SSE4.2 : {'Yes' if cpu['SSE4.2'] else 'No'}")
    print(f"AVX    : {'Yes' if cpu['AVX'] else 'No'}")
    print(f"AVX2   : {'Yes' if cpu['AVX2'] else 'No'}")
    print(f"FMA    : {'Yes' if cpu['FMA'] else 'No'}")
    print(f"AVX512 : {'Yes' if cpu['AVX512'] else 'No'}")

    print("\nRunning benchmarks...\n")

    scalar_time = benchmark(scalar_workload, N)

    jit_time = benchmark(jit_workload, N)

    simd_time = benchmark(simd_workload, N)

    parallel_simd_time = benchmark(
        parallel_simd_workload,
        N
    )

    vector_mode = detect_vector_width()

    print("Benchmark Results")
    print("-----------------")
    print(f"Scalar (Python)      : {scalar_time:.3f} s")
    print(f"JIT                  : {jit_time:.3f} s")
    print(f"SIMD (1 Core)        : {simd_time:.3f} s")
    print(f"Parallel + SIMD      : {parallel_simd_time:.3f} s")

    print("\nGenerated Vector Code")
    print("---------------------")
    print(vector_mode)

    print("\nSpeedups")
    print("---------")
    print(
        f"JIT Gain             : "
        f"{scalar_time / jit_time:.2f}x"
    )

    print(
        f"SIMD Gain            : "
        f"{scalar_time / simd_time:.2f}x"
    )

    print(
        f"Total Gain           : "
        f"{scalar_time / parallel_simd_time:.2f}x"
    )
