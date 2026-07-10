import multiprocessing as mp
import time

def workload(n):
    total = 0
    for i in range(n):
        total += i * i
    return total

if __name__ == "__main__":
    workers = int(input("how many cores? (1-14)"))
    iterations = 100_000_000

    start = time.perf_counter()

    with mp.Pool(workers) as pool:
        pool.map(workload, [iterations] * workers)

    elapsed = time.perf_counter() - start

    print(f"CPU cores used: {workers}")
    print(f"Execution time: {elapsed:.3f} seconds")
