"""Functions for running benchmarks for the kmeans algorithm"""
from timeit import default_timer as timer
from typing import List

from sklearn.cluster import KMeans

from data import load

PARAMS = {
    'n_clusters': 4,
    'n_jobs': -1,
    'n_init': 10,
    'max_iter': 300
}
NUM_LOOPS = 100
data = load.MLData('iris.csv')


def run_training(size: int = 1000):
    """Run kmeans for the specified size of the dataset"""
    # Load data
    X, _ = data.get_training_data(size)

    num_rows = len(X)

    benchmark_times: List[float] = []
    individual_times: List[float] = []
    for _ in range(NUM_LOOPS):

        km = KMeans(**PARAMS)
        start_time = timer()
        km.fit(X)
        end_time = timer()

        total_time = end_time - start_time
        benchmark_times.append(total_time*10e3)  # miliseconds

        individual_time = total_time*(10e6)/num_rows  # microseconds
        individual_times.append(individual_time)

    return benchmark_times, individual_times
