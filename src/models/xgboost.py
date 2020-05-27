"""Functions for running benchmarks for the kmeans algorithm"""
from timeit import default_timer as timer
from typing import List

import xgboost as xgb

from data import load

PARAMS = {
    'objective': 'reg:squarederror',
    'alpha': 0.9,
    'max_bin': 256,
    'scale_pos_weight': 2,
    'learning_rate': 0.1,
    'subsample': 1,
    'reg_lambda': 1,
    'min_child_weight': 0,
    'max_depth': 8,
    'max_leaves': 2**8,
    'tree_method': 'hist',
    'predictor': 'cpu_predictor'
}
NUM_LOOPS = 100
data = load.MLData('iris.csv')


def run_training(size: int = 1000):
    """Run kmeans for the specified size of the dataset"""
    # Load data
    X, y = data.get_training_data(size)
    num_rows = len(X)

    train_df = xgb.DMatrix(data=X, label=y)

    benchmark_times: List[float] = []
    individual_times: List[float] = []
    for _ in range(NUM_LOOPS):

        start_time = timer()
        MODEL = xgb.train(params=PARAMS, dtrain=train_df)
        end_time = timer()

        total_time = end_time - start_time
        benchmark_times.append(total_time*10e3)  # miliseconds

        individual_time = total_time*(10e6)/num_rows  # microseconds
        individual_times.append(individual_time)

    return benchmark_times, individual_times
