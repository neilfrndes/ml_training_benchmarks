import argparse
import importlib
import logging

import common


# Setup logging
logger = logging.getLogger('__name__')
logging.getLogger().setLevel(logging.INFO)


# Initialize Command Line arguments
parser = argparse.ArgumentParser(description='Runs inference on models.')
parser.add_argument(
    '-m', '--model', type=str, default="kmeans",
    help="Type of model to run")
parser.add_argument(
    '-o', '--observations', type=int, default=1e6,
    help="Max batch size")
parser.add_argument(
    '-t', '--train', type=bool, default=True,
    help="If true runs training else inferencing")
args = parser.parse_args()


# main module
if __name__ == '__main__':
    # Run
    try:
        logging.info(f"Loading model {args.model}")
        model = importlib.import_module('models.' + args.model)
    except ModuleNotFoundError:
        logging.error(f"Model {args.model} not found.")

    if args.train:
        logging.info(f"Running training benchmark for {args.model}...")
        logging.info(common.get_header())
        logging.info(common.get_underline())
        batch_size = 10
        while batch_size <= args.observations:
            total_times, observation_times = model.run_training(batch_size)
            stats = common.calculate_stats(observation_times)
            logging.info(common.format_stats(batch_size, stats))
            batch_size *= 10
    # else:
    #     logging.info(f"Running testing benchmark for {args.model}...")
    #     logging.info(common.STATS)
    #     batch_size = 1
    #     while batch_size <= args.observations:
    #         model.run_inference(batch_size)
    #         batch_size *= 10
else:
    logging.error(f"Could not find benchmark for {model}")

