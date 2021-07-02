"""
BOHB - scan.
Essentially a modified version of the MNIST example from
https://github.com/automl/HpBandSter/blob/master/hpbandster/examples/example_5_mnist.py
"""

import os
import pickle
import argparse

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpbandster.optimizers import BOHB
from CICYworker import CICYWorker as worker

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import logging
logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Learning hyperparameters for CICY 4-fold predictors.')
    parser.add_argument('--min_budget',
        type=float, help='Minimum number of epochs for training.',
        default=5)
    parser.add_argument('--max_budget',
        type=float, help='Maximum number of epochs for training.',
        default=250)
    parser.add_argument('--n_iterations', type=int,
        help='Number of iterations performed by the optimizer',
        default=10)
    parser.add_argument('--worker',
        help='Flag to turn this into a worker process',
        action='store_true')
    parser.add_argument('--run_id',
        type=str, help='A unique run id for this optimization run. \
            An easy option is to use the job id of the clusters scheduler.')
    parser.add_argument('--nic_name', type=str,
        help='Which network interface to use for communication.',
        default='lo')
    parser.add_argument('--shared_directory',type=str,
        help='A directory that is accessible for all processes,\
             e.g. a NFS share.', default='bohb')
    parser.add_argument('--classification', type=int,
        help='1-classification or 0-regression', default=1)
    parser.add_argument('--nfold', type=int, default=4)
    parser.add_argument('--hodge', type=int, default=2,
        help='if -1 tries to predict all hodge numbers.')
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--perc', type=int, default=30)
    args=parser.parse_args()

    # Every process has to lookup the hostname
    host = hpns.nic_name_to_host(args.nic_name)

    # give the three file paths
    if args.nfold == 4:
        cname = 'data/conf.npy'
        hname = 'data/hodge.npy'
        dname = 'data/direct.npy'
    else:
        cname = 'data/conf3.npy'
        hname = 'data/hodge3.npy'
        dname = 'data/direct3.npy'

    train_ratio = [args.perc, 10, 90-args.perc]

    if args.worker:
        import time
        # artificial delay so that nameserver is already running
        time.sleep(3)
        w = worker(cname, hname, dname, args.classification,
            train_ratio=args.perc, h=args.hodge, run_id=args.run_id,
            host=host, timeout=120)
        w.load_nameserver_credentials(
            working_directory=args.shared_directory)
        w.run(background=False)
        exit(0)

    result_logger = hpres.json_result_logger(
        directory=args.shared_directory, overwrite=False)

    # Start a nameserver:
    NS = hpns.NameServer(run_id=args.run_id, host=host, port=0,
        working_directory=args.shared_directory)
    ns_host, ns_port = NS.start()

    # Start local worker
    w = worker(cname, hname, dname, args.classification, h=args.hodge,
        run_id=args.run_id, host=host, nameserver=ns_host,
        nameserver_port=ns_port, timeout=120)
    w.run(background=True)

    # Run an optimizer
    if os.path.exists(args.load):
        # Load old run to use its results as priors
        # Note that the search space has to be identical though!
        previous_run = hpres.logged_results_to_HBS_result(
            args.previous_run_dir)
        bohb = BOHB(
            configspace = worker.get_configspace(),
            run_id = args.run_id,
            host=host,
            nameserver=ns_host,
            nameserver_port=ns_port,
            result_logger=result_logger,
            min_budget=args.min_budget, max_budget=args.max_budget,
            previous_result = previous_run,
        )

    else:
        bohb = BOHB(
            configspace = worker.get_configspace(),
            run_id = args.run_id,
            host=host,
            nameserver=ns_host,
            nameserver_port=ns_port,
            result_logger=result_logger,
            min_budget=args.min_budget, max_budget=args.max_budget,
        )
    res = bohb.run(n_iterations=args.n_iterations)

    # store results
    with open(os.path.join(
            args.shared_directory, 'results.pkl'), 'wb') as fh:
        pickle.dump(res, fh)

    # shutdown
    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()