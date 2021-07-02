"""
Running full set of experiments with configuration specified in
'configs/best.in'
"""
from CICYworker import CICYWorker
import os as os
import json
from ast import literal_eval
import argparse

# hack to make gpu work on my machine
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='4-fold experiments.')
    parser.add_argument('--budget', type=int, default=100,
                        help='max compute budget.')
    parser.add_argument('--regression', type=int, default=0,
                        help='adding regression runs.')
    parser.add_argument('--classification', type=int, default=1,
                        help='adding classification runs.')
    parser.add_argument('--dconfig', type=str, default='configs',
                        help='dir name which contains config')
    parser.add_argument('--fconfig', type=str, default='best.in',
                        help='file name which contains config')
    parser.add_argument('--hodge', type=int, default=-1,
                        help='which hodge numbers.')
    parser.add_argument('--outdir', type=str, default='results',
                        help='dir name for output files.')
    parser.add_argument('--tails', type=int, default=0,
                        help='0-use outliers, else discard')
    args = parser.parse_args()
    
    maxcompute = args.budget

    config_path = os.path.join(args.dconfig, args.fconfig)

    with open(config_path) as f:
        sample = literal_eval(f.readlines()[0])

    splits = [[80,10,10], [50,10,40], [30,10,60], [10,10,80]]
    models = []
    if args.classification:
        models += ['class']
    if args.regression:
        models += ['reg']
    if args.hodge == -1:
        hodge = [0,1,2,3]
    else:
        hodge = [args.hodge]
    if not args.tails:
        tails_path = os.path.join('data','conf_tails.npy')
    else:
        tails_path = ''

    for t in models:
        for h in hodge:
            directory = os.path.join(args.outdir, t+str(h))
            for l in splits:
                w = CICYWorker(
                    os.path.join('data','conf.npy'),
                    os.path.join('data','hodge.npy'),
                    os.path.join('data','direct.npy'),
                    1 if t == 'class' else 0,
                    train_ratio=l,
                    tails=tails_path,
                    h=h, run_id=1
                )
                w.verbose = 2
                sub_dir = os.path.join(directory, str(l[0]))
                if not os.path.exists(sub_dir):
                    os.makedirs(sub_dir)
                compute = int(maxcompute*80/l[0])
                results = w.compute(sample, compute, sub_dir)
                print(results)
                with open(os.path.join(sub_dir, t+'.json'), 'w') as f:
                    json.dump(results, f)
