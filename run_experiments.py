BATCH_SIZE = 16

import multiprocessing as mp
import os
from itertools import product

num_workers = mp.cpu_count() // 2

def spawn_worker(args):
    os.makedirs("./stats/", exist_ok=True)
    max_batches_ae, use_attention_entropy, seed, proportion, run = args
    command = f"python3 train_cifar10.py --net vit --max_batches_ae {max_batches_ae} --use_attention_entropy {use_attention_entropy} --run {run} --bs {BATCH_SIZE} --num_workers {num_workers} --random_seed {seed} --proportion {proportion}"
    os.system(command)

use_attention_entropy_ = [0, 1]
seeds = list(range(10))
max_batches_ae_ = [50, 200]
proportions = [0.01, 0.05, 0.15, 0.4, 0.7, 1.0]

args_ = []
for run, (max_batches_ae, use_attention_entropy, prop, seed) in enumerate(product(max_batches_ae_, use_attention_entropy_, proportions, seeds)):
    args_.append([max_batches_ae, use_attention_entropy, seed, prop, run])
args_.reverse()
# args_ = sorted(args_, key=lambda x: x[0])
print("Total models:", len(args_))
with mp.Pool(2) as p:
    p.map(spawn_worker, args_)
