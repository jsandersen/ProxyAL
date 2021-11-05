import sys, os
cwd = os.getcwd()
sys.path.append(cwd + '/../.') 

import argparse
from src.EvaluationLoop import EvaluationLoop
from src.training_loop2 import QueryStrategy
from data.datasets import Datasets, Encodings


parser = argparse.ArgumentParser(description='Run Active Learning with SBERT Encoder')
parser.add_argument("--name", type=str, default='test')
parser.add_argument("--dataset", type=Datasets.from_string, choices=list(Datasets))
parser.add_argument("--encoding", type=Encodings.from_string, choices=list(Encodings))
parser.add_argument("--steps", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--save_repeat", type=int, default=1)
parser.add_argument('--query_strategy', type=QueryStrategy.from_string, choices=list(QueryStrategy))
parser.add_argument('--random', default=False, action="store_true")
parser.add_argument('--warmstart', type=int, default=10)

if __name__ == "__main__":
    args = parser.parse_args()
    name = args.name
    dataset = args.dataset
    encoding = args.encoding
    al_steps = args.steps
    save_and_repeat = args.save_repeat
    sample_size_per_step = args.batch_size
    query_strategy = args.query_strategy
    warmstart = args.warmstart
    random = args.random
    
    name_postfix = ""
    
    print('#############')
    print(random)
    print('#############')
    
    if random:
        print('Random query strategy is used. Selection is ignored.')
        name_postfix = "_rnd"
        query_strategy = QueryStrategy.unc
    
    save_dir = f'./res/{dataset}_{encoding}_{str(query_strategy)}_{name}{name_postfix}'
    
    el = EvaluationLoop(save_dir, dataset, encoding, sample_size_per_step, al_steps, save_and_repeat, query_strategy, warmstart=warmstart, random=random)
    el.run()
    
    