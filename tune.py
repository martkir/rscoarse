import click
import json
import multiprocessing as mp
import os
import numpy as np
import pandas as pd
import time
import random
from tqdm import tqdm


def run(cmd):
    step = 1 / 5  # not necessary to do the sleep thing.
    i, cmd_str = cmd
    time.sleep(i * step)
    os.system(cmd_str)


def get_job_ids(root_dir):
    job_ids = [job_id for job_id in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, job_id))]
    return job_ids


def get_tune_id():
    tune_id = random.randint(0, 1000000)
    if __name__ == '__main__':
        print('tune id: {}'.format(tune_id))
    return tune_id


def default_search_space():
    num_factors = [i * 4 for i in range(1, 5)]
    num_factors = json.dumps(num_factors)
    return num_factors


num_factors = default_search_space()
tune_id = get_tune_id()


@click.command()
@click.option('--data_name', type=str, default='ml-100k')
@click.option('--num_factors', type=str, default=num_factors)
@click.option('--num_epochs', type=int, default=1)
@click.option('--num_cores', type=int, default=2)
def main(data_name, num_factors, num_epochs, num_cores):
    num_factors = json.loads(num_factors)
    cmds = []
    for i in range(len(num_factors)):
        job_id = random.randint(0, 1000000)
        job_dir = 'output/tune_{}/train_fm_{}'.format(tune_id, job_id)
        cmd = \
            'python fm.py --data_name {} --n_factors {} --num_epochs {} --use_gpu {} --job_dir {} --log False'.\
                format(data_name, num_factors[i], num_epochs, False, job_dir)
        cmds.append((i, cmd))

    pool = mp.Pool(processes=num_cores)
    for _ in tqdm(pool.imap_unordered(run, cmds), total=len(cmds)):
        pass
    pool.close()
    pool.join()

    # """
    # table to create:
    # - num_factors
    # - MSE
    # - lr
    # - batch_size
    # - use_content.
    # """
    #
    # entries = []
    # job_ids = get_job_ids(root_dir)
    # for job_id in job_ids:
    #     job_dir = os.path.join(root_dir, '{}'.format(job_id))
    #     result = BacktestResult(job_dir)
    #     trades = result.trades
    #     net_returns = np.array(trades['net_return'])
    #     gross_returns = np.array(trades['gross_return'])
    #     entry = result.metadata
    #     entry['avg_net_return'] = np.mean(net_returns)
    #     entry['avg_gross_return'] = np.mean(gross_returns)
    #     entry['std_gross_return'] = np.std(gross_returns)
    #     entries.append(entry)
    #
    # table = pd.DataFrame(entries)
    # table.to_csv('{}/summary.csv'.format(root_dir), index=False, mode='w+')


if __name__ == '__main__':
    main()