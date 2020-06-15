import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
import sys
from collections import OrderedDict
from collections import defaultdict
import torch
import click
import datasets
import random
from torch import optim
import pandas as pd
import json


def set_device(use_gpu):
    device = torch.device('cpu')
    if use_gpu:
        if not torch.cuda.is_available():
            raise Exception('CUDA capable GPU not available.')
        else:
            device = torch.device('cuda:{}'.format(0))
    try:
        print('Job initialized with device {}'.format(torch.cuda.get_device_name(device)))
    except (AssertionError, ValueError):
        print('Job initialized with CPU.')
    return device


class StatComputer(object):
    def __init__(self, loss_type, eval_acc=True):
        self.eval_acc = eval_acc
        self.loss_type = loss_type
        if loss_type == 'mse':
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

    def compute_loss(self, scores, targets):
        if self.loss_type == 'mse':
            loss = self.criterion(scores, targets.float())
        else:
            loss = self.criterion(scores, targets.reshape(-1))

        return loss

    def compute_preds(self, scores):
        if self.loss_type == 'mse':
            preds = scores.flatten().round()
        else:
            _, preds = torch.max(scores, dim=1)  # (b, 1).
            preds = preds.float()

        return preds

    def compute_accuracy(self, scores, targets):
        preds = self.compute_preds(scores)
        targets = targets.float().flatten()  # (b,)
        n_correct = (preds == targets).sum().float()
        n_total = torch.tensor(targets.shape[0]).float()
        accuracy = torch.div(n_correct, n_total)
        return accuracy

    def compute_train_stats(self, scores, targets):
        output = {}
        loss = self.compute_loss(scores, targets)
        output['train_loss'] = loss.data.cpu().numpy()
        output['train_mse'] = loss.data.cpu().numpy()
        if self.eval_acc:
            accuracy = self.compute_accuracy(scores, targets)
            output['train_acc'] = accuracy.detach().cpu().numpy()
        return output

    def compute_test_stats(self, scores, targets):
        loss = self.compute_loss(scores, targets)
        output = {}
        output['valid_loss'] = loss.data.cpu().numpy()
        output['valid_mse'] = loss.data.cpu().numpy(),
        if self.eval_acc:
            accuracy = self.compute_accuracy(scores, targets)
            output['valid_acc'] = accuracy.detach().cpu().numpy()
        return output


class PairwiseDotProduct(nn.Module):
    def __init__(self):
        super(PairwiseDotProduct, self).__init__()

    def forward(self, x, y):
        x = x.unsqueeze(1)  # (b, 1, k).
        y = y.unsqueeze(2)  # (b, k, 1).
        scores = torch.bmm(x, y)  # (b, 1, 1).
        scores = scores.reshape(x.shape[0], 1)
        return scores


class FMBaseline(nn.Module):

    def __init__(self, x_dim, u_dim, n_factors, device):
        super(FMBaseline, self).__init__()
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.factors = n_factors
        self.device = device
        self.criterion = nn.MSELoss()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(x_dim, 1)
        self.item_embedder = nn.Linear(x_dim, n_factors)
        self.user_embedder = nn.Linear(u_dim, n_factors)
        self.pdist_dot = PairwiseDotProduct()
        self.pdist_euclid = nn.PairwiseDistance(keepdim=True)
        self.init_params = {
            'x_dim': x_dim,
            'u_dim': u_dim,
            'n_factors': n_factors,
            'device': device
        }
        self.stat_computer = StatComputer(loss_type='mse', eval_acc=False)

    def forward(self, x, u):
        scores_lin = self.linear(x)  # (b, 1).
        u_embed = self.user_embedder(u)  # (b, n_factors).
        x_embed = self.item_embedder(x)  # (b, n_factors).
        scores_inter = self.pdist_dot(u_embed, x_embed)  # (b, 1).
        # scores_inter = self.pdist_euclid(u_embed, x_embed)
        # scores_inter = self.pdist_rbf(u_embed, x_embed)
        scores = scores_lin + scores_inter
        # scores = scores_lin + 1 - scores_inter
        # scores = self.sigmoid(scores)
        return scores

    def forward_special(self, x, u_embed):
        x_embed = self.item_embedder(x)  # (b, n_factors).
        scores_lin = self.linear(x)
        scores_inter = self.pdist_dot(u_embed, x_embed)  # (b, 1).
        scores = scores_lin + scores_inter
        return scores

    def process_batch(self, batch):
        """
        transforms batch returned by a sampler into correct format for the forward function of the model. the
        factorization machine uses a Sampler from setup_fm.
        params:
            batch: (x, z, u, y).
                x (tensor): (batch_size, x_dim).
                y (tensor): (batch_size, 1).
        """
        x, u, targets = batch
        x = x.to(device=self.device)
        u = u.to(device=self.device)
        targets = targets.to(device=self.device)
        forward_params = {'x': x, 'u': u}
        return forward_params, targets


class ExperimentIO(object):

    def __init__(self):
        pass

    @staticmethod
    def load_model(model_class, filename):
        state = torch.load(f=filename)
        init_params = state['init_params']
        model = model_class(**init_params)
        model.load_state_dict(state_dict=state['network'])
        return model

    @staticmethod
    def load_checkpoint(model, optimizer, filename):
        state = torch.load(f=filename)
        model.load_state_dict(state_dict=state['network'])
        optimizer.load_state_dict(state_dict=state['optimizer'])
        return model, optimizer

    @staticmethod
    def save_checkpoint(model, optimizer, current_epoch, dirname):
        state = dict()
        state['network'] = model.state_dict()  # save network parameter and other variables.
        state['init_params'] = model.init_params
        state['optimizer'] = optimizer.state_dict()

        filename = os.path.join(dirname, 'epoch_{}'.format(current_epoch))
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        torch.save(state, f=filename)

    @staticmethod
    def save_epoch_stats(epoch_stats, filename, first_line=False):
        """
        tasks:
        - if directory does not exist it will be created.
        - if file already exists then content get's
        params:
            epoch_stats: dict {}
        remarks:
            (1) use mode = +w to overwrite.
            (2) such that column names are in a desired order.
        """

        if type(epoch_stats) is not OrderedDict:  # (2)
            raise Exception('epoch_stats must be an ordered dict. got: {}'.format(type(epoch_stats)))

        if first_line:
            mode = '+w'
        else:
            mode = '+a'

        dirname = os.path.dirname(filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        with open(filename, mode) as f:
            if first_line:
                header = ','.join([k for k in epoch_stats.keys()])
                f.write(header + '\n')

            line = ','.join(['{:.4f}'.format(value) for value in epoch_stats.values()])
            f.write(line + '\n')


class UserItemSampler(object):
    def __init__(self, rels, user_ids, item_ids, device, use_content, item_cat_seq_path):
        self.obs = iter(rels)

        self.user_ids = user_ids
        self.item_ids = item_ids
        self.use_content = use_content
        self.item_cat_seq_path = item_cat_seq_path
        self.device = device
        self.uid2idx = {user_id: i for i, user_id in enumerate(self.user_ids)}
        self.xid2idx = {item_id: i for i, item_id in enumerate(self.item_ids)}
        self.num_users = len(self.user_ids)
        self.num_items = len(self.item_ids)
        self._size = len(rels)
        self.user_matrix = torch.eye(self.num_users).to(device=self.device)
        self.item_matrix = torch.eye(self.num_items).to(device=self.device)
        self.item_cat_dict = self.create_item_cat_dict()

    def __len__(self):
        return self._size

    def create_item_cat_dict(self):
        df = pd.read_csv(self.item_cat_seq_path)
        records = df.to_dict('records')
        out = {}
        for i in range(len(records)):
            item_id = records[i]['item_id']
            item_cat_seq_json = records[i]['item_cat_seq']
            item_cat_vec = torch.tensor(json.loads(item_cat_seq_json)).reshape(1, -1)
            out[item_id] = item_cat_vec.float().to(device=self.device)
        return out

    def batch(self, batch_size=1):

        while True:
            u = []
            x = []
            targets = []

            for _ in range(batch_size):
                try:
                    user_id, item_id, rel = next(self.obs)
                    user_vec = self.user_matrix[self.uid2idx[user_id]].reshape(1, self.user_matrix.shape[1])
                    item_vec = self.item_matrix[self.xid2idx[item_id]].reshape(1, self.item_matrix.shape[1])
                    if self.use_content:
                        item_vec = torch.cat([item_vec, self.item_cat_dict[item_id]], dim=1)
                    u.append(user_vec)
                    x.append(item_vec)
                    targets.append(torch.tensor(rel, dtype=torch.long).reshape(1, 1))
                except StopIteration:
                    break

            if len(targets) > 0:
                u = torch.cat(u, dim=0)
                x = torch.cat(x, dim=0)
                targets = torch.cat(targets, dim=0).to(device=self.device)
                batch = (x, u, targets)
                yield batch
            else:
                break


class Tester(object):
    def __init__(self, partition, batch_size, device, use_content, item_cat_seq_path):
        self.partition = partition
        self.batch_size = batch_size
        self.device = device
        self.use_content = use_content
        self.item_cat_seq_path = item_cat_seq_path

    @staticmethod
    def _valid_iter(model, batch):
        model.eval()
        forward_params, targets = model.process_batch(batch)
        scores = model.forward(**forward_params)
        output = model.stat_computer.compute_test_stats(scores, targets)
        return output

    def reset_sampler(self):
        self.sampler = UserItemSampler(
            rels=self.partition.valid.obs,
            user_ids=self.partition.train.user_ids,
            item_ids=self.partition.train.item_ids,
            device=self.device,
            use_content=self.use_content,
            item_cat_seq_path=self.item_cat_seq_path
        )

    def __call__(self, model, current_epoch):
        batch_stats = defaultdict(lambda: [])
        results = OrderedDict({})
        self.reset_sampler()

        with tqdm(total=len(self.sampler)) as pbar:
            for i, batch in enumerate(self.sampler.batch(self.batch_size)):
                output = self._valid_iter(model, batch)
                for k, v in output.items():
                    batch_stats[k].append(output[k])
                description = 'epoch: {} '.format(current_epoch) + \
                              ' '.join(["{}: {:.4f}".format(k, np.mean(v)) for k, v in batch_stats.items()])
                pbar.update(batch[0].shape[0])
                pbar.set_description(description)

        for k, v in batch_stats.items():
            results[k] = np.around(np.mean(v), decimals=4)

        return results


class Trainer(object):
    def __init__(self, partition, batch_size, device, use_content, item_cat_seq_path):
        self.partition = partition
        self.batch_size = batch_size
        self.device = device
        self.use_content = use_content
        self.item_cat_seq_path = item_cat_seq_path

    @staticmethod
    def _train_iter(batch, model, optimizer):
        model.train()
        forward_params, targets = model.process_batch(batch)
        scores = model.forward(**forward_params)
        output = model.stat_computer.compute_train_stats(scores, targets)
        loss = model.stat_computer.compute_loss(scores, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return output, model, optimizer

    def reset_sampler(self):
        self.sampler = UserItemSampler(
            rels=self.partition.train.obs,
            user_ids=self.partition.train.user_ids,
            item_ids=self.partition.train.item_ids,
            device=self.device,
            use_content=self.use_content,
            item_cat_seq_path=self.item_cat_seq_path
        )

    def __call__(self, model, optimizer, current_epoch):
        batch_stats = defaultdict(lambda: [])
        epoch_stats = OrderedDict({})
        self.reset_sampler()

        with tqdm(total=len(self.sampler)) as pbar_train:
            for i, batch in enumerate(self.sampler.batch(self.batch_size)):
                output, model, optimizer = self._train_iter(batch, model, optimizer)

                for k, v in output.items():
                    batch_stats[k].append(output[k])
                description = \
                    'epoch: {} '.format(current_epoch) + \
                    ' '.join(["{}: {:.4f}".format(k, np.mean(v)) for k, v in batch_stats.items()])
                pbar_train.update(batch[0].shape[0])
                pbar_train.set_description(description)

        for k, v in batch_stats.items():
            epoch_stats[k] = np.around(np.mean(v), decimals=4)

        return model, optimizer, epoch_stats


class Experiment(object):

    def __init__(self, data_name, train_name, valid_name, batch_size, num_epochs, lr, n_factors, use_content,
                 item_cat_seq_path, use_gpu, job_dir):

        """Performs a training and validation experiment.

        Args:
            model: The model to perform the experiment with.
            model_params: The parameters to initialize the model with. These are saved when creating a checkpoint.
            optimizer: The optimizer used during training.
            num_epochs: The number of epochs to train for.
            trainer: The <Trainer>. Responsible for performing a training epoch.
            tester: The <Tester>. Responsible for performing a testing/ validation epoch.
            experiment_dir: This directory contains the checkpoints and the results of an experiment.
        """

        self.data_name = data_name
        self.train_name = train_name
        self.valid_name = valid_name
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.n_factors = n_factors
        self.use_content = use_content
        self.item_cat_seq_path = item_cat_seq_path
        self.use_gpu = use_gpu
        self.device = set_device(self.use_gpu)

        self.data = datasets.Data(data_name)
        train_obs = parse(self.data.get_ratings(train_name))  # [(user_id, item_id, rating)]
        valid_obs = parse(self.data.get_ratings(valid_name))
        self.partition = Partition(train_obs, valid_obs)

        self.trainer = Trainer(self.partition, self.batch_size, self.device, self.use_content, self.item_cat_seq_path)
        self.tester = Tester(self.partition, self.batch_size, self.device, self.use_content, self.item_cat_seq_path)

        self.model = FMBaseline(
            x_dim=self.get_item_num_features(),
            u_dim=len(self.partition.train.user_ids),
            n_factors=self.n_factors,
            device=self.device
        )

        self.model.to(device=self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # todo: model_params.

        self.job_dir = job_dir
        self.results_path = os.path.join(self.job_dir, 'results.txt')
        self.checkpoints_dir = os.path.join(self.job_dir, 'checkpoints')
        os.makedirs(self.checkpoints_dir, exist_ok=False)

    def get_item_num_features(self):
        if self.use_content:
            return len(self.partition.train.item_ids) + self.data.get_num_categories()
        else:
            return len(self.partition.train.item_ids)

    def run(self):
        for current_epoch in range(self.num_epochs):
            # valid_results = self.tester_module(self.model, current_epoch)
            self.model, self.optimizer, train_results = self.trainer(self.model, self.optimizer, current_epoch)
            if current_epoch % 1 == 0 and current_epoch > -1:
                valid_results = self.tester(self.model, current_epoch)
            else:
                valid_results = {}
            sys.stderr.write('\n')
            results = OrderedDict({})
            results['current_epoch'] = current_epoch
            for k in train_results.keys():
                results[k] = train_results[k]
            for k in valid_results.keys():
                results[k] = valid_results[k]
            ExperimentIO.save_checkpoint(self.model, self.optimizer, current_epoch, dirname=self.checkpoints_dir)
            ExperimentIO.save_epoch_stats(results, self.results_path, first_line=(current_epoch == 0))


class Subset(object):
    def __init__(self):
        self.user_ids = []
        self.item_ids = []
        self.obs = []

    def add(self, user_id, item_id, rating):
        self.obs.append((user_id, item_id, rating))
        if user_id not in self.user_ids:
            self.user_ids.append(user_id)
        if item_id not in self.item_ids:
            self.item_ids.append(item_id)


class Partition(object):
    def __init__(self, train_obs, valid_obs):
        self.train = Subset()
        self.valid = Subset()
        for user_id, item_id, rating in train_obs:
            self.train.add(user_id, item_id, rating)
        for user_id, item_id, rating in valid_obs:
            if user_id in self.train.user_ids and item_id in self.train.item_ids:
                self.valid.add(user_id, item_id, rating)


def parse(df_ratings):
    # ratings are df header = user_id,item_id,rating,time
    # output: [(user_id, item_id, rating)]
    obs = []
    records = df_ratings.to_dict(orient='records')
    for record in records:
        obs.append((record['user_id'], record['item_id'], record['rating']))
    return obs


job_id = random.randint(0, 1000000)
print('job id: {}'.format(job_id))


@click.command()
@click.option('--data_name', type=str, default='ml-100k')
@click.option('--train_name', type=str, default='train_0')
@click.option('--valid_name', type=str, default='valid_0')
@click.option('--batch_size', type=int, default=256)
@click.option('--num_epochs', type=int, default=10)
@click.option('--lr', type=float, default=0.01)
@click.option('--n_factors', type=int, default=4)
@click.option('--use_content', type=bool, default=True)
@click.option('--item_cat_seq_path', type=str, default='data/rs_data/ml-100k/item_cat_seq.csv')
@click.option('--use_gpu', type=bool, default=True)
@click.option('--job_dir', type=str, default='training/fm_{}'.format(job_id))
def main(data_name, train_name, valid_name, batch_size, num_epochs, lr, n_factors, use_content, item_cat_seq_path,
         use_gpu, job_dir):

    e = Experiment(
        data_name=data_name,
        train_name=train_name,
        valid_name=valid_name,
        batch_size=batch_size,
        num_epochs=num_epochs,
        lr=lr,
        n_factors=n_factors,
        use_content=use_content,
        item_cat_seq_path=item_cat_seq_path,
        use_gpu=use_gpu,
        job_dir=job_dir
    )
    e.run()


if __name__ == '__main__':
    main()
