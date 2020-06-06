import wget
from zipfile import ZipFile
import os
import json
import pandas as pd
from sklearn.model_selection import KFold


def create(name):
    if name == 'ml-100k':
        d = MovieLens100k()
        d.create()


class Data(object):
    def __init__(self, name):
        self.name = name
        try:
            present = set(next(os.walk('data/rs_data'))[1])  # e.g. ml-100k, ml-1m
        except StopIteration:
            present = set()
        if self.name not in present:
            create(self.name)

    def get_ratings(self, which):
        path = 'data/rs_data/{}/{}.csv'.format(self.name, which)
        return pd.read_csv(path, sep=',')


class MovieLens100k(object):
    def __init__(self):
        self.rs_dir_path = 'data/rs_dir'
        self.name = 'ml-100k'
        self.rs_dir_path = os.path.join(self.rs_dir_path, 'ml-100k')
        self.raw_data_path = os.path.join('data/raw_data', 'ml-100k.zip')
        self.download_url = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'

    def create(self):
        os.makedirs(os.path.dirname(self.raw_data_path), exist_ok=True)
        wget.download(
            url=self.download_url,
            out=self.raw_data_path
        )
        if not os.path.isdir(self.rs_dir_path):
            os.makedirs(self.rs_dir_path)
        self.create_ratings()
        self.create_folds()
        self.create_item_categories()
        print('Finished creating {}.'.format(self.name))

    def create_ratings(self):
        lines = ['user_id,item_id,rating,time\n']
        with ZipFile(self.raw_data_path, mode='r') as archive:
            for line in archive.open('ml-100k/u.data'):
                record = line.decode('utf-8').strip('\n').split('\t')
                line = ','.join(record) + '\n'
                lines.append(line)
        with open(os.path.join(self.rs_dir_path, 'ratings.csv'), 'w+') as file:
            file.writelines(lines)

    def create_folds(self):
        print('Creating folds ... ')
        path = os.path.join(self.rs_dir_path, self.name, 'ratings.csv')
        df = pd.read_csv(path, sep=',')
        df.to_csv()
        kf = KFold(n_splits=5, random_state=0, shuffle=True)
        kf2 = KFold(n_splits=20)
        for i, (train_full_idx, test_idx) in enumerate(kf.split(df)):
            train_idx, valid_idx = next(kf2.split(train_full_idx))
            df_train_full = df.iloc[train_full_idx, :]
            df_test = df.iloc[test_idx, :]
            df_train = df.iloc[train_idx, :]
            df_valid = df.iloc[valid_idx, :]
            df_train_full.to_csv(os.path.join(self.rs_dir_path, self.name, 'train_full_{}.csv'.format(i)), index=False,
                                 mode='w+')
            df_test.to_csv(os.path.join(self.rs_dir_path, self.name, 'test_{}.csv'.format(i)), index=False, mode='w+')
            df_train.to_csv(os.path.join(self.rs_dir_path, self.name, 'train_{}.csv'.format(i)), index=False, mode='w+')
            df_valid.to_csv(os.path.join(self.rs_dir_path, self.name, 'valid_{}.csv'.format(i)), index=False, mode='w+')
            print('Finished creating fold {}'.format(i))
        return pd.read_csv(path, sep=',')

    def create_item_categories(self):
        lines = ['item_id,item_cat_seq\n']
        with ZipFile(self.raw_data_path, mode='r') as archive:
            for line in archive.open('ml-100k/u.item'):
                record = line.decode('iso-8859-1').strip('\n').split('|')
                item_id = int(record[0])
                genres = [int(record[i]) for i in range(5, len(record))]
                line = '{},"{}"'.format(item_id, json.dumps(genres))
                lines.append(line + '\n')
        with open(os.path.join(self.rs_dir_path, 'item_cat_seq.csv'), 'w+') as file:
            file.writelines(lines)
