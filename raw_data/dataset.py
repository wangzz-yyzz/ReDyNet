import datetime
import os
import random

import numpy as np
import pandas as pd
import torch

from raw_data.normalization import StandardScaler, NormalScaler, NoneScaler, MinMax01Scaler, MinMax11Scaler, LogScaler
from raw_data.utils import generate_dataloader


class Dataset:
    def __init__(self, args):
        self.weather_dim = None
        self.date_dim = None
        self.feature_dim = None
        self.num_batches = None
        self.dataset = args.dataset
        self.data_path = 'raw_data/' + self.dataset + '/'

        self.input_window = args.input_window
        self.output_window = args.output_window

        self.train_rate = args.train_rate
        self.valid_rate = args.valid_rate
        self.test_rate = args.test_rate

        self.scaler_type = args.scaler_type
        self.output_dim = args.output_dim

        self.batch_size = args.batch_size
        self.pad_with_last_sample = args.pad_with_last_sample
        self.feature_name = {'X': 'float', 'y': 'float'}
        self.num_workers = args.num_workers

        self.weight_col = 'connection'

        self.bidir_adj_mx = args.bidir_adj_mx
        self.set_weight_link_or_dist = args.set_weight_link_or_dist
        self.init_weight_inf_or_zero = args.init_weight_inf_or_zero
        self.calculate_weight_adj = args.calculate_weight_adj
        self.weight_adj_epsilon = args.weight_adj_epsilon

        self.adj_mx = None
        self._load_rel()

        self.load_external = args.load_external
        self.ext_dim = args.ext_dim
        self.add_day_in_week = args.add_day_in_week
        self.add_time_in_day = args.add_time_in_day
        self.add_weather = args.add_weather
        self.normal_external = args.normal_external
        self.ext_scaler_type = args.ext_scaler_type

    def _load_geo(self):
        geofile = pd.read_csv(self.data_path + self.dataset + '.geo')
        self.geo_ids = list(geofile['geo_id'])
        self.num_nodes = len(self.geo_ids)
        self.geo_to_ind = {}
        for index, idx in enumerate(self.geo_ids):
            self.geo_to_ind[idx] = index

    def _load_dyna(self):
        self._load_geo()
        if self.dataset == 'HZMETRO' or self.dataset == 'BJMETRO':
            dynafile = pd.read_csv(self.data_path + self.dataset + '_new.dyna')
        else:
            dynafile = pd.read_csv(self.data_path + self.dataset + '.dyna')
        selected = ['time', 'entity_id', 'inflow', 'outflow']
        dynafile = dynafile[selected]
        self.timeslots = list(dynafile['time'][:int(dynafile.shape[0] / len(self.geo_ids))])
        print('timeslots:', len(self.timeslots))
        self.timeslots = list(map(lambda x: x.replace('T', ' ').replace('Z', ''), self.timeslots))
        self.timeslots = np.array(self.timeslots, dtype='datetime64[ns]')

        feature_dim = len(dynafile.columns) - 2
        df = dynafile[dynafile.columns[-feature_dim:]]
        len_time = len(self.timeslots)
        data = []
        for i in range(0, df.shape[0], len_time):
            data.append(df[i:i + len_time].values)
        data = np.array(data, dtype=float)
        data = data.swapaxes(0, 1)

        return data

    def _add_external_information_3d(self, df):
        num_samples, num_nodes, feature_dim = df.shape
        is_time_nan = np.isnan(self.timeslots).any()
        data_list = [df]
        date_list = []
        if self.add_time_in_day and not is_time_nan:
            time_in_day = self.timeslots.astype("datetime64[h]").astype(int) % 24
            time_in_day = time_in_day.astype(int)
            time_in_day = time_in_day.reshape(-1, 1)
            date_list.append(time_in_day)
            time_in_day = np.tile(time_in_day, [1, num_nodes])
            time_in_day = np.expand_dims(time_in_day, axis=-1)
            data_list.append(time_in_day)

        if self.add_day_in_week and not is_time_nan:
            y_m_d = self.timeslots.astype("datetime64[D]").astype(int)
            y_m_d = list(map(lambda x: datetime.datetime.utcfromtimestamp(x * 3600 * 24), y_m_d))
            day_in_week = list(map(lambda x: x.weekday(), y_m_d))
            day_in_week = np.array(day_in_week)
            day_in_week = day_in_week.astype(int)
            day_in_week = day_in_week.reshape(-1, 1)
            date_list.append(day_in_week)

            day_in_week = np.tile(day_in_week, [1, num_nodes])
            day_in_week = np.expand_dims(day_in_week, axis=-1)
            data_list.append(day_in_week)

        if self.add_weather:
            if not os.path.exists(self.data_path + self.dataset + '.env'):
                weather_raw = pd.read_csv(self.data_path + self.dataset + '.wea')
                weather_raw = weather_raw[["date", "T", "P", "U", "Ff", "RRR"]]
                weather_raw["date"] = pd.to_datetime(weather_raw["date"], format='%Y-%m-%d %H:%M:%S')
                weather_raw["date"] = weather_raw["date"].values.astype("datetime64[s]")
                weather = []
                for i in range(self.timeslots.shape[0]):
                    time = self.timeslots[i]
                    nearest_time = weather_raw["date"].values[np.argmin(np.abs(weather_raw["date"].values - time))]
                    weather_info = weather_raw[weather_raw["date"] == nearest_time]
                    weather_info = weather_info.values[0][1:]
                    weather.append(weather_info)
                weather = np.array(weather, dtype=float)
                print('weather_raw:', weather.shape)
                pd.DataFrame(weather).to_csv(self.data_path + self.dataset + '.env', index=False)
            else:
                weather = pd.read_csv(self.data_path + self.dataset + '.env').values
            weather = weather.astype(float)
            weather = np.expand_dims(weather, axis=1)
            weather = np.tile(weather, [num_nodes, 1])
            data_list.append(weather)

        data = np.concatenate(data_list, axis=-1)
        print("zero ratio:", np.sum(np.isnan(data)) / data.size, "nan ratio:", np.sum(np.isnan(data)) / data.size)
        data[np.isnan(data)] = 0
        return data

    def _generate_input_data(self, df):
        num_samples = df.shape[0]
        x_offsets = np.sort(np.concatenate((np.arange(-self.input_window + 1, 1, 1),)))
        y_offsets = np.sort(np.arange(1, self.output_window + 1, 1))

        x, y = [], []
        min_t = abs(min(x_offsets))
        max_t = abs(num_samples - abs(max(y_offsets)))
        for t in range(min_t, max_t):
            x_t = df[t + x_offsets, ...]
            y_t = df[t + y_offsets, ...]
            x.append(x_t)
            y.append(y_t)
        x = np.stack(x, axis=0)
        y = np.stack(y, axis=0)
        return x, y

    def _generate_data(self):
        x_list, y_list = [], []
        df = self._load_dyna()
        if self.load_external:
            df = self._add_external_information_3d(df)
        x, y = self._generate_input_data(df)
        x_list.append(x)
        y_list.append(y)
        x = np.concatenate(x_list)
        y = np.concatenate(y_list)
        return x, y

    def _split_train_val_test(self, x, y):
        num_samples = x.shape[0]
        num_test = round(num_samples * self.test_rate)
        num_train = round(num_samples * self.train_rate)
        num_val = num_samples - num_test - num_train

        # train
        x_train, y_train = x[:num_train], y[:num_train]
        # val
        x_val, y_val = x[num_train: num_train + num_val], y[num_train: num_train + num_val]
        # test
        x_test, y_test = x[-num_test:], y[-num_test:]

        return x_train, y_train, x_val, y_val, x_test, y_test

    def _generate_train_val_test(self):
        x, y = self._generate_data()
        return self._split_train_val_test(x, y)

    def _get_scalar(self, scaler_type, x_train, y_train):
        if scaler_type == "normal":
            scaler = NormalScaler(maxx=max(x_train.max(), y_train.max()))
        elif scaler_type == "standard":
            scaler = StandardScaler(mean=x_train.mean(), std=x_train.std())
        elif scaler_type == "minmax01":
            scaler = MinMax01Scaler(
                maxx=max(x_train.max(), y_train.max()), minn=min(x_train.min(), y_train.min()))
        elif scaler_type == "minmax11":
            scaler = MinMax11Scaler(
                maxx=max(x_train.max(), y_train.max()), minn=min(x_train.min(), y_train.min()))
        elif scaler_type == "log":
            scaler = LogScaler()
        elif scaler_type == "none":
            scaler = NoneScaler()
        else:
            raise ValueError('Scaler type error!')
        return scaler

    def get_data(self):
        x_train, y_train, x_val, y_val, x_test, y_test = self._generate_train_val_test()
        self.feature_dim = x_train.shape[-1]
        self.ext_dim = self.feature_dim - self.output_dim
        self.date_dim = 2
        self.weather_dim = self.ext_dim - self.date_dim
        self.scaler = self._get_scalar(self.scaler_type, x_train[..., :self.output_dim], y_train[..., :self.output_dim])
        if self.normal_external and self.load_external:
            print("norm weather dim:", self.weather_dim, "ext_dim:", self.ext_dim)
            self.ext_scaler = self._get_scalar(self.ext_scaler_type,
                                               x_train[..., -self.weather_dim:], y_train[..., -self.weather_dim:])
        x_train[..., :self.output_dim] = self.scaler.transform(x_train[..., :self.output_dim])
        y_train[..., :self.output_dim] = self.scaler.transform(y_train[..., :self.output_dim])
        x_val[..., :self.output_dim] = self.scaler.transform(x_val[..., :self.output_dim])
        y_val[..., :self.output_dim] = self.scaler.transform(y_val[..., :self.output_dim])
        x_test[..., :self.output_dim] = self.scaler.transform(x_test[..., :self.output_dim])
        y_test[..., :self.output_dim] = self.scaler.transform(y_test[..., :self.output_dim])
        # external normal
        if self.normal_external and self.load_external:
            x_train[..., -self.weather_dim:] = self.ext_scaler.transform(x_train[..., -self.weather_dim:])
            y_train[..., -self.weather_dim:] = self.ext_scaler.transform(y_train[..., -self.weather_dim:])
            x_val[..., -self.weather_dim:] = self.ext_scaler.transform(x_val[..., -self.weather_dim:])
            y_val[..., -self.weather_dim:] = self.ext_scaler.transform(y_val[..., -self.weather_dim:])
            x_test[..., -self.weather_dim:] = self.ext_scaler.transform(x_test[..., -self.weather_dim:])
            y_test[..., -self.weather_dim:] = self.ext_scaler.transform(y_test[..., -self.weather_dim:])

        train_data = list(zip(x_train, y_train))
        eval_data = list(zip(x_val, y_val))
        test_data = list(zip(x_test, y_test))
        train_dataloader, eval_dataloader, test_dataloader = generate_dataloader(
            train_data, eval_data, test_data, self.feature_name,
            self.batch_size, self.num_workers)
        self.num_batches = len(train_dataloader)

        return train_dataloader, eval_dataloader, test_dataloader

    def _load_rel(self):
        self._load_geo()
        relfile = pd.read_csv(self.data_path + self.dataset + '.rel')
        edges_num = 0

        self.distance_df = relfile[~relfile[self.weight_col].isna()][[
            'origin_id', 'destination_id', self.weight_col]]

        self.adj_mx = np.zeros((len(self.geo_ids), len(self.geo_ids)), dtype=np.float32)
        if self.init_weight_inf_or_zero.lower() == 'inf' and self.set_weight_link_or_dist.lower() != 'link':
            self.adj_mx[:] = np.inf
        for row in self.distance_df.values:
            if row[0] not in self.geo_to_ind or row[1] not in self.geo_to_ind:
                continue
            if row[2] == 0:
                continue
            if self.set_weight_link_or_dist.lower() == 'dist':
                self.adj_mx[self.geo_to_ind[row[0]], self.geo_to_ind[row[1]]] = row[2]
                if self.bidir_adj_mx:
                    self.adj_mx[self.geo_to_ind[row[1]], self.geo_to_ind[row[0]]] = row[2]
            else:
                self.adj_mx[self.geo_to_ind[row[0]], self.geo_to_ind[row[1]]] = 1
                if self.bidir_adj_mx:
                    self.adj_mx[self.geo_to_ind[row[1]], self.geo_to_ind[row[0]]] = 1
            edges_num += 1

        print('edges_num:', edges_num)
        if self.calculate_weight_adj:
            self._calculate_adjacency_matrix()

    def _calculate_adjacency_matrix(self):
        distances = self.adj_mx[~np.isinf(self.adj_mx)].flatten()
        std = distances.std()
        self.adj_mx = np.exp(-np.square(self.adj_mx / std))
        self.adj_mx[self.adj_mx < self.weight_adj_epsilon] = 0

    def get_data_feature(self):
        print('adj_mx:', self.adj_mx.shape, 'num_nodes:', self.num_nodes, 'feature_dim:', self.feature_dim,
              'output_dim:', self.output_dim, 'ext_dim:', self.ext_dim, 'num_batches:', self.num_batches)
        return {"scaler": self.scaler, "adj_mx": self.adj_mx, "ext_dim": self.ext_dim,
                "num_nodes": self.num_nodes, "feature_dim": self.feature_dim,
                "output_dim": self.output_dim, "num_batches": self.num_batches}


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
