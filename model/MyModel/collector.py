import csv
import atexit
import os

import numpy as np


class Collector:
    def __init__(self, path):
        self.path = path
        self.data = {}
        self.np_array = {}
        atexit.register(self.save)
        self.index = {}

    def write(self, data, var_name, write_every=200):
        if var_name not in self.data:
            self.data[var_name] = []
            self.index[var_name] = 0

        if self.index[var_name] % write_every == 0:
            self.data[var_name].append(data)
        self.index[var_name] += 1

    def write_np_array(self, data, var_name, write_every=200):
        if var_name not in self.np_array:
            self.np_array[var_name] = []
            self.index[var_name] = 0

        if self.index[var_name] % write_every == 0:
            self.np_array[var_name].append(data)
        self.index[var_name] += 1

    def save(self):
        for key in self.data:
            with open(self.path + os.sep + key + '.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(self.data[key])

        for key in self.np_array:
            self.np_array[key] = np.concatenate(self.np_array[key], axis=0)
            np.save(self.path + os.sep + key + '.npy', self.np_array[key])

