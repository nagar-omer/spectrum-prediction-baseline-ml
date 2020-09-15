from copy import deepcopy
from sklearn import preprocessing
import torch
from torch.utils.data import Dataset
import numpy as np
from blackbody_power_density import gaussian_power_density, um
import json
from scipy import stats
import pickle


class PhotocurrentData(Dataset):
    def __init__(self, path_to_data, params):
        self._params = params if type(params) is dict else json.load(open(params, "rt"))
        self._wave_lengths = np.linspace(um(self._params["lambda_min"]),
                                         um(self._params["lambda_max"]),
                                         self._params["n_lambda"])
        self._displacement_fields, self._temperatures, self._full_data = self._load_data(path_to_data)
        # pickle.dump((self._displacement_fields, self._temperatures, self._full_data), open("full_data.pkl", "wb"))
        # self._displacement_fields, self._temperatures, self._full_data = pickle.load(open("full_data.pkl", "rb"))
        if self._params["mode"] == "P2D":  # if spectrum to photocurrent than reverse data
            self._reverse_p2d()
        self._data = deepcopy(self._full_data)
        self._temperatures_idx = {temp: i for i, temp in enumerate(self._temperatures)}
        self._norm_data()

    def _reverse_p2d(self):
        data = []
        for x, y in self._full_data:
            data.append((y, x))
        self._full_data = data

    def filter(self, indices):
        data = []
        for d in indices:
            data.append(self._full_data[d])
        self._data = data
        self._norm_data()

    @property
    def wavelengths(self):
        return self._wave_lengths

    def _planck(self, temperature):
        return gaussian_power_density(temperature=temperature,
                                      wavelengths=self._wave_lengths)

    def _load_data(self, path_to_data):
        """
        load spectral responsivity csv file as matrix
        :param path_to_data: path to csv file
        :return: tmeperatures values, displacement field values and photocurrent matrix
        """
        data_file = open(path_to_data, "rt", encoding="utf-8")
        # extract displacement fields header
        D = [float(x) for x in data_file.readline().strip().split(",")[1:]]
        temperatures = []  # temperature header
        photocurrent = []  # photocurrent matrix
        for row in data_file:
            row = row.strip().split(",")
            temperatures.append(float(row[0]))  # build temperature header
            photocurrent.append(([float(x) for x in row[1:]], self._planck(float(row[0]))))  # build matrix
        return D, temperatures, photocurrent

    def _norm_data(self):
        X = np.asarray([d[0] for d in self._data])
        Y = np.asarray([d[1] for d in self._data])
        # Y = Y / Y.max()
        X = stats.zscore(X)
        # Y = stats.zscore(Y)
        data = []
        for x, y in zip(X, Y):
            data.append((x, y))
        self._data = data

    def __getitem__(self, item):
        return torch.Tensor(self._data[item][0]), torch.Tensor(self._data[item][1])

    def __len__(self):
        return len(self._data)


def sanity_test():
    from torch.utils.data import DataLoader
    ds = PhotocurrentData("Spectral Responsivity Data Summary.csv", params="model_params.json")

    dl = DataLoader(ds,
                    shuffle=True,
                    batch_size=2)

    for src_, dst_ in dl:
        print(src_)
        print(dst_)


if __name__ == '__main__':
    sanity_test()
