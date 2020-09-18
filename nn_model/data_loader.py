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

    def data_std(self):
        X = np.asarray([d[0] for d in self._data])
        Y = np.asarray([d[1] for d in self._data])
        return np.mean(X, axis=0), np.std(X, axis=0), np.mean(Y, axis=0), np.std(Y, axis=0),

    def _reverse_p2d(self):
        data = []
        for x, y in self._full_data:
            data.append((y, x))
        self._full_data = data

    def filter(self, indices):
        data = []
        for d in indices:
            data.append(self._full_data[d])
        self._data = deepcopy(data)
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

    def normlize(self, X):
        x = deepcopy(X)
        if self._params["normalization"] == "center":
            mean_data = np.asarray([d[0] for d in self._full_data]).mean(axis=0).tolist()
            x = X - np.asarray([mean_data for _ in range(21)])
        if self._params["normalization"] == "zscore":
            ss = preprocessing.StandardScaler()
            ss.fit([d[0] for d in self._full_data])
            ss.fit(np.asarray([d[0] for d in self._full_data]))
            x = ss.transform(X)
        if "scale" in self._params["normalization"]:
            scale = float(self._params["normalization"][6:-1])
            x = X / scale
        return x

    def _norm_data(self):
        X = self.normlize(np.asarray([d[0] for d in self._data]))
        Y = np.asarray([d[1] for d in self._data])

        data = []
        for x, y in zip(X, Y):
            data.append((x, y))
        self._data = data

    def __getitem__(self, item):
        return torch.Tensor(self._data[item][0]), torch.Tensor(self._data[item][1])

    def __len__(self):
        return len(self._data)


def reduce_tick(ticks, k=10):
    new_ticks = []
    for i in range(0, len(ticks), len(ticks) // k):
        new_ticks += [str(ticks[i])] + [""] * (len(ticks) // k - 1)
    new_ticks = new_ticks[:len(ticks)]
    return new_ticks


def sanity_test():
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import seaborn as sns

    params = {
              "lambda_min": 2,
              "lambda_max": 9.5,
              "n_lambda": 35,
              "normalization": "scale(1000)",
              "n_displacement_fields": 41,
              "mode": "D2P"
            }
    ds = PhotocurrentData("Spectral Responsivity Data Summary.csv", params=params)

    dl = DataLoader(ds,
                    shuffle=True,
                    batch_size=2)

    for src_, dst_ in dl:
        print(src_)
        print(dst_)

    D = torch.Tensor([d[0] for d in ds._data])
    P = torch.Tensor([d[1] for d in ds._data])
    ax = sns.heatmap(D,
                     yticklabels=reduce_tick(np.linspace(400, 800, 21).round(2)),
                     xticklabels=reduce_tick(np.linspace(0.1, 0.78, 41).round(2)))
    plt.title("Iph (nA)")
    plt.xlabel("D (V/nm)")
    plt.ylabel("Temperature")
    ax.invert_yaxis()
    plt.show()

    ax = sns.heatmap(P,
                     yticklabels=reduce_tick(np.linspace(400, 800, 21).round(2)),
                     xticklabels=reduce_tick(np.linspace(2, 9.5, 35).round(2)))
    plt.title("Power Density")
    plt.xlabel("Wavelength (\u03BCm)")
    plt.ylabel("Temperature")
    ax.invert_yaxis()
    plt.show()


if __name__ == '__main__':
    sanity_test()
