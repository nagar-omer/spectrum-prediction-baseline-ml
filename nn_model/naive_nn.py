from torch.nn import Module, Linear
import json
from torch.distributions.normal import Normal
import torch


class NaiveSpectralModel(Module):
    def __init__(self, wavelengths, params):
        super().__init__()
        self._wavelengths = wavelengths.tolist()
        self._wavelengths = [self._wavelengths[0] - (self._wavelengths[1] - self._wavelengths[0])] + self._wavelengths
        self._params = params if type(params) is dict else json.load(open(params, "rt"))
        if self._params["mode"] == "D2P":
            self._linear_1 = Linear(self._params["n_displacement_fields"], self._params["n_lambda"], bias=True)
        else:
            self._linear_1 = Linear(self._params["n_lambda"], self._params["n_displacement_fields"], bias=True)

        if self._params["mode"] == "D2P":
            self._linear_2 = Linear(self._params["n_lambda"], self._params["n_lambda"])
        else:
            self._linear_2 = Linear(self._params["n_displacement_fields"], self._params["n_displacement_fields"])

    def forward(self, photocurrent):
        # x = torch.tanh(self._linear_1(photocurrent))
        # return self._linear_2(x)
        x = self._linear_1(photocurrent)
        return x


if __name__ == '__main__':
    from data_loader import PhotocurrentData
    from torch.utils.data import DataLoader

    ds = PhotocurrentData("Spectral Responsivity Data Summary.csv", params="model_params.json")
    dl = DataLoader(ds,
                    shuffle=True,
                    batch_size=2)
    model = NaiveSpectralModel(wavelengths=ds.wavelengths, params="model_params.json")
    for src_, dst_ in dl:
        model(src_)
