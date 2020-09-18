from time import sleep

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, MeanSquaredError
import torch
from sklearn.model_selection import KFold
from torch.nn.functional import mse_loss
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from data_loader import PhotocurrentData, reduce_tick
from naive_nn import NaiveSpectralModel
from torch.nn import MSELoss, SmoothL1Loss
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
LOG_INTERVAL = 1


def get_trainer(model, train_loader, val_loader, criterion, optimizer, metrics):
    trainer = create_supervised_trainer(model, optimizer, criterion)
    evaluator = create_supervised_evaluator(model, metrics=metrics)

    @trainer.on(Events.ITERATION_COMPLETED(every=LOG_INTERVAL))
    def log_training_loss(trainer):
        print("Epoch[{}] Loss: {:.2f}".format(trainer.state.epoch, trainer.state.output))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        print("Training Results - Epoch: {}  Avg MSE: {:.2f}"
              .format(trainer.state.epoch, metrics["MSE"]))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        print("Validation Results - Epoch: {}  Avg MSE: {:.2f}"
              .format(trainer.state.epoch, metrics["MSE"]))

    return trainer


class SmoothWeightsLoss:
    def __init__(self, model, weights=None, lambda_l1=1e-3, lambda_rows=1e-1, lambda_cols=1e-1,
                 lambda_norm_rows=1e-1, lambda_norm_cols=1e-1):
        self._lambda_rows = lambda_rows
        self._lambda_cols = lambda_cols
        self._lambda_norm_rows = lambda_norm_rows
        self._lambda_norm_cols = lambda_norm_cols
        self._lambda_l1 = lambda_l1
        self._model = model
        self._weights = weights

    def __call__(self, y, y_hat):
        # mse = mse_loss(x, y)
        mse = (torch.mean(torch.Tensor(self._weights) * (y_hat - y) ** 2)) if self._weights is not None else \
            mse_loss(y, y_hat)
        diff_rows = 0
        norm_rows = 0
        norm_cols = 0

        for i, j in zip(self._model._linear_1.weight,
                        self._model._linear_1.weight[1:]):
            diff_rows += torch.dist(i, j)
            norm_rows += torch.norm(i)
        norm_rows += torch.norm(j)

        diff_cols = 0
        for i, j in zip(self._model._linear_1.weight.T,
                        self._model._linear_1.weight.T[1:]):
            diff_cols += torch.dist(i, j)
            norm_cols += torch.norm(i)
        norm_cols += torch.norm(j)

        l1 = self._model._linear_1.weight.norm().square()
        # l1 = self._model._linear_1.weight.norm().square() + self._model._linear_2.weight.norm().square()
        return mse + self._lambda_l1*l1 + \
               self._lambda_rows*diff_rows + \
               self._lambda_cols*diff_cols + \
               self._lambda_norm_rows / norm_rows + \
               self._lambda_norm_cols / norm_cols


def load_test(file_name):
    D, iph = [], []
    for measure in csv.DictReader(open(file_name, "rt")):
        D.append(float(measure['D (V/nm)']))
        iph.append(float(measure['Iph (nA)']))
    return D, iph


def plot_weights_R_inv(model):
    W = np.asarray(model._linear_1.weight.T.tolist())
    B = np.asarray(model._linear_1.bias.tolist())
    ax = sns.heatmap(W,
                     yticklabels=reduce_tick(np.linspace(0.1, 0.78, W.shape[0]).round(2)),
                     xticklabels=reduce_tick(np.linspace(2, 9.5, W.shape[1]).round(2)))
    plt.title("R⁻¹")
    plt.xlabel("Wavelength (\u03BCm)")
    plt.ylabel("D (V/nm)")
    ax.invert_yaxis()
    plt.show()

    ax.plot(np.linspace(2, 9.5, W.shape[1]), B)
    plt.title("R⁻¹ Bias")
    plt.xlabel("Wavelength (\u03BCm)")


def plot_prediction_R_inv(model, ds):
    X = torch.Tensor([d[0] for d in ds._data])
    Y = torch.Tensor([d[1] for d in ds._data])
    Y_hat = np.asarray(model(X).tolist())

    ax = sns.heatmap(Y,
                     yticklabels=reduce_tick(np.linspace(673, 1073, Y.shape[0]).round(2)),
                     xticklabels=reduce_tick(np.linspace(2, 9.5, Y.shape[1]).round(2)))
    plt.title("Power Density - Ground Truth")
    plt.xlabel("Wavelength (\u03BCm)")
    plt.ylabel("Temperature")
    ax.invert_yaxis()
    plt.show()

    ax = sns.heatmap(Y_hat,
                     yticklabels=reduce_tick(np.linspace(673, 1073, Y_hat.shape[0]).round(2)),
                     xticklabels=reduce_tick(np.linspace(2, 9.5, Y_hat.shape[1]).round(2)))
    plt.title("Power Density - Model Prediction")
    plt.xlabel("Wavelength (\u03BCm)")
    plt.ylabel("Temperature")
    ax.invert_yaxis()
    plt.show()


def plot_test_R_inv(model, file_name, title="Test", normlizer=lambda x: [x]):
    d_test, iph_test = load_test(file_name)
    iph_hat = np.asarray(model(torch.Tensor(normlizer(np.asarray([d_test])))).tolist())
    print("="*100, np.linspace(2, 9.5, iph_hat.shape[1])[iph_hat[0].argmax()])
    plt.plot(np.linspace(2, 9.5, iph_hat.shape[1]), iph_hat[0].tolist(), label="Prediction")
    # plt.plot(np.linspace(2, 9.5, 41), iph_test, label="Ground Truth")
    plt.title(title)
    plt.xlabel("Wavelength (\u03BCm)")
    plt.ylabel("Power Density")
    plt.show()


def fit_naive_model():
    train_ds = PhotocurrentData("Spectral Responsivity Data Summary.csv", params="model_params.json")
    eval_ds = PhotocurrentData("Spectral Responsivity Data Summary.csv", params="model_params.json")
    _, _, y_mean, _ = train_ds.data_std()
    # criterion = MSELoss()
    # criterion = SmoothL1Loss()

    metrics = {
        "MSE": MeanSquaredError(),
    }

    train_loader = DataLoader(train_ds,
                              shuffle=True,
                              batch_size=2)
    val_loader = DataLoader(eval_ds,
                            shuffle=True,
                            batch_size=2)

    model = NaiveSpectralModel(wavelengths=train_ds.wavelengths, params="model_params.json")
    optimizer = Adam(model.parameters(), lr=1e-2)
    # criterion = SmoothWeightsLoss(model, weights=1/y_mean, lambda_l1=1e-2, lambda_rows=1e-3, lambda_cols=1e-3, lambda_norm=0)
    criterion = SmoothWeightsLoss(model, lambda_l1=1e-3, lambda_rows=0, lambda_cols=0,
                                  lambda_norm_rows=0, lambda_norm_cols=0)
    trainer = get_trainer(model, train_loader, val_loader, criterion, optimizer, metrics)
    trainer.run(train_loader, max_epochs=200)

    plot_weights_R_inv(model)
    plot_prediction_R_inv(model, train_ds)
    plot_test_R_inv(model, "Reconstruction Data Summary - BP-5um data.csv",
                    title="BP-5\u03BCm",
                    normlizer=train_ds.normlize)
    # plot_test_R_inv(model, "Reconstruction Data Summary - BP-CO2 data.csv",
    #                 title="BP-CO2",
    #                 normlizer=train_ds.normlize)


if __name__ == '__main__':
    fit_naive_model()
