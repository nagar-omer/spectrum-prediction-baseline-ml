from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, MeanSquaredError
import torch
from sklearn.model_selection import KFold
from torch.nn.functional import mse_loss
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from data_loader import PhotocurrentData
from naive_nn import NaiveSpectralModel
from torch.nn import MSELoss, SmoothL1Loss
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    def __init__(self, model, lambda_l1=1e-3, lambda_rows=1e-1, lambda_cols=1e-1, lambda_norm=1e-1):
        self._lambda_rows = lambda_rows
        self._lambda_cols = lambda_cols
        self._lambda_norm = lambda_norm
        self._lambda_l1 = lambda_l1
        self._model = model

    def __call__(self, x, y):
        mse = mse_loss(x, y)
        diff_rows = 0
        norm_rows = 0

        for i, j in zip(self._model._linear_1.weight,
                        self._model._linear_1.weight[1:]):
            diff_rows += torch.dist(i, j)
            norm_rows += torch.norm(i)
        norm_rows += torch.norm(j)

        diff_cols = 0
        for i, j in zip(self._model._linear_1.weight.T,
                        self._model._linear_1.weight.T[1:]):
            diff_cols += torch.dist(i, j)

        l1 = self._model._linear_1.weight.norm().square()
        return mse + self._lambda_l1*l1 + \
               self._lambda_rows*diff_rows + \
               self._lambda_cols*diff_cols + \
               self._lambda_norm / norm_rows


def fit_naive_model():
    train_ds = PhotocurrentData("Spectral Responsivity Data Summary.csv", params="model_params.json")
    eval_ds = PhotocurrentData("Spectral Responsivity Data Summary.csv", params="model_params.json")
    # criterion = MSELoss()
    # criterion = SmoothL1Loss()
    # criterion = SmoothWeightsLoss()

    metrics = {
        "MSE": MeanSquaredError(),
    }
    Y_data = {}
    Y_hat_data = {}
    for i, (train_index, test_index) in enumerate(KFold(10, shuffle=True).split(list(range(21)))):
        # train_ds.filter(train_index)
        # eval_ds.filter(test_index)
        train_loader = DataLoader(train_ds,
                                  shuffle=True,
                                  batch_size=2)
        val_loader = DataLoader(eval_ds,
                                shuffle=True,
                                batch_size=2)

        model = NaiveSpectralModel(wavelengths=train_ds.wavelengths, params="model_params.json")
        optimizer = Adam(model.parameters(), lr=0.01)
        criterion = SmoothWeightsLoss(model, lambda_l1=5e-2, lambda_rows=1e-1, lambda_cols=1e-1, lambda_norm=5e-2)
        # criterion = MSELoss()
        trainer = get_trainer(model, train_loader, val_loader, criterion, optimizer, metrics)
        trainer.run(train_loader, max_epochs=1000)

        inv_r = np.asarray(model._linear_1.weight.tolist())
        r = np.linalg.pinv(inv_r)

        ax = sns.heatmap(np.asarray(inv_r))
        ax.invert_yaxis()
        plt.show()
        ax = sns.heatmap(np.asarray(r))
        ax.invert_yaxis()
        plt.show()

        X = torch.Tensor([d[0] for d in eval_ds._data])
        Y = torch.Tensor([d[1] for d in eval_ds._data])
        Y_hat = model(X)
        ax = sns.heatmap(np.asarray(Y_hat.tolist()))
        ax.invert_yaxis()
        plt.show()

        for i, real_index in enumerate(test_index):
            Y_data[real_index] = Y[i]
            Y_hat_data[real_index] = Y_hat[i]

    # Y = np.vstack([Y_data[i] for i in sorted(Y_data)])
    # Y_hat = np.vstack([Y_data[i] for i in sorted(Y_hat_data)])
    #
    # ax = sns.heatmap(np.asarray(Y))
    # plt.show()
    # ax = sns.heatmap(np.asarray(Y_hat))
    # plt.show()
    e = 0


if __name__ == '__main__':
    fit_naive_model()
