from sklearn.model_selection import KFold
import numpy as np
from blackbody_power_density import gaussian_power_density, um
from sklearn import linear_model
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def load_data(path):
    """
    load spectral responsivity csv file as matrix
    :param path: path to csv file
    :return: tmeperatures values, displacement field values and photocurrent matrix
    """
    data_file = open(path, "rt", encoding="utf-8")
    # extract displacement fields header
    D = [float(x) for x in data_file.readline().strip().split(",")[1:]]
    temperatures = []   # temperature header
    photocurrent = []   # photocurrent matrix
    for row in data_file:
        row = row.strip().split(",")
        temperatures.append(float(row[0]))                  # build temperature header
        photocurrent.append([float(x) for x in row[1:]])    # build matrix
    return D, temperatures, np.asarray(photocurrent)


def calculate_power_density(temperatures, n_lambda=41):
    """
    calculate power density of given temperature, according to planck's function
    """
    pb = []
    for temp in temperatures:
        power_density = gaussian_power_density(temperature=temp, wavelengths=np.linspace(um(2), um(9.5), n_lambda))
        pb.append(power_density)
    return np.asarray(pb)


def ridge_fit(power_density, photocurrent, l1_reg=1.):
    """
    fit a linear function (Responsivity), use l2 regularization with coeff l2_reg
    the regression is done n times and minimizes:
        ||W*P(lambda) - I||^2_2 + l2_reg * ||W||^2_2
    """
    responsivity = []
    for displasment_field in photocurrent.T:
        # weights = [1 / i ** 2 for i in photocurrent]
        clf = linear_model.Ridge(alpha=l1_reg, solver='lsqr', tol=1e-1, fit_intercept=True)
        responsivity.append(clf.fit(power_density, displasment_field).coef_)
    return np.vstack(responsivity).T


def fit_linear_resposivity_model(kfold=11):
    """
    fit a linear model with cross validation on Photocurrent data
    """
    D, temperatures, photocurrent = load_data("Spectral Responsivity Data Summary.csv")         # load data
    power_density = calculate_power_density(temperatures, n_lambda=41)                                       # planck power density
    best_mse = 1e10
    for i, (train_index, test_index) in enumerate(KFold(kfold, shuffle=True).split(power_density)):
        # split to train and test
        power_density_train, power_density_test = power_density[train_index], power_density[test_index]
        photocurrent_train, photocurrent_test = photocurrent[train_index], photocurrent[test_index]

        # fit predict
        responsivity = ridge_fit(power_density_train, photocurrent_train, l1_reg=1e+3)
        y_pred = np.matmul(power_density_test, responsivity)

        res_inv = np.linalg.pinv(responsivity)
        p = np.matmul(photocurrent, res_inv)

        # save best result
        mse = mean_squared_error(y_pred, photocurrent_test)
        if mse < best_mse:
            plt.clf()
            # ax = sns.heatmap(p)
            ax = sns.heatmap(responsivity[:, :38], xticklabels=[round(x, 2) for x in D[:38]],
                             yticklabels=np.linspace(2, 9.5, 41).round(2))
            ax.invert_yaxis()
            plt.show()

        # print
        print("Kfold ", i, "test: ", test_index, end="\n" + "="*50 + "\n")
        print("mse: ",  mse, "\n")

    plt.show()
    e = 0


if __name__ == '__main__':
    fit_linear_resposivity_model()
