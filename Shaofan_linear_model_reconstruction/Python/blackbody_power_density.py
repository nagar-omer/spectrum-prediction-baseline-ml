import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
import scipy.stats

C1 = 2 * (6.63 * 1e-34) * (3 * 1e8)**2              # 2hC^2 the first radiation constant
C2 = (6.63 * 1e-34) * (3 * 1e8) / (1.38 * 1e-23)    # hc/kB, the second radiation constant


# f_bb = @(t, l) (c1. / (l. ^ 5). / (exp(c2. / (l * t)) - 1) / 1e9)
def planck_func(temperature, wavelength):
    """
    calculate planck function
    returns the power density of blackbody at temperature T and wavelength lambda
    """
    exponent = np.exp(C2/(temperature * wavelength))
    return C1 / (wavelength**5 * (exponent - 1)) * 1e-9


def gaussian_power_density(temperature, wavelengths):
    """
    calculate the power density as a gaussian over the planck's power density
    PB(T, LAMBDA) = INTEGRAL (lambda_min ... lambda_max) { P(T, lambda) * Gauss(LAMBDA, sigma)(lambda)}
    :param wavelengths: a list of lambda values
    :return:
    """
    power_density = []
    # set sigma for gaussian distribution
    sigma = (max(wavelengths) - min(wavelengths)) / (len(wavelengths) - 1)
    # calculate normal-power-density for each lambda
    for wavelength in wavelengths:
        # gaussian(LAMBDA, sigma)
        gaussian_wavelength = scipy.stats.norm(wavelength, sigma)
        # calculate Integral according to simpson rule
        # y = P(T, lambda) * Gauss(LAMBDA, sigma)(lambda)
        normed_wavelength_func = lambda x: planck_func(temperature, x) * gaussian_wavelength.pdf(x)
        wavelength_power, err = integrate.quad(
            func=normed_wavelength_func,
            a=min(wavelengths) - sigma,
            b=max(wavelengths) + sigma,
        )
        power_density.append(wavelength_power)
    # power_density = np.asarray(power_density)
    return power_density


def um(x):
    return 1e-6 * x


def sanity_check():
    import seaborn as sns
    pb = []
    for temp in np.linspace(673, 1073, 21):
        power_density = gaussian_power_density(temperature=temp, wavelengths=np.linspace(um(1), um(9.5), 20))
        pb.append(power_density)
        # plt.plot(np.linspace(um(1), um(9.5), 200), power_density, label='linear')
    ax = sns.heatmap(pb)
    plt.show()
    e = 0


if __name__ == '__main__':
    sanity_check()
