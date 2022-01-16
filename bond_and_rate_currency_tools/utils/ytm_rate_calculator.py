import scipy.optimize as optimize
import numpy as np


def bond_ytm(price, par, t, coup, freq=2, guess=0.05):
    freq = float(freq)
    periods = t * 2
    coupon = coup / 100 * par
    dt = [(i + 1) / freq for i in range(int(periods))]
    ytm_func = lambda y: \
        sum([coupon / freq / (1 + y / freq) ** (freq * t2) for t2 in dt]) + \
        par / (1 + y / freq) ** (freq * t) - price
    return optimize.newton(ytm_func, guess)


def vasicek(r0, k, theta, sigma, t=1., n=10, seed=777):
    np.random.seed(seed)
    dt = t / float(n)
    rates = [r0]
    for i in range(n):
        dr = k * (theta - rates[-1]) * dt + sigma * np.sqrt(dt) * np.random.normal()
        rates.append(rates[-1] + dr)
    return range(n + 1), rates


def CIR(r0, k, theta, sigma, t=1., n=10, seed=777):
    np.random.seed(seed)
    dt = t / float(n)
    rates = [r0]
    for i in range(n):
        dr = k * (theta - rates[-1]) * dt + sigma * np.sqrt(dt) * np.random.normal() * np.sqrt(dt)
        rates.append(rates[-1] + dr)
    return range(n + 1), rates


def rendleman_bartter(r0, theta, sigma, t=1., n=10, seed=777):
    np.random.seed(seed)
    dt = t / float(n)
    rates = [r0]
    for i in range(n):
        dr = theta * rates[-1] * dt + sigma * np.sqrt(dt) * np.random.normal()
        rates.append(rates[-1] + dr)
    return range(n + 1), rates


def brennan_schwartz(r0, k, theta, sigma, t=1., n=10, seed=777):
    np.random.seed(seed)
    dt = t / float(n)
    rates = [r0]
    for i in range(n):
        dr = k * (theta - rates[-1]) * dt + sigma * rates[-1] * np.sqrt(dt) * np.random.normal()
        rates.append(rates[-1] + dr)
    return range(n + 1), rates


if __name__ == '__main__':
    ytm = bond_ytm(95.0428, 100, 1.5, 5.75, 2)
    print(ytm)
    print(vasicek(0.005, 0.2, 0.15, 0.05, t=10, n=200))
    print(CIR(0.05, 0.2, 0.15, 0.05, t=10, n=200))
    print(rendleman_bartter(0.05, 0.1, 0.05, t=10, n=200))
    print(brennan_schwartz(0.05, 0.2, 0.15, 0.05, t=10, n=200))