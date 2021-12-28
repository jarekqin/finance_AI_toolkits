import numpy as np
import pandas as pd
import scipy.stats as stats


def VaR_VarCov(market_value, weighted_matrix, return_vector, days, percentile):
    """
    calculate value at risk with variance and convarance
    :param market_value: market value of assets
    :param weighted_matrix: weighte matrix of assets
    :param return_vector: return series of assets
    :param days: holding periods (days)
    :param percentile: percentile number
    :return: value at risk matrix
    """
    if isinstance(market_value, pd.DataFrame) or isinstance(market_value, pd.Series):
        market_value = market_value.values
    if isinstance(return_vector, pd.DataFrame) or isinstance(return_vector, pd.Series):
        return_vector = return_vector.values
    return_mean = return_vector.mean().values
    r_cov = return_vector.cov().values
    return_daily = np.sum(weighted_matrix * return_mean)
    Vp = np.sqrt(np.dot(weighted_matrix, np.dot(r_cov, weighted_matrix.T)))
    z = stats.norm.ppf(q=1 - percentile)
    z = np.abs(z)
    Var = np.sqrt(days) * market_value * (z * Vp - return_daily)
    return Var


def Var_His(market_value, weighted_matrix, return_vector, days, percentile, single_underlying=False):
    """
    calculate value at risk with historical method
    :param market_value: market value of assets
    :param weighted_matrix: weighte matrix of assets
    :param return_vector: return series of assets
    :param days: holding periods (days)
    :param percentile: percentile number
    :param singal_underlying: verifying wehther put 1 asset
    :return: value at risk matrix
    """
    # simulate assets historical value
    if isinstance(return_vector, pd.DataFrame) or isinstance(return_vector, pd.Series):
        return_vector = return_vector.values
    value_assets = market_value * weighted_matrix
    if not single_underlying:
        return_history = np.dot(return_vector, value_assets)
    else:
        return_history = return_vector * value_assets

    # calculate var
    var_1day = np.abs(np.percentile(a=return_history, q=(1 - percentile) * 100))
    var_ndays = np.sqrt(days) * var_1day
    return var_ndays


def Var_MCSM(market_value, price_matrix, weighted_matrix, return_vector, days, percentile, path=10000, year=1,
             singal_underlying=False,std_value=None):
    """
    calculate value at risk with MCSM
    :param market_value: market value of assets
    :param market_value: newest value of assets
    :param weighted_matrix: weighte matrix of assets
    :param return_vector: return series of assets
    :param days: holding periods (days)
    :param percentile: percentile number
    :param year: MCSM simulating times
    :param singal_underlying: verifying wehther put 1 asset
    :return: value at risk matrix
    """
    if isinstance(market_value, pd.DataFrame) or isinstance(market_value, pd.Series):
        market_value = market_value.values
    if isinstance(return_vector, pd.DataFrame) or isinstance(return_vector, pd.Series):
        return_vector = return_vector.values
    if isinstance(price_matrix, pd.DataFrame) or isinstance(price_matrix, pd.Series):
        price_matrix = price_matrix.values

    if not singal_underlying and std_value is None:
        r_mean = return_vector.mean(axis=1) * 252 * year
        r_vol = return_vector.std(axis=1) * np.sqrt(year * 252)
        n_assets = len(return_vector)
    else:
        r_mean = np.array([return_vector])
        r_vol = np.array([std_value])
        n_assets = 1
    dt = 1 / (year * 252)
    epsilon = np.random.standard_normal(size=path)
    p_new = np.zeros(shape=(path, n_assets))
    for i in range(n_assets):
        p_new[:, i] = price_matrix[i] * np.exp(
            (r_mean[i] - 0.5 * r_vol[i] ** 2) * dt + r_vol[i] * epsilon * np.sqrt(dt))
    s_delta = (np.dot(p_new / (price_matrix.T) - 1, weighted_matrix)) * market_value
    var_1day = np.abs(np.percentile(a=s_delta, q=(1 - percentile) * 100))
    var_ndays = np.sqrt(days) * var_1day
    return var_ndays


if __name__ == '__main__':
    value_port = 1e9
    weighted_matrix = np.array([1] * 4)
    return_vector = pd.Series([.017, 0.027, 0.022, -0.018])
    price_matrix = pd.Series([77.8, 65.2, 89.2, 88.1])
    days = 10
    percentile = 0.95
    print('single_asset_his_test: ',Var_His(value_port, weighted_matrix, return_vector, days, percentile, single_underlying=True))
    print('*' * 50)
    value_port = 1e8
    weighted_matrix = np.array([0.22, 0.18, 0.16, 0.14, 0.12, 0.08])
    price_matrix = pd.Series([100.1, 98.2, 97.87, 94.54, 99.07, 101.2])
    return_vector = pd.DataFrame(
        [
            [.017, 0.027, 0.082, -0.018, -0.069, 0.014],
            [.027, 0.037, 0.062, -0.028, -0.059, 0.024],
            [.037, 0.047, 0.052, -0.038, -0.049, 0.034],
            [.047, 0.057, 0.042, -0.048, -0.039, 0.044],
            [.057, 0.067, 0.032, -0.058, -0.029, 0.054],
            [.067, 0.077, 0.012, -0.068, -0.019, 0.064],
        ])
    path = 10000
    percentile = 0.95
    print('multiple asset msmc test: ', Var_MCSM(value_port, price_matrix, weighted_matrix, return_vector, 1, percentile, path, 1))
    print('*' * 50)
    value_port = 1e8
    weighted_matrix = np.array([1])
    price_matrix = np.array([100])
    return_vector = 0.017
    std_value=0.2
    path = 10000
    percentile = 0.95
    print('single_asset msmc test: ',
          Var_MCSM(value_port, price_matrix, weighted_matrix, return_vector, 1, percentile, path, 1, True,std_value))
