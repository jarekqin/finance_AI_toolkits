import numpy as np
from scipy.stats import norm
import pandas as pd
import math


def implied_volatility(options_price, underlying_price, strike_price, risk_free, t0, t1, options_type):
    """
    calculate implied volatility with bional method
    :param options_price: options price
    :param underlying_price: underlying price
    :param strike_price: strike price on options
    :param risk_free: risk free rate
    :param t0: time 0
    :param t1: time 1
    :param options_type: options_type
    :return: implied volatility
    """
    t0, t1 = pd.to_datetime(t0).date(), pd.to_datetime(t1).date()
    t = (t1 - t0).days / 365
    sigma_min = 1e-5
    sigma_max = 1.
    sigma_mid = (sigma_min + sigma_max) / 2

    if options_type in ['call', 'CALL', 'Call']:
        def call_bs(s, k, sigma, r, t):
            d1 = (np.log(s / k) + (r + pow(sigma, 2) / 2) * t) / (sigma * np.sqrt(t))
            d2 = d1 - sigma * np.sqrt(t)
            call_value = s * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)
            return call_value

        call_min = call_bs(underlying_price, strike_price, sigma_min, risk_free, t)
        call_max = call_bs(underlying_price, strike_price, sigma_max, risk_free, t)
        call_mid = call_bs(underlying_price, strike_price, sigma_mid, risk_free, t)
        diff = options_price - call_mid
        if options_price < call_min or options_price > call_max:
            raise ValueError('options price error!')
        while abs(diff) > 1e-6:
            diff = options_price - call_bs(underlying_price, strike_price, sigma_mid, risk_free, t)
            sigma_mid = (sigma_min + sigma_max) / 2
            call_mid = call_bs(underlying_price, strike_price, sigma_mid, risk_free, t)
            if options_price > call_mid:
                sigma_min = sigma_mid
            else:
                sigma_max = sigma_mid
    else:
        def put_bs(s, k, sigma, r, t):
            d1 = (np.log(s / k) + (r + pow(sigma, 2) / 2) * t) / (sigma * np.sqrt(t))
            d2 = d1 - sigma * np.sqrt(t)
            put_value = k * np.exp(-r * t) * norm.cdf(-d2) - s * norm.cdf(-d1)
            return put_value

        put_min = put_bs(underlying_price, strike_price, sigma_min, risk_free, t)
        put_max = put_bs(underlying_price, strike_price, sigma_max, risk_free, t)
        put_mid = put_bs(underlying_price, strike_price, sigma_mid, risk_free, t)
        diff = options_price - put_mid
        if options_price < put_min or options_price > put_max:
            raise ValueError('options price error!')
        while abs(diff) > 1e-6:
            diff = options_price - put_bs(underlying_price, strike_price, sigma_mid, risk_free, t)
            sigma_mid = (sigma_min + sigma_max) / 2
            put_mid = put_bs(underlying_price, strike_price, sigma_mid, risk_free, t)
            if options_price > put_mid:
                sigma_min = sigma_mid
            else:
                sigma_max = sigma_mid
    return sigma_mid


def BSM(s, k, sigma, r, t0, t1, types):
    """
    calculate options price with BSM model
    :param s: underlying price
    :param k: strike price
    :param sigma: underlying returns volatility
    :param r: risk free rate
    :param t0: time start
    :param t1: time to maturity
    :param types: options type
    :return: options price
    """
    t0, t1 = pd.to_datetime(t0).date(), pd.to_datetime(t1).date()
    t = (t1 - t0).days / 365
    d1 = (np.log(s / k) + (r + pow(sigma, 2) / 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    if types in ['call', 'CALL', 'Call']:
        options_price = s * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)
    elif types in ['put', 'PUT', 'Put']:
        options_price = k * np.exp(-r * t) * norm.cdf(-d2) - s * norm.cdf(-d1)
    else:
        raise TypeError('types only supports "call/put"!')
    return options_price


def options_letters(s, k, sigma, r, t0, t1, letter, types):
    """
    calculate options price with BSM model
    :param s: underlying price
    :param k: strike price
    :param sigma: underlying returns volatility
    :param r: risk free rate
    :param t0: time start
    :param t1: time to maturity
    :param letter: letter type
    :param types: options type
    :return: options letter value
    """
    t0, t1 = pd.to_datetime(t0).date(), pd.to_datetime(t1).date()
    t = (t1 - t0).days / 365
    d1 = (np.log(s / k) + (r + pow(sigma, 2) / 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    result = None

    if letter in ['Delta', 'delta', 'DELTA']:
        if types in ['call', 'CALL', 'Call']:
            result = norm.cdf(d1)
        elif types in ['put', 'PUT', 'Put']:
            result = norm.cdf(d1) - 1
        else:
            raise TypeError('types only supports "call/put"!')

    elif letter in ['Gamma', 'gamma', 'GAMMA']:
        result = np.exp(-pow(d1, 2) / 2) / (s * sigma * np.sqrt(2 * np.pi * t))

    elif letter in ['Theta', 'theta', 'THETA']:
        if types in ['call', 'CALL', 'Call']:
            result = -(s * sigma * np.exp(-pow(d1, 2) / 2)) / (2 * np.sqrt(2 * np.pi * t)) - r * k * np.exp(
                (-r * t) * norm.cdf(d2))
        elif types in ['put', 'PUT', 'Put']:
            result = -(s * sigma * np.exp(-pow(d1, 2) / 2)) / (2 * np.sqrt(2 * np.pi * t)) + r * k * np.exp(
                (-r * t) * norm.cdf(-d2))

    elif letter in ['Vega', 'vega', 'VEGA']:
        result = s * np.sqrt(t) * np.exp(-pow(d1, 2) / 2) / np.sqrt(2 * np.pi)

    elif letter in ['Rho', 'rho', 'RHO']:
        if types in ['call', 'CALL', 'Call']:
            result = k * t * np.exp(-r * t) * norm.cdf(d2)
        elif types in ['put', 'PUT', 'Put']:
            result = -k * t * np.exp(-r * t) * norm.cdf(-d2)
        else:
            raise TypeError('types only supports "call/put"!')

    else:
        raise TypeError('wrong type of options letter!')

    return result


def best_hedged_ratio(hedge_ratio, asset_price, futures_contract_fv):
    """
    calculate best hedged ratio between underlying and futures
    :param hedge_ratio: ratio of hedge
    :param asset_price: underlying asset
    :param futures_contract_fv: futures fv
    :return: best ratio
    """
    N = hedge_ratio * asset_price / futures_contract_fv
    if math.modf(N)[0] > 0.5:
        N = math.ceil(N)
    else:
        N = math.floor(N)
    return N



if __name__ == '__main__':
    print('call options price: ', BSM(325, 350, 0.291239, 0.0231393, '2019-10-02', '2020-04-02', 'call'))
    print('put options price: ', BSM(325, 310, 0.291239, 0.0233875, '2019-10-02', '2020-10-02', 'put'))
    print('*' * 50, 'beautiful line', '*' * 50)
    t_cal = '2019-06-28'
    t_maturity_call = '2019-08-28'
    t_maturity_put = '2019-09-25'
    vol = 0.234811
    underlying_price = 2.950

    k_call = 3.2
    k_put = 2.8
    risk_free = 0.02708

    print(options_letters(underlying_price, k_call, vol, risk_free, t_cal, t_maturity_call, 'delta', 'call'))
    print(options_letters(underlying_price, k_call, vol, risk_free, t_cal, t_maturity_call, 'gamma', 'call'))
    print(options_letters(underlying_price, k_call, vol, risk_free, t_cal, t_maturity_call, 'vega', 'call'))
    print(options_letters(underlying_price, k_call, vol, risk_free, t_cal, t_maturity_call, 'theta', 'call'))
    print(options_letters(underlying_price, k_call, vol, risk_free, t_cal, t_maturity_call, 'rho', 'call'))

    print('*' * 50, 'beautiful line', '*' * 50)
    print(options_letters(underlying_price, k_put, vol, risk_free, t_cal, t_maturity_put, 'delta', 'put'))
    print(options_letters(underlying_price, k_put, vol, risk_free, t_cal, t_maturity_put, 'gamma', 'put'))
    print(options_letters(underlying_price, k_put, vol, risk_free, t_cal, t_maturity_put, 'vega', 'put'))
    print(options_letters(underlying_price, k_put, vol, risk_free, t_cal, t_maturity_put, 'theta', 'put'))
    print(options_letters(underlying_price, k_put, vol, risk_free, t_cal, t_maturity_put, 'rho', 'put'))

    print('*' * 50, 'beautiful line', '*' * 50)
    print(best_hedged_ratio(1, 22650000, 300 * 1000))

    print('*' * 50, 'beautiful line', '*' * 50)
    from datetime import datetime
    p_etf_apr25=2.913
    shibor_apr25=0.0288
    t_calculate=datetime(2019,4,25)
    t_mature=datetime(2019,12,25)

    k_c28=2.8
    k_c30=3.0
    k_c32=3.2

    p_c28_apr25=0.3432
    p_c30=apr25=0.2438
    p_c32_apr25=0.1688

    print(implied_volatility(p_c28_apr25,p_etf_apr25,k_c28,shibor_apr25,t_calculate,t_mature,'call'))