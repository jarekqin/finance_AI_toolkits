import numpy as np


def irs(float_leg, fix_leg, principle, frequency, position):
    """
    calculate interest rate swap
    :param float_leg: rate on float
    :param fix_leg: rate on fix
    :param principle: principle on swap
    :param frequency: payoff frequency
    :param position: long or short
    :return: swap value
    """
    if position == 'long':
        return (float_leg - fix_leg) * principle / frequency
    else:
        return (fix_leg - float_leg) * principle / frequency


def swap_rate(frequency, rate_on_delivery, maturity):
    """
    calculate swap rate
    :param frequency: payoff frequency
    :param rate_on_delivery: rate on delivery
    :param maturity: contract time
    :return: rate
    """
    n_list = np.arange(1, frequency * maturity + 1)
    t = n_list / frequency
    q = np.exp(-rate_on_delivery * maturity)
    rate = frequency * (1 - q[-1]) / np.sum(q) if len(q) > 1 else frequency * (1 - q) / np.sum(q)
    return rate


def swap_value(fix_leg, float_leg, maturity, ytm_rate, frequency, principle, position):
    """
    calculate swap value
    :param fix_leg: fix leg rate
    :param float_leg: float leg rate
    :param maturity: contract maturity
    :param contract_rate: contract rate
    :param frequency: payoff frequency
    :param principle: principle on swap
    :param position: holding direction
    :return: swap value
    """
    b_fix = (fix_leg * np.sum(np.exp(-ytm_rate * maturity)) / frequency + np.exp(
        ytm_rate[-1] * maturity[-1])) * principle if len(ytm_rate)>1 and len(maturity)>1 else \
        (fix_leg * np.sum(np.exp(-ytm_rate * maturity)) / frequency + np.exp(
            ytm_rate * maturity)) * principle
    b_float=(float_leg/frequency+1)*principle*np.exp(ytm_rate[0]*maturity[0]) if len(ytm_rate)>1 and len(maturity)>1 \
        else (float_leg/frequency+1)*principle*np.exp(ytm_rate*maturity)
    if position=='long':
        return b_float-b_fix
    else:
        return b_fix-b_float




if __name__ == '__main__':
    rate_fix = 0.037
    par = 1e8
    frequency = 2
    rate_float = np.array([0.03197, 0.032, 0.029823, 0.030771, 0.04451, 0.047093, 0.04304, 0.03275, 0.02963, 0.01566])
    print(irs(rate_float, rate_fix, par, frequency, 'long'))
    print(irs(rate_float, rate_fix, par, frequency, 'short'))
    r_list = np.array([0.020579, 0.021276, 0.022080, 0.022853, 0.023527, 0.024036])
    print(swap_rate(2, r_list, 3))
    par=1e8
    rate_fix=0.0241
    rate_float=0.02178
    frequency=3
    t_list=np.array([0.47945205,0.97945205,1.47945205,1.97945205,2.47945205,2.97945205])
    ytm_list=np.array([0.027564,0.022548,0.03125,0.014755,0.021432,0.0268853])
    print(swap_value(rate_fix,rate_float,t_list,ytm_list,frequency,par,'long'))

