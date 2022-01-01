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
        ytm_rate[-1] * maturity[-1])) * principle if len(ytm_rate) > 1 and len(maturity) > 1 else \
        (fix_leg * np.sum(np.exp(-ytm_rate * maturity)) / frequency + np.exp(
            ytm_rate * maturity)) * principle
    b_float = (float_leg / frequency + 1) * principle * np.exp(ytm_rate[0] * maturity[0]) if len(ytm_rate) > 1 and len(
        maturity) > 1 \
        else (float_leg / frequency + 1) * principle * np.exp(ytm_rate * maturity)
    if position == 'long':
        return b_float - b_fix
    else:
        return b_fix - b_float


def CCS_double_fixed_cashflow(cur1, cur2, leg_rate1, leg_rate2, frequency, maturity, trader, principle):
    """
    calculate double fixed rate currency swap
    :param cur1: currency 1
    :param cur2: currency 2
    :param leg_rate1: fixed rate on leg1
    :param leg_rate2: fixed rate on leg2
    :param frequency: payoff frequency
    :param maturity: contract maturity
    :param trader: trader side
    :param principle: principle on swap
    :return: swap cashflow
    """
    cashflow = np.zeros(frequency * maturity + 1)
    if principle in ['LA', 'la', 'La']:
        cashflow[0] = -cur1
        cashflow[1:-1] = leg_rate1 * cur1 / frequency
        cashflow[-1] = (leg_rate1 / frequency + 1) * cur1
        if trader in ['A', 'a']:
            return cashflow
        else:
            return -cashflow
    else:
        cashflow[0] = cur2
        cashflow[1:-1] = -leg_rate2 * cur2 / frequency
        cashflow[-1] = -(leg_rate2 / frequency + 1) * cur2
        if trader in ['A', 'a']:
            return cashflow
        else:
            return -cashflow


def CCS_fixed_float_cashflow(cur1, cur2, leg_rate1, leg_rate2, frequency, maturity, trader, principle):
    """
    calculate on fixed-float rate currency swap
    :param cur1: currency 1
    :param cur2: currency 2
    :param leg_rate1: fixed rate on leg1
    :param leg_rate2: float rate on leg2
    :param frequency: payoff frequency
    :param maturity: contract maturity
    :param trader: trader side
    :param principle: principle on swap
    :return: swap cashflow
    """
    cashflow = np.zeros(frequency * maturity + 1)
    if principle in ['LA', 'la', 'La']:
        cashflow[0] = -cur1
        cashflow[1:-1] = leg_rate1 * cur1 / frequency
        cashflow[-1] = (leg_rate1 / frequency + 1) * cur1
        if trader in ['A', 'a']:
            return cashflow
        else:
            return -cashflow
    else:
        cashflow[0] = cur2
        cashflow[1:-1] = -leg_rate2[:-1] * cur2 / frequency
        cashflow[-1] = -(leg_rate2[-1] / frequency + 1) * cur2
        if trader in ['A', 'a']:
            return cashflow
        else:
            return -cashflow


def CCS_double_float_cashflow(cur1, cur2, leg_rate1, leg_rate2, frequency, maturity, trader, principle):
    """
    calculate on float-float rate currency swap
    :param cur1: currency 1
    :param cur2: currency 2
    :param leg_rate1: fixed rate on leg1
    :param leg_rate2: float rate on leg2
    :param frequency: payoff frequency
    :param maturity: contract maturity
    :param trader: trader side
    :param principle: principle on swap
    :return: swap cashflow
    """
    cashflow = np.zeros(frequency * maturity + 1)
    if principle in ['LA', 'la', 'La']:
        cashflow[0] = -cur1
        cashflow[1:-1] = leg_rate1[:-1] * cur1 / frequency
        cashflow[-1] = (leg_rate1[-1] / frequency + 1) * cur1
        if trader in ['A', 'a']:
            return cashflow
        else:
            return -cashflow
    else:
        cashflow[0] = cur2
        cashflow[1:-1] = -leg_rate2[:-1] * cur2 / frequency
        cashflow[-1] = -(leg_rate2[-1] / frequency + 1) * cur2
        if trader in ['A', 'a']:
            return cashflow
        else:
            return -cashflow


def CCS_value(types, leg1, leg2, rate1, rate2, ytm1, ytm2, spot_rate, frequency, maturity, trader):
    """
    calculate currency-curency swap value
    :param types: swap type
    :param leg1: leg 1 principle
    :param leg2: leg 2 principle
    :param rate1: rate on leg 1
    :param rate2: rate on leg 2
    :param ytm1: ytm on leg1  list type
    :param ytm2: ytm on leg2  list type
    :param spot_rate: spot rate
    :param frequency: payoff frequency
    :param maturity: maturity on contract
    :param trader: trader side
    :return: ccs value
    """
    if types == "double_fixed":
        bond_a = (rate1 * np.sum(np.exp(-ytm1 * maturity)) / frequency + np.exp(-ytm1[-1] * maturity[-1])) * leg1
        bond_b = (rate2 * np.sum(np.exp(-ytm2 * maturity)) / frequency + np.exp(-ytm2[-1] * maturity[-1])) * leg2
        if trader not in ['A', 'a']:
            swap_value = bond_a - bond_b * spot_rate
        else:
            swap_value = bond_b - bond_a / spot_rate
    elif types == 'double_float':
        bond_a = (rate1 / frequency + 1) * np.exp(-ytm1[0] * maturity[0]) * leg1
        bond_b = (rate2 / frequency + 1) * np.exp(-ytm2[0] * maturity[0]) * leg2
        if trader not in ['A', 'a']:
            swap_value = bond_a - bond_b * spot_rate
        else:
            swap_value = bond_b - bond_a / spot_rate
    elif types == 'fix_float':
        bond_a = (rate1 * np.sum(np.exp(-ytm1 * maturity)) / frequency + np.exp(-ytm1[-1] * maturity[-1])) * leg1
        bond_b = (rate2 / frequency + 1) * np.exp(-ytm2[0] * maturity[0]) * leg2
        if trader in ['A', 'a']:
            swap_value = bond_a - bond_b * spot_rate
        else:
            swap_value = bond_b - bond_a / spot_rate

    else:
        raise TypeError('types only supports "double_float/double_fix/fix_float!"')

    return swap_value


if __name__ == '__main__':
    rate_fix = 0.037
    par = 1e8
    frequency = 2
    rate_float = np.array([0.03197, 0.032, 0.029823, 0.030771, 0.04451, 0.047093, 0.04304, 0.03275, 0.02963, 0.01566])
    print(irs(rate_float, rate_fix, par, frequency, 'long'))
    print(irs(rate_float, rate_fix, par, frequency, 'short'))
    r_list = np.array([0.020579, 0.021276, 0.022080, 0.022853, 0.023527, 0.024036])
    print(swap_rate(2, r_list, 3))
    par = 1e8
    rate_fix = 0.0241
    rate_float = 0.02178
    frequency = 3
    t_list = np.array([0.47945205, 0.97945205, 1.47945205, 1.97945205, 2.47945205, 2.97945205])
    ytm_list = np.array([0.027564, 0.022548, 0.03125, 0.014755, 0.021432, 0.0268853])
    print(swap_value(rate_fix, rate_float, t_list, ytm_list, frequency, par, 'long'))
    rmb = 6.4e8
    usd = 1e8
    rate_rmb = 0.02
    rate_usd = 0.01
    frequency = 2
    maturity = 5
    print(CCS_double_fixed_cashflow(rmb, usd, rate_rmb, rate_usd, frequency, maturity, 'A', 'LA'))
    print(CCS_double_fixed_cashflow(rmb, usd, rate_rmb, rate_usd, frequency, maturity, 'A', 'LB'))
    rmb = 6.9e8
    usd = 1e8
    hkd = 2e8
    frequency = 2
    maturity = 3
    rate_fix = 0.03
    libor = np.array([0.01291, 0.014224, 0.016743, 0.024744, 0.028946, 0.0251661])
    print(CCS_fixed_float_cashflow(rmb, usd, rate_fix, libor, frequency, maturity, 'a', 'La'))
    print(CCS_fixed_float_cashflow(rmb, usd, rate_fix, libor, frequency, maturity, 'a', 'Lb'))
    rmb = 1.8e8
    frequency = 1
    maturity = 4
    shibor = np.array([0.0316, 0.046329, 0.03527, 0.03122])
    hibor = np.array([0.013295, 0.015057, 0.026593, 0.023743])
    print(CCS_double_float_cashflow(rmb, hkd, shibor, hibor, frequency, maturity, 'a', 'La'))
    print(CCS_double_float_cashflow(rmb, hkd, shibor, hibor, frequency, maturity, 'a', 'Lb'))
    frequency = 1
    maturity = 3
    rate_rmb = 0.02
    rate_fx = 7.0903
    usd = 1e8
    rmb = usd * 7.0771
    libor = 0.010024
    ytm_rmb = np.array([0.021156, 0.023294, 0.023811])
    ytm_usd = np.array([0.0019, 0.0019, 0.0022])
    t_list = np.array([0.78630137, 1.78630137, 2.78630137])
    print(CCS_value('fix_float', rmb, usd, rate_rmb, libor, ytm_rmb, ytm_usd, rate_fx, frequency, t_list, 'A'))
