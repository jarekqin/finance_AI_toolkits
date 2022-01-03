import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt


def bond_price(face_value, rate_freq, ytm, t0, t1, return_true_value=False, par_value=None, unit_par_value=100):
    """
    calculate bond price
    :param face_value: face value of a bond
    :param rate_freq: bonus paid frequency
    :param ytm: yield of zero bond vector (must be ndarray)
    :param t0: bond priced date
    :param t1: ytm of bond
    :return: bond price
    """
    if not isinstance(ytm, np.ndarray):
        ytm = np.array(ytm)
    t0, t1 = pd.to_datetime(t0).date(), pd.to_datetime(t1).date()
    tensor = (t1 - t0).days / 365
    if math.modf(tensor)[0] > 0.5:
        n = rate_freq * math.ceil(tensor)
    else:
        n = rate_freq * math.floor(tensor) + 1

    t_list = np.arange(n) / rate_freq
    t_list = np.sort(tensor - t_list)
    price = 100 * (np.sum(np.exp(-ytm * t_list)) * face_value / rate_freq + np.exp(-ytm[-1] * t_list[-1]))

    if return_true_value and par_value != None:
        return price * par_value / unit_par_value
    else:
        return price


def fra_value(forward_rate, rate_on_time, risk_free, principle, t1, t2, position):
    """
    calculate FRA
    :param forward_rate: future rate
    :param rate_on_time: rate on now
    :param principle: principle cash
    :param t1: time 1 length that must be float or integer
    :param t2: time 2 length that must be float or integer
    :param position: holding position
    :param when: only supports 'begin' and 'end'
    :return: cash flow
    """
    if position == 'long':
        return principle * (risk_free - forward_rate) * (t2 - t1) * np.exp(-rate_on_time * t2)
    else:
        return principle * (forward_rate - risk_free) * (t2 - t1) * np.exp(-rate_on_time * t2)


def cal_forward_rate(r1, r2, t1, t2):
    """
    calculate forward rate
    :param r1: zero rate on t1
    :param r2: zero rate on t2
    :param t1: time 1
    :param t2: time 2
    :return: forward rate(usually is list/ndarray)
    """
    return r2 + (r2 - r1) * t1 / (t2 - t1)


def Cash_FRA(forward_rate, rate_on_time, principle, t1, t2, position, when):
    """
    cash FRA payment on cashflow
    :param forward_rate: future rate
    :param rate_on_time: rate on now
    :param principle: principle cash
    :param t1: time 1 length that must be float or integer
    :param t2: time 2 length that must be float or integer
    :param position: holding position
    :param when: only supports 'begin' and 'end'
    :return: cash flow
    """
    if position == 'long':
        if when == 'begin':
            return ((rate_on_time - forward_rate) * (t2 - t1) * principle) / (1 + (t2 - t1) * rate_on_time)
        else:
            return (rate_on_time - forward_rate) * (t2 - t1) * principle
    elif position == 'short':
        if when == 'begin':
            return ((forward_rate - rate_on_time) * (t2 - t1) * principle) / (1 + (t2 - t1) * rate_on_time)
        else:
            return (forward_rate - rate_on_time) * (t2 - t1) * principle
    else:
        raise TypeError('position only supports "long"/"short"')


def fx_forward(spot, r1, r2, time):
    """
    calculate forward rate between rate 1 and rate 2
    :param spot: spot rate
    :param r1: forward rate 1
    :param r2: forward rate 2
    :param time: delta time
    :return: forward rate between rate 1 and rate 2
    """
    return spot * (1 + r1 * time) / (1 + r2 * time)


def fx_forward_value(forward_rate1, forward_rate2, spot_rate, par, risk_free, time, currency_a, currency_b, position):
    """
    calculate fx forward contract value
    :param forward_rate1: forward rate 1
    :param forward_rate2: forward rate 2
    :param spot_rate: spot rate
    :param par: par principle
    :param risk_free: risk free rate
    :param time: time period
    :param currency_a: currency type a
    :param currency_b: currency type b
    :param position: holding direction,which only supports lower letter
    :return: forward contract value
    """
    if currency_a == 'A':
        if position == 'long':
            if currency_b == 'A':
                return spot_rate * (par / forward_rate2 - par / forward_rate1) * np.exp(-risk_free * time)
            else:
                return (par / forward_rate2 - par / forward_rate1) * np.exp(-risk_free * time)
        else:
            if currency_b == 'A':
                return spot_rate * (par / forward_rate1 - par / forward_rate2) * np.exp(-risk_free * time)
            else:
                return (par / forward_rate1 - par / forward_rate2) * np.exp(-risk_free * time)
    else:
        if position == 'long':
            if currency_b == 'A':
                return (par * forward_rate2 - par * forward_rate1) * np.exp(-risk_free * time)
            else:
                return (par * forward_rate2 - par * forward_rate1) * np.exp(-risk_free * time) / spot_rate
        else:
            if currency_b == 'A':
                return (par * forward_rate1 - par * forward_rate2) * np.exp(-risk_free * time)
            else:
                return (par * forward_rate1 - par * forward_rate2) * np.exp(-risk_free * time) / spot_rate


def currency_exchange(currency_rate, foreign, domestic, quote):
    """
    calculate exchange value between foreign currency and domestic currency
    :param currency_rate: exchange rate
    :param foreign: foreign money amounts
    :param domestic: domestic money amounts
    :param quote: calculated method
    :return: exchange value
    """
    if foreign is None:
        if quote == 'direct':
            value = domestic * currency_rate
        else:
            value = domestic / currency_rate
    else:
        if quote == 'direct':
            value = foreign / currency_rate
        else:
            value = foreign * currency_rate
    return value


def tri_currency_arbitrage(cur_rate1, cur_rate2, cur_rate3, amounts, cur_type1, cur_type2, cur_type3):
    """
    calculate triangle path through 3 types of currency
    :param cur_rate1: rate on currency 1
    :param cur_rate2: rate on currency 2
    :param cur_rate3: rate on currency 3
    :param amounts: principle amounts
    :param cur_type1: currency type 1
    :param cur_type2: currency type 2
    :param cur_type3: currency type 3
    :return: profit,path
    """
    cur_3_new = cur_rate1 * cur_rate2
    if cur_3_new > cur_rate3:
        profit = amounts * (cur_3_new / cur_rate3 - 1)
        sequence = ['path: ', cur_type1, '->', cur_type3, '->', cur_type2, '->', cur_type1]
    elif cur_3_new < cur_rate3:
        profit = amounts * (cur_rate3 / cur_3_new - 1)
        sequence = ['path: ', cur_type1, '->', cur_type2, '->', cur_type3, '->', cur_type1]
    else:
        profit = 0
        sequence = ['path: ', 'no path!']
    return [profit, sequence]


def cal_fra_pay_libor(risk_free, rate_on_time, principle, t1, t2, position):
    """
    calculate fra payment under LIBOR
    :param risk_free: risk free rate
    :param rate_on_time: rate on observed time
    :param principle: principle
    :param t1: time 1
    :param t2: time 2
    :param position: holding position (could be both -1/1)
    :return: fra payoff
    """
    t1, t2 = pd.to_datetime(t1).date(), pd.to_datetime(t2).date()
    tensor = (t2 - t1).days / 365
    payoff = ((rate_on_time - risk_free) * tensor * principle) / (1 + tensor * rate_on_time)
    if position == 1:
        return payoff
    elif position == -1:
        return -payoff
    else:
        raise TypeError('position only supports 1/-1!')


def cal_fra_pay_shibor(risk_free, future_rate, period_risk_free, principle, t0, t1, t2, position):
    """
    calculate frq payoff under SHIBOR
    :param risk_free: risk free rate
    :param future_rate: future rate between t1 adn t2
    :param period_risk_free: risk free rate between t1 and t2
    :param principle: principles
    :param t0: time start
    :param t1: time 1
    :param t2: time 2
    :param position: holding position (could be both -1/1)
    :return: fra payoff
    """
    t0, t1, t2 = pd.to_datetime(t0).date(), pd.to_datetime(t1).date(), pd.to_datetime(t2).date()
    tensor1 = (t2 - t1).days / 365
    tensor2 = (t2 - t0).days / 365
    payoff = principle * (future_rate - risk_free) * tensor1 * np.exp(-period_risk_free * tensor2)
    if position == 1:
        return payoff
    elif position == -1:
        return -payoff
    else:
        raise TypeError('position only supports 1/-1!')


def bond_price_one_discount(bond_rate, fv, frequency, ytm, maturity):
    """
    calculate bond price with one discount
    :param bond_rate: rate of bond
    :param fv: face value
    :param frequency: frequency for payment
    :param ytm: yield to maturity
    :param maturity: maturity
    :return: bond price
    """
    if bond_rate == 0:
        return np.exp(-ytm * maturity) * fv
    else:
        coupon = np.ones_like(maturity) * fv * bond_rate / frequency
        NPV_coupon = np.sum(coupon * np.exp(-ytm * maturity))
        NPV_par = fv * np.exp(-ytm * maturity[-1]) if len(maturity) > 1 else fv * np.exp(-ytm * maturity)
    return NPV_coupon + NPV_par


def bond_price_diff_discount(bond_rate, fv, frequency, ytm, maturity):
    """
    calculate bond price with variable discount
    :param bond_rate: rate of bond
    :param fv: face value
    :param frequency: frequency for payment
    :param ytm: yield to maturity
    :param maturity: maturity
    :return: bond price
    """
    if bond_rate == 0:
        return np.exp(-ytm * maturity) * fv
    else:
        coupon = np.ones_like(maturity) * fv * bond_rate / frequency
        NPV_coupon = np.sum(coupon * np.exp(-ytm * maturity))
        NPV_par = fv * np.exp(-ytm[-1] * maturity[-1]) if len(maturity) > 1 and len(ytm) > 1 else \
            fv * np.exp(-ytm * maturity)
    return NPV_coupon + NPV_par


def cal_ytm(bond_price,face_rate,par,frequency,maturity):
    """
    calculate ytm
    :param bond_price:bond price
    :param face_rate: face rate
    :param par: principle
    :param frequency: freqency of payment
    :param maturity: maturity
    :return: bond ytm
    """
    import scipy.optimize as so
    def f(ytm):
        coupon = np.ones_like(maturity) * par * face_rate / frequency
        NPV_coupon = np.sum(coupon * np.exp(-ytm * maturity))
        NPV_par = par * np.exp(-ytm * maturity[-1]) if len(maturity) > 1 else \
            par * np.exp(-ytm * maturity)
        return NPV_coupon+NPV_par-bond_price
    if face_rate==0:
        return (np.log(par/bond_price))/maturity
    else:
        return so.fsolve(func=f,x0=0.1)[0]

def MAC_Duration(fv,par,frequency,ytm,time):
    """
    calculate single bond mac duration
    :param fv: face value
    :param par: par value
    :param frequency: payoff frequency
    :param ytm: yield to maturity
    :param time: time period
    :return: mac duration
    """
    if fv==0:
        duration=time
    else:
        coupon=np.ones_like(time)*par*fv/frequency
        npv_coupon=np.sum(coupon*np.exp(-ytm*time))
        npv_par=par*np.exp(-ytm*time[-1]) if len(time)>1 else par*np.exp(-ytm*time)
        bond_value=npv_par+npv_coupon
        cashflow=coupon
        if len(cashflow)>1:
            cashflow[-1]=par*(1+fv/frequency)
        else:
            cashflow = par * (1 + fv / frequency)
        weight=cashflow*np.exp(-ytm*time)/bond_value
        duration=np.sum(time*weight)
    return duration

def Modified_Duration(fv,par,frequency1,frequency2,ytm,time):
    """
    calculate single bond modified duration
    :param fv: face value
    :param par: par value
    :param frequency: payoff frequency
    :param ytm: yield to maturity
    :param time: time period
    :param time2: time2 period
    :return: mac duration
    """
    mac_duration=MAC_Duration(fv,par,frequency1,ytm,time)
    return mac_duration/(1+ytm/frequency2)

def Dollar_Duration(fv,par,frequency1,frequency2,ytm,time):
    """
    calculate single bond dollar duration
    :param fv: face value
    :param par: par value
    :param frequency: payoff frequency
    :param ytm: yield to maturity
    :param time: time period
    :param time2: time2 period
    :return: mac duration
    """
    r=frequency2*np.log(1+ytm/frequency2)
    if fv==0:
        price=par*np.exp(-r*time)
        mac_duration=time
    else:
        coupon=np.ones_like(time)*par*fv/frequency1
        npv_coupon=np.sum(coupon*np.exp(-r*time))
        npv_par=par*np.exp(-r*time[-1]) if len(time)>1 else par*np.exp(-r*time)
        price=npv_par+npv_coupon
        cashflow=coupon
        if len(cashflow)>1:
            cashflow[-1]=par*(1+fv/frequency1)
        weight=cashflow*np.exp(-r*time)/price
        mac_duration=np.sum(time*weight)
    modified_duration=mac_duration/(1+ytm/frequency2)
    return price*modified_duration

def Convexity(fv,par,frequency,ytm,time):
    """
    calculate single bond convexity
    :param fv: face value
    :param par: par value
    :param frequency: payoff frequency
    :param ytm: yield to maturity
    :param time: time period
    :return: mac duration
    """
    if fv==0:
        convexity=time
    else:
        coupon=np.ones_like(time)*par*fv/frequency
        npv_coupon=np.sum(coupon*np.exp(-ytm*time))
        npv_par=par*np.exp(-ytm*time[-1]) if len(time)>1 else par*np.exp(-ytm*time)
        bond_value=npv_par+npv_coupon
        cashflow=coupon
        if len(cashflow)>1:
            cashflow[-1]=par*(1+fv/frequency)
        else:
            cashflow = par * (1 + fv / frequency)
        weight=cashflow*np.exp(-ytm*time)/bond_value
        convexity=np.sum(pow(time,2)*weight)
    return convexity

def default_prob(y1,y2,rate_of_default,time):
    """
    calculate probability of bond
    :param y1: ytm on time 1
    :param y2: ytm on time 2
    :param rate_of_default: raate of default
    :param time: time period
    :return: probability
    """
    A=(np.exp(-y2*time)-rate_of_default*np.exp(-y1*time))/(1-rate_of_default)
    prob=-np.log(A)/time-y1
    return prob

def cal_Maculay_duration(par_value, payoff_freq, ytm, t0, t1):
    """
    calculate Maculay duration
    :param par_value: par value of bond
    :param payoff_freq: payoff frequency
    :param ytm: yield to maturity
    :param t0: time 0
    :param t1: time 1
    :return: duration
    """
    t0, t1 = pd.to_datetime(t0).date(), pd.to_datetime(t1).date()
    tensor = (t1 - t0).days / 365
    if math.modf(tensor)[0] > 0.5:
        n = payoff_freq * math.ceil(tensor)
    else:
        n = payoff_freq * math.floor(tensor) + 1

    t_list = np.arange(n) / payoff_freq
    t_list = np.sort(tensor - t_list)

    face_value = 100
    price = face_value * (np.sum(np.exp(-ytm * t_list)) * par_value / payoff_freq +
                          np.exp(-ytm * t_list[-1]))
    coupon = np.sum(t_list * np.exp(-ytm * t_list) * face_value * par_value / payoff_freq)
    par = t_list[-1] * face_value * np.exp(-ytm * t_list[-1])
    duration = (coupon + par) / price
    return duration


def cal_Modified_duration(par_value, payoff_freq, ytm, t0, t1):
    """
    calculate Modified duration
    :param par_value: par value of bond
    :param payoff_freq: payoff frequency
    :param ytm: yield to maturity
    :param t0: time 0
    :param t1: time 1
    :return: duration
    """
    t0, t1 = pd.to_datetime(t0).date(), pd.to_datetime(t1).date()
    tensor = (t1 - t0).days / 365
    if math.modf(tensor)[0] > 0.5:
        n = payoff_freq * math.ceil(tensor)
    else:
        n = payoff_freq * math.floor(tensor) + 1

    t_list = np.arange(n) / payoff_freq
    t_list = np.sort(tensor - t_list)

    face_value = 100
    y_continous = payoff_freq * np.log(1 + ytm / payoff_freq)
    price = face_value * (np.sum(np.exp(-y_continous * t_list)) * par_value / payoff_freq +
                          np.exp(-y_continous * t_list[-1]))
    coupon = np.sum(t_list * np.exp(-y_continous * t_list) * face_value * par_value / payoff_freq)
    par = t_list[-1] * face_value * np.exp(-ytm * t_list[-1])
    duration = (coupon + par) / price
    return duration / (1 + ytm / payoff_freq)


def cal_Dollar_duration(par_value, payoff_freq, ytm, t0, t1):
    """
    calculate Modified duration
    :param par_value: par value of bond
    :param payoff_freq: payoff frequency
    :param ytm: yield to maturity
    :param t0: time 0
    :param t1: time 1
    :return: duration
    """
    t0, t1 = pd.to_datetime(t0).date(), pd.to_datetime(t1).date()
    tensor = (t1 - t0).days / 365
    if math.modf(tensor)[0] > 0.5:
        n = payoff_freq * math.ceil(tensor)
    else:
        n = payoff_freq * math.floor(tensor) + 1

    t_list = np.arange(n) / payoff_freq
    t_list = np.sort(tensor - t_list)

    face_value = 100
    y_continous = payoff_freq * np.log(1 + ytm / payoff_freq)
    price = face_value * (np.sum(np.exp(-y_continous * t_list)) * par_value / payoff_freq +
                          np.exp(-y_continous * t_list[-1]))
    coupon = np.sum(t_list * np.exp(-y_continous * t_list) * face_value * par_value / payoff_freq)
    par = t_list[-1] * face_value * np.exp(-ytm * t_list[-1])
    mac_duration = (coupon + par) / price
    modi_duration = mac_duration / (1 + ytm / payoff_freq)
    return price * modi_duration


def cal_bond_value_change(par_value, payoff_freq, ytm, ytm_change, t0, t1, principles):
    """
    calculate bond value change with ytm change
    :param par_value: par value of bond
    :param payoff_freq: payoff frequency
    :param ytm: yield to maturity
    :param t0: time 0
    :param t1: time 1
    :return: duration
    """
    t0, t1 = pd.to_datetime(t0).date(), pd.to_datetime(t1).date()
    tensor = (t1 - t0).days / 365
    if math.modf(tensor)[0] > 0.5:
        n = payoff_freq * math.ceil(tensor)
    else:
        n = payoff_freq * math.floor(tensor) + 1

    t_list = np.arange(n) / payoff_freq
    t_list = np.sort(tensor - t_list)

    face_value = 100
    price = face_value * (np.sum(np.exp(-ytm * t_list)) * par_value / payoff_freq +
                          np.exp(-ytm * t_list[-1]))
    coupon = np.sum(t_list * np.exp(-ytm * t_list) * face_value * par_value / payoff_freq)
    par = t_list[-1] * face_value * np.exp(-ytm * t_list[-1])
    duration = (coupon + par) / price
    value_change = -price * duration * ytm_change * (principles / face_value)
    return value_change


def cal_bond_convexity(par_value, payoff_freq, ytm, t0, t1):
    """
    calculate convexity
    :param par_value: par value of bond
    :param payoff_freq: payoff frequency
    :param ytm: yield to maturity
    :param t0: time 0
    :param t1: time 1
    :return: duration
    """
    t0, t1 = pd.to_datetime(t0).date(), pd.to_datetime(t1).date()
    tensor = (t1 - t0).days / 365
    if math.modf(tensor)[0] > 0.5:
        n = payoff_freq * math.ceil(tensor)
    else:
        n = payoff_freq * math.floor(tensor) + 1

    t_list = np.arange(n) / payoff_freq
    t_list = np.sort(tensor - t_list)

    face_value = 100
    price = face_value * (np.sum(np.exp(-ytm * t_list)) * par_value / payoff_freq +
                          np.exp(-ytm * t_list[-1]))
    coupon = np.sum(t_list ** 2 * np.exp(-ytm * t_list) * face_value * par_value / payoff_freq)
    par = (t_list[-1] ** 2) * face_value * np.exp(-ytm * t_list[-1])
    convexity = (coupon + par) / price
    return convexity


def cal_bond_convexity_change(par_value, payoff_freq, ytm, ytm_change, bond_face_value, t0, t1):
    """
    calculate convexity change influencing bond price
    :param par_value: par value of bond
    :param payoff_freq: payoff frequency
    :param ytm: yield to maturity
    :param ytm_change: yield to maturity change
    :param face_value: bond_face_value of bond
    :param t0: time 0
    :param t1: time 1
    :return: duration
    """
    t0, t1 = pd.to_datetime(t0).date(), pd.to_datetime(t1).date()
    tensor = (t1 - t0).days / 365
    if math.modf(tensor)[0] > 0.5:
        n = payoff_freq * math.ceil(tensor)
    else:
        n = payoff_freq * math.floor(tensor) + 1

    t_list = np.arange(n) / payoff_freq
    t_list = np.sort(tensor - t_list)

    face_value = 100
    price = face_value * (np.sum(np.exp(-ytm * t_list)) * par_value / payoff_freq +
                          np.exp(-ytm * t_list[-1]))
    coupon1 = np.sum(t_list * np.exp(-ytm * t_list) * face_value * par_value / payoff_freq)
    par1 = t_list[-1] * face_value * np.exp(-ytm * t_list[-1])
    D = (coupon1 + par1) / price
    coupon2 = np.sum(t_list ** 2 * np.exp(-ytm * t_list) * face_value * par_value / payoff_freq)
    par2 = (t_list[-1] ** 2) * face_value * np.exp(-ytm * t_list[-1])
    convexity = (coupon2 + par2) / price
    value_change = (-D * price * ytm_change + 0.5 * convexity * price * ytm_change ** 2) * bond_face_value / face_value
    return value_change


def plot_forward_rate_line(spot_rate, forward_rate, time_list, x_lalbe, y_label, title, save_path=None):
    plt.figure(figsize=(30, 15))
    plt.plot(time_list, spot_rate, 'r-', label='spot rate line', lw=2.5)
    plt.plot(time_list, spot_rate, 'bo', label='zero rate')
    plt.plot(time_list, forward_rate, 'c-', label='forward rate line', lw=2.5)
    plt.plot(time_list, forward_rate, 'mo', label='forward rate')
    plt.xticks(fontsize=22)
    plt.xlabel(x_lalbe, fontsize=30)
    plt.yticks(fontsize=22)
    plt.ylabel(y_label, fontsize=30, rotation=90)
    plt.title(title, fontsize=30)
    plt.legend(fontsize=22, loc=4)
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


if __name__ == '__main__':
    r_list = [0.02461687, 0.026397, 0.02697966, 0.02741662, 0.02812507, 0.02903182]
    t0 = '2019-05-31'
    t1 = '2020-05-25'
    t2 = '2020-11-28'
    coupon1 = 0.0271
    coupon2 = 0.036
    m1 = 2
    m2 = 2
    par1 = 3e7
    par2 = 5e7
    par = 100
    print('bond pricing test')
    print(bond_price(coupon1, m1, r_list[0:2], t0, t1))
    print(bond_price(coupon2, m2, r_list[0:3], t0, t2))
    print(bond_price(coupon1, m1, r_list[0:2], t0, t1, True, par1, par))
    print(bond_price(coupon2, m2, r_list[0:3], t0, t2, True, par2, par))
    print('-' * 50, 'beautifully color line', '-' * 50)
    test_list = np.arange(1, 11)
    spot_rate_jun = np.array(
        [0.03159, 0.032568, 0.033124, 0.033367, 0.033507, 0.034313, 0.034852, 0.034822, 0.034773, 0.034756])
    forward_rate_jun = cal_forward_rate(spot_rate_jun[:9], spot_rate_jun[1:], test_list[:9], test_list[1:])
    forward_rate_jun = np.append(spot_rate_jun[0], forward_rate_jun)
    print('forward rate test', forward_rate_jun)
    plot_forward_rate_line(spot_rate_jun, forward_rate_jun, test_list, 'time period', 'rate', '2018 Jun Bond rate line')
    print('-' * 50, 'beautifully color line', '-' * 50)
    print('libor test', cal_fra_pay_libor(0.027, 0.0280763, 6e7, '2018-12-31', '2019-3-31', 1))
    print('-' * 50, 'beautifully color line', '-' * 50)
    print('shibor test: ', cal_fra_pay_shibor(0.038, cal_forward_rate(0.03286, 0.03466, 6 / 12, 9 / 12)
                                              , 0.03466, 1e8, '2019-01-03', '2019-07-01', '2019-09-30', 1))
    print('-' * 50, 'beautifully color line', '-' * 50)
    print('duration test: ', cal_Maculay_duration(0.029, 2, 0.032397, '2019-07-31', '2026-05-05'))
    print('-' * 50, 'beautifully color line', '-' * 50)
    print('bond price change test: ', cal_bond_value_change(0.029, 2, 0.032397, 0.001, '2019-07-31', '2026-05-05', 2e8))
    print('-' * 50, 'beautifully color line', '-' * 50)
    print('modified duration test: ', cal_Maculay_duration(0.044, 1, 0.027225, '2019-08-30', '2020-02-02'))
    print('-' * 50, 'beautifully color line', '-' * 50)
    print('dollar duration test: ', cal_Dollar_duration(0.044, 1, 0.027225, '2019-08-30', '2020-02-02'))
    print('-' * 50, 'beautifully color line', '-' * 50)
    print('convexity test: ', cal_bond_convexity(0.0401, 2, 0.038319, '2019-09-30', '2025-07-31'))
    print('-' * 50, 'beautifully color line', '-' * 50)
    print('convexity change test: ',
          cal_bond_convexity_change(0.0401, 2, 0.038319, 0.0015, 1.8e8, '2019-09-30', '2025-07-31'))
    print('-' * 50, 'beautifully color line', '-' * 50)
    print(Cash_FRA(0.02, 0.02756, 1e8, 1, 1.25, 'long', 'end'))
    print(Cash_FRA(0.02, 0.02756, 1e8, 1, 1.25, 'long', 'begin'))
    print(Cash_FRA(0.02, 0.02756, 1e8, 1, 1.25, 'short', 'end'))
    print(Cash_FRA(0.02, 0.02756, 1e8, 1, 1.25, 'short', 'begin'))
    print('-' * 50, 'beautifully color line', '-' * 50)
    print(fra_value(0.03, 0.02939, 0.024477, 2e8, 0.5, 0.75, 'short'))
    print(fra_value(0.03, 0.02939, 0.024477, 2e8, 0.5, 0.75, 'long'))
    print('-' * 50, 'beautifully color line', '-' * 50)
    print(currency_exchange(7.1277, None, 6e6, 'direct'))
    print(currency_exchange(1.1135, None, 8e6, 'undirect'))
    print('-' * 50, 'beautifully color line', '-' * 50)
    print(tri_currency_arbitrage(7.0965, 1 / 68.4562, 1 / 9.7150, 1e8, 'RMB', 'USD', 'RUB'))
    print('-' * 50, 'beautifully color line', '-' * 50)
    print(fx_forward(7.0965, 0.015820, 0.001801, 1 / 12))
    print('-' * 50, 'beautifully color line', '-' * 50)
    cal_rate1 = fx_forward(7.0066, 0.0256, 0.013973, 6 / 12)
    cal_rate2 = fx_forward(7.1277, 0.0143, 0.0035, 3 / 12)
    print(cal_rate1, cal_rate2)
    print('short_value: ', fx_forward_value(cal_rate1, cal_rate2, 7.1277, 1e8, 0.003494, 3 / 12, 'A', 'A', 'short'))
    print('long_value: ', fx_forward_value(cal_rate1, cal_rate2, 7.1277, 1e8, 0.003494, 3 / 12, 'A', 'B', 'long'))
    print('-' * 50, 'beautifully color line', '-' * 50)
    print(bond_price_one_discount(0, 100, 0, 0.01954, 0.5))
    print('-' * 50, 'beautifully color line', '-' * 50)
    print(cal_ytm(104.802,0.0369,100,2,np.arange(1,2*4+1)/2))
    print('-' * 50, 'beautifully color line', '-' * 50)
    list1=np.arange(1,2*4+1)/2
    print(MAC_Duration(0.0369,100,2,0.024,list1))
    print('-' * 50, 'beautifully color line', '-' * 50)
    print(Modified_Duration(0.0369,100,2,2,0.024145,list1))
    print('-' * 50, 'beautifully color line', '-' * 50)
    print(Dollar_Duration(0.0369,100,2,2,0.024145,list1))
    print('-' * 50, 'beautifully color line', '-' * 50)
    print(Convexity(0.0369,100,2,0.024145,list1))
    print('-' * 50, 'beautifully color line', '-' * 50)
    print(default_prob(0.02922,0.073611,0.381,3))