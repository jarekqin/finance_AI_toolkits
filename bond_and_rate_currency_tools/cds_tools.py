import numpy as np


def cds_spread(Lambda,recover_rate,zero_rate,contract_maturity,frequency):
    """
    calculate cds spread
    :param Lambda: default rate
    :param recover_rate: recover rate
    :param zero_rat: zero rate
    :param contract_maturity: contract maturity
    :param frequency: payoff frequency
    :return: cds spread
    """
    t_list=np.arange(frequency*contract_maturity+1)/frequency
    a=np.sum(np.exp(-Lambda*t_list[:-1]-zero_rate*t_list[1:]))
    b=np.sum(np.exp(-(Lambda+zero_rate)*t_list[1:]))
    spread=frequency*(1-recover_rate)*(a/b-1)
    return spread

def cds_cashflow(spread,frquency,t1,t2,principle,recovery_rate,trader,event=False):
    """
    calculate cds cashflow
    :param spread: spread on cds
    :param frquency: payoff frequency
    :param t1: contract time
    :param t2: default happened time
    :param principle: principle on contract
    :param recovery_rate: recovery rate after default
    :param trader: trader side
    :param event: default event happened or not
    :return: cash flow
    """
    if not event:
        n=frequency*t1
        cashflow=spread*principle*np.ones(n)/frequency
        if trader=='buyer':
            cf=-cashflow
        else:
            cf=cashflow
    else:
        default_pay=(1-recovery_rate)*principle
        if frequency==1:
            n=int(t2)*frequency+1
            cashflow=spread*principle*np.ones(n)/frequency
            spread_end=(t2-int(t2))*spread*principle
            cashflow[-1]=spread_end-default_pay
            if trader=='buyer':
                cf=-cashflow
            else:
                cf=cashflow
        else:
            n=(int(t2)+1)/frequency
            cashflow=spread*principle*np.ones(n)/frequency
            spread_end=(t2-int(t2)-0.5)*spread*principle
            cashflow[-1]=spread_end-default_pay
            if trader=='buyer':
                cf=-cashflow
            else:
                cf=cashflow
    return cf




if __name__=='__main__':
    zero_rate=np.array([0.021276,0.022853,0.024036,0.025010,0.025976])
    recovery_rate=0.4
    frequency=1
    maturity=5
    Lambda=0.03
    print(cds_spread(Lambda,recovery_rate,zero_rate,maturity,frequency))

    spread=0.012
    frequency=1
    maturity=3
    par=1e8
    print(cds_cashflow(spread,frequency,maturity,None,par,None,'buyer',False))

    t_default=28/12
    rate=0.35
    print(cds_cashflow(spread, frequency, maturity, t_default, par, rate, 'buyer', True))
    print(cds_cashflow(spread, frequency, maturity, t_default, par, rate, 'seller', True))
















