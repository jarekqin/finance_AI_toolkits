import numpy as np
from scipy.stats import norm


def CVAR(principle,recovery_rate,copula_rho,default_rate,contract_holding_period,available_scale=0.05):
    """
    credit value at risk
    :param principle: contract principle
    :param recovery_rate: recovery rate when seeing a default
    :param copula_rho: copula function rho
    :param default_rate: default rate
    :param contract_holding_period: contract holding period
    :param available_scale: available scale on normal distribution
    :return: credit value at risk
    """
    c=1-np.exp(-default_rate*contract_holding_period)
    v=norm.cdf((norm.ppf(c)+pow(copula_rho,0.5)*norm.ppf(available_scale))/pow(1-copula_rho,0.5))
    var=principle*(1-recovery_rate)*v
    return var


if __name__=='__main__':
    print(CVAR(2e11,0.5,0.2,0.015,1,0.999))