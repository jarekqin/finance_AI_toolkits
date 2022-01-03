import numpy as np

def CAMP(beta,rm,rf):
    """
    calculate CAPM model
    :param beta: beta from market
    :param rm: return rate from market
    :param rf: risk rate
    :return: CAPM value
    """
    return rf+beta*(rm-rf)


def sharp_atio(rp,rf,vol):
    """
    calculate sharp ratio
    :param rp: return of portfolio
    :param rf: risk free
    :param vol: volatility of portfolio
    :return: sharp ratio
    """
    return (rp-rf)/vol

def sortino_ratio(rp,rf,vol_lower):
    """
    calculate sortino ratio
    :param rp: return of portfolio
    :param rf: risk free
    :param vol_lower: volatility of portfolio that less than mean
    :return: sharp ratio
    """
    return (rp-rf)/vol_lower


def treynor_ratio(rp,rf,beta):
    """
    calculate treynor_ratio
    :param rp: return of portfolio
    :param rf: risk free
    :param beta: beta from market
    :return: treynor ratio
    """
    return (rp-rf)/beta

def calmar_ratio(rp,max_drowback):
    """
    calculate calmar_ratio
    :param rp: return of portfolio
    :param max_drowback: max drowback ratio
    :return: calmar ratio
    """
    return rp/max_drowback

def max_drowback(data):
    """
    calcuate data max drowback
    :param data: series or dateframe
    :return: max drowback
    """
    n=len(data)
    dd=np.zeros((n-1,n-1))
    for i in range(n-1):
        pi=data.iloc[i]
        for j in range(i+1,n):
            pj=data.iloc[j]
            dd[i,j-1]=(pi-pj)/pi
    max_dd=np.mx(dd)
    return max_dd

def inform_ratio(rp,rb,trace_error):
    """
    calculate inform ratio
    :param rp: return of portfolio
    :param rb: return of benchmark
    :param trace_error: trace error
    :return: inform ratio
    """
    return (rp-rb)/trace_error


if __name__=='__main__':
    print(inform_ratio(0.15,0.12,0.2))