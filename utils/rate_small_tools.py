import numpy as np


def FV(A,n,r,m):
    """
    calcuate fave value
    :param A: principle
    :param n: investment years
    :param r: annual rate
    :param m: frequency
    :return: face value
    """

    if m=='year':
        value=A*pow(1+r,n)
    elif m=='semi-year':
        value=A*pow(1+r/2,n*2)
    elif m=='quater':
        value=A*pow(1+r/4,n*4)
    elif m=='month':
        value=A*pow(1+r/12,n*12)
    elif m=='week':
        value=A*pow(1+r/52,n*52)
    else:
        value=A*pow(1+r/365,n*365)
    return value

def R_m2(r_m1,m1,m2):
    """
    calculate known rate of r_m1 for equal new rate on m2 times
    :param r_m1: known rate1
    :param m1: frequency of r_m1
    :param m2: frequency of new rate
    :return: new rate on m2 frequency
    """
    r=m2*(pow(1+r_m1/m1,m1/m2)-1)
    return r

def Rc(rm,m):
    """
    calculate continous rate
    :param rm: known rate
    :param m: frequency
    :return: new rate
    """
    r=m.np.log(1+rm)/m
    return r

