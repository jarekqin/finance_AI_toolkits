import numpy as np
import pandas as pd

import multiprocessing

def pcaWeights(cov,riskDist=None,riskTarget=1.):
    """
    Using R distribution to calculate PCA weights
    :param cov: covirance matrix
    :param riskDist: self-defined risk distribution. if this is None,then we are going to use zeros matrix
    :param riskTarget: risk targeted value
    :return: PCA weights
    """
    eVal,eVec=np.linalg.eight(cov)
    indices=eVal.argsort()[::-1]
    Val,eVec=eVal[indices],eVec[indices]
    if riskDist is None:
        riskDist=np.zeros(cov.shape[0])
        riskDist[-1]=1.
    loads=riskTarget*(riskDist/eVal)**0.5
    weights=np.dot(eVec,np.reshape(loads,(-1,1)))
    return weights

def getTEvents(gRaw,h):
    """
    Cusum filter
    :param gRaw:origonal datatime series
    :param h: threshold value
    :return: signal list
    """
    tEvents,sPos,sNeg=[],0,0

    diff=gRaw.diff()
    for i in diff.index[1:]:
        sPos,sNeg=max(0,sPos+diff.loc[i]),min(0,sNeg+diff.loc[i])
        if sNeg<-h:
            sNeg=0
            tEvents.append(i)
        elif sPos>h:
            sPos=0
            tEvents.append(i)
    return pd.DatetimeIndex(tEvents)


def getDailyVol(close,span0=100):
    """
    according to close series to calculate vol based on ewm model
    :param close: close time series
    :param span0: span value for ewm model
    :return: daily vol
    """
    df0=close.index.searchsorted(close.index-pd.Timedelta(days=1))
    df0=df0[df0>0]
    df0=pd.Series(close.index[df0-1],index=close.index[close.shape[0]-df0.shape[0]:])
    df0=close.loc[df0.index]/close.loc[df0.values].value-1
    df0=df0.ewm(span=span0).std()
    return df0

def applyStSlOnT1(close,events,ptS1,molecule):
    """
    triple obstcles labeling method
    :param close: close time series
    :param events: dataframe data column
    :param ptS1: obstcles width
    :param molecule: signle threat solver
    :return: labels
    """
    events_=events.loc[molecule]
    out=events_[['t1']].copy(deep=True)
    if ptS1[0]>0:
        pt=ptS1[0]*events_['trgt']
    else:
        pt=pd.Series(index=events.index)
    if ptS1[1]>0:
        s1=-ptS1[1]*events_['trgt']
    else:
        s1=pd.Series(index=events.index)

    for loc,t1 in events_['t1'].fillna(close.index[-1]).iteritems():
        df0=close[loc:t1]
        df0=(df0-close[loc]-1)*events_.loc[loc,'side']
        out.loc[loc,'s1']=df0[df0<s1[loc]].index.min()
        out.loc[loc,'pt']=df0[df0>pt[loc]].index.min()
    return out


# this function can only be seen as an example
def getEvents(close,tEvents,ptS1,trgt,minRet,numThreads,t1=False):
    trgt=trgt.loc[tEvents]
    trgt=trgt[trgt>minRet]
    if t1 is False:
        t1=pd.Series(pd.NaT,index=tEvents)
    side_=pd.Series(1.,index=trgt.index)
    events=pd.concat({'t1':t1,'trgt':trgt,'side':side_})
    df0=multiprocessing.Process(func=applyStSlOnT1,target={'molecule',events.index},
                                numThreads=numThreads,close=close,events=events,ptS1=[ptS1,ptS1]
                                )
    events['t1']=df0.dropna(how='all').min(axis=1)
    events=events.drop('side',axis=1)
    return events