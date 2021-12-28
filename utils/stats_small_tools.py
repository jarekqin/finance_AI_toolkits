import numpy as np
import statsmodels.api as sma
import matplotlib.pyplot as plt
import arch

def OLS_model(x,y,return_summary=False):
    x_consts=sma.add_constant(x)
    model=sma.OLS(endog=y,exog=x_consts)
    returned_model = model.fit()

    if return_summary:
        print(returned_model.summay())
        return returned_model
    else:
        return returned_model

def S_plot(x,y,model,title,x_label,y_label,save_path=None):
    plt.figure(figsize=(15,8))
    plt.scatter(x,y,c='b',marker='o')
    plt.plot(x,model.params[0]+model.paras[1]*x,'r-',lw=2.5)
    plt.xticks(fontsize=18)
    plt.xlabel('%s' % x_label,fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel('%s' % y_label,fontsize=18)
    plt.title('%s' % title,fontsize=18)
    plt.grid(True)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)

def ARCH_model(x,y,mean,lags,vol,p,o,q,dist='normal',show=True,*args,**kwargs):
    if len(x)==0:
        model=arch.arch_model(y=y,mean=mean,lags=lags,vol=vol,p=p,o=o,q=q,dist=dist,*args,**kwargs)
    else:
        model = arch.arch_model(y=y,x=x, mean=mean, lags=lags, vol=vol, p=p, o=o, q=q, dist=dist,*args,**kwargs)
    model_=model.fit(*args,**kwargs)
    if show:
        model_.summary()
        print('*'*50)
        model_.plot()
    else:
        return model_
