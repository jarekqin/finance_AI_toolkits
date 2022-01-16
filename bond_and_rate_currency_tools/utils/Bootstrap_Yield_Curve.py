import numpy as np

class BoostrapYieldCurve(object):
    def __init__(self):
        self.zeros_rates=dict()
        self.instruments=dict()

    def add_instrument(self,par,t,coup,price,compounding_freq=2):
        self.instruments[t]=(par,coup,price,compounding_freq)

    def get_maturity(self):
        return sorted(self.instruments.keys())

    def get_zero_rate(self):
        self.bootstrap_zero_coupons()
        self.get_bond_spot_rates()
        return [self.zeros_rates[t] for t in self.get_maturity()]

    def bootstrap_zero_coupons(self):
        for (t,instrument) in self.instruments.items():
            (par,coup,price,freq)=instrument
            if coup==0:
                spot_rate=self.zero_coupon_spot_rate(par,price,t)
                self.zeros_rates[t]=spot_rate

    def zero_coupon_spot_rate(self,par,price,t):
        spot_rate=np.log(par/price)/t
        return spot_rate

    def get_bond_spot_rates(self):
        for t in self.get_maturity():
            instrument=self.instruments[t]
            (par,coup,parice,freq)=instrument
            if coup!=0:
                spot_rate=self.calculate_bond_spot_rate(t,instrument)
                self.zeros_rates[t]=spot_rate

    def calculate_bond_spot_rate(self,t,instrument):
        try:
            (par,coup,price,freq)=instrument
            periods=t*freq
            value=price
            per_coupon=coup/freq
            for i in range(int(periods)-1):
                t=(i+1)/float(freq)
                spot_rate=self.zeros_rates[t]
                discounted_coupon=per_coupon*np.exp(-spot_rate*t)
                value-=discounted_coupon
            last_period=int(periods)/float(freq)
            spot_rate=-np.log(value/(par+per_coupon))/last_period
            return spot_rate
        except:
            print("Error: spot rate not found for t=",t)


if __name__=='__main__':
    yield_curve=BoostrapYieldCurve()
    yield_curve.add_instrument(100,0.25,0.,97.5)
    yield_curve.add_instrument(100,0.5,0.,94.9)
    yield_curve.add_instrument(100,1.,0.,90.)
    yield_curve.add_instrument(100,1.5,8,96.,2)
    yield_curve.add_instrument(100,2.,12.,101.6,2)

    print(yield_curve.get_zero_rate())
    print(yield_curve.get_maturity())