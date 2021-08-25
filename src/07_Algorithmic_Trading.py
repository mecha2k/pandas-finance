#!/usr/bin/env python
# coding: utf-8

# In[1]:


# setup most of the environment for the examples

# pandas and numpy
import pandas as pd
import pandas.io.data as web
import numpy as np

# dates
from datetime import datetime

# plotting
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# formatting options
pd.set_option('display.notebook_repr_html', False)
pd.set_option('display.max_columns', 8)
pd.set_option('display.max_rows', 10) 
pd.set_option('display.width', 78) 
pd.set_option('precision', 6)


# # Moving Average

# In[2]:


# get stock data for microsoft
msft = web.DataReader("MSFT", "yahoo", 
                      datetime(2000, 1, 1), 
                      datetime(2014, 12, 31))
msft[:5]


# In[3]:


# calculate 7, 30, 90 and 120 rolling windows
msft['MA7'] = pd.rolling_mean(msft['Adj Close'], 7)
msft['MA30'] = pd.rolling_mean(msft['Adj Close'], 30)
msft['MA90'] = pd.rolling_mean(msft['Adj Close'], 90)
msft['MA120'] = pd.rolling_mean(msft['Adj Close'], 120)


# In[4]:


# plot the price against the various rolling windows
msft['2014'][['Adj Close', 'MA7', 
              'MA30', 'MA120']].plot(figsize=(12,8));
plt.savefig('5104OS_07_01.png', bbox_inches='tight', dpi=300)


# In[5]:


# plot the price against the various rolling windows
msft['2002'][['Adj Close', 'MA7', 
              'MA30', 'MA120']].plot(figsize=(12,8));
plt.savefig('5104OS_07_02.png', bbox_inches='tight', dpi=300)


# # Exponential Weighted Moving Average

# In[6]:


periods = 10
alpha = 2.0/(periods +1)
factors = (1-alpha) ** np.arange(1, 11)
sum_factors = factors.sum()
weights = factors/sum_factors
weights


# In[7]:


# calculate EWMA relative to MA for 90 days
span = 90
msft_ewma = msft[['Adj Close']].copy()
msft_ewma['MA90'] = pd.rolling_mean(msft_ewma, span)
msft_ewma['EWMA90'] = pd.ewma(msft_ewma['Adj Close'], 
                              span=span)
msft_ewma['2014'].plot(figsize=(12, 8));
plt.savefig('5104OS_07_11.png', bbox_inches='tight', dpi=300)


# # Technical analysis techniques

# ## Cross-overs

# In[8]:


# plot MSFT July 2002 through Sept 2002
# there are several simple 30 day moving average cross overs
msft['2002-1':'2002-9'][['Adj Close', 
                         'MA30']].plot(figsize=(12,8));
plt.savefig('5104OS_07_12.png', bbox_inches='tight', dpi=300)


# In[9]:


# plot 30 and 90 day moving average for MSFT Jan 2002 through June 2002
# there is one cross-over
msft['2002-1':'2002-6'][['Adj Close', 'MA30', 'MA90']
                       ].plot(figsize=(12,8));
plt.savefig('5104OS_07_13.png', bbox_inches='tight', dpi=300)


# # Zipline

# ## Algo trading with Zipline

# ### Zipline: buy apple

# In[10]:


import zipline as zp


# In[11]:


class BuyApple(zp.TradingAlgorithm):
    """ Simple trading algorithm that does nothing
    but buy one share of AAPL every trading period.
    """
    
    trace=False
    
    def __init__(self, trace=False):
        BuyApple.trace = trace
        super(BuyApple, self).__init__()
    
    def initialize(context):
        if BuyApple.trace: print("---> initialize")
        if BuyApple.trace: print(context)
        if BuyApple.trace: print("<--- initialize")
        
    def handle_data(self, context):
        if BuyApple.trace: print("---> handle_data")
        if BuyApple.trace: print(context)
        self.order("AAPL", 1)
        if BuyApple.trace: print("<-- handle_data")  


# In[12]:


import zipline.utils.factory as zpf
# zipline has its own method to load data from Yahoo! Finance
data = zpf.load_from_yahoo(stocks=['AAPL'], 
                           indexes={}, 
                           start=datetime(1990, 1, 1),
                           end=datetime(2014, 1, 1), 
                           adjusted=False)
data.plot(figsize=(12,8));
plt.savefig('5104OS_07_15.png', bbox_inches='tight', dpi=300)


# In[13]:


result = BuyApple(trace=True).run(data['2000-01-03':'2000-01-07'])


# In[14]:


# orders for the first day of the simulation
result.iloc[0].orders


# In[15]:


# orders for day 2
result.iloc[1].orders


# In[16]:


# what is our starting and ending cash in each day of trading,
# along with the value of our investments?
result[['starting_cash', 'ending_cash', 'ending_value']]


# In[17]:


# what is our portfolio value each day of trading?
# that is, how much cash on hand plus the value of our investments
pvalue = result.ending_cash + result.ending_value
pvalue


# In[18]:


# zipline already calculates this for each day
result.portfolio_value


# In[19]:


# what were our daily rates of return?
result.portfolio_value.pct_change()


# In[20]:


# daily returns are already included in the results
result['returns']


# In[21]:


# run the simulation for all of 2000
result_for_2000 = BuyApple().run(data['2000'])


# In[22]:


# take a look at our cash and investment value
result_for_2000[['ending_cash', 'ending_value']]


# In[23]:


# visualize our portfolio value
result_for_2000.portfolio_value.plot(figsize=(12,8));
plt.savefig('5104OS_07_16.png', bbox_inches='tight', dpi=300)


# In[24]:


# run it over five years
result = BuyApple().run(data['2000':'2004']).portfolio_value
result.plot(figsize=(12,8));
plt.savefig('5104OS_07_17.png', bbox_inches='tight', dpi=300)


# # Dual Moving Average Cross-Over

# In[25]:


# reminder of the AAPL data
sub_data = data['1990':'2002-01-01']
sub_data.plot(figsize=(12,8));
plt.savefig('5104OS_07_18.png', bbox_inches='tight', dpi=300)


# In[26]:


"""
The following algorithm implements a double moving average cross 
over. Investments will be made whenever the short moving average 
moves across the long moving average. We will trade only at the 
cross, not continuously buying or selling until the next cross. 
If trending down, we will sell all of our stock.  If trending up, 
we buy as many shares as possible up to 100. The strategy will 
record our buys and sells in extra data return from the simulation.
"""
class DualMovingAverage(zp.TradingAlgorithm):
    def initialize(context):
        # we need to track two moving averages, so we will set
        #these up in the context the .add_transform method 
        # informs zipline to execute a transform on every day 
        # of trading
        
        # the following will set up a MovingAverge transform, 
        # named short_mavg, accessing the .price field of the 
        # data, and a length of 100 days
        context.add_transform(zp.transforms.MovingAverage, 
                              'short_mavg', ['price'],
                              window_length=100)

        # and the following is a 400 day MovingAverage
        context.add_transform(zp.transforms.MovingAverage,
                              'long_mavg', ['price'],
                              window_length=400)

        # this is a flag we will use to track the state of 
        # whether or not we have made our first trade when the 
        # means cross.  We use it to identify the single event 
        # and to prevent further action until the next cross
        context.invested = False

    def handle_data(self, data):
        # access the results of the transforms
        short_mavg = data['AAPL'].short_mavg['price']
        long_mavg = data['AAPL'].long_mavg['price']
        
        # these flags will record if we decided to buy or sell
        buy = False
        sell = False

        # check if we have crossed
        if short_mavg > long_mavg and not self.invested:
            # short moved across the long, trending up
            # buy up to 100 shares
            self.order_target('AAPL', 100)
            # this will prevent further investment until 
            # the next cross
            self.invested = True
            buy = True # records that we did a buy
        elif short_mavg < long_mavg and self.invested:
            # short move across the long, tranding down
            # sell it all!
            self.order_target('AAPL', -100)
            # prevents further sales until the next cross
            self.invested = False
            sell = True # and note that we did sell

        # add extra data to the results of the simulation to 
        # give the short and long ma on the interval, and if 
        # we decided to buy or sell
        self.record(short_mavg=short_mavg,
                    long_mavg=long_mavg,
                    buy=buy,
                    sell=sell)


# In[27]:


# run the simulation
results = DualMovingAverage().run(sub_data)


# In[28]:


# draw plots of the results
def analyze(data, perf):
    fig = plt.figure() # create the plot
    
    # the top will be a plot of long/short ma vs price
    ax1 = fig.add_subplot(211,  ylabel='Price in $')
    data['AAPL'].plot(ax=ax1, color='r', lw=2.)
    perf[['short_mavg', 'long_mavg']].plot(ax=ax1, lw=2.)

    # the following puts an upward triangle at each point 
    # we decided to buy
    ax1.plot(perf.ix[perf.buy].index, perf.short_mavg[perf.buy],
             '^', markersize=10, color='m')
    # and the following a downward triangle where we sold
    ax1.plot(perf.ix[perf.sell].index, perf.short_mavg[perf.sell],
             'v', markersize=10, color='k')

    # bottom plot is the portfolio value
    ax2 = fig.add_subplot(212, ylabel='Portfolio value in $')
    perf.portfolio_value.plot(ax=ax2, lw=2.)

    # and also has the marks for buy and sell points
    ax2.plot(perf.ix[perf.buy].index, 
             perf.portfolio_value[perf.buy],
             '^', markersize=10, color='m')
    ax2.plot(perf.ix[perf.sell].index, 
             perf.portfolio_value[perf.sell],
             'v', markersize=10, color='k')

    # and set the legend position and size of the result
    plt.legend(loc=0)
    plt.gcf().set_size_inches(14, 10)
    plt.savefig('5104OS_07_19.png', bbox_inches='tight', dpi=300)


# In[29]:


# visually analyze the results
analyze(sub_data, results)


# ## Pair trading

# In[30]:


# load data for Coke and Pepsi and visualize
data = zpf.load_from_yahoo(stocks=['PEP', 'KO'], 
                           indexes={},
                           start=datetime(1997, 1, 1), 
                           end=datetime(1998, 6, 1), 
                           adjusted=True)
data.plot(figsize=(12,8));
plt.savefig('5104OS_07_20.png', bbox_inches='tight', dpi=300)


# In[31]:


# calculate and plot the spread
data['Spread'] = data.PEP - data.KO
data['1997':].Spread.plot(figsize=(12,8))
plt.ylabel('Spread')
plt.axhline(data.Spread.mean());
plt.savefig('5104OS_07_21.png', bbox_inches='tight', dpi=300)


# In[32]:


import statsmodels.api as sm
@zp.transforms.batch_transform
def ols_transform(data, ticker1, ticker2):
    """Compute the ordinary least squares of two series.
    """
    p0 = data.price[ticker1]
    p1 = sm.add_constant(data.price[ticker2], prepend=True)
    slope, intercept = sm.OLS(p0, p1).fit().params

    return slope, intercept


# In[33]:


class Pairtrade(zp.TradingAlgorithm):
    """ Pairtrade algorithm for two stocks, using a window 
    of 100 days for calculation of the z-score and 
    normalization of the spread. We will execute on the spread 
    when the z-score is > 2.0 or < -2.0. If the absolute value 
    of the z-score is < 0.5, then we will empty our position 
    in the market to limit exposure.
    """
    def initialize(self, window_length=100):
        self.spreads=[]
        self.invested=False
        self.window_length=window_length
        self.ols_transform=             ols_transform(refresh_period=self.window_length,
                          window_length=self.window_length)

    def handle_data(self, data):
        # calculate the regression, will be None until 100 samples
        params=self.ols_transform.handle_data(data, 'PEP', 'KO')
        if params:
            # get the intercept and slope
            intercept, slope=params

            # now get the z-score
            zscore=self.compute_zscore(data, slope, intercept)

            # record the z-score
            self.record(zscore=zscore)

            # execute based upon the z-score
            self.place_orders(data, zscore)

    def compute_zscore(self, data, slope, intercept):
        # calculate the spread
        spread=(data['PEP'].price-(slope*data['KO'].price+ 
                                       intercept))
        self.spreads.append(spread) # record for z-score calc
        self.record(spread = spread)
        
        # now calc the z-score
        spread_wind=self.spreads[-self.window_length:]
        zscore=(spread - np.mean(spread_wind))/np.std(spread_wind)
        return zscore

    def place_orders(self, data, zscore):
        if zscore>=2.0 and not self.invested:
            # buy the spread, buying PEP and selling KO
            self.order('PEP', int(100/data['PEP'].price))
            self.order('KO', -int(100/data['KO'].price))
            self.invested=True
            self.record(action="PK")
        elif zscore<=-2.0 and not self.invested:
            # buy the spread, buying KO and selling PEP
            self.order('PEP', -int(100 / data['PEP'].price))
            self.order('KO', int(100 / data['KO'].price))
            self.invested = True
            self.record(action='KP')
        elif abs(zscore)<.5 and self.invested:
            # minimize exposure
            ko_amount=self.portfolio.positions['KO'].amount
            self.order('KO', -1*ko_amount)
            pep_amount=self.portfolio.positions['PEP'].amount
            self.order('PEP', -1*pep_amount)
            self.invested=False
            self.record(action='DE')
        else:
            # take no action
            self.record(action='noop')


# In[34]:


perf = Pairtrade().run(data['1997':])


# In[35]:


# what actions did we take?
selection = ((perf.action=='PK') | (perf.action=='KP') |
             (perf.action=='DE'))
actions = perf[selection][['action']]
actions


# In[36]:


# plot prices
ax1 = plt.subplot(411)
data[['PEP', 'KO']].plot(ax=ax1)
plt.ylabel('Price')

# plot spread
ax2 = plt.subplot(412, sharex=ax1)
data.Spread.plot(ax=ax2)
plt.ylabel('Spread')

# plot z-scores
ax3 = plt.subplot(413)
perf['1997':].zscore.plot()
ax3.axhline(2, color='k')
ax3.axhline(-2, color='k')
plt.ylabel('Z-score')

# plot portfolio value
ax4 = plt.subplot(414)
perf['1997':].portfolio_value.plot()
plt.ylabel('Protfolio Value')

# draw lines where we took actions
for ax in [ax1, ax2, ax3, ax4]:
    for d in actions.index[actions.action=='PK']:
        ax.axvline(d, color='g')
    for d in actions.index[actions.action=='KP']:
        ax.axvline(d, color='c')
    for d in actions.index[actions.action=='DE']:
        ax.axvline(d, color='r')

plt.gcf().set_size_inches(16, 12)
plt.savefig('5104OS_07_22.png', bbox_inches='tight', dpi=300)

