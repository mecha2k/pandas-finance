# #!/usr/bin/env python
# # coding: utf-8
#
# # # Options
#
# # In[1]:
#
#
# # various pandas, numpy
# import pandas as pd
# import numpy as np
# import pandas.io.data as web
# from datetime import datetime
# from datetime import date
#
# # plotting
# import matplotlib.pyplot as plt
#
# # make plots inline
# get_ipython().run_line_magic('matplotlib', 'inline')
#
# # formatting options
# pd.set_option('display.notebook_repr_html', False)
#
# pd.set_option('display.max_columns', 7)
# pd.set_option('display.max_rows', 10)
# pd.set_option('display.width', 82)
# pd.set_option('precision', 3)
#
#
# # # Options data from Yahoo!
#
# # In[2]:
#
#
# # retrieve options from Yahoo for AAPL
# #aapl_options = web.Options('AAPL', 'yahoo')
#
# # data is not actually retrieved until explicitly requesting.
# #aapl_options = aapl_options.get_all_data().reset_index()
#
#
# # In[3]:
#
#
# # save to a snapshot for easy reuse / offline processing
# #aapl_options.to_csv('aapl_options.csv')
#
#
# # In[4]:
#
#
# # or, load snapshot
# aapl_options = pd.read_csv('aapl_options.csv',
#                            parse_dates=['Expiry'])
#
#
# # In[5]:
#
#
# # let's restructure and tidy this data to be useful in the examples
# aos = aapl_options.sort(['Expiry', 'Strike'])[
#     ['Expiry', 'Strike', 'Type', 'IV', 'Bid',
#      'Ask', 'Underlying_Price']]
# aos['IV'] = aos['IV'].apply(lambda x: float(x.strip('%')))
#
#
# # In[6]:
#
#
# # take a peek at the options data
# aos[:5]
#
#
# # In[7]:
#
#
# # we have the following distinct dates for options available on AAPL
# aos['Expiry'].unique()
#
#
# # In[8]:
#
#
# # call option
# aos.loc[158]
#
#
# # ## Implied volatility
#
# # In[9]:
#
#
# # all calls on expiry date of 2015-02-27
# calls1 = aos[(aos.Expiry=='2015-02-27') & (aos.Type=='call')]
# calls1[:5]
#
#
# # In[10]:
#
#
# # IV tends to be minimized at the underlying price
# ax = aos[(aos.Expiry=='2015-02-27') & (aos.Type=='call')]         .set_index('Strike')[['IV']].plot(figsize=(12,8))
# ax.axvline(calls1.Underlying_Price.iloc[0], color='g');
# #plt.savefig('5104OS_08_01.png', bbox_inches='tight', dpi=300)
#
#
# # # Calculating payoff on options
#
# # ## Call option payoff calculation
#
# # In[11]:
#
#
# def call_payoff(price_at_maturity, strike_price):
#     return max(0, price_at_maturity - strike_price)
#
#
# # In[12]:
#
#
# # out-of-the-money
# call_payoff(25, 30)
#
#
# # In[13]:
#
#
# # in-the-money
# call_payoff(35, 30)
#
#
# # In[14]:
#
#
# def call_payoffs(min_maturity_price, max_maturity_price,
#                  strike_price, step=1):
#     """
#     Calculate the payoffs for a range of maturity prices at
#     a given strike price
#     """
#     maturities = np.arange(min_maturity_price,
#                            max_maturity_price + step, step)
#     payoffs = np.vectorize(call_payoff)(maturities, strike_price)
#     df = pd.DataFrame({'Strike': strike_price, 'Payoff': payoffs},
#                       index=maturities)
#     df.index.name = 'Maturity Price'
#     return df
#
#
# # In[15]:
#
#
# # calculate call payoffs for...
# call_payoffs(10, 25, 15)
#
#
# # In[16]:
#
#
# def plot_call_payoffs(min_maturity_price, max_maturity_price,
#                       strike_price, step=1):
#     """
#     Plot a canonical call option payoff graph
#     """
#     payoffs = call_payoffs(min_maturity_price, max_maturity_price,
#                            strike_price, step)
#     plt.figure(figsize=(12,8))
#     plt.ylim(payoffs.Payoff.min() - 10, payoffs.Payoff.max() + 10)
#     plt.ylabel("Payoff")
#     plt.xlabel("Maturity Price")
#     plt.title('Payoff of call option, Strike={0}'
#               .format(strike_price))
#     plt.xlim(min_maturity_price, max_maturity_price)
#     plt.plot(payoffs.index, payoffs.Payoff.values);
#
#
# # In[17]:
#
#
# plot_call_payoffs(10, 25, 15)
# #plt.savefig('5104OS_08_07.png', bbox_inches='tight', dpi=300)
#
#
# # ## Put option payoff calculation
#
# # In[18]:
#
#
# def put_payoff(price_at_maturity, strike_price):
#     """
#     return the put payoff
#     """
#     return max(0, strike_price - price_at_maturity)
#
#
# # In[19]:
#
#
# # out-of-the-money
# put_payoff(25, 20)
#
#
# # In[20]:
#
#
# # in-the-money
# put_payoff(15, 20)
#
#
# # In[21]:
#
#
# def put_payoffs(min_maturity_price, max_maturity_price,
#                 strike_price, step=1):
#     """
#     Calculate the payoffs for a range of maturity prices
#     at a given strike price
#     """
#     maturities = np.arange(min_maturity_price,
#                            max_maturity_price + step, step)
#     payoffs = np.vectorize(put_payoff)(maturities, strike_price)
#     df = pd.DataFrame({'Payoff': payoffs, 'Strike': strike_price},
#                       index=maturities)
#     df.index.name = 'Maturity Price'
#     return df
#
#
# # In[22]:
#
#
# # calculate call payoffs for
# put_payoffs(10, 25, 15)
#
#
# # In[23]:
#
#
# def plot_put_payoffs(min_maturity_price,
#                      max_maturity_price,
#                      strike_price,
#                      step=1):
#     """
#     Plot a canonical call option payoff graph
#     """
#     payoffs = put_payoffs(min_maturity_price,
#                           max_maturity_price,
#                           strike_price, step)
#     plt.figure(figsize=(12,8))
#     plt.ylim(payoffs.Payoff.min() - 10, payoffs.Payoff.max() + 10)
#     plt.ylabel("Payoff")
#     plt.xlabel("Maturity Price")
#     plt.title('Payoff of put option, Strike={0}'
#               .format(strike_price))
#     plt.xlim(min_maturity_price, max_maturity_price)
#     plt.plot(payoffs.index, payoffs.Payoff.values);
#
#
# # In[24]:
#
#
# plot_put_payoffs(10, 25, 15)
# #plt.savefig('5104OS_08_08.png', bbox_inches='tight', dpi=300)
#
#
# # # Profit and loss calculation
#
# # ## Call option PnL for the Buyer
#
# # In[25]:
#
#
# def call_pnl_buyer(premium, strike_price, min_maturity_price,
#                    max_maturity_price, step = 1):
#     payoffs = call_payoffs(min_maturity_price, max_maturity_price,
#                            strike_price)
#     payoffs['Premium'] = premium
#     payoffs['PnL'] = payoffs.Payoff - premium
#     return payoffs
#
#
# # In[26]:
#
#
# pnl_buyer = call_pnl_buyer(12, 15, 10, 35)
# pnl_buyer
#
#
# # In[27]:
#
#
# def plot_pnl(pnl_df, okind, who):
#     plt.figure(figsize=(12,8))
#     plt.ylim(pnl_df.Payoff.min() - 10, pnl_df.Payoff.max() + 10)
#     plt.ylabel("Profit / Loss")
#     plt.xlabel("Maturity Price")
#     plt.title('Profit and loss of {0} option, {1}, Premium={2} Strike={3}'
#               .format(okind, who, pnl_df.Premium.iloc[0],
#                       pnl_df.Strike.iloc[0]))
#     plt.ylim(pnl_df.PnL.min()-3, pnl_df.PnL.max() + 3)
#     plt.xlim(pnl_df.index[0], pnl_df.index[len(pnl_df.index)-1])
#     plt.plot(pnl_df.index, pnl_df.PnL)
#     plt.axhline(0, color='g');
#
#
# # In[28]:
#
#
# plot_pnl(pnl_buyer, "put", "Buyer")
# #plt.savefig('5104OS_08_09.png', bbox_inches='tight', dpi=300)
#
#
# # ## Call option Profit and Loss for the seller
#
# # In[29]:
#
#
# def call_pnl_seller(premium, strike_price, min_maturity_price,
#                     max_maturity_price, step = 1):
#     payoffs = call_payoffs(min_maturity_price, max_maturity_price,
#                            strike_price)
#     payoffs['Premium'] = premium
#     payoffs['PnL'] = premium - payoffs.Payoff
#     return payoffs
#
#
# # In[30]:
#
#
# pnl_seller = call_pnl_seller(12, 15, 10, 35)
# pnl_seller
#
#
# # In[31]:
#
#
# plot_pnl(pnl_seller, "call", "Seller")
# #plt.savefig('5104OS_08_10.png', bbox_inches='tight', dpi=300)
#
#
# # In[32]:
#
#
# def plot_combined_pnl(pnl_df):
#     plt.figure(figsize=(12,8))
#     plt.ylim(pnl_df.Payoff.min() - 10, pnl_df.Payoff.max() + 10)
#     plt.ylabel("Profit / Loss")
#     plt.xlabel("Maturity Price")
#     plt.title('Profit and loss of call option Strike={0}'
#               .format(pnl_df.Strike.iloc[0]))
#     plt.ylim(min(pnl_df.PnLBuyer.min(), pnl_df.PnLSeller.min()) - 3,
#              max(pnl_df.PnLBuyer.max(), pnl_df.PnLSeller.max()) + 3)
#     plt.xlim(pnl_df.index[0], pnl_df.index[len(pnl_df.index)-1])
#     plt.plot(pnl_df.index, pnl_df.PnLBuyer, color='b')
#     plt.plot(pnl_df.index, pnl_df.PnLSeller, color='r')
#     plt.axhline(0, color='g');
#
#
# # In[33]:
#
#
# pnl_combined = pd.DataFrame({'PnLBuyer': pnl_buyer.PnL,
#                              'PnLSeller': pnl_seller.PnL,
#                              'Premium': pnl_buyer.Premium,
#                              'Strike': pnl_buyer.Strike,
#                              'Payoff': pnl_buyer.Payoff})
# pnl_combined
#
#
# # In[34]:
#
#
# plot_combined_pnl(pnl_combined)
# #plt.savefig('5104OS_08_11.png', bbox_inches='tight', dpi=300)
#
#
# # ## Put option profit and loss for the Buyer
#
# # In[35]:
#
#
# def put_pnl_buyer(premium, strike_price, min_maturity_price,
#                   max_maturity_price, step = 1):
#     payoffs = put_payoffs(min_maturity_price, max_maturity_price,
#                           strike_price)
#     payoffs['Premium'] = premium
#     payoffs['Strike'] = strike_price
#     payoffs['PnL'] = payoffs.Payoff - payoffs.Premium
#     return payoffs
#
#
# # In[36]:
#
#
# pnl_put_buyer = put_pnl_buyer(2, 15, 10, 30)
# pnl_put_buyer
#
#
# # In[37]:
#
#
# plot_pnl(pnl_put_buyer, "put", "Buyer")
# #plt.savefig('5104OS_08_12.png', bbox_inches='tight', dpi=300)
#
#
# # ## Put Option PnL for the Seller
#
# # In[38]:
#
#
# def put_pnl_seller(premium, strike_price, min_maturity_price,
#                    max_maturity_price, step = 1):
#     payoffs = put_payoffs(min_maturity_price, max_maturity_price,
#                           strike_price)
#     payoffs['Premium'] = premium
#     payoffs['Strike'] = strike_price
#     payoffs['PnL'] = payoffs.Premium - payoffs.Payoff
#     return payoffs
#
#
# # In[39]:
#
#
# pnl_put_seller = put_pnl_seller(2, 15, 10, 30)
# pnl_put_seller
#
#
# # In[40]:
#
#
# plot_pnl(pnl_put_seller, "put", "Seller")
# #plt.savefig('5104OS_08_13.png', bbox_inches='tight', dpi=300)
#
#
# # # Black-Scholes using Mibian
#
# # In[41]:
#
#
# aos[aos.Expiry=='2016-01-15'][:2]
#
#
# # In[42]:
#
#
# date(2016, 1, 15) - date(2015, 2, 25)
#
#
# # In[43]:
#
#
# import mibian
# c = mibian.BS([128.79, 34.29, 1, 324], 57.23)
#
#
# # In[44]:
#
#
# c.callPrice
#
#
# # In[45]:
#
#
# c.putPrice
#
#
# # In[46]:
#
#
# c = mibian.BS([128.79, 34.29, 1, 324],
#               callPrice=94.878970089456217 )
# c.impliedVolatility
#
#
# # ### Charting option price change over time
#
# # In[47]:
#
#
# df = pd.DataFrame({'DaysToExpiry': np.arange(364, 0, -1)})
# df
#
#
# # In[48]:
#
#
# bs_v1 = mibian.BS([128.79, 34.29, 1, 324], volatility=57.23)
# calc_call = lambda r: mibian.BS([128.79, 34.29, 1,
#                                  r.DaysToExpiry],
#                                 volatility=57.23).callPrice
# df['CallPrice'] = df.apply(calc_call, axis=1)
# df
#
#
# # In[49]:
#
#
# df[['CallPrice']].plot(figsize=(12,8));
# #plt.savefig('5104OS_08_40.png', bbox_inches='tight', dpi=300)
#
#
# # ## The Greeks
#
# # ### Calculation and visualization
#
# # In[50]:
#
#
# greeks = pd.DataFrame()
# delta = lambda r: mibian.BS([r.Price, 60, 1, 180],
#                             volatility=30).callDelta
# gamma = lambda r: mibian.BS([r.Price, 60, 1, 180],
#                             volatility=30).gamma
# theta = lambda r: mibian.BS([r.Price, 60, 1, 180],
#                             volatility=30).callTheta
# vega = lambda r: mibian.BS([r.Price, 60, 1, 365/12],
#                            volatility=30).vega
#
# greeks['Price'] = np.arange(10, 70)
# greeks['Delta'] = greeks.apply(delta, axis=1)
# greeks['Gamma'] = greeks.apply(gamma, axis=1)
# greeks['Theta'] = greeks.apply(theta, axis=1)
# greeks['Vega'] = greeks.apply(vega, axis=1)
# greeks[:5]
#
#
# # In[51]:
#
#
# greeks[['Delta', 'Gamma', 'Theta', 'Vega']].plot(figsize=(12,8));
# #plt.savefig('5104OS_08_54.png', bbox_inches='tight', dpi=300)
#
