import pandas as pd
import numpy as np
import yfinance as yf
import mplfinance as mpf
import matplotlib.pyplot as plt
import scipy.stats as stats

from datetime import datetime
from icecream import ic

# Set some Pandas options
pd.set_option("display.notebook_repr_html", False)
pd.set_option("display.max_columns", 15)
pd.set_option("display.max_rows", 8)
pd.set_option("precision", 3)

idx = pd.IndexSlice

if __name__ == "__main__":
    src_data = "data/yf_data.pkl"
    start = datetime(2012, 1, 1)
    end = datetime(2014, 12, 31)
    try:
        data = pd.read_pickle(src_data)
        sp_500 = pd.read_csv("data/sp500_price.csv")
        print("data reading from file...")
    except FileNotFoundError:
        tickers = ["AAPL", "MSFT", "GE", "IBM", "AA", "DAL", "UAL", "PEP", "KO"]
        get_yfinance = lambda ticker: yf.download(ticker, start=start, end=end)
        data = map(get_yfinance, tickers)
        data = pd.concat(data, keys=tickers, names=["Ticker", "Date"])
        data.to_pickle(src_data)
        sp_500 = yf.download("^GSPC", start=start, end=end)
        sp_500.to_csv("data/sp500_price.csv")
    ic(data.head())
    ic(sp_500.head())

    # reset the index to make everything columns
    just_closing_prices = data[["Adj Close"]].reset_index()
    ic(just_closing_prices[:5])

    # now pivot Date to the index, Ticker values to columns
    daily_close_px = just_closing_prices.pivot("Date", "Ticker", "Adj Close")
    ic(daily_close_px[:5])

    # _ = daily_close_px["AAPL"].plot(figsize=(6, 4))
    # plt.savefig("images/5104OS_05_01.png", bbox_inches="tight", dpi=300)
    #
    # # plot all the stock closing prices against each other
    # _ = daily_close_px.plot(figsize=(6, 4))
    # plt.savefig("images/5104OS_05_02.png", bbox_inches="tight", dpi=300)

    # ## Plotting volumes series data
    # msftV = data.Volume.loc["MSFT"]
    # plt.bar(msftV.index, msftV)
    # plt.gcf().set_size_inches(12, 6)
    # plt.savefig("images/5104OS_05_03.png", bbox_inches="tight", dpi=300)
    #
    # ## Combined Price and Volumes
    # top = plt.subplot2grid((4, 4), (0, 0), rowspan=3, colspan=4)
    # top.plot(daily_close_px.index, daily_close_px["MSFT"], label="Adjusted Close")
    # plt.title("MSFT Adjusted Close Price from 2011 - 2014")
    # plt.legend(loc=2)
    # bottom = plt.subplot2grid((4, 4), (3, 0), rowspan=1, colspan=4)
    # bottom.bar(msftV.index, msftV)
    # plt.title("Microsoft Daily Trading Volume")
    # plt.gcf().set_size_inches(12, 8)
    # plt.subplots_adjust(hspace=0.75)
    # plt.savefig("images/5104OS_05_04.png", bbox_inches="tight", dpi=300)

    ## Plotting candlesticks
    subset = data.loc["MSFT"].loc["2014-12":"2014-12"]
    ic(subset[:5])

    # # convert our dates matplotlib formatters representation
    # import matplotlib.dates as mdates
    #
    # subset["date_num"] = subset["Date"].apply(lambda date: mdates.date2num(date.to_pydatetime()))
    # ic(subset[:5])
    #
    # # extract only values required, in order, as tuples
    # subset_as_tuples = [
    #     tuple(x) for x in subset[["date_num", "Open", "High", "Low", "Close"]].values
    # ]
    # ic(subset_as_tuples[:5])
    #
    # # required imports for fomatting
    # from matplotlib.dates import DateFormatter
    #
    # week_formatter = DateFormatter("%b %d")  # e.g., Jan 12
    #
    # # We want to only display labels for Mondays
    # from matplotlib.dates import WeekdayLocator, MONDAY
    #
    # mondays = WeekdayLocator(MONDAY)  # major ticks on the mondays
    #
    # # now draw the plot
    # plt.figure(figsize=(6, 4))
    # fig, ax = plt.subplots()
    # # set the locator and formatter for the x-axis
    # ax.xaxis.set_major_locator(mondays)
    # ax.xaxis.set_major_formatter(week_formatter)

    # draw the candlesticks
    ic(mpf.available_styles())
    colorset = mpf.make_marketcolors(up="tab:red", down="tab:blue", volume="tab:blue")
    style = mpf.make_mpf_style(marketcolors=colorset)
    # fig = mpf.figure(figsize=(6, 4))
    mpf.plot(
        subset,
        type="candle",
        style=style,
        mav=(3, 6, 9),
        volume=True,
        savefig="images/5104OS_05_05.png",
    )

    ## Calculating daily percentage change
    # calc % change from day 0 to day 1
    AA_p_t0 = daily_close_px.iloc[0]["AA"]  # Pt-1
    AA_p_t1 = daily_close_px.iloc[1]["AA"]  # Pt
    r_t1 = AA_p_t1 / AA_p_t0 - 1  # returns
    ic(AA_p_t0, AA_p_t1, r_t1)

    # we can apply this to everything with the following using slices
    dpc_1 = daily_close_px.iloc[1:] / daily_close_px.iloc[:-1].values - 1
    ic(dpc_1.loc[:, "AA":"AAPL"])

    # show the DataFrame that is the numerator
    price_matrix_minus_day1 = daily_close_px.iloc[1:]
    ic(price_matrix_minus_day1[:5])

    # the numerator is a 2-d array, but excludes the last day
    ic(daily_close_px.iloc[:-1].values)

    # or using the shift function
    dpc_2 = daily_close_px / daily_close_px.shift(periods=1) - 1
    ic(dpc_2.iloc[:, 0:2][:5])

    # to make this easy, pandas has .pct_change() baked in
    daily_pct_change = daily_close_px.pct_change()
    ic(daily_pct_change.iloc[:, 0:2][:5])

    # set NaN's to 0
    daily_pct_change.fillna(0, inplace=True)
    ic(daily_pct_change.iloc[:5, :5])

    ## Calculating simple daily cumulative returns
    # calc the cumulative daily returns
    cum_daily_return = (1 + daily_pct_change).cumprod()
    ic(cum_daily_return.iloc[:, :2][:5])

    # plot all the cumulative returns
    cum_daily_return.plot(figsize=(6, 4))
    _ = plt.legend(loc=2)
    plt.savefig("images/5104OS_05_08.png", bbox_inches="tight", dpi=300)

    ## Analyzing distribution of returns
    # plot daily % change values histogram for AAPL using 50 bins
    aapl = daily_pct_change["AAPL"]
    ic(aapl.head())
    # _ = aapl.hist(bins=50, figsize=(6, 4))
    # plt.savefig("images/5104OS_05_09.png", bbox_inches="tight", dpi=300)

    aapl.describe()
    aapl.describe(percentiles=[0.025, 0.5, 0.975])

    # # plot all the cumulative return distributions
    # _ = daily_pct_change.hist(bins=50, sharex=True, figsize=(6, 4))
    # plt.savefig("images/5104OS_05_10.png", bbox_inches="tight", dpi=300)
    #
    # ### QQ-Plots
    # # create a qq-plot of AAPl returns vs normal
    # f = plt.figure(figsize=(6, 4))
    # ax = f.add_subplot(111)
    # stats.probplot(aapl, dist="norm", plot=ax)
    # plt.savefig("images/5104OS_05_11.png", dpi=300)
    #
    # ## Box and whisker plots
    # # create a box and whisker for the AAPL returns
    # _ = daily_pct_change[["AAPL"]].plot(kind="box", figsize=(3, 6))
    # plt.savefig("images/5104OS_05_12.png", dpi=300)
    #
    # # examine all the returns
    # daily_pct_change.plot(kind="box", figsize=(6, 4))
    # plt.savefig("images/5104OS_05_13.png", dpi=300)
    #
    # ## Comparison of daily percentage change between stocks
    #
    # def render_scatter_plot(data, x_stock_name, y_stock_name, xlim=None, ylim=None):
    #     fig = plt.figure(figsize=(6, 4))
    #     ax = fig.add_subplot(111)
    #     ax.scatter(data[x_stock_name], data[y_stock_name])
    #     if xlim is not None:
    #         ax.set_xlim(xlim)
    #     ax.autoscale(False)
    #     # horiz and v lines at 0
    #     ax.vlines(0, -10, 10)
    #     ax.hlines(0, -10, 10)
    #     # this line would be perfect correlation
    #     ax.plot((-10, 10), (-10, 10))
    #     # label axes
    #     ax.set_xlabel(x_stock_name)
    #     ax.set_ylabel(y_stock_name)
    #
    # # MSFT vs AAPL
    # limits = [-0.15, 0.15]
    # render_scatter_plot(daily_pct_change, "MSFT", "AAPL", xlim=limits)
    # plt.savefig("images/5104OS_05_14.png", bbox_inches="tight", dpi=300)
    #
    # # DAL vs UAL
    # render_scatter_plot(daily_pct_change, "DAL", "UAL", xlim=limits)
    # plt.savefig("images/5104OS_05_15.png", bbox_inches="tight", dpi=300)
    #
    # # all stocks against each other, with a KDE in the diagonal
    # _ = pd.scatter_matrix(daily_pct_change, diagonal="kde", alpha=0.1, figsize=(12, 12))
    # plt.savefig("images/5104OS_05_16.png", bbox_inches="tight", dpi=300)
    #
    # ## Moving Windows
    # msftAC = msft["2012"]["Adj Close"]
    # ic(msftAC[:5])
    #
    # sample = msftAC["2012"]
    # sample.plot(figsize=(6, 4))
    # plt.savefig("images/5104OS_05_17.png", bbox_inches="tight", dpi=300)
    #
    # sample.plot(figsize=(6, 4))
    # pd.rolling_mean(sample, 5).plot(figsize=(6, 4))
    # plt.savefig("images/5104OS_05_18.png", bbox_inches="tight", dpi=300)
    #
    # sample.plot(figsize=(6, 4))
    # pd.rolling_mean(sample, 5).plot(figsize=(6, 4))
    # pd.rolling_mean(sample, 10).plot(figsize=(6, 4))
    # pd.rolling_mean(sample, 20).plot(figsize=(6, 4))
    # plt.savefig("images/5104OS_05_19.png", bbox_inches="tight", dpi=300)
    #
    # mean_abs_dev = lambda x: np.fabs(x - x.mean()).mean()
    # pd.rolling_apply(sample, 5, mean_abs_dev).plot(figsize=(6, 4))
    # plt.savefig("images/5104OS_05_20.png", bbox_inches="tight", dpi=300)
    #
    # expanding_mean = lambda x: pd.rolling_mean(x, len(x), min_periods=1)
    # sample.plot()
    # pd.expanding_mean(sample).plot(figsize=(6, 4))
    # plt.savefig("images/5104OS_05_21.png", bbox_inches="tight", dpi=300)
    #
    # ## Volatility calculation
    # # use a minimum of 75 days
    # min_periods = 75
    # # calculate the rolling standard deviation
    # vol = pd.rolling_std(daily_pct_change, min_periods) * np.sqrt(min_periods)
    # # plot it
    # _ = vol.plot(figsize=(6, 6))
    # plt.savefig("images/5104OS_05_22.png", bbox_inches="tight", dpi=300)
    #
    # ## Rolling correlation of returns
    # # one year (252 days) rolling correlation of AAPL and MSFT
    # rolling_corr = pd.rolling_corr(
    #     daily_pct_change["AAPL"], daily_pct_change["MSFT"], window=252
    # ).dropna()
    # ic(rolling_corr[251:])  # first 251 are NaN
    #
    # # plot the rolling correlation
    # _ = rolling_corr.plot(figsize=(6, 4))
    # plt.savefig("images/5104OS_05_23.png", bbox_inches="tight", dpi=300)
    #
    # ## Least squares regression of returns (beta)
    # # least squares on the returns of AAPL and MSFT
    # model = pd.ols(y=daily_pct_change["AAPL"], x={"MSFT": daily_pct_change["MSFT"]}, window=250)
    # ic(model)
    #
    # # what is the beta?
    # ic(model.beta[0:5])
    #
    # _ = model.beta["MSFT"].plot(figsize=(12, 8))
    # plt.savefig("images/5104OS_05_24.png", bbox_inches="tight", dpi=300)
    #
    # # Comparing stocks to the S&P 500
    # # we need to calculate the pct change on the close for S&P 500
    # sp_500_dpc = sp_500["Adj Close"].pct_change().fillna(0)
    # ic(sp_500_dpc[:5])
    #
    # # now concat the S&P data with the other daily pct values
    # dpc_all = pd.concat([sp_500_dpc, daily_pct_change], axis=1)
    # dpc_all.rename(columns={"Adj Close": "SP500"}, inplace=True)
    # ic(dpc_all[:5])
    #
    # # from all the daily, calculate the cumulative
    # cdr_all = (1 + dpc_all).cumprod()
    # ic(cdr_all[:5])
    #
    # # calculate the correlations
    # dpc_corrs = dpc_all.corr()
    # ic(dpc_corrs)
    #
    # # how well did each stock relate to the S&P 500?
    # ic(dpc_corrs.ix["SP500"])
    #
    # # plot GE/UAL against S&P500
    # _ = cdr_all[["SP500", "GE", "UAL"]].plot(figsize=(6, 4))
    # plt.savefig("images/5104OS_05_25.png", bbox_inches="tight", dpi=300)
    #
    # # GE vs S&P 500
    # render_scatter_plot(dpc_all, "GE", "SP500")
    # plt.savefig("images/5104OS_05_26.png", bbox_inches="tight", dpi=300)
    #
    # # and UAL vs S&P 500
    # render_scatter_plot(dpc_all, "UAL", "SP500")
    # plt.savefig("images/5104OS_05_27.png", bbox_inches="tight", dpi=300)
