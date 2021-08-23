import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from datetime import datetime
from icecream import ic

# Set some Pandas options
pd.set_option("display.notebook_repr_html", False)
pd.set_option("display.max_columns", 15)
pd.set_option("display.max_rows", 8)
pd.set_option("precision", 3)

idx = pd.IndexSlice

if __name__ == "__main__":
    # Time-series data and the DatetimeIndex
    # create a a DatetimeIndex from an array of datetime's
    dates = [datetime(2014, 8, 1), datetime(2014, 8, 2)]
    dti = pd.DatetimeIndex(dates)
    ic(dti)

    # a Series given a datetime list will automatically create
    # a DatetimeIndex as its index
    np.random.seed(42)
    ts = pd.Series(np.random.randn(2), dates)
    ic(type(ts.index))

    # retrieve a value using a datetime object
    ic(ts[datetime(2014, 8, 2)])

    # this can also be performed with a string
    ic(ts["2014-8-2"])

    # create a Series with a DatetimeIndex using strings as dates
    np.random.seed(42)
    dates = ["2014-08-01", "2014-08-02"]
    ts = pd.Series(np.random.randn(2), dates)
    ic(ts)

    # convert a list of items to a DatetimeIndex
    # NaT : not-a-time value
    # dti = pd.to_datetime(["Aug 1, 2014", "2014-08-02", "2014.8.3", None])
    # ic(dti)

    # watch out as a failure to convert an item on the list
    # to a date/time will result in the return value being a
    # NumPy array instead of a DatetimeIndex
    # dti2 = pd.to_datetime(["Aug 1, 2014", "foo"])
    # ic(type(dti2))

    # coerce pandas to convert all to datetime and a DatetimeIndex
    # substituting NaT where values can not be converted
    ic(pd.to_datetime(["Aug 1, 2014", "foo"], errors="coerce"))

    # demonstrate two representations of the same date, one
    # month first, the other day first, converting to the
    # same date representation in pandas
    dti1 = pd.to_datetime(["8/1/2014"])
    dti2 = pd.to_datetime(["1/8/2014"], dayfirst=True)
    ic(dti1[0], dti2[0])

    # create a Series with a DatetimeIndex starting at 8/1/2014
    # and consisting of 10 consequtive days
    dates = pd.date_range("8/1/2014", periods=10)
    s1 = pd.Series(np.random.randn(10), dates)
    ic(s1[:5])

    # for examples of data retrieval / slicing, we will use the
    # following data from Yahoo! Finance
    msft = pd.read_csv("data/msft.csv", index_col=0, parse_dates=True)
    aapl = pd.read_csv("data/aapl.csv", index_col=0, parse_dates=True)
    ic(msft.head())

    # extract just the Adj Close values
    msftAC = msft["Adj Close"]
    ic(msftAC.head(3))

    # slicing using a DatetimeIndex nicely works with dates
    # passed as strings
    ic(msft["2012-01-01":"2012-01-05"])

    # returns a Series representing all the values of the
    # single row indexed by the column names
    ic(msft.loc["2012-01-03"])

    # this is an error as this tries to retrieve a column
    # named '2012-01-03'
    # msft['2012-01-03'] # commented to prevent killing the notebook

    # this is a Series, so the lookup works
    ic(msftAC.loc["2012-01-03"])

    # we can lookup using partial date specifications such as only year and month
    ic(msft.loc["2012-02"].head(5))

    # slice starting at the beginning of Feb 2012 and end on Feb 9 2012
    ic(msft.loc["2012-02":"2012-02-09"])

    # create a time-series with one minute frequency
    bymin = pd.Series(
        np.arange(0, 90 * 60 * 24), pd.date_range("2014-08-01", "2014-10-29 23:59:00", freq="T")
    )
    ic(bymin)

    # slice at the minute level
    ic(bymin["2014-08-01 12:30":"2014-08-01 12:59"])

    # create a period representing a start of 2014-08 and for a duration of one month
    aug2014 = pd.Period("2014-08", freq="M")
    ic(aug2014)

    # pandas determined the following start and end for the period
    ic(aug2014.start_time, aug2014.end_time)

    # what is the one month period following the given period?
    sep2014 = aug2014 + 1
    ic(sep2014)

    # the calculated start and end are
    ic(sep2014.start_time, sep2014.end_time)

    # create a pandas PeriodIndex
    mp2013 = pd.period_range("1/1/2013", "12/31/2013", freq="M")
    ic(mp2013)

    # dump all the calculated periods
    for p in mp2013:
        print("{0} {1} {2} {3}".format(p, p.freq, p.start_time, p.end_time))

    # and now create a Series using the PeriodIndex
    np.random.seed(42)
    ps = pd.Series(np.random.randn(12), mp2013)
    ic(ps)

    # Shifting and lagging time-series data
    # refresh our memory on the data in the MSFT closing prices Series
    ic(msftAC[:5])

    # shift the prices one index position forward
    shifted_forward = msftAC.shift(1)
    ic(shifted_forward[:5])

    # the last item is also shifted away
    ic(msftAC.tail(5), shifted_forward.tail(5))

    # shift backwards 2 index labels
    shifted_backwards = msftAC.shift(-2)
    ic(shifted_backwards[:5])

    # this has resulted in 2 NaN values at
    # the end of the resulting Series
    ic(shifted_backwards.tail(5))

    # shift by a different frequency does not realign and ends up
    # essentially changing the index labels by the specific amount of time
    ic(msftAC.shift(periods=1, freq="S"))

    # resulting Series has one day added to all index labels
    ic(msftAC.shift(periods=1, freq="D"))

    # calculate the percentage change in closing price
    ic(msftAC.shift(periods=1))
    ic(msftAC / msftAC.shift(periods=1))
    ic(msftAC / msftAC.shift(periods=1) - 1)

    # Frequency conversion of time-series data
    # take a two item sample of the msftAC data for demonstrations
    sample = msftAC[:2]
    ic(sample)

    # demonstrate resampling to hour intervals realignment causes many NaN's
    ic(sample.asfreq("H"))

    # fill NaN's with the last know non-NaN valuen
    ic(sample.asfreq("H", method="ffill"))

    # fill with the "next known" value
    ic(sample.asfreq("H", method="bfill"))

    ## Up and down resampling of time-series
    # calculate the cumulative daily returns for MSFT
    msft_cum_ret = (1 + (msftAC / msftAC.shift() - 1)).cumprod()
    ic(msft_cum_ret)

    # resample to a monthly cumulative return
    msft_monthly_cum_ret = msft_cum_ret.resample("M")
    ic(msft_monthly_cum_ret)

    # verify the monthly average for 2012-01
    ic(msft_cum_ret["2012-01"].mean())

    # verify that the default resample techique is mean
    ic(msft_cum_ret.resample(rule="M").mean())

    # resample to monthly and give us open, high, low, close
    ic(msft_cum_ret.resample(rule="M").ohlc()[:5])

    # this will return an index with periods instead of timestamps
    by_periods = msft_cum_ret.resample(rule="M", kind="period").mean()
    for i in by_periods.index[:5]:
        ic("{0}:{1} {2}".format(i.start_time, i.end_time, by_periods[i]))

    # upsampling will be demonstrated using the second
    # and third values (first is NaN)
    sample = msft_cum_ret[1:3]
    ic(sample)

    # upsampling this will have a lot of NaN's
    by_hour = sample.resample("H")
    ic(by_hour)
    ic(by_hour.interpolate())
