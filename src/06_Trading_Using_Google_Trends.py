import pandas as pd
import numpy as np
import yfinance as yf
import mplfinance as mpf
import matplotlib.pyplot as plt
import scipy.stats as stats
import quandl
import os
import io
from dotenv import load_dotenv
from datetime import datetime
from icecream import ic

pd.set_option("display.notebook_repr_html", False)
pd.set_option("display.max_columns", 15)
pd.set_option("display.max_rows", 8)
pd.set_option("precision", 3)


if __name__ == "__main__":
    idx = pd.IndexSlice
    src_data = "data/yf_data.pkl"
    start = datetime(2012, 1, 1)
    end = datetime(2014, 12, 31)

    # this is the same DJIA data from the authors
    paper = pd.read_csv(
        "data/PreisMoatStanley2013.dat", delimiter=" ", parse_dates=[0, 1, 100, 101]
    )
    ic(paper[:5])

    data = pd.DataFrame(
        {
            "GoogleWE": paper["Google End Date"],
            "debt": paper["debt"].astype(np.float64),
            "DJIADate": paper["DJIA Date"],
            "DJIAClose": paper["DJIA Closing Price"].astype(np.float64),
        }
    )
    ic(data[:5])

    ## Gathering our own DJIA data from Quandl
    # load_dotenv(dotenv_path="../.env", verbose=True)
    # quandl.ApiConfig.api_key = os.getenv("Quandl")
    #
    # # get the DJIA from Quandl for 2004-01-01 to 2011-02-28
    # djia = quandl.get("YAHOO/INDEX_DJI", trim_start="2004-01-01", trim_end="2011-03-05")
    # ic(djia)

    djia = pd.read_csv("data/djia.csv", index_col=0)
    ic(djia[:3])

    djia_closes = djia["Close"].reset_index()
    ic(djia_closes[:3])

    data = pd.merge(data, djia_closes, left_on="DJIADate", right_on="Date")
    data.drop(["DJIADate"], inplace=True, axis=1)
    data = data.set_index("Date")
    ic(data[:3])

    # examine authors versus our DJIA data
    data[["DJIAClose", "Close"]].plot(figsize=(12, 8))
    plt.savefig("images/ch06/5104OS_06_02.png", bbox_inches="tight", dpi=300)

    ic((data["DJIAClose"] - data["Close"]).describe())
    ic(data[["DJIAClose", "Close"]].corr())

    ## Google trends data
    with open("data/trends_report_debt.csv") as f:
        data_section = f.read().split("\n\n")[1]
        trends_data = pd.read_csv(
            io.StringIO(data_section),
            header=1,
            index_col="Week",
            converters={"Week": lambda x: pd.to_datetime(x.split(" ")[-1])},
        )
    our_debt_trends = trends_data["2004-01-01":"2011-02-28"].reset_index()
    ic(our_debt_trends[:5])

    final = pd.merge(
        data.reset_index(),
        our_debt_trends,
        left_on="GoogleWE",
        right_on="Week",
        suffixes=["P", "O"],
    )
    final.drop("Week", inplace=True, axis=1)
    final.set_index("Date", inplace=True)
    ic(final[:5])

    combined_trends = final[["GoogleWE", "debtP", "debtO"]].set_index("GoogleWE")
    ic(combined_trends[:5])
    ic(combined_trends.corr())
    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax1.plot(combined_trends.index, combined_trends.debtP, color="b")
    ax2 = ax1.twinx()
    ax2.plot(combined_trends.index, combined_trends.debtO, color="r")
    plt.savefig("images/ch06/5104OS_06_05.png", bbox_inches="tight", dpi=300)
    plt.show()

    # Generate the order signals
    base = final.reset_index().set_index("GoogleWE")
    base.drop(["DJIAClose"], inplace=True, axis=1)
    ic(base[:3])

    # calculate the rolling mean of the previous three weeks for each week
    base["PMA"] = pd.rolling_mean(base.debtP.shift(1), 3)
    base["OMA"] = pd.rolling_mean(base.debtO.shift(1), 3)
    ic(base[:5])

    # calculate the order signals for the papers data
    base["signal0"] = 0  # default to 0
    base.loc[base.debtP > base.PMA, "signal0"] = -1
    base.loc[base.debtP < base.PMA, "signal0"] = 1

    # and for our trend data
    base["signal1"] = 0
    base.loc[base.debtO > base.OMA, "signal1"] = -1
    base.loc[base.debtO < base.OMA, "signal1"] = 1
    ic(base[["debtP", "PMA", "signal0", "debtO", "OMA", "signal1"]])

    # Computing Returns :  add in next week's percentage change to each week of data
    base["PctChg"] = base.Close.pct_change().shift(-1)
    ic(base[["Close", "PctChg", "signal0", "signal1"]][:5])

    # calculate the returns
    base["ret0"] = base.PctChg * base.signal0
    base["ret1"] = base.PctChg * base.signal1
    ic(base[["Close", "PctChg", "signal0", "signal1", "ret0", "ret1"]][:5])

    # Cumulative returns and the result of the strategy
    # calculate and report the cumulative returns
    base["cumret0"] = (1 + base.ret0).cumprod() - 1
    base["cumret1"] = (1 + base.ret1).cumprod() - 1
    ic(base[["cumret0", "cumret1"]])

    # show graph of growth for the papers data
    base["cumret0"].plot(figsize=(12, 4))
    plt.savefig("images/ch06/5104OS_06_06.png", bbox_inches="tight", dpi=300)

    # show graph of growth for the papers data
    base[["cumret0", "cumret1"]].plot(figsize=(12, 4))
    plt.savefig("images/ch06/5104OS_06_07.png", bbox_inches="tight", dpi=300)
