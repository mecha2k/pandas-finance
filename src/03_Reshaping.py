import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from icecream import ic

# Set some Pandas options
pd.set_option("display.notebook_repr_html", False)
pd.set_option("display.max_columns", 15)
pd.set_option("display.max_rows", 8)
pd.set_option("precision", 3)

if __name__ == "__main__":
    idx = pd.IndexSlice
    start = datetime.datetime(2012, 1, 1)
    end = datetime.datetime(2012, 12, 30)

    # read the Microsoft and Apple data from file
    msft = pd.read_csv("data/msft.csv", index_col=0, parse_dates=True)
    aapl = pd.read_csv("data/aapl.csv", index_col=0, parse_dates=True)

    ic(msft[:3])
    # get MSFT adjusted close data for Jan and Feb 2012
    msftA01 = msft.loc["2012-01", "Adj Close"]
    msftA02 = msft.loc["2012-02", "Adj Close"]
    ic(msftA01[:3])

    # combine the first three rows of each of msftA01 and msftA02
    ic(pd.concat([msftA01.head(3), msftA02.head(3)]))

    # Extract only the Jan 2012 AAPL values.
    aaplA01 = aapl.loc["2012-01", "Adj Close"]
    # now concat the AAPL and MSFT Jan 2012 data
    # there will be duplicate index labels
    withDups = pd.concat([msftA01[:3], aaplA01[:3]])
    ic(withDups)

    # show the two records for data of 2012-01-03
    ic(withDups.loc["2012-01-03"])

    # demonstrate concat with a specification of the
    # stock tickets being part of the index
    # this help disambiguate the duplicate dates using
    # a hierarchal index
    closes = pd.concat([msftA01[:3], aaplA01[:3]], keys=["MSFT", "AAPL"])
    ic(closes)
    ic(closes.index)

    # extract just MSFT values using .ix
    ic(closes["MSFT"][:2])
    ic(closes.loc["MSFT"][:2])

    # demonstrate concatenation using two DataFrame's
    # that each have two columns.  pandas will align the
    # data in columns by the column names (labels)
    msftAV = msft[["Adj Close", "Volume"]]
    aaplAV = aapl[["Adj Close", "Volume"]]
    ic(pd.concat([msftAV, aaplAV]))

    # demonstrate concatenation with DataFrame objects
    # that do not have the same set of columns
    # this demonstrates pandas filling in NaN values
    aaplA = aapl[["Adj Close"]]
    ic(pd.concat([msftAV, aaplA]))

    # perform an inner join on the DataFrame's
    # since aaplA does not have a Volume column, pandas
    # will not include that column in the result
    ic(pd.concat([msftAV, aaplA], join="inner"))

    # concat along the rows, causing duplicate columns to be created in the result
    msftA = msft[["Adj Close"]]
    closes = pd.concat([msftA, aaplA], axis=1)
    ic(closes[:3])

    # concat along rows using two DataFrame objects with
    # different number of rows. This demonstrates how
    # NaN values will be filled in those rows for AAPL
    # which only hase three rows as compared to 5 for MSFT
    ic(pd.concat([msftAV[:5], aaplAV[:3]], axis=1, keys=["MSFT", "AAPL"]))

    # inner join can also be used along this axis
    # this will not include rows with index labels that do
    # not exist in both DataFrame objects
    ic(pd.concat([msftA[:5], aaplA[:3]], axis=1, join="inner", keys=["MSFT", "AAPL"]))

    # ignore indexes and just concatenate the data and
    # have the result have a default integer index
    ic(pd.concat([msftA[:3], aaplA[:3]], ignore_index=True))

    ## Merging DataFrame objects
    # we will merge these two DataFrame objects,
    # so lets peek at the data to remind ourselves
    # of what they contain
    msftAR = msftA.reset_index()
    msftVR = msft[["Volume"]].reset_index()
    ic(msftAR[:3])

    # merge the two.  pandas finds the columns in common,
    # in this case Date, and merges on that column and adds
    # a column for all the other columns in both DataFrame's
    msftCVR = pd.merge(msftAR, msftVR)
    ic(msftCVR[:5])

    # we will demonstrate join semantics using this DataFrame
    msftAR0_5 = msftAR[0:5]
    msftVR2_4 = msftVR[2:4]

    # merge semantics using default inner join
    ic(pd.merge(msftAR0_5, msftVR2_4))

    # same joing but using
    ic(pd.merge(msftAR0_5, msftVR2_4, how="outer"))

    ## Pivoting
    ic(msft)
    # need to insert Symbol column before combining
    ic(msft.insert(0, "Symbol", "MSFT"))
    ic(msft)
    ic(aapl.insert(0, "Symbol", "AAPL"))

    # concatenate the MSFT and AAPL data
    # index will consist of the Date column, which we will sort
    combined = pd.concat([msft, aapl]).sort_index()
    ic(combined)

    # this pushes the index into a column and resets to a
    # default integer index
    s4p = combined.reset_index()
    ic(s4p[:5])

    # pivot Date into the Index, make the columns match the
    # unique values in the Symbol column, and the values
    # will be the AdjClose values
    closes = s4p.pivot(index="Date", columns="Symbol", values="Adj Close")
    ic(closes[:3])

    ## Stacking and Unstacking
    # stack the first level of columns into the index
    # essentially, moves AAPL and MSFT into the index
    # leaving a single colum which is the AdjClose values
    stackedCloses = closes.stack()
    ic(stackedCloses)
    ic(stackedCloses.index)

    # using .ix we can retrieve close values by
    # specifying both the date and ticker
    ic(stackedCloses["2012-01-03"]["AAPL"])

    # lookup on just the date, which will give us two values
    # one each for AAPL and MSFT.
    ic(stackedCloses["2012-01-03"])

    # this looks up all values for the MSFT symbol
    ic(stackedCloses.loc[:, "MSFT"])

    # pivots the last level of the index back into a column
    unstackedCloses = stackedCloses.unstack()
    ic(unstackedCloses[:3])

    ## Melting
    # melt making id_vars of Date and Symbol, making the
    # column names the variable and the for each the value
    melted = pd.melt(s4p, id_vars=["Date", "Symbol"])
    ic(melted[:5])

    # extract the values for the data for MSFT on 2012-01-03
    ic(melted[(melted.Date == "2012-01-03") & (melted.Symbol == "MSFT")])

    # Grouping and aggregation
    ## Splitting
    # construct a DataFrame to demonstrate splitting
    # extract from combined the Symbol and AdjClose, and reset the index
    s4g = combined[["Symbol", "Adj Close"]].reset_index()
    # now, add two columns, year and month, using the year and month
    # portions of the data as integers
    s4g.insert(1, "Year", pd.DatetimeIndex(s4g["Date"]).year)
    s4g.insert(2, "Month", pd.DatetimeIndex(s4g["Date"]).month)
    ic(s4g[:5])

    # group by the Symbol column
    ic(s4g.groupby("Symbol"))

    # group again, but save the result this time
    grouped = s4g.groupby("Symbol")
    # the groupby object has a property groups, which shows how
    # all rows will in mapped into the groups.
    # the type of this object is a python dict
    ic(type(grouped.groups))

    # show the mappings of rows to groups
    ic(grouped.groups)

    # these report the number of groups that resulted from
    # the grouping
    ic(len(grouped), grouped.ngroups)

    # this function will print the contents of a group
    def print_groups(groupobject):
        for name, group in groupobject:
            print(name)
            print(group.head())

    # examine our resulting groups
    print_groups(grouped)
    # .size will tell us the count of items in each group
    ic(grouped.size())

    # a specific group can be retrieved using .get_group()
    # which returns a DataFrame representing the specified group
    ic(grouped.get_group("MSFT"))

    # group by three different fields and print the result
    mcg = s4g.groupby(["Symbol", "Year", "Month"])
    print_groups(mcg)

    # set the index of the data to be the following three fields
    # we are creating a multiindex
    mi = s4g.set_index(["Symbol", "Year", "Month"])
    ic(mi)

    # now we can group based upon values in the actual index
    # the following groups by level 0 of the index (Month)
    mig_l1 = mi.groupby(level=0)
    print_groups(mig_l1)

    # group by three levels in the index using their names
    mig_l12 = mi.groupby(level=["Symbol", "Year", "Month"])
    print_groups(mig_l12)

    # Aggregation
    # this will apply the mean function to each group
    ic(mig_l12.agg(np.mean))

    # example of groupby that also ignores the index
    # resulting in a default integer index
    # this also has the mean function applied
    ic(s4g.groupby(["Symbol", "Year", "Month"], as_index=False).agg(np.mean)[:5])

    # apply multiple functions to each group in one call
    ic(mig_l12.agg([np.mean, np.std]))
