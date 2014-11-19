# portfolio_value.py
'''
Software used:
http://wiki.quantsoftware.org/index.php?title=QSTK_License

Created on October 16, 2014
Updated on October 29, 2014

@author: James Corsini
@summary: Takes buy and sell data in csv format and outputs the portfolio's
value. Should be used in conjunction with portfolio_analyze.py to get
stats and plots.
'''

# QSTK Imports
import QSTK.qstkutil.qsdateutil as du
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkutil.DataAccess as da

# Third Party Imports
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
import math
from numpy import genfromtxt
import time
import os
import shutil
from portfolio_functions import *


def readcsvtona(csvfile):
    '''
    @summary: Reads in data in from csv (hist_orders format)
    @param csvfile: CSV file in the format
    year, month, day, stock, order (buy/sell), shares
    @return: Numpy arrays of the dates, stocks, orders, and shares
    '''
    
    # Reading orders from the csv file.
    na_data = np.loadtxt(csvfile, delimiter=",",usecols=(0,1,2,5))
    na_dates = np.int_(na_data[:,0:3])
    na_share = na_data[:,3]
    na_share = na_share.reshape(-1,1)
    na_stock = np.genfromtxt(csvfile, delimiter=",",usecols=(3),dtype=str)
    na_ordpos = np.genfromtxt(csvfile, delimiter=",",usecols=(4),dtype=str)

    return [na_dates, na_stock, na_ordpos, na_share]



def natodict(na_dates,na_stock,na_ordpos,na_share):
    '''
    @summary: Converts numpy arrays into dict while obtaining Yahoo data
    @param na_dates: Array of dates when transaction occurred
    @param na_stock: Array of stocks bought/ sold and cash deposited/ withdrawn
    @param na_ordpos: Array of transaction types (Buy or Sell)
    @param na_share: Array of shares or cash transacted
    @return: Dict of symbols (d_data) with index called ldt_timestamps
    list of symbols (not repeted) and ldt_trans which is an index for the
    transaction dates
    '''

    # Creating the timestamps from dates read
    ldt_trans = []

    #print na_stock
    for i in range(0, len(na_stock)):
        ldt_trans.append(dt.datetime(na_dates[i, 0],
                        na_dates[i, 1], na_dates[i, 2],16,0,0))
    
    # get stock data
    ldt_stocks = list(set(na_stock))

    #print ldt_stocks
    ls_symbols = ldt_stocks

    # Remove _CASH from this array
    if '_CASH' in ls_symbols:
        ls_symbols.remove('_CASH')

    # Creating an object of the dataaccess class with Yahoo as the source.
    c_dataobj = da.DataAccess('Yahoo')
    
    # Creating the timestamps from dates read
    ldt_timestamps = []
    for i in range(0, na_dates.shape[0]):
        ldt_timestamps.append(dt.date(na_dates[i, 0],
                        na_dates[i, 1], na_dates[i, 2]))

    # Start and End date of the charts
    dt_end = ldt_timestamps[-1]
    dt_start = ldt_timestamps[0]

    # We need closing prices so the timestamp should be hours=16.
    dt_timeofday = dt.timedelta(hours=16)

    # Get a list of trading days between the start and the end.
    ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt_timeofday)

    # Keys to be read from the data, it is good to read everything in one go.
    ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']

    # Reading the data, now d_data is a dictionary with the keys above.
    # Timestamps and symbols are the ones that were specified before.
    ldf_data = c_dataobj.get_data(ldt_timestamps, ls_symbols, ls_keys)
    d_data = dict(zip(ls_keys, ldf_data))

    # Filling the data for NAN
    for s_key in ls_keys:
        d_data[s_key] = d_data[s_key].fillna(method='ffill')
        d_data[s_key] = d_data[s_key].fillna(method='bfill')
        d_data[s_key] = d_data[s_key].fillna(1.0)

    return [d_data,ldt_timestamps,ls_symbols,ldt_trans]


def closeprices(d_data,ldt_timestamps,ls_symbols):
    '''
    @summary: Get only the close prices from the dict for all symbols
    @param d_data: Dict containing all of the Yahoo data
    @param ldt_timestamps: List of timestamps from the begining of the portfolio
    period until the end
    @param ls_symbols: List of symbols in the dict
    @return: The close prices in numpy array and dataframe
    '''

    # Getting the numpy ndarray of close prices.
    na_price = d_data['close'].values
    # Converting na_price to dataframe
    df_price = pd.DataFrame(na_price, index=ldt_timestamps,columns = ls_symbols)

    return [na_price, df_price]
    


def cashval(df_price, ls_symbols, ldt_timestamps, na_stock, na_share, na_ordpos, ldt_trans):
    '''
    @summary: Calculates the cash and value of the portfolio for each day
    @param df_price: Yahoo prices of all the sympbols
    @param ls_symbols: List of symbols in the portfolio
    @param ldt_timestamps: List of dates and times from the start of the portfolio to the end
    @param na_stock: Array of stocks in order of transaction
    @param na_share: Array of shares/ cash in order of transaction
    @param na_ordpos: Array of orders (Buy/Sell/Deposit/Withdraw) in order of transaction
    @param ldt_trans: List of the dates and times of the transactions
    @return:  The daily value of the portfolio
    '''

    # Initialize variables
    # @future: Can make these inputs to the function

    # Value for commission (if wanted)
    fl_commission = 7.95

    # Value for starting cash
    startcash = 0 #1000000


    # Making the cash and value array
    # this will have rows equal dates and the columns are ls_valsym
    ls_valsym = ["PortCash","Value"]

    # Create dataframe for value array with 0s
    df_val = pd.DataFrame(index=ldt_timestamps, columns=ls_valsym)
    df_val = df_val.fillna(0.0)

    # Setting first value to be starting cash ammount 
    df_val['PortCash'].ix[ldt_timestamps[0]] = startcash

    # Create dataframe for shares filled with 0s
    # indexed by the timestamps and columns are the symbols in the portfolio
    df_shares = pd.DataFrame(index=ldt_timestamps, columns=ls_symbols)
    df_shares = df_shares.fillna(0)


    # Main loop to calculate the shares map (assigns appropriate number of shares
    # to each stock for each day.  Also, calulates the cash for each day that the
    # portfolio is active
    for k in range(0,len(na_ordpos)):
    
        rownum = df_shares.index.get_loc(ldt_trans[k])

        if na_ordpos[k] == 'Buy':
            if na_stock[k]=='_CASH':
                #print 'add cash'
                df_val['PortCash'].ix[ldt_timestamps[rownum:len(ldt_timestamps)]] = df_val['PortCash'].ix[ldt_timestamps[rownum]]+na_share[k,0]
            else:
                df_shares[na_stock[k]].ix[ldt_timestamps[rownum:len(ldt_timestamps)]] = df_shares[na_stock[k]].ix[ldt_trans[k]]+na_share[k,0]
                df_val['PortCash'].ix[ldt_timestamps[rownum:len(ldt_timestamps)]] = df_val['PortCash'].ix[ldt_timestamps[rownum]]-(na_share[k,0]*df_price[na_stock[k]].ix[ldt_timestamps[rownum]])-fl_commission
                #print "BUY; ", na_share[k,0], " ", na_stock[k]], " at ", df_price[na_stock[k]].ix[ldt_timestamps[rownum]]
       
        if na_ordpos[k] == 'Sell':
            if na_stock[k]=='_CASH':
                #print 'withdraw cash'
                df_val['PortCash'].ix[ldt_timestamps[rownum:len(ldt_timestamps)]] = df_val['PortCash'].ix[ldt_timestamps[rownum]]-na_share[k,0]
            else:
                df_shares[na_stock[k]].ix[ldt_timestamps[rownum:len(ldt_timestamps)]] = df_shares[na_stock[k]].ix[ldt_trans[k]]-na_share[k,0]
                df_val['PortCash'].ix[ldt_timestamps[rownum:len(ldt_timestamps)]] = df_val['PortCash'].ix[ldt_timestamps[rownum]]+(na_share[k,0]*df_price[na_stock[k]].ix[ldt_timestamps[rownum]])
                #print "SELL; ", na_share[k,0], " ", na_stock[k]], " at ", df_price[na_stock[k]].ix[ldt_timestamps[rownum]]

            
    # Finding the value of the portfolio for each day
    for j in range(0,len(ldt_timestamps)):
        for sym in ls_symbols:
            df_val['Value'].ix[ldt_timestamps[j]] = df_val['Value'].ix[ldt_timestamps[j]]+(df_shares[sym].ix[ldt_timestamps[j]]*df_price[sym].ix[ldt_timestamps[j]])


    # To get value of portfolio for specific day
    #valdate = [yyyy,mm,dd,16,0,0]
    #print "Value for portfolio at ", valdate, ": ", df_val.index.get_loc(valdate)


    # Uses numpy to sum the cash and value columns to obtain portfolio value
    na_portval = np.sum(df_val, axis=1)

    # Converts portfolio value (value + cash) back to dataframe
    df_portval = pd.DataFrame(na_portval,index=ldt_timestamps, columns=['PortValue'])

    return df_portval


def printvalcsv(df_portval, ldt_timestamps):
    '''
    @summary: Creates CSV file and installs it in the proper directory.
    @param df_portval: Dataframe of the portfolio values
    @param ldt_timestamps: List of date times in timestamps
    @return:  Nothing
    '''
    
    # Creates directory if it doesn't exist
    # Warning if backup folder does not exist but output folder does
    # then on first run a backup will not take place.
    # In this case, run it twice to save it
    if not os.path.exists('./output/backup'):
        os.makedirs('./output/backup')
    else:
        shutil.copy2('./output/values.csv', './output/backup/values' + timestamp() + '.csv')
    
    # Writes CSV file
    csv_value = csv.writer(open("./output/values.csv","wb"), delimiter=',',quoting=csv.QUOTE_NONE)
    for j in range(0,len(ldt_timestamps)):
        csv_value.writerow([ldt_timestamps[j].year,ldt_timestamps[j].month,ldt_timestamps[j].day,df_portval['PortValue'].ix[ldt_timestamps[j]]])

    return



if __name__ == '__main__':

    
    start_value = time.time()

    [na_dates,na_stock,na_ordpos,na_share] = readcsvtona('./hist_orders/hist_orders.csv')
    [d_data,ldt_timestamps,ls_symbols,ldt_trans] = natodict(na_dates,na_stock,na_ordpos,na_share)
    [na_price, df_price] = closeprices(d_data,ldt_timestamps,ls_symbols)
    df_portval = cashval(df_price, ls_symbols, ldt_timestamps, na_stock, na_share, na_ordpos, ldt_trans)
    printvalcsv(df_portval, ldt_timestamps)

    print "Portfolio Value: ", df_portval['PortValue'][-1]

    print "Portfolio_value run in: " , (time.time() - start_value) , " seconds."; 

    print "Done"

    
