# portfolio_analyze.py
'''
Software used:
http://wiki.quantsoftware.org/index.php?title=QSTK_License

Created on October 16, 2014
Updated on October 29, 2014

@author: James Corsini
@summary: Takes a portfolio value and compares it to benchmarks.  Outputs
comparison stats and plots
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


def analyze(values,benchmark,start=1,end=1,plots=False):
    '''
    @summary analyze: Compares values of portfolio to key benchmarks
    @param values: DataFrame containing the dates and values of given portfolio
    @param benchmark: List of tickers to be used as benchmarks
    @param start: Start date in the form [yyyy,mm,dd]
    @param end: End date in the form [yyyy,mm,dd]
    @param plots: Show and save plots (true/ false)
    @output analyze: No return.  Outputs key metrics and plots of portfolio vs. benchmarks 
    '''
    
    # Intitialize variables
    RiskFreeRate = 0
    

    # Import dates and values from CSV
    if start == 1:
        temp=1
    else:
        na_dates = np.genfromtxt(values, delimiter=",",dtype=int,usecols=(0,1,2))
        
    if end ==1:
        temp=2
    else:
        na_value = np.genfromtxt(values, delimiter=",",usecols=(3))
    
    # Reshape to a proper numpy array (x,1)
    na_value = na_value.reshape(-1,1)

    # Start and End date of the charts
    dt_start = dt.datetime(na_dates[0,0],na_dates[0,1],na_dates[0,2],16,0,0)
    dt_end = dt.datetime(na_dates[-1,0],na_dates[-1,1],na_dates[-1,2],16,0,0)

    # Add error handling for dates outside portfolio (at least start)

    
    # List of symbols
    ls_benchmark = benchmark

    # We need closing prices so the timestamp should be hours=16.
    dt_timeofday = dt.timedelta(hours=16)

    # Get a list of trading days between the start and the end.
    ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt_timeofday)

    # Creating an object of the dataaccess class with Yahoo as the source.
    c_dataobj = da.DataAccess('Yahoo', cachestalltime=0)

    # Keys to be read from the data, it is good to read everything in one go.
    ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']

    # Reading the data, now d_data is a dictionary with the keys above.
    # Timestamps and symbols are the ones that were specified before.
    ldf_data = c_dataobj.get_data(ldt_timestamps, ls_benchmark, ls_keys)
    d_data = dict(zip(ls_keys, ldf_data))

    # Filling the data for NAN
    for s_key in ls_keys:
        d_data[s_key] = d_data[s_key].fillna(method='ffill')
        d_data[s_key] = d_data[s_key].fillna(method='bfill')
        d_data[s_key] = d_data[s_key].fillna(1.0)

    # Getting the numpy ndarray of close prices.
    na_price = d_data['close'].values
   


    # Portfolio calculations
    

    # Calculate daily returns on portfolio
    na_portrtn = na_value.copy()
    tsu.returnize0(na_portrtn);
 
    # Calculate volatility (stdev) of daily returns of portfolio
    fl_portvol = np.std(na_portrtn);

    #Calculate standard deviation of portfolio prices
    fl_portstd = np.std(na_value)
 
    #Calculate average daily returns of portfolio
    fl_portdayrtn = np.mean(na_portrtn);
 
    #Calculate portfolio sharpe ratio (avg portfolio return / portfolio stdev) * sqrt(252)
    fl_portshrp = (fl_portdayrtn / fl_portvol) * np.sqrt(252);

    # Calculate total return of porfolio 
    fl_portrtn_total = 1+((na_value[-1,0]-na_value[0,0])/na_value[0,0])


    
    # Benchmark calculations

    
    #Calculate daily returns on benchmark
    na_benchrtn = na_price.copy()
    tsu.returnize0(na_benchrtn);
 
    #Calculate volatility (stdev) of daily returns of benchmark
    fl_benchvol = np.std(na_benchrtn);

    #Calculate standard deviation of benmark prices
    fl_benchstd = np.std(na_price)
 
    #Calculate average daily returns of benchmark
    fl_benchdayrtn = np.mean(na_benchrtn);
 
    #Calculate benchmark sharpe ratio (avg benchmark return / benchmark stdev) * sqrt(252)
    fl_benchshrp = (fl_benchdayrtn / fl_benchvol) * np.sqrt(252);

    # Calculate total return of benchmark 
    fl_benchrtn_total = 1+((na_price[-1,0]-na_price[0,0])/na_price[0,0])



    print "Start Date: ", dt_start
    print "End Date: ", dt_end

    print "Benchmark: ", ls_benchmark

    print "Sharpe Ratio of Fund: ", fl_portshrp
    print "Sharpe Ratio of Index: ", fl_benchshrp
    
    print "Total Return of Fund: ", fl_portrtn_total
    print "Total Return of Index: ", fl_benchrtn_total

    print "Standard Deviation of Fund: ", fl_portvol
    print "Standard Deviation of Index: ", fl_benchvol

    print "Average Daily Return of Fund: ", fl_portdayrtn #np.sum(valsumrtn)/len(na_price)
    print "Average Daily Return of Index; ", fl_benchdayrtn #np.sum(sum_yrrtn)/len(na_price)
    


    # Plots

    if plots == True:
        


        # Normalizing the prices to start at 1 and see relative returns
        na_normalized_price = na_price / na_price[0, :]
        na_normalized_values = na_value / na_value[0,:]
        

        # Needs to be updated!

        # Plotting the prices with x-axis=timestamps
        plt.clf()
        #print na_normalized_price*na_csvdata1
        plt.plot(ldt_timestamps, na_normalized_price)
                 #na_normalized_price*na_csvdata1)#na_price)
        plt.plot(ldt_timestamps, na_normalized_values)#na_normalized_values
        plt.legend(ls_benchmark)
        plt.ylabel('Normalized Close')
        plt.xlabel('Date')
        plt.legend([ls_benchmark,'Fund Value'])
        plt.savefig('normalized.pdf', format='pdf')
        plt.show()



        # Copy the normalized prices to a new ndarry to find returns.
        na_rets = na_normalized_price.copy()
        na_vals = na_normalized_values.copy()

        # Calculate the daily returns of the prices. (Inplace calculation)
        # returnize0 works on ndarray and not dataframes.
        tsu.returnize0(na_rets)
        tsu.returnize0(na_vals)

        # Plotting the scatter plot of daily returns between XOM VS $SPX
        # Used for finding correlated stocks - needs to be in a line
        # Are there any methods for finding the linear equation and R^2???
        plt.clf()
        plt.scatter(na_rets[:, 0], na_vals[:, 0], c='blue') #(x,y)
        plt.ylabel('Portfolio')
        plt.xlabel('$SPX')
        plt.savefig('scatterSPXvXOM.pdf', format='pdf')
        plt.show()



if __name__ == '__main__':

    time_analyze = time.time()

    values = "./output/values.csv"
    benchmark = ['$SPX']
    start = [2013,10,20]
    end = [2014,10,2]

    analyze(values,benchmark,start,end,False)

    print "Portfolio_Analyze run in: " , (time.time() - time_analyze) , " seconds."; 

    print "Done"

    
