
'''
Software used:
http://wiki.quantsoftware.org/index.php?title=QSTK_License

Created on October 16, 2014

@author: James Corsini
@summary: Takes buy and sell data in csv format and outputs the portfolio's value.
Adding functoinality to analyze the portfolio and compare with Index.
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



def timestamp():
    '''
    @summary: Create string for timestamp.
    @return:  Timestamp string
    '''

    # Create timestamp string
    tod = dt.datetime.today()

    year = tod.year
    month = tod.month
    day = tod.day
    hour = tod.hour
    minute = tod.minute
    second = tod.second

    stamp = str(year) + str(month) + str(day) + str(hour) + str(minute) + str(second)

    return stamp 


def copyFile(src, dest):
    '''
    @summary: Copies a file from source to destination.
    @param src: Source directory
    @param dest: Destination directory
    @return:  Nothing
    '''
        
    try:
        shutil.copy(src, dest)
    # eg. src and dest are the same file
    except shutil.Error as e:
        print('Error: %s' % e)
    # eg. source or destination doesn't exist
    except IOError as e:
        print('Error: %s' % e.strerror)




def bollinger(array,ldt_timestamps,period):

    rollingmean = pd.rolling_mean(array,period,min_periods=period)
    rollingstd = pd.rolling_std(array,period,min_periods=period)

    bollinger = (array-rollingmean)/rollingstd
    k=1
    if k==1:
        # Plotting the prices with x-axis=timestamps
        plt.clf()
        '''
        plt.plot(ldt_timestamps, array)
        plt.plot(ldt_timestamps, rollingmean)
        plt.plot(ldt_timestamps, rollingmean+rollingstd)
        plt.plot(ldt_timestamps, rollingmean-rollingstd)
        plt.plot(ldt_timestamps, bollinger)
        plt.legend(ls_symbols)
        plt.ylabel('Normalized Close')
        plt.xlabel('Date')
        plt.legend([symbols,'Fund Value'])
        plt.savefig('normalized.pdf', format='pdf') 
        plt.show()'''

    #print bollinger['AAPL'].ix[dt.datetime(2010,6,14,16,0,0)]
    #print bollinger['MSFT'].ix[dt.datetime(2010,5,12,16,0,0)]

    return [bollinger, rollingmean, rollingstd]

