#!/bin/env python
# Created on March 25, 2020
#  by Keith Cherkauer
#
# This script serves as the solution set for assignment-10 on descriptive
# statistics and environmental informatics. See the assignment documention 
# and repository at:
# https://github.com/Environmental-Informatics/assignment-10.git for more
# details about the assignment.
#
"""
Edited by Eric Kong on April 7th, 2020.

Description: Data is imported and stored as a dataframe. Functions are written
            to calculate descriptive statistics and environmental metric of the data. 
            Calculated results are outputted as CSV and TXT files. 

References:
    https://stackoverflow.com/questions/48312655/replace-dataframe-column-negative-values-with-nan-in-method-chain?noredirect=1&lq=1
    https://towardsdatascience.com/using-the-pandas-resample-function-a231144194c4
    https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#resampling
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.apply.html
    https://stackoverflow.com/questions/19914937/applying-function-with-multiple-arguments-to-create-a-new-pandas-column
    https://kite.com/python/answers/how-to-apply-a-function-with-multiple-arguments-to-a-pandas-dataframe-in-python
    https://stackoverflow.com/questions/17071871/how-to-select-rows-from-a-dataframe-based-on-column-values
    https://stackoverflow.com/questions/26105804/extract-month-from-date-in-python/26105888
"""

import pandas as pd
import scipy.stats as stats
import numpy as np 

# column headers for new dataframes
annualcolumnheader = ['Mean Flow','Peak Flow','Median Flow','Coeff Var','Skew','Tqmean','R-B Index','7Q','3xMedian'];
monthlycolumnheader = ['Mean Flow','Coeff Var','Tqmean','R-B Index'];

def ReadData( fileName ):
    """This function takes a filename as input, and returns a dataframe with
    raw data read from that file in a Pandas DataFrame.  The DataFrame index
    should be the year, month and day of the observation.  DataFrame headers
    should be "agency_cd", "site_no", "Date", "Discharge", "Quality". The 
    "Date" column should be used as the DataFrame index. The pandas read_csv
    function will automatically replace missing values with np.NaN, but needs
    help identifying other flags used by the USGS to indicate no data is 
    available.  Function returns the completed DataFrame, and a dictionary 
    designed to contain all missing value counts that is initialized with
    days missing between the first and last date of the file."""
    
    # define column names
    colNames = ['agency_cd', 'site_no', 'Date', 'Discharge', 'Quality']

    # open and read the file
    DataDF = pd.read_csv(fileName, header=1, names=colNames,  
                         delimiter=r"\s+",parse_dates=[2], comment='#',
                         na_values=['Eqp'])
    DataDF = DataDF.set_index('Date')
    
    # check for negative streamflow 
    DataDF.loc[~(DataDF['Discharge'] > 0), 'Discharge'] = np.nan
    
    # quantify the number of missing and negative values
    MissingValues = DataDF["Discharge"].isna().sum()
    
    # remove invalid stream flow data values and remove negative values
    #DataDF = DataDF.dropna(subset=['Discharge'])
    
    return( DataDF, MissingValues )

def ClipData( DataDF, startDate, endDate ):
    """This function clips the given time series dataframe to a given range 
    of dates. Function returns the clipped dataframe and and the number of 
    missing values."""
    
    # isolate the date range we want to work with
    DataDF = DataDF.loc[startDate:endDate] # start and end date defined in line 273
    
    MissingValues = DataDF["Discharge"].isna().sum() # quantify the number of missing values
    
    return( DataDF, MissingValues )

def CalcTqmean(Qvalues):
    """This function computes the Tqmean of a series of data, typically
       a 1 year time series of streamflow, after filtering out NoData
       values. Tqmean is the fraction of time that daily streamflow
       exceeds mean streamflow for each year. Tqmean is based on the
       duration rather than the volume of streamflow. The routine returns
       the Tqmean value for the given data array."""
    
    # Qvalues is the series of streamflow values 
    # Count all the times flow is larger than the mean flow
    # then divide by the # of data points to determine how often this happens
    
    Tqmean = ((Qvalues > Qvalues.mean()).sum() / len(Qvalues))
    
    return ( Tqmean )

def CalcRBindex(Qvalues):
    """This function computes the Richards-Baker Flashiness Index
       (R-B Index) of an array of values, typically a 1 year time
       series of streamflow, after filtering out the NoData values.
       The index is calculated by dividing the sum of the absolute
       values of day-to-day changes in daily discharge volumes
       (pathlength) by total discharge volumes for each year. The
       routine returns the RBindex value for the given data array."""
    
    # Qvalues is the series of streamflow values
    storage = int(0) # initalized as zero, this variable stores the value from  
                # the previous day-to-day change calulation so it can be added during the summation
    
    remove_nan = len(Qvalues.dropna())
    if remove_nan > 0:   # run if the length of the flow values, after nan's are removed is greater than 0
        for position in range(1, remove_nan): # loop for all values
            storage = storage + abs(Qvalues.iloc[position-1] - Qvalues.iloc[position]) # abs value of day-to-day change  
        RBindex = storage / sum(Qvalues.dropna()) # summed day-to-day changes divided by sum of flow         
    else: 
            RBindex = np.nan    # combatting the divide by zero error that occurs because of the nan values    
    return ( RBindex )

def Calc7Q(Qvalues):
    """This function computes the seven day low flow of an array of 
       values, typically a 1 year time series of streamflow, after 
       filtering out the NoData values. The index is calculated by 
       computing a 7-day moving average for the annual dataset, and 
       picking the lowest average flow in any 7-day period during
       that year. The routine returns the 7Q (7-day low flow) value
       for the given data array."""
       
    # Qvalues is the series of streamflow values
    # calculate average of the weekly flow data for a given period
    # set val7Q to the lowest value 
    val7Q =  Qvalues.rolling(window=7).mean().min()
    
    return ( val7Q )

def CalcExceed3TimesMedian(Qvalues):
    """This function computes the number of days with flows greater 
       than 3 times the annual median flow. The index is calculated by 
       computing the median flow from the given dataset (or using the value
       provided) and then counting the number of days with flow greater than 
       3 times that value. The routine returns the count of events greater 
       than 3 times the median annual flow value for the given data array."""
    
    # Qvalues is the series of streamflow values
    # Count all the times flow is larger than three times the median flow
    # similar layout to Tqmean code 
    median3x = (Qvalues > (Qvalues.median()*3)).sum()
    
    return ( median3x )

def GetAnnualStatistics(DataDF):
    """This function calculates annual descriptive statistcs and metrics for 
    the given streamflow time series. Values are retuned as a dataframe of
    annual values for each water year. Water year, as defined by the USGS,
    starts on October 1."""
    
    annual_index = DataDF.resample('AS-OCT').mean() # resample index - yearly based on start of water year, October
    WYDataDF = pd.DataFrame(index = annual_index.index, columns = annualcolumnheader) # setting new DF with proper index and headers
    ADF = DataDF.resample('AS-OCT') # resampled DF stored as a simple variable name 
    
    # statistics 
    WYDataDF['Mean Flow'] = ADF['Discharge'].mean()
    WYDataDF['Peak Flow'] = ADF['Discharge'].max()
    WYDataDF['Median Flow'] = ADF['Discharge'].median()
    WYDataDF['Coeff Var'] = (ADF['Discharge'].std() / ADF['Discharge'].mean()) * 100
    WYDataDF['Skew'] = ADF.apply({'Discharge':lambda x: stats.skew(x,bias=False)})
    
    # metrics
    # applying the custom built functions
    WYDataDF['Tqmean'] = ADF.apply({'Discharge':lambda x: CalcTqmean(x)})
    WYDataDF['R-B Index'] = ADF.apply({'Discharge':lambda x: CalcRBindex(x)})
    WYDataDF['7Q'] = ADF.apply({'Discharge':lambda x: Calc7Q(x)})
    WYDataDF['3xMedian'] = ADF.apply({'Discharge':lambda x: CalcExceed3TimesMedian(x)})
    
    return ( WYDataDF )

def GetMonthlyStatistics(DataDF):
    """This function calculates monthly descriptive statistics and metrics 
    for the given streamflow time series.  Values are returned as a dataframe
    of monthly values for each year."""

    monthly_index = DataDF.resample('MS').mean() # resample index - monthly
    MoDataDF = pd.DataFrame(index = monthly_index.index, columns = monthlycolumnheader) # setting new DF with proper index and headers
    MDF = DataDF.resample('MS') # resampled DF stored as a simple variable name 
    
    #metrics and statistics 
    MoDataDF['Mean Flow'] = MDF['Discharge'].mean()
    MoDataDF['Coeff Var'] = (MDF['Discharge'].std() / MDF['Discharge'].mean()) * 100
    MoDataDF['Tqmean'] = MDF.apply({'Discharge':lambda x: CalcTqmean(x)})
    MoDataDF['R-B Index'] = MDF.apply({'Discharge':lambda x: CalcRBindex(x)})

    return ( MoDataDF )

def GetAnnualAverages(WYDataDF):
    """This function calculates annual average values for all statistics and
    metrics. The routine returns an array of mean values for each metric
    in the original dataframe."""
    
    # Annual average function should return a Series with a single average value for each statistic of metric
    AnnualAverages = WYDataDF.mean(axis=0)
    
    return( AnnualAverages )

def GetMonthlyAverages(MoDataDF):
    """This function calculates annual average monthly values for all 
    statistics and metrics. The routine returns an array of mean values 
    for each metric in the original dataframe."""
    
    # Monthly average function should return a DataFrame with 12 monthly values for each metric
    isolated_months = MoDataDF.index.month # tip for extracting month from Cherkauer
    MonthlyAverages = MoDataDF.groupby(isolated_months).mean() # group the new DF using the months and find the mean of that data
    
    return( MonthlyAverages )

# the following condition checks whether we are running as a script, in which 
# case run the test code, otherwise functions are being imported so do not.
# put the main routines from your code after this conditional check.

if __name__ == '__main__':

    # define filenames as a dictionary
    # NOTE - you could include more than just the filename in a dictionary, 
    #  such as full name of the river or gaging site, units, etc. that would
    #  be used later in the program, like when plotting the data.
    fileName = { "Wildcat": "WildcatCreek_Discharge_03335000_19540601-20200315.txt",
                 "Tippe": "TippecanoeRiver_Discharge_03331500_19431001-20200315.txt" }
    
    # define blank dictionaries (these will use the same keys as fileName)
    DataDF = {}
    MissingValues = {}
    WYDataDF = {}
    MoDataDF = {}
    AnnualAverages = {}
    MonthlyAverages = {}
    
    # process input datasets
    for file in fileName.keys():
        
        print( "\n", "="*50, "\n  Working on {} \n".format(file), "="*50, "\n" )
        
        DataDF[file], MissingValues[file] = ReadData(fileName[file])
        print( "-"*50, "\n\nRaw data for {}...\n\n".format(file), DataDF[file].describe(), "\n\nMissing values: {}\n\n".format(MissingValues[file]))
        
        # clip to consistent period
        DataDF[file], MissingValues[file] = ClipData( DataDF[file], '1969-10-01', '2019-09-30' )
        print( "-"*50, "\n\nSelected period data for {}...\n\n".format(file), DataDF[file].describe(), "\n\nMissing values: {}\n\n".format(MissingValues[file]))
        
        # calculate descriptive statistics for each water year
        WYDataDF[file] = GetAnnualStatistics(DataDF[file])
        
        # calculate the annual average for each stistic or metric
        AnnualAverages[file] = GetAnnualAverages(WYDataDF[file])
        
        print("-"*50, "\n\nSummary of water year metrics...\n\n", WYDataDF[file].describe(), "\n\nAnnual water year averages...\n\n", AnnualAverages[file])

        # calculate descriptive statistics for each month
        MoDataDF[file] = GetMonthlyStatistics(DataDF[file])

        # calculate the annual averages for each statistics on a monthly basis
        MonthlyAverages[file] = GetMonthlyAverages(MoDataDF[file])
        
        print("-"*50, "\n\nSummary of monthly metrics...\n\n", MoDataDF[file].describe(), "\n\nAnnual Monthly Averages...\n\n", MonthlyAverages[file])
        
        
################################ Outputting Results to files, indented so it rules for both files########################################################
# script will keep adding data to the same text file if run multiple times 

# annual metrics CSV - WYDataDF
        WYDataDF[file].insert(0, 'Station', file)
        WYDataDF[file].to_csv('Annual_Metrics.csv', mode='a', sep=",") # append
     
# monthly metric CSV - MoDataDF
        MoDataDF[file].insert(0, 'Station', file)
        MoDataDF[file].to_csv('Monthly_Metrics.csv', mode='a', sep=",")         
        
        
# avg monthly metric TAB - MonthlyAverages
        MonthlyAverages[file].insert(0, 'Station', file)
        MonthlyAverages[file].to_csv('Average_Monthly_Metrics.txt', mode='a', sep="\t") 
        
# avg annual metrics TAB - AnnualAverages
# AnnualAverages is a dictionary of series 
    strip_wildcat = AnnualAverages['Wildcat'].to_frame() # convert the stripped dictionary entry to DF
    strip_tippe = AnnualAverages['Tippe'].to_frame()

    repeat_tippe = ['Tippe']*9 # list of station name
    repeat_wildcat = ['Wildcat']*9

    strip_tippe['Station'] = repeat_tippe # adding station name to stripped DF
    strip_wildcat['Station'] = repeat_wildcat

    combined = [strip_wildcat, strip_tippe] # combine DF's
    endgame = pd.concat(combined) 

    endgame.to_csv('Average_Annual_Metrics.txt', mode='a', sep="\t") 
        