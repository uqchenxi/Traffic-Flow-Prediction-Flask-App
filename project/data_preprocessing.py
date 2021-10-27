import pandas as pd
import numpy as np
import re
import holidays
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt

NUMLIST = [str(i) for i in range(10)]
PATTERN = re.compile(r'\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}')

"""
Construct a function to extract a stop_id from alighting stop in the traffic dataset

Parameters:
    stop_string: The alighting stop string to be extracted
Return:
    s_num: the stop_id number in stop string
"""
def getNum(stop_string):
    status = False
    s_num = ''
    # extract the stop id e.g. ABC [1234]
    for i, num in enumerate(stop_string):
        if num == '[':
            status = True

        if status == True:
            if num in NUMLIST:
                s_num += num

        if num == ']':
            status = False
    return s_num

"""
Construct a function to create split the traffic data by the time interval

Parameters:
    start_date: The start point for date slicing
    end_date: The end point for date slicing
    time_interval: The time interval to be sliced 
Return:
    time_slices: a time index to group the traffic data into time interval
"""
def date_slice(start_date, end_date, time_interval):
    time_slices = pd.Series()
    start_date = pd.to_datetime(start_date, format='%Y/%m/%d %H:%M') # transfer the start date string to the date
    end_date = pd.to_datetime(end_date, format='%Y/%m/%d %H:%M') # transfer the end date string to the date
    time_slices = time_slices.append(pd.Series(start_date), ignore_index=True)

    while end_date > start_date: # generate the time slice index
        interval = timedelta(minutes=time_interval)
        start_date = start_date + interval
        time_slices = time_slices.append(pd.Series(start_date), ignore_index=True)
    return time_slices

"""
Construct a function to create future prediction of the date slicing 

Parameters:
    start_date: The start date time string for the future prediction
    time_interval: The start date time string for the end prediction
    step: The steps of the passenger flow to be predicted
Return:
    dates: a time index to group the traffic data into 15 minutes time interval
"""
def date_slice_extent(start_date, time_interval, step):
    dates = pd.bdate_range(start_date, periods=time_interval, freq=str(step) + 'min')
    dates = pd.Series(dates)
    dates = dates.dt.strftime('%Y-%m-%d %H:%M').values.tolist()
    return dates

"""
Construct a function to to check whether the date is in AU public holidays
If is in the public holiday, the value will 1 else will be 0 using one-hot encoding

Parameters:
    dataframe: The dataframe to be checked
Return:
    dataframe: The processed dataframe with is_holiday column
"""
def is_holiday(dataframe):
    au_holidays = []
    for date in holidays.AU(years=2019).items():
        au_holidays.append(str(date[0]))
    dataframe['Is_holiday'] = [1 if str(val).split()[0]
                                    in au_holidays else 0 for val in dataframe['Date']]
    return dataframe

"""
Construct a function to save the database as csv file

Parameters:
    dataset: The dataset to be saved
    filename: The filename of dataset
"""
def save_csv(dataset, filename):
    dataset.to_csv(filename)

"""
Construct a function to read the destination file

Parameters:
    filename: The filename of text
Return:
    destination_set: the destination set to be delivered
"""
def read_text_file(filename):
    # extract the stop information from stop destination text
    destination = pd.read_csv(filename, sep=',', engine='python', header=None,
                              skiprows=1, names=['stop_id', 'stop_code', 'stop_name', 'stop_desc',
                                                 'stop_lat', 'stop_lon', 'zone_id', 'stop_url',
                                                 'location_type', 'parent_station', 'platform_code'])
    destination = destination.drop(destination[destination['stop_code'].isnull()].index)
    destination_set = destination[['stop_id', 'stop_code', 'stop_name', 'stop_lat', 'stop_lon']]
    return destination_set

"""
The function to cast the attributes as float and string

Parameters:
    aggregate_result: The result to be cast
Return:
    aggreagate_result: The result has been cast
"""
def cast_aggregate_result(aggregate_result):
    aggregate_result = aggregate_result.dropna()
    aggregate_result[['Passengers']] = aggregate_result[['Passengers']].astype('float32')
    return aggregate_result

"""
The function to read csv file and do data cleaning for the dataset

Parameters:
    filename: The filename of traffic data
Return:
    dataframe: The dataframe of traffic dataset to be analyzed
"""
def read_traffic_csv(filename):
    dataframe = pd.read_csv(filename, low_memory=False)
    # fill missing values in Passengers with its median value
    dataframe['Passengers'] = dataframe['Passengers'].fillna(dataframe['Passengers'].median())
    dataframe.loc[dataframe['Passengers'] == 0, 'Passengers'] = dataframe['Passengers'].median()
    dataframe = dataframe.drop(dataframe[dataframe['Passengers'] < 0].index)
    dataframe = dataframe.drop(dataframe[dataframe['Alighting Stop'].isnull()].index)
    dataframe = dataframe.drop(dataframe[dataframe['Boarding Stop'].isnull()].index)
    # transfer the data type of alighting time into date type (datetime)
    dataframe['Alighting Time'] = pd.to_datetime(dataframe['Alighting Time'], infer_datetime_format=True)
    dataframe['Boarding Time'] = pd.to_datetime(dataframe['Boarding Time'], infer_datetime_format=True)
    dataframe['Alighting stop_id'] = dataframe['Alighting Stop'].apply(lambda x: getNum(x))
    dataframe['Boarding stop_id'] = dataframe['Boarding Stop'].apply(lambda x: getNum(x))
    return dataframe

"""
The function to aggregate the whole traffic dataset without attribute constraints

Parameters:
    dataframe: The dataframe to be aggragated
    start_date: The start date of the dataframe to be preprocessed
    end_date: The start date of the dataframe to be preprocessed
    time_interval: the time period for each record
    timing: The timing choices of alighting time and boarding time
Return:
    aggregate_result: The aggregated result to be processed in deep learning model
"""
def aggregate_dataframe(dataframe, start_date, end_date, time_interval, timing):
    pd.set_option('mode.chained_assignment', None)
    dataframe = dataframe[['Operations Date', timing
                           + ' Time', 'Passengers', timing + ' Stop', timing + ' stop_id', 'Route']]
    # get date index for aggregation function
    date_index = date_slice(start_date=start_date, end_date=end_date, time_interval=time_interval)
    dataframe['Time Interval'] = pd.cut(dataframe[timing + ' Time'], bins=date_index)
    # perform the aggregate operation
    aggregate_result = pd.DataFrame(dataframe.groupby(['Time Interval'], as_index=False)['Passengers'].sum())
    aggregate_result = aggregate_result.dropna()
    aggregate_result['Date'] =  aggregate_result['Time Interval'].astype('str').apply(lambda x: re.findall(PATTERN, x)[0])
    # check whether the row is in AU public holiday
    aggregate_result = is_holiday(aggregate_result)
    aggregate_result = cast_aggregate_result(aggregate_result)
    return aggregate_result

"""
The function to aggregate the traffic dataset with stop attribute constraints

Parameters:
    dataframe: The dataframe to be aggragated
    start_date: The start date of the dataframe to be preprocessed
    end_date: The start date of the dataframe to be preprocessed
    stop_id: The stop_id to be aggragated
    time_interval: the time period for each record
    timing: The timing choices of alighting time and boarding time
Return:
    aggregate_result: The aggregated result to be processed in deep learning model
"""
def aggregate_stop_attribute(dataframe, start_date, end_date, stop_id, time_interval, timing):
    pd.set_option('mode.chained_assignment', None)
    dataframe = dataframe[['Operations Date', timing +
                           ' Time', 'Passengers', timing + ' Stop', timing + ' stop_id', 'Route']]
    # extract the records by stop_id constraints
    dataframe = dataframe[dataframe[timing + ' stop_id'] == stop_id]
    date_index = date_slice(start_date=start_date, end_date=end_date, time_interval=time_interval)
    # get the time interval from the dataframe
    dataframe['Time Interval'] = pd.DataFrame(pd.cut(dataframe[timing + ' Time'], bins=date_index))
    # perform aggregate operation
    aggregate_result = pd.DataFrame(dataframe.groupby(['Time Interval'], as_index=False)['Passengers'].sum())
    aggregate_result['Date'] = aggregate_result['Time Interval'].astype('str').apply(lambda x: re.findall(PATTERN, x)[0])
    aggregate_result = is_holiday(aggregate_result)
    aggregate_result = cast_aggregate_result(aggregate_result)
    return aggregate_result

"""
Construct a function to aggregate the traffic dataset with route constraints

Parameters:
    dataframe: The dataframe to be aggragated
    start_date: The start date of the dataframe to be preprocessed
    end_date: The start date of the dataframe to be preprocessed
    route_id: The route_id to be aggragated
    time_interval: the time period for each record
    timing: The timing choices of alighting time and boarding time
Return:
    aggregate_result: The aggregated result to be processed in deep learning model
"""
def aggregate_route_attribute(dataframe, start_date, end_date, route_id, time_interval, timing):
    dataframe = dataframe[dataframe['Route'] == route_id] # extract route attributes
    aggregate_result = aggregate_dataframe(dataframe, start_date, end_date, time_interval, timing)
    return aggregate_result

"""
The function to aggregate the traffic dataset with both stop and route constraints.

Parameters:
    dataframe: The dataframe to be aggragated
    start_date: The start date of the dataframe to be preprocessed
    end_date: The start date of the dataframe to be preprocessed
    time_interval: the time period for each record
    timing: The timing choices of alighting time and boarding time
Return:
    aggregate_result: The aggregated result to be processed in deep learning model
"""
def aggregate_double_attribute(dataframe, start_date, end_date, stop_id, route_id, time_interval, timing):
    dataframe = dataframe[dataframe['Route'] == route_id]
    aggregate_result = aggregate_stop_attribute(dataframe, start_date, end_date, stop_id, time_interval, timing)
    return aggregate_result

"""
The function to cast time interval for viualization panel

Parameters:
    time interval: the interval to be processed
Return:
    date string: the date string in the diagram
"""
def cast_time_interval(time_interval):
    if (0 < time_interval < 60):
        return str(time_interval) + ' minutes'
    elif (60 <= time_interval < 720):
        return str(time_interval / 60) + ' hours'
    else:
        return str(time_interval / 1440) + ' days'

"""
The function to get statistics for visualization panel

Parameters:
    parameters: The dictionary to be stored
Return:
    parameters: The same processed dictionary to be returned
"""
def get_statistics(parameters):
    parameters['max'] = max(parameters['list']) # get the max value
    parameters['min'] = min(parameters['list']) # get the min value
    parameters['mean'] = int(np.mean(parameters['list'])) # get the average value
    parameters['max index'] = parameters['dates'][parameters['list'].index(max(parameters['list']))]
    parameters['min index'] = parameters['dates'][parameters['list'].index(min(parameters['list']))]
    return parameters

"""
The function to sort the route passenger values for top busy routes

Parameters:
    dateframe: The dateframe to be pre-processed
    start_date: The start date of the dateframe
    end_date: The end date of the dateframe
Return:
    top_routes: The top busy routes for passengers
    passenger_list: The list of passenger number of each route
    date_list: The date list for each route
    total: The total passenger number for each route
"""
def sort_routes(dataframe, start_date, end_date):
    # use group by function to get the aggregated passenger number for all routes
    aggregate_result = pd.DataFrame(dataframe.groupby(['Route'], as_index=False)['Passengers'].sum())
    # sort the passenger number for all routes
    aggregate_result = aggregate_result.sort_values(by='Passengers', ascending=False).reset_index()
    # get top five busy routes passenger number
    top_routes = aggregate_result.loc[:4, 'Route'].values.tolist()
    total = aggregate_result.loc[:4, 'Passengers'].values.tolist()
    passenger_list = []
    date_list = []
    # get the passenger number for each routes
    for route in top_routes:
        out = aggregate_route_attribute(
            dataframe, start_date, end_date, route, 120, 'Alighting')
        passengers = out['Passengers'].values.tolist()
        date_list = out['Date'].values.tolist()
        passenger_list.append(passengers)
    return top_routes, passenger_list, date_list, total

"""
The function generate information for heat map

Parameters:
    dateframe: The dateframe to be pre-processed
    start_date: The start date of the dateframe
    end_date: The end date of the dateframe
    time_interval: The time interval of the data
Return:
    stop_info: A list of stops with longitude and latitude
"""
def heatmap_stop(dataframe, start_date, end_date, time_interval):
    stops = dataframe['Alighting stop_id'].unique().tolist()
    # get the longitude and latitude from the destination file
    destination = pd.read_csv('Destination.csv')
    # use date slice to aggregate the dataframe
    date_index = date_slice(start_date=start_date, end_date=end_date, time_interval=time_interval)
    dataframe['Time Interval'] = pd.DataFrame(pd.cut(dataframe['Alighting' + ' Time'], bins=date_index))
    aggregate_result = pd.DataFrame(dataframe.groupby(['Time Interval', 'Alighting stop_id'], as_index=False)['Passengers'].sum())
    aggregate_result['Passengers'] = aggregate_result['Passengers'].apply(lambda x: cast_heatmap_weight(x))
    # use dictionary to get all heat map information
    stops_info = []
    for index, element in enumerate(stops):
        stop = destination[destination['stop_id'] == element]
        if stop.empty == False:
            dic = {}
            dic['stop_name'] = stop['stop_name'].values[0] # get the stop name
            dic['stop_lat'] = stop['stop_lat'].values[0] # get the longitude
            dic['stop_lon'] = stop['stop_lon'].values[0] # get the latitude
            dic['weight'] = aggregate_result[aggregate_result['Alighting stop_id'] == element]['Passengers'].values.tolist()
            stops_info.append(dic)
    return stops_info

"""
The function to generate heap map weight by passenger number

Parameters:
    passengers: The passenger number to be processed
Return:
    weight: The weight processed for the heat map
"""
def cast_heatmap_weight(passengers):
    if passengers < 100:
        weight = 1
    elif (passengers >= 100) & (passengers < 200):
        weight = 2
    elif (passengers >= 200) & (passengers < 300):
        weight = 3
    elif (passengers >= 300) & (passengers < 400):
        weight = 4
    else:
        weight = 5
    return weight

if __name__ == '__main__':
    # dataframe = read_traffic_csv('May 2019 TransactionReport-314828-1.csv')
    dataframe = read_traffic_csv('April 2019 TransactionReport-314829-1.csv')
    # dataframe = read_traffic_csv('March 2019 TransactionReport-314830-1.csv')
    stop_UQ = aggregate_stop_attribute(dataframe, '2019/03/01 06:00', '2019/03/30 23:59', '1882', 30, 'Boarding')
    # stop_UQ = aggregate_dataframe(dataframe, '2019/05/01 06:00', '2019/05/30 23:59', 120, 'Alighting')
    # stop_UQ = aggregate_dataframe(dataframe, '2019/04/01 06:00', '2019/04/07 23:59', 15, 'Alighting')
    dates = stop_UQ['Date'].values.tolist()
    set = np.array(stop_UQ['Passengers'].values)
    # plt.figure(figsize=(12, 5))
    #
    # plt.plot(dates, set.flatten(), 'b', label='Passenger Flow')
    # plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(200))
    # plt.ylabel('Passenger Number')
    # plt.xlabel('Time Interval: 30 minutes / Duration: 2019/03/01 06:00 --- 2019/03/30 23:59')
    # plt.title('UQ Lake Stop [1882] March')
    # plt.legend(loc='best')
    plt.show()




