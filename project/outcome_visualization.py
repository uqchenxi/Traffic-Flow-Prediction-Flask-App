import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

NUMLIST = [str(i) for i in range(10)] # the number str from 1 to 9

"""
Construct a function to extract passenger flow and corresponding date from a stop

Parameters:
    dataframe: The traffic dataframe to be extract
    stop_id: The stop to be analyzed
Return:
    stop: the customized data only contains time interval and passengers
"""
def create_stop_data(stop_id, dataframe):
    stop = dataframe[dataframe['stop_id'] == stop_id]
    stop = stop[['Time Interval', 'Passengers']]
    stop['Time Interval'] = stop['Time Interval'].apply(lambda x: get_time_period(x))
    stop = stop.astype('float32')
    return stop


"""
Construct a function to extract passenger flow and corresponding date from a stop

Parameters:
    dataframe: The traffic dataframe to be extract
    route_id: The route to be analyzed
Return:
    stop: the customized data only contains time interval and passengers
"""
def create_route_data(route_id, dataframe):
    route = dataframe[dataframe['Route'] == route_id]
    route = route[['Time Interval', 'Passengers']]
    route['Time Interval'] = route['Time Interval'].apply(lambda x: get_time_period(x))
    route = route.astype('float32')
    return route

"""
Construct a function to extract hours and minutes from Time Interval str

Parameters:
    time_str: the time string to be extract:
Returns:
    s_num: the expected number of string
"""
def get_time_period(time_str):
    status = False
    s_num = ''

    for i, num in enumerate(time_str):
        if num == ' ':
            status = True

        if status == True:
            if num in NUMLIST:
                s_num += num

        if len(s_num) == 4:
            status = False

    return s_num

# # read the processed data from passenger flow file and destination file
# dataframe = pd.read_csv('./Aggregate_Passenger_Flow_April_route.csv')
# dataframe = dataframe.dropna()
# destination = pd.read_csv('./Destination.csv')
# # dataframe['stop_id'] = dataframe['stop_id'].astype(str)
# #
# # # combine the passenger flow file and destination file
# # combined_data = pd.merge(destination, dataframe, on = 'stop_id')
# #
# # # extract the dataset of UQ lake on April 1st, 2019
# # stop_UQ = create_stop_data('1882', dataframe)
# #route_412 = create_route_data('412', dataframe)
# route_411 = create_route_data('412', dataframe)
#
# # plot of the stop passenger from April 2019
# #plt.figure('Passenger Flow for UQ Lake Station [1882] during April 2019')
# plt.figure('Passenger Flow for Route [412] during April 1st 2019')
# # x_list = stop_UQ.index
# # y_list = stop_UQ['Passengers']
# x_list = route_411.index
# y_list = route_411['Passengers']
# # plt.scatter(x_list, y_list)
# ax = plt.gca()
# ax.plot(x_list, y_list, color='b', linewidth=1, alpha=0.6)
#
# ax.set_ylabel('Passenger Number')
# ax.set_xlabel('Time Interval : 15 minutes / Duration: April 1st to April 30th')
# #ax.set_xlabel('Time Interval : 15 minutes / Duration: April 1st 06:00am to 23:59pm')
# #ax.set_xlabel('Time Interval : 15 minutes / Duration: April 1st to April 7th')
# #ax.set_title('Passenger Flow for UQ Lake Station [1882] during April, 2019')
# #ax.set_title('Passenger Flow for UQ Lake Station [1882] during April, 2019')
# #ax.set_title('Passenger Flow for Boggo Road station [10795] during April 1st, 2019')
# #ax.set_title('Passenger Flow for Boggo Road station [10795] during April 1st to April 7th, 2019')
# #ax.set_title('Passenger Flow for Boggo Road station [10795] during April 1st to April 30th, 2019')
# #ax.set_title('Passenger Flow for Route [412] during April 1st to 7th, 2019')
# #ax.set_title('Passenger Flow for Route [412] during April 1st, 2019')
# ax.set_title('Passenger Flow for Route [412] during April, 2019')
#ax.set_title('Passenger Flow for Route [411] during April 1st to 7th, 2019')
#ax.set_title('Passenger Flow for Route [411] during April 1st, 2019')
#ax.set_title('Passenger Flow for Route [411] during April, 2019')