import json
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import numpy as np
import math
import re

studentid = os.path.basename(sys.modules[__name__].__file__)


def log(question, output_df, other):
    print("--------------- {}----------------".format(question))

    if other is not None:
        print(question, other)
    if output_df is not None:
        df = output_df.head(5).copy(True)
        for c in df.columns:
            df[c] = df[c].apply(lambda a: a[:20] if isinstance(a, str) else a)

        df.columns = [a[:10] + "..." for a in df.columns]
        print(df.to_string())

def question_1(file):
    """
    :return: df1
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """
    data = pd.read_csv(file)
    # passenger_in_out
    data['passenger_in_out'] = data['Passengers_In'] - data['Passengers_Out']
    data['passenger_in_out'] = data['passenger_in_out'].apply(lambda x : "IN" if x > 0 else ("OUT" if x < 0 else "SAME"))

    #freight_in_out
    data['freight_in_out'] = data['Freight_In_(tonnes)'] - data['Freight_Out_(tonnes)']
    data['freight_in_out'] = data['freight_in_out'].apply(lambda x : "IN" if x > 0 else ("OUT" if x < 0 else "SAME"))

    #mail_in_out
    data['mail_in_out'] = data['Mail_In_(tonnes)'] - data['Mail_Out_(tonnes)']
    data['mail_in_out'] = data['mail_in_out'].apply(lambda x : "IN" if x > 0 else ("OUT" if x < 0 else "SAME"))

    log("QUESTION 1", output_df=data[["AustralianPort", "ForeignPort", "passenger_in_out", "freight_in_out", "mail_in_out"]], other=data.shape)
    return data


def question_2(df):
    """
    :param df1: the dataframe created in question 1
    :return: dataframe df2
            Please read the assignment specs to know how to create the output dataframe
    """

    df2 = df.loc[:, ['AustralianPort']]
    df2["PassengerInCount"] = df["passenger_in_out"].apply(lambda x: x=="IN")
    df2["PassengerOutCount"] = df["passenger_in_out"].apply(lambda x: x=="OUT")
    df2["FreightInCount"] = df["freight_in_out"].apply(lambda x:x=="IN")
    df2["FreightOutCount"] = df["freight_in_out"].apply(lambda x:x=="OUT")
    df2["MainInCount"] = df["mail_in_out"].apply(lambda x:x=="IN")
    df2["MainOutCount"] = df["mail_in_out"].apply(lambda x:x=="OUT")
    df2 = df2.groupby("AustralianPort").sum().reset_index()
    log("QUESTION 2", output_df=df2, other=df2.shape)
    return df2

def question_3(df):
    """
    :param df1: the dataframe created in question 1
    :return: df3
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """

    df3 = df.loc[:,["Country", "Passengers_In", "Passengers_Out","Freight_In_(tonnes)", "Freight_Out_(tonnes)", "Mail_In_(tonnes)", "Mail_Out_(tonnes)"]]
    df3 = df3.groupby("Country").mean().reset_index()
    df3=df3.sort_values(by="Passengers_In").round(2)

    df3.columns = ["Country", "Passengers_in_average", "Passengers_out_average", "Freight_in_average","Freight_out_average", "Mail_in_average", "Mail_out_average"]

    log("QUESTION 3", output_df=df3, other=df3.shape)
    return df3

def question_4(df):
    """
    :param df1: the dataframe created in question 1
    :return: df4
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """

    temp = df.loc[df["Passengers_Out"]>0, ["AustralianPort", "Country", "Month", "ForeignPort"]] # at least one passenger
    df4 = pd.DataFrame({"counts": temp.groupby(["AustralianPort","Country", "Month"]).size()}).reset_index()
    
    df4 = df4[df4["counts"]>1] # more than one ForeignPort used 
    df4 = df4.groupby("Country").size().reset_index() # group by country, then count
    df4.columns = ["Country", "Unique_ForeignPort_Count"]
    df4.sort_values(["Unique_ForeignPort_Count",'Country'], inplace=True,ascending=[False,True])
    log("QUESTION 4", output_df=df4, other=df4.shape)
    return df4.head()

def question_5(path):
    """
    :param seats : the path to dataset
    :return: df5
            Data Type: dataframe
            Please read the assignment specs to know how to create the  output dataframe
    """
    df5 = pd.read_csv(path)
    df5["Source_City"] = df5.apply(lambda x : x["International_City"] if x["In_Out"]=="I" else x["Australian_City"], axis=1)
    df5["Destination_City"] = df5.apply(lambda x : x["International_City"] if x["In_Out"]=="O" else x["Australian_City"], axis=1)
    
    log("QUESTION 5", output_df=df5, other=df5.shape)
    return df5

def question_6(df):
    """
    :param df5: the dataframe created in question 5
    :return: df6
    """
    
    # Comment:
    #     We focus on the data between 2015 and 2019, this is because the early flight information does not reveal the current market staus, and the COVID pandemic had a huge influence on international flight, so the data before 2015 and after 2019 is dropped.

    #     The route is defined as a city-pair, regardless of the number of stops in-between. For example, a real route Sydney-Melbourne-Nadi is considered as a Sydney-Nadi route and a Melbourne-Nadi route. 
        
    #     The airline company can extract the flight information for specific route using the index "<source>-<destination>", i.e. data.loc['Sydney-Nadi']
    

    cols = ['Source_City', 'Destination_City', 'Port_Country', 'Port_Region', 'Service_Country', 'Service_Region', 'Airline', 'Route', 'Stops', 'All_Flights', 'Year', 'Month_num' ]
    df = df.loc[:, cols]
    conditions = (df['Year'] >= 2015) & (df['Year'] < 2020)
    df = df[conditions]
    df['Stops'] = df['Stops'].apply(lambda x : str(x) + ' stop(s)')
    df['Route'] = df.apply(lambda x : x['Source_City'] + "-" + x['Destination_City'], axis=1)
    df6 = np.round(pd.pivot_table(df, index=['Route', 'Airline'], columns=['Year', 'Stops'], values='All_Flights', aggfunc=np.sum, fill_value=0))
    
    log("QUESTION 6", output_df=None, other=df6.shape)
    print(df6.head())
    return df6
    
def question_7(city_pairs, seats):
    """
    :param seats: the path to dataset
    :param city_pairs : the path to dataset
    :return: nothing, but saves the figure on the disk
    """
    
    # Comment:
    #     The chart displays the seat utilisation rate for each Port_Region between the period of Sep 2003 and Sep 2022. 

    #     Two dataframes are created for inbound travel and outbound travel respectively.

    #     Seat utilisation rate is cauculated as passenger_count/total_max_seats, where passenger_count can be Passengers_In or Passengers_Out, total_max_count is the summation of Max_Seats of all flights from the source city to destination city in a unique month.

    #     The seat utilisation rate is calculated for each city-pair in a unique month, and the average rate is aggregated for each port region.
        
    
    # extract needed columns, and filter date
    col1 = ['Passengers_In', 'Passengers_Out', 'AustralianPort', 'ForeignPort', 'Month']
    col2 = ['Source_City','Destination_City', 'Port_Region', 'Max_Seats', 'Month']

    city_pairs = city_pairs.loc[:, col1]
    city_pairs['Month'] = city_pairs['Month'].apply(lambda x : "01-"+x)
    city_pairs['Month'] = pd.to_datetime(city_pairs['Month']).dt.date
    city_pairs = city_pairs[(city_pairs['Month'] >= pd.to_datetime("9-2003").date()) & (city_pairs['Month'] <= pd.to_datetime("9-2022").date())]

    seats = seats.loc[:, col2]
    seats['Month'] = seats['Month'].apply(lambda x : "01-"+x)
    seats['Month'] = pd.to_datetime(seats['Month']).dt.date
    seats = seats[(seats['Month'] >= pd.to_datetime("9-2003").date()) & (seats['Month'] <= pd.to_datetime("9-2022").date())]

    # get total max seats
    seats = seats.groupby(['Month', 'Port_Region', 'Source_City','Destination_City']).sum().reset_index()

    # get dataframe for inbound and outbound travel
    df_In= city_pairs.merge(seats, left_on=['Month', 'AustralianPort', 'ForeignPort'], right_on=['Month', 'Destination_City', 'Source_City'])
    df_Out= city_pairs.merge(seats, left_on=['Month', 'AustralianPort', 'ForeignPort'], right_on=['Month', 'Source_City', 'Destination_City'])

    # calculate seat utilisation rate
    df_In['Seat_Rate'] = np.round(df_In['Passengers_In'] * 100 / df_In['Max_Seats'],2)
    df_Out['Seat_Rate'] = np.round(df_Out['Passengers_Out'] * 100 / df_Out['Max_Seats'],2)

    # drop column
    df_In.drop('Passengers_Out', axis=1, inplace=True)
    df_Out.drop('Passengers_In', axis=1, inplace=True)

    # calculate average seat utilisation rate for each port region
    df_In = df_In.groupby(['Port_Region', 'Month'])['Seat_Rate'].mean().reset_index()
    df_Out = df_Out.groupby(['Port_Region', 'Month'])['Seat_Rate'].mean().reset_index()

    # group by port region for charts
    grouped_In = df_In.groupby('Port_Region')
    grouped_Out = df_Out.groupby('Port_Region')
    fig, axes = plt.subplots(5,2, figsize=(15,25))

    for (key, ax) in zip(grouped_In.groups.keys(), axes.flatten()):
        grouped_In.get_group(key).plot(x='Month', y='Seat_Rate', ax=ax, legend=True)
        grouped_Out.get_group(key).plot(x='Month', y='Seat_Rate', ax=ax)
        ax.legend(['Inbound', 'Outbound'])
        ax.set_title(key)
        ax.set_ylabel('Seat Utilisation Rate (%)')
        ax.set_xlabel('Time')
        ax.set_ylim(0,100)
        
    
    plt.subplots_adjust(hspace = 0.3)
    fig.suptitle("Australian International Flight Seat Utilisation Rate between Sep 2003 to Sep 2022")
    plt.savefig("{}-Q7.png".format(studentid))


if __name__ == "__main__":
    df1 = question_1("city_pairs.csv")
    df2 = question_2(df1.copy(True))
    df3 = question_3(df1.copy(True))
    df4 = question_4(df1.copy(True))
    df5 = question_5("seats.csv")
    df6 = question_6(df5.copy(True))
    question_7(df1.copy(True),df5.copy(True))
