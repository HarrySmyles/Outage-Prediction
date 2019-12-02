import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta, date, time
import statistics
from sklearn.linear_model import LinearRegression
from sklearn import metrics

lr = LinearRegression()

#load data
location = 'outage_data.csv'
df = pd.DataFrame()
df = pd.read_csv(location)


#All States in Continental US
states = ["Alabama","Arizona","Arkansas","California","Colorado",
  "Connecticut","Delaware","Florida","Georgia","Idaho","Illinois",
  "Indiana","Iowa","Kansas","Kentucky","Louisiana","Maine","Maryland",
  "Massachusetts","Michigan","Minnesota","Mississippi","Missouri","Montana",
  "Nebraska","Nevada","New Hampshire","New Jersey","New Mexico","New York",
  "North Carolina","North Dakota","Ohio","Oklahoma","Oregon","Pennsylvania",
  "Rhode Island","South Carolina","South Dakota","Tennessee","Texas","Utah",
  "Vermont","Virginia","Washington","West Virginia","Wisconsin","Wyoming"]


us_state_region = {
    'Alabama': 'South',
    'Arizona': 'West',
    'Arkansas': 'South',
    'California': 'West',
    'Colorado': 'West',
    'Connecticut': 'Northeast',
    'Delaware': 'South',
    'District of Columbia': 'South',
    'Florida': 'South',
    'Georgia': 'South',
    'Idaho': 'West',
    'Illinois': 'Midwest',
    'Indiana': 'Midwest',
    'Iowa': 'Midwest',
    'Kansas': 'Midwest',
    'Kentucky': 'South',
    'Louisiana': 'South',
    'Maine': 'Northeast',
    'Maryland': 'South',
    'Massachusetts': 'Northeast',
    'Michigan': 'Midwest',
    'Minnesota': 'Midwest',
    'Mississippi': 'South',
    'Missouri': 'Midwest',
    'Montana': 'West',
    'Nebraska': 'Midwest',
    'Nevada': 'West',
    'New Hampshire': 'Northeast',
    'New Jersey': 'Northeast',
    'New Mexico': 'West',
    'New York': 'Northeast',
    'North Carolina': 'South',
    'North Dakota': 'Midwest',
    'Ohio': 'Midwest',
    'Oklahoma': 'South',
    'Oregon': 'West',
    'Pennsylvania': 'Northeast',
    'Rhode Island': 'Northeast',
    'South Carolina': 'South',
    'South Dakota': 'Midwest',
    'Tennessee': 'South',
    'Texas': 'South',
    'Utah': 'West',
    'Vermont': 'Northeast',
    'Virginia': 'South',
    'Washington': 'West',
    'West Virginia': 'South',
    'Wisconsin': 'Midwest',
    'Wyoming': 'West',
}


"""Functions"""


#Convert calculate outage duration from start date and time and end date and time
def outageDuration(startDate, startTime, endDate, endTime):
    outageList = []
    for i in range(0, len(startDate)):
        if str(startTime[i]).lower() == "midnight":
            startTime[i] = "12:00 am"
        if str(endTime[i]).lower() == "midnight":
            endTime[i] = "12:00 am"
        try:
            splitDate = startDate[i].split("/")
            date1 = date(int(splitDate[2]), int(splitDate[0]), int(splitDate[1]))
            splitDate = endDate[i].split("/")
            date2 = date(int(splitDate[2]), int(splitDate[0]), int(splitDate[1]))
            splitTime = startTime[i].split(":")
            splitTime[1] = splitTime[1].replace(".","").replace("noon", "pm")
            if splitTime[1][-2:] != "am" and splitTime[0] == "12":
                splitTime[0] = "0"
            h = int(splitTime[0]) if (splitTime[1][-2:].lower() == "am" or splitTime[0] == '12') else (int(splitTime[0]) + 12)
            m = int(splitTime[1][0:2])
            time1 = time(hour=h, minute=m)
            splitTime = endTime[i].split(":")
            splitTime[1] = splitTime[1].replace(".","").replace("noon", "pm")
            if splitTime[1][-2:] != "am" and splitTime[0] == "12":
                splitTime[0] = "0"
            h = int(splitTime[0]) if (splitTime[1][-2:].lower() == "am" or splitTime[0] == '12') else (int(splitTime[0]) + 12)
            m = int(splitTime[1][0:2])
            time2 = time(hour=h, minute=m)
            datetime = timedelta(hours=time2.hour-time1.hour, minutes = time2.minute-time1.minute)
            datetime = datetime + (date2 - date1)
            if datetime.total_seconds()/3600 < 0:  #Remove durations less than 0 from invalid data
                outageList.append('')
            else:
                outageList.append(datetime.total_seconds()/3600)
        except:
            outageList.append('')
    return outageList

"""Cleaning Data"""

#Function to plot the scatter plot and return prediction data
def plot_data(stateList, outageList, tagList, criteria, setlist, customerCount):
    totallist = []
    returnlist = {}
    for j, state in enumerate(setlist):
        outlist = []
        customerlist = []
        for i, tag in enumerate(tagList):
            if criteria in str(tag).lower() and state == stateList[i]:
                outlist.append(outageList[i])
                totallist.append(outageList[i])
                customerlist.append(int(customerCount[i].replace(",","")))
        outlist2 = np.array([outlist])
        customerlist2 = np.array([customerlist])
        outlist2 = outlist2.reshape(len(outlist),1)
        customerlist2 = customerlist2.reshape(len(customerlist),1)
        lr.fit(customerlist2, outlist2)
        y_pred = lr.predict(customerlist2)
        plt.subplot(2, 2, j+1)
        plt.scatter(customerlist, outlist)
        plt.plot(customerlist, y_pred, label = "Regression Line", color="red")
        returnlist[state] = [metrics.mean_squared_error(outlist2, y_pred), lr.coef_[0][0], lr.intercept_[0]]
        plt.xlabel("Customers Out")
        plt.ylabel("Outage Duration(Hours)")
        plt.title(state)
        plt.legend(loc='upper left')
    plt.suptitle(criteria.upper())
    plt.show()
    return returnlist

#Remove locations not in continental US
row = 0
check = 0
for geo in df['Geographic Areas']:
    for state in states:
        if state.lower() in str(geo).lower():
            df['Geographic Areas'].loc[row] = state
            check = 1
    if check != 1:
        df['Geographic Areas'].loc[row] = ''
    check = 0
    row = row + 1
mask = df['Geographic Areas'] == ''
df = df.loc[~mask, :]
df = df.reset_index(drop=True)

#Remove rows where there is no customer data
for i, customer_count in enumerate(df["Number of Customers Affected"]):
    if type(customer_count) == str:
        data = customer_count.split(' ')
        for word in data:
            if not word.replace(",","").isdigit():
                df["Number of Customers Affected"][i] = ''
                break
            if int(word.replace(",", "")) == 0:  #Remove rows where number of customers out = 0 because no customers were out to experience outage
                df["Number of Customers Affected"][i] = ''
                break
    else:
        df["Number of Customers Affected"][i] = ''

mask = df['Number of Customers Affected'] == ''
df = df.loc[~mask, :]
df = df.reset_index(drop=True)


#Add a new column to dataset that has outage duration and remove rows that do not have a valid duration
df['Outage Duration'] = outageDuration(df["Date Event Began"], df["Time Event Began"], df["Date of Restoration"], df["Time of Restoration"])
mask = df['Outage Duration'] == ''
df = df.loc[~mask, :]
df = df.reset_index(drop=True)

#Drop extra columns
df = df.drop(["Date Event Began", "Time Event Began", "Date of Restoration", "Time of Restoration", "Respondent", "NERC Region", "Demand Loss (MW)"], axis=1)

#Fill Unknown in null values
df["Tags"].fillna(value="Unknown")
tagdict = {}
for tag in df["Tags"]:
    tagsplit = str(tag).lower().split(", ")
    for i in tagsplit:
        if i in tagdict:
            tagdict[i] = tagdict[i] + 1
        else:
            tagdict[i] = 1

#Change states to regions as just state data pool is too small
for i, state in enumerate(df["Geographic Areas"]):
    df["Geographic Areas"][i] = us_state_region[state]

#Create list of all states mentioned in dataset
states = set()
for state in df["Geographic Areas"]:
    states.update({state})
states = list(states)
states.sort()

#replace fuel supply emergency related outages with load shedding as they are related
for i, tag in enumerate(df["Tags"]):
    if type(tag) == str:
        if "fuel supply emergency" in tag:
            df["Tags"][i] = "load shedding"


"""Show Data"""
request = input("What data do you want to see? enter number:\n(1) severe weather\n(2) vandalism\n(3) load shedding\n\nInput: ")
key = {'1': 'severe weather', '2': 'vandalism', '3': 'load shedding'}
predicted_data = plot_data(df["Geographic Areas"], df["Outage Duration"], df["Tags"], key[request], states, df["Number of Customers Affected"])

requested_region = int(input("What Region to predict Outage time given customer count?\n(1) West\n(2) Midwest\n(3) South\n(4) Northeast\n\nInput: "))
customercount = int(input("Enter number of customer's out:  "))

if requested_region == 1:
    print("Estimated Outage Duration: ", customercount * predicted_data["West"][1] + predicted_data["West"][2], " hours")
    print("Mean Squared Error: ", predicted_data['West'][0])

if requested_region == 2:
    print("Estimated Outage Duration: ", customercount * predicted_data["Midwest"][1] + predicted_data["Midwest"][2], " hours")
    print("Mean Squared Error: ", predicted_data['West'][0])

if requested_region == 3:
    print("Estimated Outage Duration: ", customercount * predicted_data["South"][1] + predicted_data["South"][2], " hours")
    print("Mean Squared Error: ", predicted_data['West'][0])

if requested_region == 4:
    print("Estimated Outage Duration: ", customercount * predicted_data["Northeast"][1] + predicted_data["Northeast"][2], " hours")
    print("Mean Squared Error: ", predicted_data['West'][0])

