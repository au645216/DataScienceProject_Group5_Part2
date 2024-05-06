#import libraries
import requests
import pandas as pd
import datetime
import os

# Write url code for api
url = "https://skyscanner80.p.rapidapi.com/api/v1/flights/search-one-way"

# Write query string
querystring = {
    "fromId": "eyJzIjoiQkxMIiwiZSI6Ijk1NjczOTg5IiwiaCI6IjI3NTM5NDY2In0=",
    "toId": "eyJzIjoiU1ROIiwiZSI6Ijk1NTY1MDUyIiwiaCI6IjI3NTQ0MDA4In0=",
    "departDate": "2024-06-06",
    "adults": "1",
    "currency": "DKK"
}

# Credentials
headers = {
    "X-RapidAPI-Key": "29c8a7e93fmshfcd61b4708b9365p1b58bcjsn640b9f086194",
    "X-RapidAPI-Host": "skyscanner80.p.rapidapi.com"
}

# Write response you would like to get
response = requests.get(url, headers=headers, params=querystring)

# Data flattening, so we can get the intenerary data
data = response.json()['data']
flattened_data = pd.json_normalize(data)
flattened_data2 = data['itineraries']

#creation of timestamp so we can see when data is drawn aka when the price was what
current_datetime = datetime.datetime.now()
timestamp = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

#creation of place for data to go
output_data = []
#extraction of data
for item in flattened_data2:
    if item["id"] == '9996-2406060915--31915-0-16574-2406060940':
        output_data.append({
            "ID": item["id"],
            "Price (Raw)": item["price"]["raw"],
            "Origin ID": item["legs"][0]["origin"]["id"],
            "Destination ID": item["legs"][0]["destination"]["id"],
            "Departure time": item["legs"][0]["departure"],
            "Arrival time": item["legs"][0]["arrival"],
            "Timestamp": timestamp
         })

#creation a DataFrame from the extracted data
df = pd.DataFrame(output_data)

#check if the file exists
if os.path.isfile("flightprices.xlsx"):
    #file exists, so write a new lines in file
    with pd.ExcelWriter("flightprices.xlsx", mode="a", if_sheet_exists="overlay") as writer:
        df.to_excel(writer, index=False, sheet_name="Sheet1", startrow=writer.sheets["Sheet1"].max_row, header=False)
else:
    #file doesn't exist, so make a new excel file and write there
    df.to_excel("flightprices.xlsx", index=False)

