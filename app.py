from flask import Flask, render_template
import requests
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import pickle

app = Flask(__name__)

loaded_model = None

# Load the model from the pickle file
with open("data/cricketPrediction.pkl", "rb") as f:
    loaded_model = pickle.load(f)
    

import requests
import pandas as pd

mainUrl = "https://garbage-cheat-reads-protect.trycloudflare.com"

getEntireLiveDataApi = "/analyzingtrends/getentirelivedata" 
getCurrentLiveDataApi = "/analyzingtrends/getcurrentlivedata"

def getAllLiveData():
  response = requests.get(mainUrl + getEntireLiveDataApi)

  liveData = []

  if response.status_code == 200:  # Check if the request was successful
      json_data = response.json()  # Convert response to JSON format
      liveData = pd.DataFrame(json_data)
  else:
      print("Failed to retrieve data from the API.")

  return liveData

def getCurrentLiveData():
  response = requests.get(mainUrl + getCurrentLiveDataApi)
  liveList = []
  liveData = None
  json_data = None

  if response.status_code == 200:  # Check if the request was successful
      json_data = response.json()  # Convert response to JSON format
      liveList.append(json_data)
      liveData = pd.DataFrame(liveList)
  else:
      print("Failed to retrieve data from the API.")

  return liveData, json_data    


def processLiveFeed():

    liveData0, liveJson = getCurrentLiveData()


    new_columns = ['match_id', 'season', 'venue', 'innings', 'ball', 'batting_team',
                'bowling_team', 'striker', 'non_striker', 'bowler', 'runs_off_bat',
                'extras', 'wides', 'noballs', 'city', 'date', 'team1', 'team2',
                'toss_winner', 'toss_decision', 'result', 'dl_applied', 'winner',
                'win_by_runs', 'win_by_wickets', 'player_of_match', 'total_runs', 'id']

    # Rename columns
    liveData0.columns = new_columns
    liveData = liveData0.filter(['batting_team', 'bowling_team', 'non_striker', 'bowler', 'team1',
        'team2', 'toss_winner', 'ball', 'runs_off_bat', 'dl_applied'])

    to_norm = (liveData.select_dtypes(include =['int64', 'float64'])) #select continuous variables

    continuous = MinMaxScaler().fit_transform(to_norm) #fit and transform min max scaler (normalizes)
    continuous = pd.DataFrame(continuous, index = liveData.index, columns = list(to_norm))

    #encode categorical variables

    to_encode = (liveData.select_dtypes(include =['object'])) #test if anything is an object or category variable

    label_encoders = [] #make new encoder for each column
    encoded_data = pd.DataFrame()

    encoding_dicts = {}

    # Iterate over each column in 'to_encode' and encode using a separate LabelEncoder
    for col in to_encode:
        # Create a new instance of LabelEncoder for the current column
        encoder = LabelEncoder()
        
        # Fit and transform the data in 'from02to22[col]'
        encoded_data[col] = encoder.fit_transform(liveData[col])
        
        # Store the encoder in the list
        label_encoders.append(encoder)

        encoding_dicts[col] = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))


    #patch the columns back together

    data = pd.concat([encoded_data.reset_index(drop=True), continuous.reset_index(drop=True)], axis = 1) #reset indices to avoid errors in concat

    predictRFC = loaded_model.predict(data)

    # Apply mapping function directly to the predicted values (assuming predictRFC is a Series or array)
    out = pd.Series(predictRFC).apply(map_team)
    out = pd.DataFrame(out)
    out['match_id'] = liveData0['match_id']

def map_team(value):
    if value == 0:
        return 'team1'
    else:
        return 'team2'





@app.route('/')
def index():
    liveData0, liveJson = getCurrentLiveData()
    return render_template('index.html', match_data = liveJson)

if __name__ == '__main__':
    app.run()
