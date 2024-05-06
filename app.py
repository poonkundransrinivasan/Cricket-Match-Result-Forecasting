from flask import Flask, render_template
import joblib
import requests
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from collections import Counter
import pickle
import json
import numpy as np

app = Flask(__name__)

import requests
import pandas as pd

loaded_model = None

with open("data/cricketPrediction.pkl", "rb") as f:
    loaded_model = pickle.load(f)

driveUrl = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSd_HEZE5_vyvgzAcOIXqaCvgefTD-B8suE03j46W_pryfgcHBNAne_mVSKNHTIAbB03BkjxbvXFHlr/pub?gid=0&single=true&output=csv"

x = pd.read_csv(driveUrl)

mainUrl = x["local"][0]


# mainUrl = "http://127.0.0.1:8100"

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
    liveData0 = getAllLiveData()

    new_columns = ['match_id', 'season', 'venue', 'innings', 'ball', 'batting_team',
                'bowling_team', 'striker', 'non_striker', 'bowler', 'runs_off_bat',
                'extras', 'wides', 'noballs', 'city', 'date', 'team1', 'team2',
                'toss_winner', 'toss_decision', 'result', 'dl_applied', 'winner',
                'win_by_runs', 'win_by_wickets', 'player_of_match', 'total_runs', 'id']

    # Rename columns
    liveData0.columns = new_columns
    liveData = liveData0.filter(['innings', 'ball', 'batting_team', 'bowling_team', 'striker',
       'non_striker', 'bowler', 'runs_off_bat', 'extras', 'team1', 'team2',
       'toss_winner', 'toss_decision', 'dl_applied'])


    liveData['toss_winner'] = liveData.apply(lambda row: 'team1' if row['toss_winner'] == row['team1'] else 'team2', axis=1)


    # Load the team map dictionary from the saved JSON file
    with open('data/team_map.json', 'r') as f:
        team_map_loaded = json.load(f)

    with open('data/player_map.json', 'r') as f:
        player_map_loaded = json.load(f)

    team_map_loaded = dict(team_map_loaded)
    player_map_loaded = dict(player_map_loaded)

    liveData['team1'] = liveData['team1'].apply(lambda x: team_map_loaded.get(x))

    liveData['team2'] = liveData['team2'].apply(lambda x: team_map_loaded.get(x))

    liveData['batting_team'] = liveData['batting_team'].apply(lambda x: team_map_loaded.get(x))

    liveData['bowling_team'] = liveData['bowling_team'].apply(lambda x: team_map_loaded.get(x))

    liveData['striker'] = liveData['striker'].apply(lambda x: player_map_loaded.get(x))

    liveData['non_striker'] = liveData['non_striker'].apply(lambda x: player_map_loaded.get(x))

    liveData['bowler'] = liveData['bowler'].apply(lambda x: player_map_loaded.get(x))

    liveData['toss_winner'] = liveData['toss_winner'].apply(lambda x: 0 if x == "team1" else 1)

    liveData['toss_decision'] = liveData['toss_decision'].apply(lambda x: 0 if x == "bat" else 1)

    loaded_min_max_scaler = joblib.load('data/min_max_scaler.joblib')

    liveData[['innings', 'ball', 'runs_off_bat', 'extras', 'dl_applied']] = loaded_min_max_scaler.transform(liveData[['innings', 'ball', 'runs_off_bat', 'extras', 'dl_applied']])


    predictRFC = loaded_model.predict(liveData)

    # Apply mapping function directly to the predicted values (assuming predictRFC is a Series or array)
    print(predictRFC)
    out = pd.Series(predictRFC).apply(map_team)
    out = pd.DataFrame(out)
    out['match_id'] = liveData0['match_id']
    print(liveData.head())
    
    # countTeamPrediction = Counter(out[0])

    # print(countTeamPrediction)

    return out, liveJson, predictRFC

def map_team(value):
    if value == 0:
        return 'team1'
    else:
        return 'team2'


@app.route('/')
def index():
    out, liveJson, predictRFC = processLiveFeed()

    liveData, t = getCurrentLiveData()


    zeros = np.count_nonzero(predictRFC == 0)
    total = len(predictRFC)

    percTeam1 = (zeros/total) * 100
    percTeam2 = 100 - percTeam1

    print(liveData.get('team1')[0])

    return render_template('index.html', match_data = liveJson, predictWinTeam1 = liveData.get('team1')[0] + " : " + str(round(percTeam1, 2)) + "%", predictWinTeam2 = liveData.get('team2')[0] + " : " + str(round(percTeam2, 2))  + "%")

if __name__ == '__main__':
    app.run()
