# Cricket Match Winner Analyser

Final project repo for INFO 523 - Spring 2024.

# Introduction

Our project predicts the match results by using historical ODI cricket data and state-of-the-art machine learning. It redefines cricket analysis by closely monitoring forecast accuracy and encouraging user engagement.

# Datasets

We have considered dataset from year 2002 to 2022 as past data. using this following data we have trainned the machine learning model to predict the current years match winner (considering 2023 as the current year).

oDI_Match_Data.csv: Provides facts about the location and season of the cricket matches along with team information and the play results from each team member. We’ll need this one to investigate partnerships between batsmen. It’s dimensions are 155432 rows of data by 23 variable columns.

ODI_Match_info.csv: Overlaps in data with the above but provides information on the umpire, performance, and the city the match took place. We’ll need this one to analyze the batting and bowling performance of each player. It’s dimensions are 2380 rows of data by 18 variable columns.

# Primary Objectives

Within this repository, In this project, we use past and current match data, train them under regression classification models, select the best model and then use it to predict the winner of a cricket match based on the live match statistics (e.g., runs scored, wickets fallen, overs bowled), that update at regular intervals.

This model will be displayed on a webpage which will refresh every time a new stat is updated, and the prediction from the model will update.

# Contributors
 
- Christian Ortmann
- Pappala Praveen Kumar
- Poonkundran Srinivasan
- Srinivasan Akash
- Theeda Gowtham
- Tiruthani Rajitha Reddy
- Bhawari Tejas



#### Disclosure:

The project is accomplished under the guidance of Dr. Greg Chism for INFO 523 - Data Mining and Discovery at the University of Arizona.