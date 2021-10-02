Predicting Various Risk Factors in the Toronto Neighborhoods: Project Overview
==============================================================================

* Objective: By analyzing the three datasets collected from the open Toronto portal, one can accurately predict what, when, and where different crimes and accidents will be likely to happen in neighborhoods in the Toronto area. This can effectively help the Toronto Police department to automate their task force scheduling by delegating police patrols efficiently; sending out the right amount of help to people at the right time, and most importantly, taking the active precautions to prevent crimes from happening. 


Datasets Overview 
----------------------------
1. Toronto Police MCI (Major Crime Indicator): This dataset includes all Major Crime Indicators (MCI) 2014 to 2019 occurrences by reported date and related offences. The Major Crime Indicators categories are Assault, Break and Enter, Auto Theft, Robbery and Theft Over (Excludes Sexual Assaults). 

   https://open.toronto.ca/dataset/major-crime-indicators/
   
2. Toronto Police KSI (Killed/Seriously Injured): This dataset includes all traffic collisions events where a person was either Killed or Seriously Injured (KSI) from 2006 – 2019.

   https://open.toronto.ca/dataset/motor-vehicle-collisions-involving-killed-or-seriously-injured-persons/

3. Toronto Neighborhoods: Boundaries of City of Toronto Neighborhoods. 

   https://open.toronto.ca/dataset/neighbourhoods/


Data Cleaning 
-------------
After importing the MSI and KSI datasets, I saved it as a Pandas Dataframe and checked the various stats of the datasets, then started cleaning them. 

I made the following changes on the MSI dataset: 
* remove missing values 
* filter only the important columns that would be used 
* change datatype 
* keep only data started from 2014
* group and count each crime by neighborhood, division, type, each day, each hour
* sort the data by dates and hours 

I also made the following changes on the KSI dataset: 
* filter only the important columns that would be used
* rename some of the columns to the same names as the MSI dataset for convenience 
* change datatype
* keep only data started from 2014
* group and count each accidents by neighborhood, division, type, each day, each hour
* sort the data by dates and hours 

Then, I concatenated the two cleaned dataframes together into a new dataframe, and output it as a csv file for later use. 


Exploratory Data Analysis 
-------------------------
To continue exploring the data, I started with aggregating each type of the KSI accidents by year and visualizing the percentage of each KSI accident among all types in pie charts by year. I also wanted to see the general trend of the number of each type of KSI collision in Toronto over the years, therefore setting up histogram as well. Same step for the MSI dataset. 

Below are some highlights from the visualization: 

[![Screen-Shot-2021-10-02-at-2-44-15-PM.png](https://i.postimg.cc/0j7t7Vpg/Screen-Shot-2021-10-02-at-2-44-15-PM.png)](https://postimg.cc/Fksg4gWG)

[![Screen-Shot-2021-10-02-at-3-08-10-PM.png](https://i.postimg.cc/Qd6wSnW1/Screen-Shot-2021-10-02-at-3-08-10-PM.png)](https://postimg.cc/xXz61551)

Moreover, it'd be useful to look closely on how the numbers of different crimes and collisions in Toronto neighborhood have changed over the years - the trends from 2014 to 2019. 

Some visualization examples are as below: 

[![Screen-Shot-2021-10-02-at-8-44-27-PM.png](https://i.postimg.cc/7P7cQkC3/Screen-Shot-2021-10-02-at-8-44-27-PM.png)](https://postimg.cc/qgJjzWxq)

Investigating how the numbers of different crimes and collisions in Toronto change during different time of the day can be helpful too. 

[![Screen-Shot-2021-10-02-at-8-55-05-PM.png](https://i.postimg.cc/Jnyg8y1n/Screen-Shot-2021-10-02-at-8-55-05-PM.png)](https://postimg.cc/Fd4GgHk5)

Crime and collision rates are also different among Toronto neighborhoods. 

[![Screen-Shot-2021-10-02-at-8-59-59-PM.png](https://i.postimg.cc/QdJVgCZH/Screen-Shot-2021-10-02-at-8-59-59-PM.png)](https://postimg.cc/nXMHpHzJ)



Model Building & Evaluation 
---------------------------
From the EDA, it is clear that the two factors that affect crime and collision occurences are time of a day and geographical regions of the neighborhoods. 

1. Model 1: Clustering (3 clusters give optimal results) 

I applied an unsupervised learning model - clustering, intending to help the Toronto police clearly identify all neighborhoods in Toronto based on the risk level cluster they belong to in different time periods of a day. The clustering method is effective in this case as it directs the police department's focus to the high risk regions so they can allocate more task force in those areas. Through evaluation, the cluster of 3 is suitable in this case.

[![Screen-Shot-2021-10-02-at-9-24-13-PM.png](https://i.postimg.cc/3wm2mYJR/Screen-Shot-2021-10-02-at-9-24-13-PM.png)](https://postimg.cc/7CPCyvz8)

2. Model 2: Time-Series Forecasting (SARIMA Model)

Besides helping police recognize the high risks neighborhoods, using time-series forecasting can help the police department to forecast the numbers of crimes and collisions that will happen in the future. 

Below is an example of the time-series forecasting for Assault, Breaking & Entering cases by month. 

[![Screen-Shot-2021-10-02-at-10-11-34-PM.png](https://i.postimg.cc/D0qdqGQ8/Screen-Shot-2021-10-02-at-10-11-34-PM.png)](https://postimg.cc/wRjmz7Gp)

[![Screen-Shot-2021-10-02-at-9-41-04-PM.png](https://i.postimg.cc/xdsqk1WM/Screen-Shot-2021-10-02-at-9-41-04-PM.png)](https://postimg.cc/5HCfTb92)
