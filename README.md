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
To continue exploring the data, I started with aggregating each type of the KSI accidents by year and visualizing the percentage of each KSI accident among all types in pie charts by year. I also wanted to see the general trend of the number of each type of KSI collision in Toronto over the years, therefore setting up a barplot as well. Same step for the MSI dataset. 

Below are some highlights from the visualization: 

[![Screen-Shot-2021-10-02-at-2-44-15-PM.png](https://i.postimg.cc/0j7t7Vpg/Screen-Shot-2021-10-02-at-2-44-15-PM.png)](https://postimg.cc/Fksg4gWG)

[![Screen-Shot-2021-10-02-at-3-08-10-PM.png](https://i.postimg.cc/Qd6wSnW1/Screen-Shot-2021-10-02-at-3-08-10-PM.png)](https://postimg.cc/xXz61551)

[![Screen-Shot-2021-10-02-at-3-07-05-PM.png](https://i.postimg.cc/0QKTzdjm/Screen-Shot-2021-10-02-at-3-07-05-PM.png)](https://postimg.cc/MMxP4ByK)

[![Screen-Shot-2021-10-02-at-3-09-06-PM.png](https://i.postimg.cc/wjBbtcbv/Screen-Shot-2021-10-02-at-3-09-06-PM.png)](https://postimg.cc/2VPxMvYR)
