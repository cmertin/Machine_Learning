About the Data
==============

This data comes from https://archive.ics.uci.edu/ml/datasets/Air+Quality

###Data Set Information:

The dataset contains 9358 instances of hourly averaged responses from an array of 5 metal oxide chemical sensors embedded in an Air Quality Chemical Multisensor Device. The device was located on the field in a significantly polluted area, at road level, within an Italian city. Data were recorded from March 2004 to February 2005 (one year) representing the longest freely available recordings of on field deployed air quality chemical sensor devices responses. Ground Truth hourly averaged concentrations for CO, Non-Metanic Hydrocarbons, Benzene, Total Nitrogen Oxides (NOx) and Nitrogen Dioxide (NO2) and were provided by a co-located reference certified analyzer. Evidences of cross-sensitivities as well as both concept and sensor drifts are present as described in De Vito _et al._, Sens. And Act. B, Vol. 129,2,2008 (citation required) eventually affecting sensors concentration estimation capabilities. Missing values are tagged with -200 value.
This dataset can be used exclusively for research purposes. Commercial purposes are fully excluded.

###Attribute Information:

0. Date (DD/MM/YYYY)
1. Time (HH.MM.SS)
2. True hourly averaged concentration CO in mg/m^3 (reference analyzer)
3. PT08.S1 (tin oxide) hourly averaged sensor response (nominally CO targeted)
4. True hourly averaged overall Non-Metanic HydroCarbons concentration in microg/m^3 (reference analyzer)
5. True hourly averaged Benzene concentration in microg/m^3 (reference analyzer)
6. PT08.S2 (titania) hourly averaged sensor response (nominally NMHC targeted)
7. True hourly averaged NOx concentration in ppb (reference analyzer)
8. PT08.S3 (tungsten oxide) hourly averaged sensor response (nominally NOx targeted)
9. True hourly averaged NO2 concentration in microg/m^3 (reference analyzer)
10. PT08.S4 (tungsten oxide) hourly averaged sensor response (nominally NO2 targeted)
11. PT08.S5 (indium oxide) hourly averaged sensor response (nominally O3 targeted)
12. Temperature in °C
13. Relative Humidity (%)
14. AH Absolute Humidity

###Extra features

I decided to create features based on the given Information to hopefully provide
more accurate results. The overall features as of right now are

0. Month
1. Day
2. Year
3. Time (Hour)
4. True hourly averaged concentration CO in mg/m^3 (reference analyzer)
5. PT08.S1 (tin oxide) hourly averaged sensor response (nominally CO targeted)
6. True hourly averaged overall Non-Metanic HydroCarbons concentration in microg/m^3 (reference analyzer)
7. True hourly averaged Benzene concentration in microg/m^3 (reference analyzer)
8. PT08.S2 (titania) hourly averaged sensor response (nominally NMHC targeted)
9. True hourly averaged NOx concentration in ppb (reference analyzer)
10. PT08.S3 (tungsten oxide) hourly averaged sensor response (nominally NOx targeted)
11. True hourly averaged NO2 concentration in microg/m^3 (reference analyzer)
12. PT08.S4 (tungsten oxide) hourly averaged sensor response (nominally NO2 targeted)
13. PT08.S5 (indium oxide) hourly averaged sensor response (nominally O3 targeted)
14. Temperature in °C
15. Relative Humidity (%)
16. AH Absolute Humidity
17. Monday (boolean)
18. Tuesday (boolean)
19. Wednesday (boolean)
20. Thursday (boolean)
21. Friday (boolean)
22. Saturday (boolean)
23. Sunday (boolean)


With this many features, I am worried about over fitting of the data, but as
this is a test, I can always change my model later. I was trying to get the
overall trend of the particulates in the past 24 hours with this technique.

Numbers 17 and 18 are there because there should *most likely* be more pollution
during the weekday than on the weekend thanks to business traffic. I was
considering using a boolean value for each day, but I thought that just
differentiating between the weekday and weekend should be good enough.

This data is stored in `AirQuality_clean.csv`

###Data Manipulation

To clean the data, I parse the original data and I remove any data points that
have more than 2 missing features. From here, I take the remaining data and I
separate it based on those with and without missing features. From here, I
iterate through all the missing features and perform K-Nearest Neighbors Regression
on it, and use the data that had all of the features, with `k = 3`. Then I took
the average of the features and assigned it to the value of the missing feature.
