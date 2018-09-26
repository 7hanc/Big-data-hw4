# Big-data-hw4

# Homework 4 (Build a predictive framework)

> Goal
* Build a predictive framework, which is able to predict "WeatherDelay" of all flights in 2008. This work should use the data from 2003 to 2007 as training including validation and 2008 as testing.
>> Dataset
* Airline on-time performance dataset: http://stat-computing.org/dataexpo/2009/
> Question
* Q1: Explain the predictive framework you designed.
  * Feature: “Month” and “DayofMonth” 
  * Algorithms: **Linear regression**.
* Q2: Explain how method you use to validate your model when training.
  * Use cross-validation to validate my model 
  * The method of cross validation: **K-fold**   
### The answer of Q1 and Q2 -> hw4.py
* Q3: Show the **evaluation results** of validation in training and prediction in testing  by following those evaluation metric:
  * MAE (平均絕對誤差) = ![mae](https://i.imgur.com/fHLGayL.png)
  * RMSE (均方根誤差) = ![rmse](https://i.imgur.com/cAdZnvD.png)
<table>
　<tr>
    <td> </td>
　  <td>average MAE (值越大, 誤差越大)</td>
    <td>average RMSE (值越大, 誤差越大)</td>
　</tr>
 <tr>
    <td>validation in training</td>
　  <td>1.612427</td>
    <td>5.016839</td>
　</tr>
 <tr>
    <td>prediction in testing</td>
　  <td>1.495521</td>
    <td>9.243812</td>
　</tr>
</table>
