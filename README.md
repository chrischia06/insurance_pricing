

for [https://pricing-game.com/](https://pricing-game.com/)

The objective is to : 
+ model insurance claim loss (regression)
+ Using your predicted prices, construct a pricing strategy which will be used to compete with others on the platform

## Analysis

+ [Exploratory Data Analysis](EDA/Insurance_Pricing_Game_EDA.md)
+ Modelling

## Timeline

+ Recorded ML experiments - 21/01/2021
+ Hyperparameter tuning for classification. need to decide what to optimise - ROC leads to class weights approx 1 - 24/01/2021
+ The Catboost Classification * Mean(Y >0)) model performed poorly on the competition leaderboard 25/01/2021. Maybe instead of mean, fit a Gamma/Lognormal and take the inverse quantile $F^-1(p)$
