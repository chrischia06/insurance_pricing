

for [https://pricing-game.com/](https://pricing-game.com/)

The objective is to : 
+ Model insurance claim loss (regression)
+ Moreover, devise a pricing rule, based on the model prediction for insurance claim, which will be used to compete with others on the platform.

## Approach

A general approach to kaggle/data science style problems:

1. Examine literature and existing code that others have done
2. [Exploratory Data Analysis](EDA/Insurance_Pricing_Game_EDA.md) ([Colab Link](https://colab.research.google.com/drive/1pNzkU904Pwm12lPYYH73lVqlCOMMiIED#scrollTo=oh7IdU-KOpBv))
2. Modelling: - Loss Function, model training metric, [Hyperparameter Tuning](#)
3. [Model Evaluation](#) Optimise for final metric pricing strategy

<!-- # Literature

+ https://freakonometrics.github.io/documents/talks/CHARPENTIER-bank-of-england-2017.pdf

+ French Third - party liabilities dataset

+ https://www.openml.org/d/41214

**Case Study: French Motor Third-Party Liability Claims**
+ https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3164764
+ https://scikit-learn.org/stable/modules/model_evaluation.html#mean-tweedie-deviance

+ https://towardsdatascience.com/insurance-risk-pricing-tweedie-approach-1d71207268fc
+ https://core.ac.uk/download/pdf/211518053.pdf

+ A Logistic Regression Based Auto InsuranceRate-Making Model Designed for the InsuranceRate Reform

+ Generalized Linear Models for Non-life Pricing  - Overlooked Facts and Implications

+ https://www.actuaries.org.uk/system/files/documents/pdf/c6-paper.pdf

+ Towards Machine Learning: Alternative Methods for Insurance Pricing – Poisson-Gamma GLM’s, Tweedie GLM’s and Artificial Neural Networks 
+ https://www.actuaries.org.uk/system/files/field/document/F7%20Navarun%20Jain.pdf
+ https://www.actuaries.org.uk/system/files/field/document/F7_Navarun%20Jain_0.pdf

# Other people's code for some ideas

https://gitlab.ethz.ch/stevenb/insurance_pricing_competition

+ the winning solution from the previous year; 
+ Preprocessing: Label + One-hot Encoding
+ 2 Stage Modelling: 
+ Balanced Random Forest to predict Default Probability $P(Y > 0 | X)$ https://imbalanced-learn.org/stable/ensemble.html
+  Gaussian Processes Regression to predict Claim size: $E[Y | Y > 0, X]$


https://github.com/KNurmik/AICrowd-Insurance-Pricing-Game/blob/main/Submission_notebook_(Python).ipynb

+ Ensemble of Random Forest + GBR

https://github.com/bpostance/aicrowd-pricing/blob/main/AIcrowd_submission_v0_0.ipynb
+ XGBoost for Claim Probability, Gamma Regression for Claim Severity, 


https://www.kaggle.com/floser/glm-neural-nets-and-xgboost-for-insurance-pricing

+ Modelling: GLM, DNN, XGBoost
+ for : https://www.kaggle.com/floser/french-motor-claims-datasets-fremtpl2freq


https://github.com/kasaai/explain-ml-pricing
+ Interpretability


https://github.com/anhdanggit/insurance-econometrics
+ GLMs : Gamma, Log-Normal prior

https://github.com/LeoPetrini/XGBoost-in-Insurance-2017

+ XGBoost + Tweedie

https://www.kaggle.com/anmolkumar/vehicle-insurance-eda-lgbm-vs-catboost-85-83


https://www.kaggle.com/c/allstate-claims-severity/discussion/24520
+ Good old Allstate Insurance Claims Severity dataset.

## Timeline

+ Recorded ML experiments - 21/01/2021
+ Hyperparameter tuning for classification. need to decide what to optimise - ROC leads to class weights approx 1 - 24/01/2021
+ The Catboost Classification * Mean(Y >0)) model performed poorly on the competition leaderboard 25/01/2021. Maybe instead of mean, fit a Gamma/Lognormal and take the inverse quantile $F^-1(p)$

### Pricing Competition  Leaderboard

**Week 4**
+ 0 trades made. Pricing is  $\hat{Y} + 1118.752708 = E[Y | X] + E[Y | Y >0]$. This suggests that prices are too high

**Week 5**
+ Negative profit. Pricing is $\hat{p} * E[Y | Y > 0]$ with CatBoost

**Week 6**
+ On the pricing leaderboard, high market share (average 0.67), but massive losses.  This corresponds to RMSE submission 5. The problem is that one-stage regression is a conditional mean model, and in insurance severity the conditional mean tends to be less ?? (mentioned in a paper I think).

**Week 7**
+ Average Loss of -413577.87056237616 in 1000 markets, participation rate around 0.36. Using LightGBM 2 step model. This has worse performance than the previous


