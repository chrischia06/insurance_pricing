# ==================== WHAT TO FILL OUT IN THIS FILE ===========================
"""
There are 3 sections that should be filled out in this file

1. Package imports below
2. The PricingModel class
3. The load function

There are also three other sections at the end of the file that can
safely be ignored. These are:

    - Probability calibration. An optional step to ensure
      probabilities predicted by your model are calibrated
      (see docstring)

    - Consistency check to make sure the code that you submit is the
      same as your trained model.

    - A main section that runs this file and produces the the trained model file
      This also checks whether your load_model function works properly
"""


# ========================== 1. PACKAGE IMPORTS ================================
# Include your package imports here

import hashlib
import numpy as np
import pandas as pd
import pickle

from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import ComplementNB 
# from sklearn.linear_model import LogisticRegressionCV, LinearRegression
# from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

# ========================= 2. THE PRICING MODEL ===============================
class PricingModel():
    """
    This is the PricingModel template. You are allowed to:
    1. Add methods
    2. Add init variables
    3. Fill out code to train the model

    You must ensure that the predict_premium method works!
    """

    def __init__(self,):
        
        # =============================================================
        # This line ensures that your file (pricing_model.py) and the saved
        # model are consistent and can be used together.
        self._init_consistency_check()  # DO NOT REMOVE
        self.booleanFeatures1 = ['pol_payd','Professional','WorkPrivate','M','order_pol_coverage','drv_drv2','diesel']#+['diesel','tourism','pol_bonus2',Retired','Maxi', 'Median1', 'Median2', 'Mini'] # omitted - + ['drv_drv2','vh_make','vh_model','canton_code','commune_code',city_district_code','regional_department_code','Biannual', 'Monthly','Quarterly']
        self.numericFeatures1 = ['pol_duration', 'vh_din', 'pol_bonus', 'population', 'pol_sit_duration', 'vh_value', 'vh_sale_begin',  'vh_cyl','vh_sale_end','vh_age', 'vh_speed', 'drv_age1', 'vh_weight','town_mean_altitude'] #['drv_age_lic1','order_pol_coverage','town_surface_area','town_mean_altitude']

        self.gbc = GradientBoostingClassifier(
            learning_rate=0.1,
            n_estimators=500,
            max_depth=6,
            n_iter_no_change=100,
            max_features='auto',
            subsample=1,
            validation_fraction=0.2)

        self.gbr = GradientBoostingRegressor(n_estimators=500,max_depth=5,loss='ls',n_iter_no_change=100)
        
    def _preprocessor(self, X_raw):
        """

        This function prepares the features of the data for training,
        evaluation, and prediction.

        Parameters
        ----------
        X_raw : Pandas dataframe
            This is the raw data features excluding claims information 

        Returns
        -------
        X: Pandas DataFrame
            A clean data set that is used for training and prediction.
        """
        # =============================================================
        # YOUR CODE HERE
        X_raw2 = X_raw.copy()
        #fill na for city data - 395 missing entries - categorical and numerical, for now assume they come from the most frequent town
        X_raw2['commune_code'] = X_raw2['commune_code'].fillna(X_raw2['commune_code'].mode()[0])
        X_raw2['canton_code'] = X_raw2['canton_code'].fillna(X_raw2['canton_code'].mode()[0])
        X_raw2['city_district_code'] = X_raw2['city_district_code'].fillna(X_raw2['city_district_code'].mode()[0])
        X_raw2['regional_department_code'] = X_raw2['regional_department_code'].fillna(X_raw2['regional_department_code'].mode()[0])
        X_raw2['population'] = X_raw2['population'].fillna(X_raw2['population'].mode()[0])
        X_raw2['town_mean_altitude'] = X_raw2['town_mean_altitude'].fillna(X_raw2['town_mean_altitude'].mode()[0])
        X_raw2['town_surface_area'] = X_raw2['town_surface_area'].fillna(X_raw2['town_surface_area'].mode()[0])
    #   

        
        
    #     #impute or remove illogical values
        X_raw2.loc[X_raw2['vh_weight']==0,'vh_weight'] = X_raw2['vh_weight'].median() # - 2959 missing entries - imputation doesnt give a better correlation
        # X_raw2 = X_raw2.loc[X_raw2['drv_age1']>=X_raw2['drv_age_lic1']] # 32 missing entires; dropped
        # X_raw2 = X_raw2.loc[X_raw2['vh_cyl']>0] #3 missing entries
        X_raw2.loc[X_raw2['vh_value']==0,'vh_value'] = 18659 #from data exploration

        

    #     #one hot encode categorical features; 
        X_raw2 = pd.concat([X_raw2,pd.get_dummies(X_raw2['pol_coverage'])],axis=1)
        X_raw2.loc[X_raw2['pol_usage']=='AllTrips','pol_usage'] = 'Professional' #- only 77 AllTrips; from data description it is similar to professional
        X_raw2 = pd.concat([X_raw2,pd.get_dummies(X_raw2['pol_usage'])],axis=1)
        X_raw2 = pd.concat([X_raw2,pd.get_dummies(X_raw2['pol_pay_freq'])],axis=1)

    #     #binarize features
        X_raw2['pol_payd'] = (X_raw2['pol_payd'] == 'Yes') * 1 # Yes/ No
        X_raw2['drv_drv2'] = (X_raw2['drv_drv2'] == 'Yes') * 1 # Yes / No
        X_raw2['tourism'] = (X_raw2['vh_type'] == 'Tourism') * 1 #tourism or commerical
        X_raw2['diesel'] = (X_raw2['vh_fuel'] == 'Diesel') * 1 #also hybrids but very low representation - 62 hybrids
        X_raw2['M'] = (X_raw2['drv_sex1'] == 'M') * 1 #Male / Female
        
    #     #ordinally encode policy coverage
        order = {'Mini':1,'Median2':2,'Median1':3,'Maxi':4}
        X_raw2['order_pol_coverage'] = X_raw2['pol_coverage'].apply(lambda x : order[x]/4)

        unwantedFeatures= ['id_policy','pol_coverage','pol_pay_freq','pol_usage','pol_insee_code','drv_sex1',
                   'drv_age2','drv_sex2','drv_age_lic2','vh_fuel','vh_make','vh_model','vh_type']+['commune_code',
       'canton_code', 'city_district_code', 'regional_department_code']+['Yearly']
    
        
        X_raw2 = X_raw2.drop(unwantedFeatures,axis=1)
        return X_raw2[self.numericFeatures1+self.booleanFeatures1]


    def fit(self, X_raw, y_made_claim, y_claims_amount):
        """
        Here you will use the fit function for your pricing model.

        Parameters
        ----------
        X_raw : Pandas DataFrame
            This is the raw data features excluding claims information 
        y_made_claim : Pandas DataFrame
            A one dimensional binary array indicating the presence of accidents
        y_claims_amount: Pandas DataFrame
            A one dimensional array which records the severity of claims (this is
            zero where y_made_claim is zero).

        """

        # YOUR CODE HERE
        
        # Remember to include a line similar to the one below
        X_clean = self._preprocessor(X_raw)
        # X_train, X_test,y_train,y_test = train_test_split(X_clean,y_made_claim,test_size=0.5,stratify=y_made_claim,shuffle=True)



        sample_weights = np.zeros(len(y_made_claim))
        sample_weights[y_made_claim == 0] = 1
        sample_weights[y_made_claim == 1] = 8


        self.gbc.fit(X_clean,y_made_claim,sample_weight=sample_weights)
        X_clean['preds'] = self.gbc.predict_proba(X_clean)[:,1]
        self.gbr.fit(X_clean,y_claims_amount)
        return None


    def predict_premium(self, X_raw):
        """Predicts premiums based on the pricing model.

        Parameters
        ----------
        X_raw : Pandas DataFrame
            This is the raw data features excluding claims information

        Returns
        -------
        Pandas DataFrame
            A one dimensional array of the same length as the input with
            values corresponding to the offered premium prices
        """
        # =============================================================
        # You can include a pricing strategy here
        # For example you could scale all your prices down by a factor

        # YOUR CODE HERE

        # Remember to include a line similar to the one below
        X_clean = self._preprocessor(X_raw)
        X_clean['preds'] = self.gbc.predict_proba(X_clean)[:,1]
        X_clean['premiums'] = self.gbr.predict(X_clean)
        return X_clean['premiums'].apply(lambda x : max(100,x))
        
        


    def save_model(self):
        """
        Saves a trained model to pricing_model.p.
        """

        # =============================================================
        # Default : pickle the trained model. Change this (and the load
        # function, below) only if the library you used does not support
        # pickling.
        with open('pricing_model.p', 'wb') as target:
            pickle.dump(self, target)


    def _init_consistency_check(self):
        """
        INTERNAL METHOD: DO NOT CHANGE.
        Ensures that the saved object is consistent with the file.
        This is done by saving a hash of the module file (pricing_model.py) as
        part of the object.
        For this to work, make sure your source code is named pricing_model.py.
        """
        try:
            with open('pricing_model.py', 'r') as ff:
                code = ff.read()
            m = hashlib.sha256()
            m.update(code.encode())
            self._source_hash = m.hexdigest()
        except Exception as err:
            print('There was an error when saving the consistency check: '
                  '%s (your model will still work).' % err)




# =========================== 3. LOAD FUNCTION =================================
def load_trained_model(filename = 'pricing_model.p'):
    """
    Include code that works in tandem with the PricingModel.save_model() method. 

    This function cannot take any parameters and must return a PricingModel object
    that is trained. 

    By default, this uses pickle, and is compatible with the default implementation
    of PricingGame.save_model. Change this only if your model does not support
    pickling (can happen with some libraries).
    """
    with open(filename, 'rb') as model:
        pricingmodel = pickle.load(model)
    return pricingmodel



# ========================= OPTIONAL CALIBRATION ===============================
def fit_and_calibrate_classifier(classifier, X, y):
    """
    Note:  This functions performs probability calibration
    This is an optional tool for you to use, it calibrates the probabilities from 
    your model if need be. 

    For more information see:
    https://scikit-learn.org/stable/modules/calibration.html 
    """
    X_train, X_cal, y_train, y_cal = train_test_split(
        X, y, train_size=0.85, random_state=0)
    classifier = classifier.fit(X_train, y_train)

    # This line does the calibration for you
    calibrated_classifier = CalibratedClassifierCV(
        classifier, method='sigmoid', cv='prefit').fit(X_cal, y_cal)
    return calibrated_classifier



# ==============================================================================

def check_consistency(trained_model, filename):
    """Returns True if the source file is consistent with the trained model."""
    # First, check that the model supports consistency checking (has _source_hash).
    if not hasattr(trained_model, '_source_hash'):
        return True  # No check was done (so we assume it's all fine).
    trained_source_hash = trained_model._source_hash
    with open(filename, 'r') as ff:
        code = ff.read()
    m = hashlib.sha256()
    m.update(code.encode())
    true_source_hash = m.hexdigest()
    return trained_source_hash == true_source_hash




# ============================ MAIN FUNCTION ================================
# Please do not write any executing code outside of the  __main__ safeguard.
# By default, this code trains your model (using training_data.csv) and saves
# it to pricing_model.p, then checks the consistency of the saved model and
# the pickle file.


if __name__ == '__main__':

    # Load the training data
    training_X_raw = pd.read_csv('training_data.csv')
    y_claims_amount = training_X_raw['claim_amount']
    y_made_claim = training_X_raw['made_claim']
    X_train = training_X_raw.drop(columns=['made_claim', 'claim_amount'])

    # Instantiate the pricing model and fit it
    my_pricing_model = PricingModel()
    my_pricing_model.fit(X_train, y_made_claim, y_claims_amount)

    # Save and load the pricing model
    my_pricing_model.save_model()
    loaded_model = load_trained_model()

    # Generate prices from the loaded model and the instantiated model
    predictions1 = my_pricing_model.predict_premium(X_train)
    predictions2 = loaded_model.predict_premium(X_train)

    # ensure that the prices are the same
    assert np.array_equal(predictions1, predictions2)