###############################################################################
# NOTE: Please make sure that all of the files related to the pricing games are
# in one folder.
#
# This file has 4 sections that you should fill out:
#    
#    Section 1: Place all the packages that you want loaded or installed in a vector
#    Section 2: Place code that transform the input dataset, if necessary, to create new features
#    Section 3: Place code that trains your models
#    Section 4: Place code that uses your trained model to make a premium prediction
#
################################## SECTION 1 ##################################
# In this section you load of the R packages that you might need
#

# YOUR CODE HERE ------------------------------------------------------
# Edit the 'packages' vector to contain the names of all packages you need
packages <- c("dplyr", "caret")

# The code below loads the required package

lapply(packages, library, character.only = TRUE)

################################## SECTION 2 ##################################
# In this section you should put all the code that you need to transform the 
# input dataset, if necessary, to create new features

preprocess <- function(data){
  # This function creates all necessary features
  # This function returns a 'data.frame' object  
  
  # YOUR CODE HERE ------------------------------------------------------

  # The output is the data frame 
  # return(data) 
}  

################################## SECTION 3 ##################################
# In this section you should put all the code that you will need to train your
# model


train <- function(data){
  # This function trains your models and returns the trained model.
  
  # First, create all new features, if necessary
  
  # data <- preprocess(data)
  
  # YOUR CODE HERE ------------------------------------------------------
  
  
  # ---------------------------------------------------------------------
  # The result trained_model is something that you will save in the next section
  # return(trained_model)
}

################################## SECTION 4 ##################################
# In this section you should edit the predict_premium function below so that it
# works with any dataset with a similar structure as the training data
# NOTE: remember that the data will not include ANY claim information
# therefore, the last two columns will not be included

predict_premium = function(data, trained_model){
  
  #   This takes in a single row of the data and outputs a premium price.
  #
  #   You should edit this function so that it works with your trained_model object
  #
  #   Inputs:
  #
  #       - data: This is data.frame for which premiums should be computed
  #       - trained_model: This is the result of the train() function 
  #
  #   Output:
  #
  #       - premium: This is a vector of prices for all individuals in 'data'
  
  
  # return(pure_premium)
  # ---------------------------------------------------------------------
}

############################## SAVING THE MODEL ###############################
# This section trains your model, saves it into a file labeled "trained_model.RData",

# DO NOT ALTER BEYOND THIS POINT ----------------------------------------------


train_and_save = function(){
  data = read.csv('training_data.csv')
  model = train(data)
  save(model, file='trained_model.RData')
}

# This trains and saves your model when you run the script
train_and_save()
