# StackedGeneralizer
## Automated Stacked Generalization 
Stacked Generalization is a ensemble learning algorithm that combines the results of base models by training a higher-level learner on the lower level outputs. This idea was first introduced by Wolpert (1992) with the basic idea as follow:

1. Split the training set into two disjoint sets.
2. Train several base learners on the first part.
3. Test the base learners on the second part.
4. Using the predictions from 3) as the inputs, and the correct responses as the outputs, train a higher level learner.

Recent interest in model stacking has grown lately as a result of widespread use and success in Kaggle competitions, with winners often combining [over 30 models](https://www.kaggle.com/c/otto-group-product-classification-challenge/discussion/14335#184498) to great sucess. This script simplifies the model selection process by allowing users to choose any number of models, number of folds used for cross-validation, the higher-level stacking model, and data stratification. The automated stacker also accomodates many different forms of machine learning tasks such as classification, regression, and image and text analysis. Additional base models not included here can also be used as long as they support `.fit()` and `.predict()` or `.predict_proba()` methods. 

## Brief Description
This script 


