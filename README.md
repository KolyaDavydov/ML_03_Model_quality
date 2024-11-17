# Model quality estimation

Summary: This project is about different validation techniques; we will discuss how to measure models quality correctly and avoid leaks. Also, we consider a few ways for model hyperparameters optimization and various methods for feature selection.

## Contents

1. [Chapter I. Preamble](#chapter-i-preamble)
2. [Chapter II. Introduction](#chapter-ii-introduction) \
    2.1. [One fold validation](#one-fold-validation) \
    2.2. [Cross-validation (N folds validation)](#cross-validation-n-folds-validation) \
    2.3. [Hyperparameters optimization](#hyperparameters-optimization) \
    2.4. [Feature selection](#feature-selection)
3. [Chapter III. Goal](#chapter-iii-goal) 
4. [Chapter IV. Instructions](#chapter-iv-instructions)
5. [Chapter V. Task](#chapter-v-task)

## Chapter I. Preamble

In previous projects we considered a lot of different examples of machine learning applications and dived into 
linear regression model creation. But we always had train and test samples that were prepared. 
In this project we will discuss how to split out the dataset into parts to fit models correctly.

Validation process is probably one of the most difficult and important parts in the modeling pipeline. 
Look at a few examples: let's imagine that you are a huge food retail company. And from the beginning of summer, 
you have a task to forecast ice cream sales in the next 3 months. You collected data from the last half of the year. 
You use the last three available months for the test set. But 3 summer months that we currently cant observe, 
3 spring months and 3 winter months from available data will have different natures. Everyone knows that sales of 
ice cream in cold months decrease.

Let’s look at a more complex example. We need to predict credit defaults in a bank. It is a widely common task in practice. 
If we would group our train and test samples by an identifier of the credit request in this task, we will make a great mistake. 
Why? One client could ask the bank for credit more than one time. 
And it could happen that a customer credit from 2020 will be in the train part, while credit of the same customer 
from 2018 will be in the test part. This means that the fitted model will already know that the client returned the 
loan from 2018. Of course this knowledge won’t be directly incorporated into the model, but this information is highly 
likely to be stored in features. This will let the model learn misleading patterns and results in underfitting. 
This situation is usually called an information leakage.

And this is not all. Do you remember that in the previous project we select special features for our models? 
What if we want to try different possible subsets and choose the best one that does not contain redundant features. 
Or you might also notice that when we add regularization to the loss it contains the weight  that we multiply on. 
This weight also influences the model's performance. So how should we optimize it? Spoiler, for such needs we should 
also split our dataset into the validation part, i.e. split into train, valid and test parts.

Thus the main goal for the validation process is to find the best way to enclose the training and testing process 
in such a way that it won’t add any mistake to the model while it will work in production. 
Let’s get through validation techniques.

## Chapter II. Introduction

### One fold validation

It is a clear idea to split our data into two parts – training and testing set. But there are a lot of ways to make it. 
The first and simplest way is a random split by some identifier with a fixed ratio of train/test set. 
For instance, split randomly by index of sample record or by identifier of the user that corresponds to a single sample 
in our dataset. It is widely used method for those cases when you have a lot of data and **data has no time relationship**. 
In this case the test part is often called out-of-fold. Fold here is the synonym to “part of the dataset”. 
So “out-of-fold” means that we check performance on the samples that were out of the part that we use for training.

Second way to split the dataset into 2 samples is to sort our data by time or date and take some last period as a test. 
Be careful with the example above. Obviously, this method could be used when you have time relationships in data. 
In this case our split methodology called out-of-time.

In practice, two folds – train and test are not enough. Later in this chapter we consider modeling pipeline parts 
like feature selection and hyperparameters optimization which require special fold for model quality estimation 
and this set is called validation set. It could be a fold that we create from the train 
using out-of-fold or out-of-time strategy. Important, that:
* on the train part of the dataset we train our model
* on the validation part of the dataset we measure quality of trained model and tune its performance varying different condition of preprocessing data or varying hyperparameters of the model
* on the test part of the dataset we measure the final quality of our model to understand the real profit of our model

So, you cannot use test data in modeling pipeline except for the final metrics measure.

Below we visualize the splitting process into train, validation and test. 

![Classic approach](misc/images/classic_approach.png)

Classic approach

![Out-of-time for test part](misc/images/out_of_time_for_test_part.png)

Out-of-time for test part

![Out-of-time both for test and valid parts](misc/images/out_of_time_both_and_valid_parts.png)

Out-of-time both for test and valid parts

Source: https://muse.union.edu/dvorakt/train-validate-and-test-with-time-series/

As we see above, we could combine these methods. But what if we have not so much data for modeling, how to avoid overfitting?

### Cross-validation (N folds validation)

In practice we could find a lot of problems where we cannot collect much data for model training, 
for example medical problems, where data collection is expensive and complex. 
For these cases we could use a cross validation scheme. First of all we split our data into N folds. 
Secondly we take the first fold and use it as the test part, while the other folds we are using to train our model. 
Then we repeat this operation for the next fold and so on. In the end we need to collect metrics from all folds 
and take a mean to evaluate the model performance. Most common number of folds are in the range from 3 till 10. 
Look at the figure below to get deeper understanding:

![Cross-validation](misc/images/grid_search_cross_validation.png)

Source: https://scikit-learn.org/stable/modules/cross_validation.html

There is a special case for cross-validation called leave-one-out validation scheme. 
It will be a task for you – find definition for this scheme and give limitations and strong sides.

In sklearn there are several special methods for cross validation – K-fold, grouped K-fold, stratified K-fold and 
TimeSeriesSplit. Let's dive into them to understand what the differences are.

![K-Fold](misc/images/k_fold.png)

**K-Fold** repeats what we describe above. Blue is the train set and the red one is the test. 
To get the performance we train and evaluate our model on 4 different splits and then take the average score. 
Alternatively, for every red piece we can remember the prediction of the corresponding model. 
If we combine these predictions together we get the vector that is called out-of-fold predictions. 
Thus we can compute our performance metric passing to the function out-of-fold predictions and true labels. 
Why do we mention this alternative way? Because out-of-fold might be useful for improving models performance but 
this theme is out of project scope. If you wish to dive deeper read about stacking.

K-fold method has a strong weakness when data has observations of one common group. 
Group here might be any important property that you believe data should be splitted together. 
It could be observations of one client in different times, or id of different airplanes where you need to detect breakages. 
In these cases, we need to split train test samples into parts that way that one client/airplane will be only 
in train or test sample – it couldn’t be intersections for clients id in train and test sets. 
This method is called **group k-fold**. As previously we split our sample into k folds, but we also grouped 
it by special parameter. Below you can see visualization for group k-fold by “group” column.

![Group-K-Fold](misc/images/group_k_fold.png)

There are few more interesting and important branches for cross validation schemes. 
These are called **Stratified K-fold** and **Stratified Group K-fold**. 
It is a task for you, give a few examples where we need stratification of target variables through the folds. 
Give us strong and weak sides of these methods.

Now we consider the last cross validation scheme, when data has a time relationship. 
It is called Time Series Split. First of all we need to sort our data by date or specified timeline. 
Define k – it will be the number of folds.

![TimeSeriesSplit](misc/images/time_series_split.png)

For the first model we take 1/k of data for the train sample and next 1/k for the test sample. 
For the second model we extend the train dataset to 2/k of data and move the sliding window to next 1/k for the test. 
And so on. In this method we train k-1 models instead of k models.

### Hyperparameters optimization

In this part we would discuss hyperparameters optimization – it is the process of finding the best combination 
of model parameters for better performance and less overfitting. There are 2 types of model parameters – internal –  
model optimizing these parameters by itself during fitting, and external – that are not updating during fitting 
(we don’t update it with gradient or somehow else). Such external params are called hyperparameters and here we 
will talk only about them. Try to give a few examples for both types of parameters.

Hyperparameters optimization is a loop process. You change one or several model params, 
fit the model on the train set and measure quality on validations set and if metrics increase you move on in 
that direction and in the opposite case you try to change it. 

Sometimes clear logic helps you to choose suitable hyperparameters to optimize. 
For example, let's imagine that we have polynomial regression as a base algorithm. 
And we have a huge gap between metrics on train and validation set. So draw a conclusion that our model is overfitted. 
Clear logic suggests us to decrease the degree of polynomial features – number of those is hyperparameter in this case. 
But how to find an optimal set of 5 or 10 independent hyperparameters.

Unfortunately, there is almost nothing better than to check all meaningful combinations of hyperparameters. 
This method is called Grid Search. But it will take a lot of time. 
If we have a limited amount of time we can use Randomized Grid Search. To understand how they work is a part of your task.

But both of these approaches have a weakness - they do not consider relationships in parameters. 
If we fit 3 models with the same degree of polynomial features varying other hyperparameters and get poor performance? 
How likely we should try another degree? The idea that solves this is applied in the Bayes optimization. 
There are two libraries in python that implement the solution: hyperopt and optuna (optuna seems to be better). 
To explain what math is under the hood of this approach is also part of your task.

### Feature selection

Next important step in the modeling process that can also be considered as hyperparameter optimization is feature selection. 
Often, we have thousands of features from raw data sources, and we could also generate a huge amount of them. 
It is an obvious question how to find more important features which have signal and remove noisy and trash columns. 
As a result we can not only speed up the model but also increase the performance. But how to make this?

The same as with hyperparameters we could brute force all possible combinations of features and find optimum, 
but it will take too much time. Luckily compared to hyperparameters tuning there are a lot of approaches to how 
to select features. To understand them all is better to use some classification:

![Feature Selection Methods](misc/images/feature_selection_methods.png)

Source: https://neptune.ai/blog/feature-selection-methods (this list is not full and classification might be not so perfect)

Division between supervised and unsupervised is the same as in machine learning tasks. 
To understand the difference between wrappers, filters and embedded is your task in this project. 
Please be sure to get acquainted with:
* All unsupervised techniques
* All wrappers methods
* Filters
  * Pearson
  * Chi2
* Embedded 
  * Lasso
  * Ridge
* Next methods we would classify in somewhere between wrappers and filters and they are not displayed on the figure above. But these methods recommended themself a lot
  * **permutation importance**
  * **shap** - https://shap.readthedocs.io/en/latest/

The last thing that we want to note before the practice is that both hyperparameters optimization 
and feature selection can be combined together with cross validation. 
It will help to make these processes fair and do not let models overfit.

## Chapter III. Goal

The goal of this task is to get a deep understanding of the validation’s schemes, hyperparameters optimization and feature selection. 

## Chapter IV. Instructions

* This project will only be evaluated by humans. You are free to organize and name
your files as you desire.
* Here and further we use Python 3 as the only correct version of Python.
* For training deep learning algorithms you can try [Google Colab](https://colab.research.google.com). It offers kernels
(Runtime) with GPU for free that is faster than CPU for such tasks.
* The norm is not applied to this project. Nevertheless, you are asked to be clear
and structured in the conception of your source code.
* Store the datasets in the subfolder data

## Chapter V. Task

We will continue our practice with a problem from Kaggle.com. 
In this chapter we will implement all the validation’s schemes, few methods for hyperparameters tune and 
feature selection methods that were described upstairs. Measure quality metrics on train and test samples. 
Will detect overfitted models and regularize these. And dive more deeply with native model estimation and comparison.
1. Answer to the questions from introduction
   1. What is leave-one-out? Provide limitations and strong sides
   2. How Grid Search, Randomized Grid Search and Bayesian optimization works?
   3. Explain the classification of feature selection methods. Explain how Pearson and Chi2 works. Explain how Lasso works. Explain what is permutation importance. Get acquainted with SHAP
2. Introduction  - make all preprocess staff from the previous lesson
   1. Read all data.
   2. Preprocess “interest level” feature.
   3. Create features:  'Elevator', 'HardwoodFloors', 'CatsAllowed', 'DogsAllowed', 'Doorman', 'Dishwasher', 'NoFee', 'LaundryinBuilding', 'FitnessCenter', 'Pre-War', 'LaundryinUnit', 'RoofDeck', 'OutdoorSpace', 'DiningRoom', 'HighSpeedInternet', 'Balcony', 'SwimmingPool', 'LaundryInBuilding', 'NewConstruction', 'Terrace'
3. Implement next methods:
   1. Split data into 2 parts randomly with parameter test_size (ratio from 0 to 1), return train and test samples 
   2. Split data into 3 parts randomly with parameters validation_size and  test_size, return train, validation and test samples 
   3. Split data into 2 parts with parameter date_split, return train and test samples splitted by date_split param
   4. Split data into 3 parts with parameters validation_date and  test_date, return train, validation and test samples  splitted by input params
4. Implement next cross validation methods:
   1. K-fold, where k is input parameter, return list of train and test indexes 
   2. Grouped K-fold, where k and group_field are input parameter, return list of train and test indexes 
   3. Stratified K-fold, where k and stratify_field are input parameter, return list of train and test indexes
   4. Time series split, where k and date_field are input parameter, return list of train and test indexes 
5. Cross-validation comparison
   1. Apply all implemented validation methods from above to our dataset.
   2. Apply the corresponding methods from sklearn.
   3. Compare the resulting distributions of features for the training part of the dataset between sklearn and your own implementation.
   4. Compare all validation schemes. Choose the best one. Explain your choice.
6. Feature selection
   1. Fit lasso regression model with normalized features. Use your method for split samples into 3 parts by field created with ratio 60/20/20 – train/validation/test.
   2. Sort features by weight coefficients from model, refit model on top 10 features and compare quality.
   3. Implement method for simple feature selection by nan ration in feature and correlation. Apply this method for the feature set and take top 10 features, refit model and measure quality.
   4. Implement permutation importance method and take top 10 features, refit model and measure quality.
   5. Import shap and also refit model on top 10 features
   6. Compare the quality of these methods for different sides – speed, metrics and stability.
7. Hyperparameters optimization
   1. Implement grid search and random search methods for alpha and l1_ratio for ElasticNet model from sklearn.
   2. Find the best combination for the model hyperparameters.
   3. Fit resulted model.
   4. Import optuna and configure the same experiment with ElasticNet.
   5. Estimate metrics and compare these approaches.
   6. Run optuna on one of the cross validation schemes

### Submission

Save your code in python JupyterNotebook. Your peer will load it and compare with basic solution. 
Your code should contain the answers to all mandatory questions. Task ‘additional’ is on your own.
