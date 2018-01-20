# Classification based on Weka

Classification tasks using different classifiers and feature ranking methods

Ranking methods:
1. ReliefF Attribute Evaluator
2. Chi-Squared Attribute Evaluator
3. SVM Attribute Evaluator

Classifier:
1. SVM
2. Multilayer Perceptron
3. Logistic
4. Naïve Bayes Multinomial
5. KNN

## Introduction

Following is my skeleton code to apply different Ranking feature selection methods with different classifiers. Kindly remind it just one possible way to use feature selection.

```
Sample_size=k;
Feature_size=f;

For each classifier as i: 
	For each ranking method as j:

		# n is between 1 and f (could set the interval if f is too large)
		For the number of top features I selected as n:   
			# 10-fold should be considered if k is too large 
			For each training data and test data in Leave-One-Out CV as TrainData and TestData:   
				# rank features
				FeaturesRank =Rank Features in TrainData by j;
				#select top n features after ranking
				Top_features= FeaturesRank(1:n);
				# set classifier 
				Classifier=i; 
				# use feature subset to train classifier 
				Model=Classifier.train(TrainData(Top_features)); 
				# Predict the Test dataset label
				Label= Model.Predict(TestData);   
				Save(Label);
			End
			# we also can get confusion matrix, recall, AUC etc. here
			Accuracy_i_j_n=correct_predction_for_all_cross_validation/k;
		End
	End
End
```

Finally, we can compare all the accuracies, and choose the highest one.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

## Prerequisites

Install Eclipse
Import Weka packages (Detail pls see: https://weka.wikispaces.com/Use+WEKA+in+your+Java+code)


## Running the tests

Run the ./src/weka_Main/ClassifierTest.java

## Authors

XU SHIHAO 

##URL
https://github.com/HareIam/weka_ML






