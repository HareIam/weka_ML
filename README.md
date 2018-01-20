# Feature selection using rank method

Open the project in eclipse

Following is my skeleton code to apply different ‘Ranking’ feature selection methods with different classifiers. Kindly remind it just one possible way to use feature selection.

Sample_size=k;
Feature_size=f;

For each classifier as i: 
   For each ranking method as j:
       For number of top features I selected as n: # n is between 1 and f (could set the interval if f is too large)
           For each training data and test data in Leave-One-Out CV as TrainData and TestData: # 10-fold should be considered if k is too large
               FeaturesRank =Rank Features in TrainData by j;  # rank features
               Top_features= FeaturesRank(1:n);  #select top n features after ranking
               Classifier=i; # set classifier
               Model=Classifier.train(TrainData(Top_features)); # use feature subset to train classifier
               Label= Model.Predict(TestData); # Predict the Test dataset label
               Save(Label);
           End
               Accuracy_i_j_n=correct_predction_for_all_cross_validation/k;  # we also can get confusion matrix, recall, AUC etc. here
       End
    End
End

Finally we can compare all the accuracies, and choose the highest one.
