# LinearClassifiers_MachineLearning_Python
Linear Classifiers on Iris Dataset with Python_KNN_SVM_RandomForest_GradientTreeBoosting
In this repository I have done Linear Classifiers such as KNN, SVM, Random Forest, GradientTreeBoosting and analysis on Iris Datasets
## IRIS DATASET
The Iris dataset from \sklearn.datasets.load iris". This dataset is a classic and fairly simple benchmark for basic machine learning algorithms. It includes different features of three Iris flower species (setosa, versicolor, virginica).
## Scatter Plot
for better understanding the dataset,I have plotted the pairs plot (scatter plot matrix) of the data. Note that the pairs plot includes the scatter plots of every dimension
versus another dimension.
## KNN
In this file I have Classifies the data using a KNN classfier. Tuned the hyperparameters of the KNN classifier using sklearn functions. Plotted the different validation accuracies against the values of the parameter and select the best hyperparameter to train the model.
1. Firstly, I divide the data into train, validation, and test sets (60%, 20%, 20%)
2. Trained the model with each classifer's default parameters, used the train set and test the model on the test set and stored the accuracy of the model.
3. Then found the best parameters of the classifers, in this case, k for KNN. For this, I used the validation set and by picking different values of K from {1, 5, 10, 15, 20, 25, 30, 35}
## SVM
In this file I have Classified data using a linear SVM classifer and Evaluated the best value for the term C: [0.1, 0.5, 1, 2, 5, 10, 20, 50]. Here I have used 10-fold cross validation. First, randomly divide data into (80%, 20%) portions of train-validation and test sets (with random state=42). Then, apply 10-fold cross validation on the train-validation set. In every fold, 90% of data is used for training and 10% of data for validation. For every C value, a mean accuracy of the folds is found. The best mean accuracy determines the best value for C.
## Tree-based Classifers
Classified the data using three tree-based classifers: Decision Trees, Random Forests and Gradient Tree Boosting. Tuned the hyper-parameters of the classifer using 10-fold cross validation and sklearn functions.

## Following links are useful for solving the above problems
[https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html]

[https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html]

[https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html]

[https://scikit-learn.org/stable/modules/generated/sklearn.model\_selection.GridSearchCV.html]
