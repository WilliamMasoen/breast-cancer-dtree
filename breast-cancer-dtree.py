# Import required libraries
from sklearn import datasets  # Load datasets from scikit-learn
import matplotlib.pyplot as plt  # For visualizing the decision tree and accuracies
from sklearn import tree  # Decision Tree Classifier
from sklearn.metrics import accuracy_score  # Evaluate accuracy
from sklearn.model_selection import train_test_split, GridSearchCV  # Split data and optimize hyperparameters
from sklearn.datasets import load_breast_cancer  # Specific dataset used in this analysis

# Load the Breast Cancer dataset
X, y = datasets.load_breast_cancer(return_X_y=True)

# Display the number of instances and features in the dataset
data_instances, data_features = X.shape  # Get instances and features from the dataset
print("There are", data_instances, "instances described by", data_features, "features.")

# Split the dataset into training and testing sets (60% training, 40% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Initialize a Decision Tree Classifier with specific hyperparameters
clf = tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=6) 
clf = clf.fit(X_train, y_train)  # Train the classifier on the training data

# Make predictions on the test set
predC = clf.predict(X_test)

# Calculate and print the accuracy of the classifier
score = accuracy_score(y_test, predC)
print('The accuracy of the classifier is', score)

# Visualize the decision tree
plt.figure(figsize=(25, 12))
tree.plot_tree(clf, feature_names=load_breast_cancer().feature_names, 
               class_names=load_breast_cancer().target_names, 
               filled=True, fontsize=12)
plt.title("Decision Tree Visualization")
plt.show()

# Analyze the effect of tree depth on classifier accuracy
trainAccuracy = []
testAccuracy = []
depthOptions = range(1, 16)  # Test depths from 1 to 15

for depth in depthOptions:
    # Train decision tree with varying depths
    cltree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=depth, min_samples_split=6)
    cltree = cltree.fit(X_train, y_train)
    
    # Calculate training and testing accuracy
    y_predTrain = cltree.predict(X_train)
    y_predTest = cltree.predict(X_test)
    trainAccuracy.append(accuracy_score(y_train, y_predTrain))
    testAccuracy.append(accuracy_score(y_test, y_predTest))

# Plot training and testing accuracy against tree depth
plt.plot(depthOptions, trainAccuracy, marker='o', color='red', label='Training Accuracy')
plt.plot(depthOptions, testAccuracy, marker='s', color='blue', label='Test Accuracy')
plt.legend(['Training Accuracy', 'Test Accuracy'])
plt.xlabel('Tree Depth')
plt.ylabel('Classifier Accuracy')
plt.title("Effect of Tree Depth on Accuracy")
plt.show()

"""
Note: The optimal tree depth, based on test accuracy, is approximately 7.
However, selecting hyperparameters using the test set can lead to overfitting.
Instead, cross-validation should be used for hyperparameter selection.
"""

# Use GridSearchCV to find the best hyperparameters
parameters = {'max_depth': list(range(1, 16)), 'min_samples_split': list(range(2, 7))}
clf = GridSearchCV(tree.DecisionTreeClassifier(criterion='entropy'), parameters, cv=3) 
clf = clf.fit(X_train, y_train)
tree_model = clf.best_estimator_

# Display the best hyperparameters
print("The optimal tree has a maximum depth of", tree_model.max_depth, 
      'and requires a minimum of', tree_model.min_samples_split, 'samples to split a node.')

# Visualize the optimal decision tree
plt.figure(figsize=(25, 12))
tree.plot_tree(tree_model, filled=True, fontsize=12)
plt.title("Optimized Decision Tree Visualization")
plt.show()

"""
Explanation: 
The GridSearchCV method finds the best hyperparameters using cross-validation. 
This ensures the model generalizes well on unseen data without overfitting.
"""

# Tenfold Stratified Cross-Validation Explanation
"""
Tenfold cross-validation splits the dataset into 10 equal parts. The model trains on 9 folds and validates on the remaining 1. 
This process repeats 10 times, ensuring each fold is used for validation exactly once. 
Stratified cross-validation ensures each fold maintains the same class distribution as the original dataset.
"""
