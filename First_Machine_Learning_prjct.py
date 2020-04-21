import pandas as pd
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'  # Data Loading
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)
print(dataset.shape)  # To know the shape of the data
print(dataset.head(20))  # Displaying the first 20 data
print(dataset.describe())  # Taking a statistical summary
print(dataset.groupby('class').size())  # Class distribution

# Univariate Plots - box and whisker plot

dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
pyplot.show()

dataset.hist()  # Plotting histogram
pyplot.show()


# Multivariate plots
scatter_matrix(dataset)
pyplot.show()

# STEP 1
# Creating a validation dataset
# Splitting the dataset
array = dataset.values
X = array[:, 0:4]
Y = array[:, 4]
X_train, X_valudation, Y_train, Y_valudation = train_test_split(X, Y, test_size=0.2, random_state=1)

# Step 2
# Logistic Regression
# Linear Discriminant analysis
# K-Nearest neighbours
# Classification and regression trees
# Gaussian Naive Bayes
# Support vector machines

models = [('LR', LogisticRegression(solver='liblinear', multi_class='ovr')), ('LDA', LinearDiscriminantAnalysis()),
          ('KNN', KNeighborsClassifier()), ('NB', GaussianNB()), ('SVM', SVC(gamma='auto'))]

# STEP 3
# EVALUATION
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# STEP 4
# Compare our model
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()

# Make Prediction on SVM
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_valudation)

# Evaluate Predictions
print(accuracy_score(Y_valudation, predictions))
print(confusion_matrix(Y_valudation, predictions))
print(classification_report(Y_valudation, predictions))
