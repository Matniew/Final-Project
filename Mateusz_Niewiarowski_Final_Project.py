import pandas as pd
import numpy as np
import csv
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics as m
from sklearn.metrics import confusion_matrix

#Creation of a function Results for a easier way to display the performance indicators.
def Results(true_value, pred) :
    
    results = {"Accuracy :" : m.accuracy_score(true_value,pred),
              "Precision : " : m.precision_score(true_value,pred),
              "Recall : " : m.recall_score(true_value,pred),
              "Cohen's Kappa : " : m.cohen_kappa_score(true_value, pred),
              "F1 : " : m.f1_score(true_value,pred)}
    for key in results:
        print (key, results[key])

data = pd.read_csv('data.csv', encoding = 'ISO-8859-1')

#print(data.head(10))
#print(data.info())

#The code will check if there is any missing values. Isnull will return a boolean value true if there is missing data.
#The double .any() will loop over all the database to a final single boolean. In case it is false there is any missing values.
any_missing_values = data.isnull().any().any()
print(f"Are there any missing values? {any_missing_values}") 

#correlation_matrix = data.corr()
#plt.figure(figsize=(20, 15))
#sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=0, vmax=1,mask=correlation_matrix.abs() <= 0.6)
#plt.show()

#Proportion of bankruptcies
#Data imbalance
Bankrupt = len(data[data["Bankrupt?"] == 1])
Not_bankrupt = len(data[data["Bankrupt?"] == 0])
Proportion_of_bankruptcies = (Bankrupt/Not_bankrupt)*100
print(f"In the database we have {Proportion_of_bankruptcies}% of bankruptcy.")

categories = ['Bankrupt?']
col_A = len(data[data["Bankrupt?"] == 1])
col_B = len(data[data["Bankrupt?"] == 0])

# Plotting a graph to analyse visually the class imbalance
x = np.arange(len(categories))

plt.bar(x - 0.2, col_A, width=0.4, label='Financial distress')
plt.bar(x + 0.2, col_B, width=0.4, label='Financial stable')


#plt.xlabel('Bankrupt?')
plt.ylabel('Count')
plt.title('Class Distribution')
plt.xticks(x, categories)  
plt.legend()
plt.show()

# I separate my independant variable on y axis which is the bankruptcy prediction.
X = data.drop('Bankrupt?', axis=1)
y = data['Bankrupt?']
#Splitting data into test set and training set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#As my data is imbalanced I use smote to rebalance my data.
smote = SMOTE(random_state=42)
X_train_SMOTE, y_train_SMOTE = smote.fit_resample(X_train, y_train)
#I have the same proportion of bankrupt and non bankrupt companies
print("\nClass distribution after SMOTE:")
print(pd.Series(y_train_SMOTE).value_counts())

#This is a set of parameter used by the random forest classifier. 
hyperparameters = {
    'n_estimators': [50, 100, 150],
    'criterion' : ["gini", "entropy", "log_loss"], 
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'random_state': [0, 42, 100],
}

clf = RandomForestClassifier()
#I train my model on SMOTE data in order to catch a trend and learn it.
clf.fit(X_train_SMOTE, y_train_SMOTE)
#Random search is going to search for the best combination of parameters.
random_search = RandomizedSearchCV(clf, param_distributions=hyperparameters, n_iter=10, cv=5, verbose=0, random_state=42, n_jobs=-1, error_score='raise')
RF_Model = random_search.fit(X_train_SMOTE, y_train_SMOTE)
#I do the prediction on my test set
RF_pred = RF_Model.predict(X_test)

Results(y_test, RF_pred)

#Plotting of the confusion matrix
conf_matrix = confusion_matrix(y_test, RF_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Classe 0', 'Classe 1'], yticklabels=['Classe 0', 'Classe 1'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Random Forest Classifier')
plt.show()

#Lets see wich variables are the most important according to random forest classifier.
#There is an embedded feature_importance extraction. Creating a DataFrame for better visualisation.
importance_df = pd.DataFrame(clf.feature_importances_, index=X_train.columns, columns=['importance'])
# Sorting in the descending way
importance_df = importance_df.sort_values('importance', ascending=False)
print(importance_df.head(10))


#Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def Results(true_value, pred) :
    
    results = {"Accuracy :" : m.accuracy_score(true_value,pred),
              "Precision : " : m.precision_score(true_value,pred),
              "Recall : " : m.recall_score(true_value,pred),
              "Cohen's Kappa : " : m.cohen_kappa_score(true_value, pred),
              "F1 : " : m.f1_score(true_value,pred)}
    for key in results:
        print (key, results[key])

data = pd.read_csv('data.csv', encoding = 'ISO-8859-1')

X = data.drop('Bankrupt?', axis=1)
y = data['Bankrupt?']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X_train_SMOTE, y_train_SMOTE = smote.fit_resample(X_train, y_train)

#Using logistic regression I decided to scale my data. All of them are on the same scale of 0-1.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_SMOTE)
X_test_scaled = scaler.transform(X_test)

print("\nDistribution des classes après SMOTE:")
print(pd.Series(y_train_SMOTE).value_counts())

#PCA hypothesis is that the center have to be 0 so we have to standardise
#Unsupervised feature reduction
pca = PCA(n_components=10)
#Alwyas testing my results on untouched test data? For PCA?  
X_test_pca = pca.fit(X_test)

model = LogisticRegression( C= 0.8, solver = 'liblinear')
model.fit(X_train_scaled, y_train_SMOTE)

y_pred = model.predict(X_test_scaled)

Results(y_test, y_pred)


conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['True 0', 'True 1'], yticklabels=['Pred 0', 'Pred 1'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Logistic Regression')
plt.show()

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

#MLP
from imblearn.over_sampling import ADASYN
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

def Results(true_value, pred) :
    
    results = {"Accuracy :" : m.accuracy_score(true_value,pred),
              "Precision : " : m.precision_score(true_value,pred),
              "Recall : " : m.recall_score(true_value,pred),
              "Cohen's Kappa : " : m.cohen_kappa_score(true_value, pred),
              "F1 : " : m.f1_score(true_value,pred)}
    for key in results:
        print (key, results[key])

data = pd.read_csv('data.csv', encoding = 'ISO-8859-1')

X = data.drop('Bankrupt?', axis=1)
y = data['Bankrupt?']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Here I use the ADASYN method for data balancing
adasyn = ADASYN()
X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)

#I use standard scaler to scale all my data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_resampled)
X_test_scaled = scaler.transform(X_test)


model = MLPClassifier(
    hidden_layer_sizes=(150, 150),  
    activation='logistic',  
    solver='sgd',  
    alpha=0.0006,  
    learning_rate='adaptive',  
    max_iter=500,   
)

model.fit(X_train_scaled, y_resampled)
y_pred = model.predict(X_test_scaled)
conf_matrix = confusion_matrix(y_test, y_pred)


plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['True 0', 'True 1'], yticklabels=['Pred 0', 'Pred 1'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Multilayer Perceptron Classifier')
plt.show()

y_prob = model.predict_proba(X_test)[:, 1]

# Ajustement du seuil de probabilité (par exemple, seuil à 0.3)
threshold = 0.6
y_pred_adjusted = (y_prob > threshold).astype(int)

Results(y_test, y_pred)

conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

#Kera Neural Network
from imblearn.over_sampling import RandomOverSampler
from keras.layers import Dense
from keras.models import Sequential
from keras.regularizers import l2


data = pd.read_csv('data.csv', encoding = 'ISO-8859-1')

X = data.drop('Bankrupt?', axis=1)
y = data['Bankrupt?']

n_cols = X.shape[1]
n_cols

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
y_train_resampled

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

model = Sequential()
model.add(Dense(units=11, input_shape=(n_cols,), activation='relu', kernel_regularizer=l2(0.07)))
model.add(Dense(units=5, activation='relu',kernel_regularizer=l2(0.07)))
model.add(Dense(units=1, activation='sigmoid'))


model.compile(optimizer='adam', loss='binary_crossentropy', metrics='accuracy')

model.fit(X_train_scaled, y_train_resampled, epochs=10, batch_size=512, verbose=1)

y_predd = (model.predict(X_test_scaled) > 0.57).astype(int)

conf_matrix = confusion_matrix(y_test, y_predd)
print(conf_matrix)

conf_matrix = confusion_matrix(y_test, y_predd)


plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['True 0', 'True 1'], yticklabels=['Pred 0', 'Pred 1'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Keras Neural Network')
plt.show()

Results(y_test, y_predd)