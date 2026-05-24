import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

# reading dataset
url = "https://raw.githubusercontent.com/dphi-official/Datasets/master/Loan_Data/loan_train.csv"

data = pd.read_csv(url)

# changing column names
data.rename(columns={
    'ApplicantIncome': 'Income',
    'LoanAmount': 'Debt',
    'Credit_History': 'PaymentHistory',
    'Loan_Status': 'Creditworthy'
}, inplace=True)

# converting target values
data['Creditworthy'] = data['Creditworthy'].map({
    'Y': 1,
    'N': 0
})

print(data.head())

# checking null values
print(data.isnull().sum())

# filling missing values
for i in data.columns:
    
    if data[i].dtype == 'object':
        data[i].fillna(data[i].mode()[0], inplace=True)
        
    else:
        data[i].fillna(data[i].mean(), inplace=True)

# encoding categorical values
le = LabelEncoder()

for i in data.columns:
    
    if data[i].dtype == 'object':
        data[i] = le.fit_transform(data[i])

# feature engineering
data['DebtIncomeRatio'] = data['Debt'] / (data['Income'] + 1)

# separating input and output
x = data.drop('Creditworthy', axis=1)
y = data['Creditworthy']

# splitting dataset
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

# scaling data
sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# logistic regression
lr = LogisticRegression()

lr.fit(x_train, y_train)

lr_pred = lr.predict(x_test)

# decision tree
dt = DecisionTreeClassifier()

dt.fit(x_train, y_train)

dt_pred = dt.predict(x_test)

# random forest
rf = RandomForestClassifier(n_estimators=100)

rf.fit(x_train, y_train)

rf_pred = rf.predict(x_test)

# function for checking results
def check_result(name, y_test, pred):

    print("\n")
    print(name)

    acc = accuracy_score(y_test, pred)
    pre = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    print("Accuracy :", acc)
    print("Precision :", pre)
    print("Recall :", rec)
    print("F1 Score :", f1)

    cm = confusion_matrix(y_test, pred)

    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(name)
    plt.show()

# checking all models
check_result("Logistic Regression", y_test, lr_pred)

check_result("Decision Tree", y_test, dt_pred)

check_result("Random Forest", y_test, rf_pred)

# roc auc score
lr_auc = roc_auc_score(y_test, lr.predict_proba(x_test)[:,1])

dt_auc = roc_auc_score(y_test, dt.predict_proba(x_test)[:,1])

rf_auc = roc_auc_score(y_test, rf.predict_proba(x_test)[:,1])

print("\nROC AUC Scores")

print("Logistic Regression :", lr_auc)
print("Decision Tree :", dt_auc)
print("Random Forest :", rf_auc)

# feature importance
importance = pd.DataFrame({
    'Feature': x.columns,
    'Importance': rf.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nFeature Importance")
print(importance)

# graph
plt.figure(figsize=(8,5))

sns.barplot(
    x='Importance',
    y='Feature',
    data=importance
)

plt.title("Feature Importance")
plt.show()