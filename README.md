# Heart-Disease-Prediction-using-Logistic-Regression

This project aims to predict the presence of heart disease using logistic regression. The dataset used contains various features related to heart health.

## Table of Contents
- [Importing Libraries](#importing-libraries)
- [Importing Data](#importing-data)
- [Dataset Desription](#dataset-description)
- [Data Exploration](#data-exploration)
- [Data Visualization](#data-visualization)
- [Data Splitting](#data-splitting)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Conclusion](#conclusion)

## Importing Libraries

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
```

## Importing Data

```python
data = pd.read_csv("heart.csv")
```
## Dataset Description
<pre>
The heart disease dataset is a widely used dataset for predicting heart disease, which contains a mix of categorical and numerical variables. It consists of 303 entries and 14 columns, each representing different attributes related to heart disease. Here is a detailed description of the dataset:

<b>Columns</b>

age: Age of the patient (integer)
sex: Gender of the patient (1 = male, 0 = female)
cp: Chest pain type (categorical: 0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic)
trestbps: Resting blood pressure (in mm Hg on admission to the hospital)
chol: Serum cholesterol in mg/dl
fbs: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
restecg: Resting electrocardiographic results (categorical: 0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy)
thalach: Maximum heart rate achieved
exang: Exercise-induced angina (1 = yes, 0 = no)
oldpeak: ST depression induced by exercise relative to rest
slope: The slope of the peak exercise ST segment (categorical: 0 = upsloping, 1 = flat, 2 = downsloping)
ca: Number of major vessels (0-3) colored by fluoroscopy
thal: Thalassemia (categorical: 1 = normal, 2 = fixed defect, 3 = reversible defect)
target: Diagnosis of heart disease (1 = presence of heart disease, 0 = absence of heart disease)

</pre>

## Data Exploration

```python
data.head()
data.info()
data.describe()
```
## Data Visualization

```python
sns.countplot(x='target', data=data)
plt.show()
```


![download](https://github.com/user-attachments/assets/765af3bf-edcf-4cde-9292-8eae0c53f196)



## Data Splitting

```python
X = data.iloc[:,:-1]
y = data.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
```

## Model Training

```python
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```

## Model Evaluation

```python
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
```

## Conclusion

After evaluating the performance of our logistic regression model on the heart disease dataset, we can draw several key conclusions:

### Model Accuracy
The model achieved an accuracy of approximately 77.05% on the test set, as calculated using the `accuracy_score` function from `sklearn.metrics`. This means that the model correctly predicts the presence or absence of heart disease about 77% of the time.

### Confusion Matrix
The confusion matrix provides a detailed breakdown of the model's performance:


![download](https://github.com/user-attachments/assets/52499f86-c4bb-4659-8249-7d3594e5b369)

Where:
- **TN (True Negative)**: Number of actual negatives correctly predicted as negatives.
- **FP (False Positive)**: Number of actual negatives incorrectly predicted as positives.
- **FN (False Negative)**: Number of actual positives incorrectly predicted as negatives.
- **TP (True Positive)**: Number of actual positives correctly predicted as positives.

### Recall
Recall, also known as sensitivity or true positive rate, is a measure of how well the model identifies positive instances (presence of heart disease). It is calculated as:
\[ \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} \]
For this model, the recall is 0.87, indicating that the model correctly identifies 87% of the actual positive cases.

### Precision
Precision measures the accuracy of positive predictions made by the model. It is calculated as:
\[ \text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} \]
For this model, the precision is 0.73, meaning that 73% of the instances predicted as positive by the model are actually positive.

### Summary
- **Accuracy**: 77.05%
- **Recall**: 87%
- **Precision**: 73%

The model demonstrates a good balance between precision and recall, with a relatively high recall indicating its effectiveness in identifying most positive cases of heart disease.
