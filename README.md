# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Start the program.

Step 2: Import the required libraries
(numpy, pandas, sklearn).

Step 3: Load the Iris dataset which contains the following attributes:

Sepal length

Sepal width

Petal length

Petal width
and the class label:

Species of Iris flower

Step 4: Separate the dataset into:

Input features (X)

Output labels (Y)

Step 5: Perform data preprocessing:

Check and handle missing values

Apply feature scaling using standardization

Step 6: Split the dataset into:

Training dataset

Testing dataset

Step 7: Initialize the SGD Classifier with suitable parameters.

Step 8: Train the classifier using the training dataset.

Step 9: Use the trained model to predict the species of Iris flowers for the test dataset.

Step 10: Evaluate the performance of the model using accuracy score or confusion matrix.

Step 11: Display the predicted Iris species.

Step 12: Stop the program.

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: 
RegisterNumber:  
*/
```
# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# ------------------------------
# Step 1: Sample dataset
# ------------------------------
data = {
    'Hours_Studied': [2, 3, 4, 5, 6, 7, 8, 9],
    'Previous_Score': [40, 50, 55, 60, 65, 70, 75, 80],
    'Internship': [0, 0, 1, 0, 1, 1, 1, 1],  # 0 = No, 1 = Yes
    'Placement': [0, 0, 0, 1, 1, 1, 1, 1]    # Target: 0 = Not Placed, 1 = Placed
}

df = pd.DataFrame(data)

# ------------------------------
# Step 2: Split into features and target
# ------------------------------
X = df[['Hours_Studied', 'Previous_Score', 'Internship']]
y = df['Placement']

# ------------------------------
# Step 3: Train-test split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# ------------------------------
# Step 4: Feature scaling
# ------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------------------
# Step 5: Create and train SGDClassifier for Logistic Regression
# ------------------------------
sgd_model = SGDClassifier(loss='log_loss',       # 'log' loss → logistic regression
                          max_iter=1000,
                          learning_rate='optimal',
                          random_state=42)
sgd_model.fit(X_train, y_train)

# ------------------------------
# Step 6: Make predictions
# ------------------------------
y_pred = sgd_model.predict(X_test)
y_prob = sgd_model.predict_proba(X_test)   # Probability of placement

# ------------------------------
# Step 7: Evaluate the model
# ------------------------------
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ------------------------------
# Step 8: Predict placement for a new student
# ------------------------------
new_student = np.array([[6, 68, 1]])  # Example: 6 hours, 68 prev score, Internship yes
new_student_scaled = scaler.transform(new_student)
placement_pred = sgd_model.predict(new_student_scaled)
placement_prob = sgd_model.predict_proba(new_student_scaled)

print(f"\nPredicted Placement Status: {'Placed' if placement_pred[0]==1 else 'Not Placed'}")
print(f"Probability of Placement: {placement_prob[0][1]:.2f}")


## Output:
![prediction of iris species using SGD Classifier](sam.png)
<img width="1720" height="543" alt="image" src="https://github.com/user-attachments/assets/6cf1989e-71e4-4337-85e9-4b236c4521f3" />



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
