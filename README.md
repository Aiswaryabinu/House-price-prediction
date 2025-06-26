# House-price-prediction
# Functions Explanation for House Price Prediction

This project is implemented mainly in a Jupyter notebook (`priceprediction.ipynb`). Below is an overview of the key functions, code blocks, and their purposes:

---

### 1. Importing Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
```
**Purpose:**  
Imports all required libraries for data manipulation, visualization, and machine learning.

---

### 2. Reading Data
```python
df = pd.read_csv('Housing.csv')
df.head()
```
**Purpose:**  
Loads the housing dataset into a pandas DataFrame and displays the first few rows.

---

### 3. Data Inspection
```python
df.info()
df.shape
```
**Purpose:**  
Checks the structure, data types, and dimensions of the dataset.

---

### 4. Data Preprocessing
#### Encoding Categorical Variables
```python
hash_set = {
    'unfurnished': 0,
    'semi-furnished': 1,
    'furnished': 2
}
for col in ['furnishingstatus']:
    data[col] = data[col].map(hash_set)
```
**Purpose:**  
Converts the `furnishingstatus` column from categories to numeric codes for model compatibility.

---

### 5. Splitting Features and Target
```python
x = data.drop('price', axis=1)
y = data['price']
```
**Purpose:**  
Separates the feature variables (`x`) from the target variable (`y`, which is the house price).

---

### 6. Train-Test Split
```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
```
**Purpose:**  
Splits the dataset into training and testing sets for model validation.

---

### 7. Model Training
```python
model = LinearRegression()
model.fit(x_train, y_train)
```
**Purpose:**  
Creates and trains a linear regression model using the training data.

---

### 8. Prediction
```python
y_pred = model.predict(x_test)
```
**Purpose:**  
Uses the trained model to predict house prices on the test set.

---

### 9. Model Evaluation
```python
print(model.score(x_train, y_train))
print(mean_squared_error(y_test, y_pred))
print(r2_score(y_test, y_pred))
```
**Purpose:**  
Evaluates the model’s accuracy and performance using metrics like R² score and mean squared error.

---

### 10. Result Visualization (optional)
```python
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()
```
**Purpose:**  
Plots the actual vs. predicted prices to visually assess model performance.

---

**Note:**  
Some code cells contain display logic or additional data cleaning which are not listed here but follow similar, self-explanatory patterns. If you want explanations for specific functions or blocks, let me know!
