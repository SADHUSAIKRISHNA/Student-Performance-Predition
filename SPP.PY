import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# Step 1: Create the dataset
data = {
    'Study_Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Score': [35, 40, 50, 55, 60, 65, 70, 78, 85, 90]
}
df = pd.DataFrame(data)
# Step 2: Data Visualization
plt.scatter(df['Study_Hours'], df['Score'])
plt.xlabel('Study Hours')
plt.ylabel('Score')
plt.title('Study Hours vs Student Score')
plt.show()
# Step 3: Split data
X = df[['Study_Hours']]   # Independent variable
y = df['Score']           # Dependent variable
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Step 4: Train the model
model = LinearRegression()
model.fit(X_train, y_train)
# Step 5: Make predictions
y_pred = model.predict(X_test)
# Step 6: Model Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R² Score:", r2)
# Step 7: Predict new value
hours = np.array([[6]])
predicted_score = model.predict(hours)
print("Predicted score for 6 study hours:", predicted_score[0])
