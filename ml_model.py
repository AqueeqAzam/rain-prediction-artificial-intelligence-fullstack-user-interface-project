import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import pickle

# Load the dataset
df = pd.read_csv("https://raw.githubusercontent.com/AqueeqAzam/data-science-and-machine-learning-datasets/main/environment.csv")

# Data Exploration and Cleaning
# Check for missing values
print(df.isnull().sum())

# Fill or drop missing values if any
# Example: df.fillna(method='ffill', inplace=True)

# Data Preprocessing
# Features and target variable
X = df.drop('rain', axis=1)
y = df['rain']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the scaler and model
scaler = StandardScaler()
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the scaler on the training data and transform it
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model.fit(X_train_scaled, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test_scaled)
accuracy =  accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the model and scaler to a pickle file
with open('rain_prediction_model.pkl', 'wb') as file:
    pickle.dump({'model': model, 'scaler': scaler}, file)

print("Model and scaler saved to 'rain_prediction_model.pkl'")