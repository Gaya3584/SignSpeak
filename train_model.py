import pandas as pd # Load data
from sklearn.model_selection import train_test_split # Split data into training and testing sets
from sklearn.ensemble import RandomForestClassifier # Import Random Forest Classifier
import joblib # Save and load models

# Load data
df = pd.read_csv("gesture_data.csv", header=None) # Read CSV file without header
X = df.iloc[:, 1:]  # Landmark features 
y = df.iloc[:, 0]   # Labels

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # Split data into 80% training and 20% testing

# Train model
model = RandomForestClassifier() # Initialize Random Forest Classifier
model.fit(X_train, y_train) # Fit model to training data

# Save model
joblib.dump(model, "gesture_model.pkl") # Save the trained model to a file

# Optional: evaluate
print("Accuracy:", model.score(X_test, y_test)) # Print accuracy of the model on the test set
