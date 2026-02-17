# Data Handling
import pandas as pd

# Machine Learning & Model Persistence
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


# Read the data into a DataFrame
df = pd.read_csv('data/penguins_cleaned.csv')

# Remove the target column that we want to predict
X_raw = df.drop('species', axis=1)
# Set the target column we want to predict
y_raw = df.species

# One-Hot Encoding: Creates new binary columns for each category
# Convert non-numeric/categorical variables into numerical vectors (0s and 1s)
features_to_encode = ['island', 'sex']
# Convert categorical variables into new columns with value or 1 (present) or 0 otherwise
X_encoded = pd.get_dummies(X_raw, columns=features_to_encode)

# Label/Ordinal Encoding: Assigns a unique integer to each category
# Create map of species (categorical) to numerical values
target_mapper = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}
# Map the target values to numerical values
y_encoded = y_raw.apply(lambda x: target_mapper[x])

# Split the dataset into train and test sets
# 70% training and 30% test
# Stratify ensures the resulting training and testing sets
# maintain the same proportion of classes as the original dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.3, random_state=42
)

# Create an instance of the random forest classifier
# n_estimators determines the number of decision trees in the forest
model_classifier = RandomForestClassifier(n_estimators=100)

# Train the classifier on the training data
model_classifier.fit(X_train, y_train)

# Display the accuracy of the model
print(f"Train accuracy: {accuracy_score(y_train, model_classifier.predict(X_train))}")
print(f"Test accuracy: {accuracy_score(y_test, model_classifier.predict(X_test))}")

# Save the model to disk
joblib.dump(model_classifier, 'penguin_classifier_model.sav')