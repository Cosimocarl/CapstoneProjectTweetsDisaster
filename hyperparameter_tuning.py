import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report

# Load your dataset
df = pd.read_csv('./data/train.csv')

# Fill missing values in 'keyword' and 'location'
df['keyword'].fillna('no_keyword', inplace=True)
df['location'].fillna('no_location', inplace=True)

# Combine 'keyword', 'location', and 'text' into a single feature
df['combined_features'] = df['keyword'] + ' ' + df['location'] + ' ' + df['text']

# Split the data into features and target
X = df['combined_features']
y = df['target']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define the hyperparameter grid for Random Forest
param_dist = {
    'n_estimators': np.arange(100, 1000, 100),
    'max_depth': [None] + list(np.arange(10, 110, 10)),
    'min_samples_split': np.arange(2, 11, 2),
    'min_samples_leaf': np.arange(1, 11, 2),
    'bootstrap': [True, False]
}

# Initialize Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Set up RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=param_dist,
    n_iter=20,  # Number of parameter settings sampled
    scoring='accuracy',
    cv=5,  # 5-fold cross-validation
    random_state=42,
    n_jobs=-1,  # Use all available cores
    verbose=2
)

# Create a pipeline with TF-IDF vectorizer and the RandomizedSearchCV
tuning_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),  # Convert text data into TF-IDF features
    ('random_search', random_search)  # Perform hyperparameter tuning
])

# Fit the pipeline to the training data
tuning_pipeline.fit(X_train, y_train)

# Best parameters found by RandomizedSearchCV
best_params = random_search.best_params_

# Train the model with the best parameters
best_model = random_search.best_estimator_
best_model.fit(X_train, y_train)

# Predict on the test set
y_pred_best = best_model.predict(X_test)

# Calculate metrics for the best model
accuracy_best = accuracy_score(y_test, y_pred_best)
f1_best = f1_score(y_test, y_pred_best)
recall_best = recall_score(y_test, y_pred_best)
classification_rep_best = classification_report(y_test, y_pred_best)

print("Best Parameters:", best_params)
print("Accuracy:", accuracy_best)
print("F1 Score:", f1_best)
print("Recall:", recall_best)
print("\nClassification Report:\n", classification_rep_best)
