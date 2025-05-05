import pandas as pd
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from google.colab import drive

# Mount Google Drive
drive.mount('/content/gdrive')

# Load JSON dataset from Google Drive
file_path = '/content/gdrive/My Drive/dataset.json'
with open(file_path, 'r') as f:
    json_data = json.load(f)

# Convert JSON to DataFrame
# Convert JSON to DataFrame
data = pd.json_normalize(json_data['intents'])
data['patterns'] = data['patterns'].astype(str)
data['responses'] = data['responses'].apply(lambda x: ' '.join(map(str, x)))  # Convert all responses to strings before joining
data['context'] = data['context'].apply(lambda x: ' '.join(map(str, x)) if isinstance(x, list) else '')  # Handle context lists

# Combine patterns and context for feature extraction
X_combined = data['patterns'] + ' ' + data['context']
y = data['tag']

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X_combined)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=8)

# Initialize models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=8),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=8),
    # "XGBoost": XGBClassifier(use_label_encoder=True, eval_metric='mlogloss', random_state=8, tree_method='gpu_hist'),
    "SVM": SVC(kernel='linear', probability=True, random_state=8),
    "Logistic Regression": LogisticRegression(random_state=8),
    "Naive Bayes": MultinomialNB(),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=8),
    "Decision Tree": DecisionTreeClassifier(random_state=8)
}

# Train and evaluate models
results = {}
confusion_matrices = {}

for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)

    # Save the model
    model_path = f"/content/gdrive/My Drive/{model_name.replace(' ', '_')}_model.pkl"
    joblib.dump(model, model_path)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[model_name] = accuracy
    confusion_matrices[model_name] = confusion_matrix(y_test, y_pred)
    print(f"{model_name} Accuracy: {accuracy:.4f}\n")

# Plot model performance
plt.figure(figsize=(10, 6))
plt.barh(list(results.keys()), list(results.values()), color='skyblue')
plt.xlabel('Accuracy')
plt.title('Model Performance')
plt.show()

# Display confusion matrices
for model_name, cm in confusion_matrices.items():
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=models.keys(), yticklabels=models.keys())
    plt.title(f"Confusion Matrix for {model_name}")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Print classification reports
for model_name, model in models.items():
    print(f"Classification Report for {model_name}:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

# Save the TF-IDF vectorizer
vectorizer_path = '/content/gdrive/My Drive/tfidf_vectorizer.pkl'
joblib.dump(vectorizer, vectorizer_path)

print("All models and vectorizer saved successfully!")

# Predict response based on patterns and context
def get_response(input_text):
    input_vector = vectorizer.transform([input_text])
    best_model_name = max(results, key=results.get)
    best_model = joblib.load(f"/content/gdrive/My Drive/{best_model_name.replace(' ', '_')}_model.pkl")
    predicted_tag = best_model.predict(input_vector)[0]
    response = data[data['tag'] == predicted_tag]['responses'].iloc[0]
    return response

# Example usage
user_input = "Attending a meeting or class 3 person"
response = get_response(user_input)
print(f"Response: {response}")
