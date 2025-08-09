import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import joblib


print("Loading dataset...")
data = pd.read_csv("emails.csv")


vocabulary = list(data.columns[1:])  
X = data.iloc[:, 1:].values  


email_numbers = data['Email No.'].str.extract('(\d+)').astype(int)
print("\nEmail number range:", email_numbers.min(), "to", email_numbers.max())


spam_threshold = 2500
y = np.where(email_numbers <= spam_threshold, 1, 0)  # Changed > to <= to fix label assignment
print("\nLabel distribution:")
print("Non-spam (0):", sum(y == 0))
print("Spam (1):", sum(y == 1))

print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def train_and_evaluate(model, name, X_train, X_test, y_train, y_test):
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"{name} Accuracy: {accuracy:.4f}")
    print(f"\n{name} Classification Report:")
    print(classification_report(y_test, predictions))
    
    
    joblib.dump(model, f"{name.lower().replace(' ', '_')}.pkl")
    return model, accuracy


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC(max_iter=2000)
}


results = {}
for name, model in models.items():
    model, accuracy = train_and_evaluate(model, name, X_train, X_test, y_train, y_test)
    results[name] = accuracy

print("\nOverall Results:")
for name, accuracy in results.items():
    print(f"{name}: {accuracy:.4f}")


joblib.dump(vocabulary, 'vocabulary.pkl')
