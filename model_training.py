import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# 1) Load
df = pd.read_csv("data/IMDB Dataset.csv")

# 2) Train-Test split
X_train, X_test, y_train, y_test = train_test_split(df.review, df.sentiment, test_size=0.2, random_state=42)

# 3) Build pipeline (TF-IDF + Logistic Regression)
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", LogisticRegression(max_iter=1000))
])

# 4) Train
model.fit(X_train, y_train)

# 5) Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# 6) Save
joblib.dump(model, "saved_model/sentiment_model.pkl")
print("Model saved!")
