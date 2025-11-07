# run_spam_classifier.py
import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# --- Configuration ---
DATA_PATH = "spam_data.csv"   # put your CSV here in the same folder
MODEL_OUT = "spam_classifier.pkl"
FIG_OUT = "spam_confusion_matrix.png"

# --- Load ---
if not os.path.exists(DATA_PATH):
    raise SystemExit(f"Dataset not found: {DATA_PATH}  → put your CSV in this folder and name it '{DATA_PATH}'")

df = pd.read_csv(DATA_PATH)
print("Raw shape:", df.shape)

# Expect columns named 'text' and 'label' (label values: 'spam' or 'ham' etc.)
if not {'text', 'label'}.issubset(df.columns):
    raise SystemExit("CSV must contain 'text' and 'label' columns.")

# --- Clean & prepare ---
df = df.dropna(subset=['text', 'label']).reset_index(drop=True)
df['text'] = df['text'].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip().str.lower()
df['label_bin'] = df['label'].apply(lambda x: 1 if str(x).lower() in ['spam','1','true','yes'] else 0)

X = df['text']
y = df['label_bin']

# --- Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
print("Train / Test sizes:", X_train.shape[0], "/", X_test.shape[0])

# --- Vectorize ---
tf = TfidfVectorizer(max_features=20000, ngram_range=(1,2), stop_words='english')
X_train_tfidf = tf.fit_transform(X_train)
X_test_tfidf  = tf.transform(X_test)
print("TF-IDF vectorization done. Feature dim:", X_train_tfidf.shape[1])

# --- Models to try ---
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

results = []

for name, model in models.items():
    print(f"\nTraining: {name}")
    model.fit(X_train_tfidf, y_train)
    preds = model.predict(X_test_tfidf)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)

    print(f"{name} -> Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")
    print("Classification report:")
    print(classification_report(y_test, preds, zero_division=0))

    results.append((name, model, preds, (acc, prec, rec, f1)))

# --- Choose best by F1 (simple) ---
best_name, best_model, best_preds, best_metrics = max(results, key=lambda x: x[3][3])
print(f"\nBest model by F1: {best_name} (F1={best_metrics[3]:.4f})")

# --- Save the pipeline pieces ---
pipe_obj = {"vectorizer": tf, "model": best_model}
with open(MODEL_OUT, "wb") as f:
    pickle.dump(pipe_obj, f)
print(f"Saved model pipeline to {MODEL_OUT}")

# --- Confusion matrix for best model ---
cm = confusion_matrix(y_test, best_preds)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["ham","spam"], yticklabels=["ham","spam"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix — {best_name}")
plt.tight_layout()
plt.savefig(FIG_OUT, dpi=150)
plt.close()
print(f"Saved confusion matrix as {FIG_OUT}")