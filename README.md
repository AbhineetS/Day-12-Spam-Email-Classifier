# Day 12 — Spam Email Classifier (NLP)

In this project, I built a simple **Natural Language Processing (NLP)** model that classifies emails as **spam** or **ham (not spam)** using machine learning techniques like **TF-IDF Vectorization** and **Naive Bayes**.

---

## 🧠 Overview
The goal was to demonstrate how raw text can be transformed into numerical data and then classified by a machine learning algorithm.  
This project introduces the fundamentals of **text preprocessing**, **vectorization**, and **binary classification** in NLP.

---

## ⚙️ Workflow
1. **Data Loading** — Used a small labeled dataset of email samples (spam & ham).  
2. **Text Vectorization** — Converted email text into numerical form using **TF-IDF**.  
3. **Model Training** — Compared two classifiers:
   - Multinomial Naive Bayes  
   - Logistic Regression  
4. **Evaluation** — Measured performance using Accuracy, Precision, Recall, and F1 Score.  
5. **Model Saving** — Exported the trained pipeline and confusion matrix for visualization.

---

## 📊 Results
| Model | Accuracy | Precision | Recall | F1 Score |
|-------|-----------|------------|---------|-----------|
| Naive Bayes | 0.67 | 0.50 | 1.00 | 0.67 |
| Logistic Regression | 0.67 | 0.50 | 1.00 | 0.67 |

✅ **Best Model:** Naive Bayes (F1 = 0.66)  
📈 Confusion matrix saved as `spam_confusion_matrix.png`  
💾 Trained model pipeline saved as `spam_classifier.pkl`

---

## 💡 Tech Stack
Python | Scikit-learn | Pandas | TF-IDF | NLP | Machine Learning

---

## ▶️ Run the Project
```bash
source ../Day-01-Titanic/venv/bin/activate
python3 run_spam_classifier.py### Update: Improved documentation formatting
