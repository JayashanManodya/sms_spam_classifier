# 📬 SMS Spam Classifier — NLP & Machine Learning Project

This project is a **binary classification system** that detects whether a given SMS message is **spam** or **ham** (not spam) using **Natural Language Processing (NLP)** and **Machine Learning (ML)**. It is built using the **SMS Spam Collection Dataset** and demonstrates how to preprocess raw text, extract features, and train a model to accurately classify text messages.

---

## 📌 Table of Contents

- [Project Objective](#project-objective)
- [Dataset Description](#dataset-description)
- [NLP Workflow](#nlp-workflow)
- [Model Building](#model-building)
- [Model Evaluation](#model-evaluation)
- [How to Run This Project](#how-to-run-this-project)
- [Technologies & Libraries Used](#technologies--libraries-used)
- [Results & Discussion](#results--discussion)
- [Conclusion](#conclusion)
- [Possible Improvements](#possible-improvements)

---

## 🎯 Project Objective

To develop a spam detection model that classifies SMS messages as either **Spam** or **Ham** using:

- NLP techniques for text preprocessing
- Feature extraction using Bag of Words
- Machine learning classifier (Naive Bayes)
- Accuracy and performance metrics for evaluation

---

## 📂 Dataset Description

- **Dataset**: SMS Spam Collection Dataset (UCI Repository)
- **Records**: 5,572 labeled SMS messages
- **Labels**:
  - `ham`: Non-spam (legit messages)
  - `spam`: Junk or unwanted messages

---

## 🧠 NLP Workflow

### 🔹 1. Text Cleaning

- Lowercasing  
- Removing punctuation and digits  
- Removing special characters  

---

### 🔹 2. Tokenization

Splits text into individual tokens (words).

```python
"This is spam" → ['this', 'is', 'spam']
```

---

### 🔹 3. Stopword Removal

Removes common, less meaningful words.

```python
['this', 'is', 'spam'] → ['spam']
```

---

### 🔹 4. Stemming

Converts words to root form using Porter Stemmer.

```python
['loved', 'loving'] → ['love']
```

---

### 🔹 5. Feature Extraction (Bag of Words)

Uses `CountVectorizer` to convert text to numeric vectors.

```python
["buy now", "win now"] →
[1, 1, 0], [0, 1, 1]
```

---

## 🤖 Model Building

- **Model Used**: Multinomial Naive Bayes (suitable for word frequencies)  
- **Why Naive Bayes?**
  - Fast and efficient for text
  - Performs well with bag-of-words data
- **Train-Test Split**: 80/20

---

## 📊 Model Evaluation

- **Accuracy**  
- **Precision**  
- **Recall**  
- **F1-score**  
- **Confusion Matrix**

✅ Achieved approximately **97% accuracy** in classification!

---

## ▶️ How to Run This Project

### ✅ Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install pandas numpy nltk scikit-learn matplotlib seaborn
```

---

### 🏃 Steps

1. Clone this repository:
```bash
git clone https://github.com/your-username/sms-spam-classifier.git
cd sms-spam-classifier
```

2. Launch the notebook:
```bash
jupyter notebook notebooks/sms_spam_classifier.ipynb
```

3. Run all cells in order.

4. Optionally, run modular scripts in `src/`.

---

## 💻 Technologies & Libraries Used

| Tool/Library      | Purpose                             |
|------------------|-------------------------------------|
| Python            | Programming Language                |
| pandas            | Data Handling                       |
| numpy             | Numerical Computations              |
| nltk              | Natural Language Processing         |
| scikit-learn      | Machine Learning                    |
| matplotlib/seaborn| Data Visualization                  |

---

## ✅ Results & Discussion

- Trained Naive Bayes model achieved **~97% accuracy**
- Very low false positive rate for ham
- Effective preprocessing and feature engineering were key

---

## 🧾 Conclusion

This project demonstrates a real-world application of NLP and ML:

- Preprocessed raw text efficiently
- Converted it into usable numeric format
- Applied a simple and fast model with high accuracy
- Identified spam reliably with minimal resources

---

## 🚀 Possible Improvements

- Use **TF-IDF Vectorizer** instead of BoW  
- Replace stemming with **lemmatization**  
- Try other ML models (e.g., Logistic Regression, SVM)  
- Deploy as a web app using Flask or Streamlit  
- Improve data visualization and EDA  

---
