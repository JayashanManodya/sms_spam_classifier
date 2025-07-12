📬 SMS Spam Classifier — NLP & Machine Learning Project
This project is a binary classification system that detects whether a given SMS message is spam or ham (not spam) using Natural Language Processing (NLP) and Machine Learning (ML). It is built using the SMS Spam Collection Dataset and demonstrates how to preprocess raw text, extract features, and train a model to accurately classify text messages.

📌 Table of Contents
Project Objective

Dataset Description

Natural Language Processing (NLP) Workflow

Data Cleaning & Preprocessing

Tokenization

Stopword Removal

Stemming

Feature Extraction (BoW)

Model Building

Model Evaluation

How to Run This Project

Technologies & Libraries Used

Results & Discussion

Conclusion

Possible Improvements

🎯 Project Objective
To develop a spam detection model that classifies SMS messages as either Spam or Ham using:

NLP techniques for text processing.

A Naive Bayes classifier for machine learning.

Accuracy and performance metrics for evaluation.

📂 Dataset Description
The dataset used is the SMS Spam Collection Dataset from the UCI Machine Learning Repository.
It contains 5,572 SMS messages, each labeled as either:

ham – non-spam

spam – unwanted commercial or scam messages

Each row contains:

A label (ham or spam)

A message text

🧠 Natural Language Processing (NLP) Workflow
Raw text data is messy and needs transformation to be used in ML models. Here's the step-by-step breakdown of the preprocessing process applied:

🔹 1. Text Cleaning
Remove unnecessary symbols or characters that do not contribute to meaning.

Steps performed:

Convert text to lowercase (HELLO → hello)

Remove punctuation and numbers using regex

Remove special characters and whitespaces

Why? Cleaned text ensures consistency, and irrelevant characters are discarded to avoid noise in analysis.

🔹 2. Tokenization
Tokenization splits each sentence into individual words or tokens.

Example:

“This is a spam message” → [‘this’, ‘is’, ‘a’, ‘spam’, ‘message’]

Theory: Tokenization is essential for turning unstructured text into a structured format that ML models can understand.

🔹 3. Stopword Removal
Stopwords are common words that carry little meaning (e.g., “the”, “is”, “and”).

Using NLTK's list, we remove them from our token list.

Example:

[‘this’, ‘is’, ‘a’, ‘spam’, ‘message’] → [‘spam’, ‘message’]

Why? Removing stopwords focuses the model on important words that carry meaning, improving accuracy.

🔹 4. Stemming
Stemming reduces words to their root form.

Example:

“loved”, “loving” → “love”
“playing”, “played” → “play”

Used: Porter Stemmer from NLTK

Why? Stemming helps group similar words, reducing the vocabulary size and improving learning efficiency.

🔹 5. Feature Extraction: Bag of Words (BoW)
The text is converted into numerical format using CountVectorizer, a method from Scikit-learn.

Bag of Words counts how often each word appears in the dataset.

Each message becomes a vector of word frequencies.

Example:

Word	"Buy now"	"now win free"
buy	1	0
now	1	1
win	0	1
free	0	1

Theory: BoW ignores grammar and word order but captures word presence and frequency, which is often enough for spam detection.

🤖 Model Building
Algorithm Used: Multinomial Naive Bayes
The Multinomial Naive Bayes model is best for discrete data (like word counts).

Why Naive Bayes?

Simple and fast

Performs well with text classification

Works under the assumption that features (words) are conditionally independent given the class (spam/ham)

Steps in notebook:

Split data into training and testing sets (80/20 split)

Fit the model on training data

Predict on test data

📊 Model Evaluation
Metrics Used:
Accuracy Score – overall performance

Confusion Matrix – true positives, false positives, etc.

Precision – how many predicted spams were actually spam

Recall – how many actual spams were correctly detected

F1-Score – harmonic mean of precision and recall

📌 Results from your notebook show very high accuracy (~97%), indicating a strong model.

▶️ How to Run This Project
🛠️ Requirements
Install these packages before running the notebook:

bash
Copy
Edit
pip install pandas numpy matplotlib seaborn sklearn nltk
🏃 Run Instructions
Clone or download this repository.

Open the sms_spam_classifier.ipynb file in:

Jupyter Notebook

VS Code (with Jupyter extension)

Google Colab

Run all cells sequentially.

Optional: Download NLTK resources if prompted (stopwords, punkt).

💻 Technologies & Libraries Used
Tool/Library	Purpose
Python	Programming language
Pandas	Data manipulation
NumPy	Array operations
NLTK	Natural language processing
Scikit-learn	ML algorithms and utilities
Matplotlib/Seaborn	Visualization

✅ Results & Discussion
The model achieved ~97% accuracy, correctly identifying most spam and ham messages.

Naive Bayes works effectively due to the independence assumption and nature of word frequency features.

Feature extraction with CountVectorizer enabled numerical representation of raw text.

🧾 Conclusion
This project successfully demonstrates a practical application of NLP in classifying text messages. By using:

Proper text preprocessing techniques

Effective feature engineering (BoW)

A simple yet powerful classifier (Naive Bayes)

We were able to build a model that performs well in identifying spam with high accuracy.

🚀 Possible Improvements
Here are a few ways to take this project further:

Use TF-IDF Vectorizer instead of BoW to weigh words based on importance.

Lemmatization instead of stemming to retain valid root words.

Try more advanced models: SVM, Logistic Regression, Random Forest.

Train on a larger dataset for better generalization.

Deploy the model using Flask or Streamlit as a web app.
