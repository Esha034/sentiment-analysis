# Sentiment Analysis Web App (Movie Reviews)

This project is an end-to-end Machine Learning application that predicts the sentiment (Positive / Negative) of movie reviews using Natural Language Processing (NLP) techniques.
The model is trained on the IMDb 50K Movie Reviews dataset and deployed as an interactive Streamlit web application.

## Live Demo (Run Globally)

👉 Streamlit App Link:
🔗 https://sentiment-analysis-webapp03.streamlit.app/

(Recruiters can directly open this link and test the model without any setup.)

## Problem Statement

Understanding customer sentiment from text data is crucial for applications like:

Product reviews analysis

Customer feedback monitoring

Social media sentiment tracking

This project solves a binary text classification problem where a given movie review is classified as Positive or Negative.

## Dataset

Source: IMDb Movie Reviews Dataset

Size: 50,000 labeled reviews

Classes: Positive, Negative

## Dataset link:
https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

## Approach & Methodology
### 1️ Text Preprocessing & Feature Extraction

Converted raw text into numerical features using TF-IDF Vectorization

Reduced the impact of common words and emphasized informative terms

### 2️ Model Selection

Used Logistic Regression, which performs well on:

-High-dimensional sparse data

-Text classification tasks

Chosen for:

-Fast training and inference

-Interpretability

-Production suitability

### 3️ Model Training & Evaluation

Train-test split: 80-20

Achieved ~89% accuracy on test data

### 4 Deployment

-Model deployed using Streamlit

-Enables real-time sentiment prediction through a web interface

### 5 Tech Stack

#### Programming Language: Python

#### Libraries:

-Pandas, NumPy

-Scikit-learn

-Streamlit

-Joblib

#### ML Techniques:

-TF-IDF

-Logistic Regression

#### Deployment: Streamlit Cloud

## 📁 Project Structure
sentiment-analysis/
│
├── data/
│   └── IMDB Dataset.csv
│
├── saved_model/
│   └── sentiment_model.pkl
│
├── app.py
├── model_training.py
├── requirements.txt
└── README.md

## 💻 Installation & Execution (Run Locally)

Follow these steps to run the project on your local machine:

### 🔹 Step 1: Clone the Repository
git clone https://github.com/<your-username>/sentiment-analysis.git
cd sentiment-analysis

### 🔹 Step 2: Create Virtual Environment
python -m venv venv


### Activate it:

-Windows

venv\Scripts\activate


-Linux / macOS

source venv/bin/activate

### 🔹 Step 3: Install Dependencies
pip install -r requirements.txt

### 🔹 Step 4: Train the Model
python model_training.py


This will train the model and save it in the saved_model/ directory.

### 🔹 Step 5: Run Streamlit App
streamlit run app.py


### Open the browser at:

http://localhost:8501

#### Sample Inputs to Test

“I absolutely loved this movie. The acting was brilliant.”

“The movie was boring and a complete waste of time.”

## Results

-High accuracy on unseen reviews

-Fast and reliable predictions

-Achieved ~89% accuracy on test data

-Lightweight and production-ready ML pipeline

## Future Enhancements

-Upgrade model using BERT / Transformer-based architectures

-Add confidence score for predictions

-Support multi-class sentiment (positive / neutral / negative)

-Deploy via Docker for scalable production use

## 👩‍💻 Author

Eshani Sinha
B.Tech – Computer Science & Engineering
Aspiring Machine Learning Engineer
