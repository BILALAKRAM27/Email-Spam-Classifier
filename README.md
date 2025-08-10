# Email Spam Classifier

An advanced machine learning-powered spam detection system that uses multiple models to classify emails as spam or legitimate. The system provides detailed explanations for its decisions and allows model performance comparison.

## 🌟 Features

### 1. Multiple Classification Models
- **Logistic Regression**: Fast and interpretable
- **Naive Bayes**: Efficient for text classification
- **Linear SVM**: Robust for high-dimensional data

### 2. Interactive Web Interface
- Clean, modern UI built with Streamlit
- Real-time email classification
- Model selection option
- Confidence score display

### 3. Explanation System
- LIME (Local Interpretable Model-agnostic Explanations)
- Highlights influential words and their impact
- Color-coded feature importance:
  - 🟢 Green: Features suggesting legitimate email
  - 🔴 Red: Features indicating spam
- Confidence scores for predictions

### 4. Model Performance Comparison
- Comprehensive metrics display:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
- Interactive confusion matrices
- Side-by-side model comparison
- Detailed performance analysis

## 📋 Requirements

```
pandas
numpy
scikit-learn
streamlit
lime
joblib
seaborn
matplotlib
```

## 🚀 Setup and Installation

1. Clone the repository
```bash
git clone <repository-url>
cd Email_spam_classiffier
```

2. Create and activate a virtual environment (optional but recommended)
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## 🎯 Usage

### 1. Training Models
Run the training script to train and save the models:
```bash
python train_models.py
```
This will:
- Load and process the email dataset
- Train three different models
- Save the models and vocabulary

### 2. Running the Web Interface
Start the Streamlit application:
```bash
streamlit run app_new.py
```

### 3. Using the Interface

#### Classify Emails Tab
1. Select a model from the dropdown
2. Input or paste email content
3. Click "Classify Email"
4. View results:
   - Classification decision
   - Confidence score
   - Feature importance explanation

#### Model Comparison Tab
1. View comprehensive metrics for all models
2. Analyze confusion matrices
3. Compare performance metrics:
   - Accuracy scores
   - Precision and recall
   - F1-scores for spam and non-spam

## 📊 Project Structure

```
Email_spam_classiffier/
├── app_new.py           # Main Streamlit application
├── train_models.py      # Model training script
├── emails.csv          # Dataset
├── requirements.txt    # Project dependencies
├── vocabulary.pkl      # Saved vocabulary
└── models/            # Saved model files
    ├── logistic_regression.pkl
    ├── naive_bayes.pkl
    └── linear_svm.pkl
```

## 🔍 Model Details

### Logistic Regression
- Linear classifier with probability output
- Max iterations: 1000
- Good for interpretability

### Naive Bayes
- Multinomial Naive Bayes
- Efficient for text classification
- Works well with high-dimensional data

### Linear SVM
- Linear Support Vector Machine
- Max iterations: 2000
- Robust for high-dimensional sparse data

## 📈 Performance Metrics Explanation

- **Accuracy**: Overall correct predictions
- **Precision**: Accuracy of positive predictions
  - How many predicted spams are actually spam?
- **Recall**: Proportion of actual positives identified
  - How many actual spams were caught?
- **F1-Score**: Harmonic mean of precision and recall
  - Balance between precision and recall

## 🤝 Contributing

Feel free to:
- Report bugs
- Suggest features
- Submit pull requests


## 🙏 Acknowledgments

- Dataset source: [mail.csv](https://www.kaggle.com/datasets/venky73/spam-mails-dataset)
- LIME package for explanations
- Streamlit for the web interface
