import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import joblib
from lime.lime_text import LimeTextExplainer
import re
import seaborn as sns
import matplotlib.pyplot as plt
import itertools


st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="collapsed"
)


st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.5rem;
        margin-bottom: 2rem;
        text-align: center;
        color: #666;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .spam {
        background-color: #ff4b4b;
        color: white;
    }
    .not-spam {
        background-color: #00cc66;
        color: white;
    }
    .confidence {
        font-size: 1.2rem;
        margin-top: 1rem;
    }
    .explanation {
        margin-top: 2rem;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 10px;
    }
    .feature {
        margin: 0.5rem 0;
        padding: 0.5rem;
        border-radius: 5px;
    }
    .positive {
        color: #00cc66;
    }
    .negative {
        color: #ff4b4b;
    }
</style>
""", unsafe_allow_html=True)


vocabulary = None

@st.cache_resource
def load_models():
    """Load the trained models and vocabulary"""
    try:
        models = {
            "Logistic Regression": joblib.load('logistic_regression.pkl'),
            "Naive Bayes": joblib.load('naive_bayes.pkl'),
            "Linear SVM": joblib.load('linear_svm.pkl')
        }
        global vocabulary
        vocabulary = joblib.load('vocabulary.pkl')
        return models, vocabulary
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

def preprocess_text(text, vocab):
    """Convert text to the same feature format as our training data"""
    # Split the text into words and create a set
    words = set(word.lower() for word in text.split())
    # Create a feature vector where 1 indicates the word is present
    return [1 if word in words else 0 for word in vocab]

def plot_confusion_matrix(cm, classes=['Not Spam', 'Spam'], title='Confusion Matrix'):
    """Plot confusion matrix with a heatmap"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    return plt

def calculate_metrics(model, X_test, y_test):
    """Calculate various metrics for model evaluation"""
    predictions = model.predict(X_test)
    cm = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions, target_names=['Not Spam', 'Spam'], output_dict=True)
    accuracy = accuracy_score(y_test, predictions)
    return cm, report, accuracy

def get_model_explanation(model, text, vocab, num_features=6):
    """Get LIME explanation for the model's prediction"""
    explainer = LimeTextExplainer(class_names=['Not Spam', 'Spam'])
    
    def predict_proba(texts):
        
        features = np.array([preprocess_text(t, vocab) for t in texts])
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(features)
        else:
           
            decisions = model.decision_function(features)
            probs = 1 / (1 + np.exp(-decisions))
            return np.vstack((1-probs, probs)).T
    
    exp = explainer.explain_instance(text, predict_proba, num_features=num_features)
    return exp

def main():
    st.markdown('<h1 class="main-header">Email Spam Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced ML-powered spam detection system</p>', unsafe_allow_html=True)
    
    
    tab1, tab2 = st.tabs(["📧 Classify Email", "📊 Model Comparison"])
    
   
    models, vocabulary = load_models()
    if not models or not vocabulary:
        st.error("Failed to load models. Please ensure you have trained the models first.")
        st.error("Run train_models.py before using this application.")
        return

   
    try:
        data = pd.read_csv("emails.csv")
        email_numbers = data['Email No.'].str.extract('(\d+)').astype(int)
        X = data.iloc[:, 1:].values
        y = np.where(email_numbers > 2500, 1, 0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    except Exception as e:
        st.error(f"Error loading test data: {str(e)}")
        return
    
    with tab1:
        
        model_name = st.selectbox(
            "Select Model",
            ["Logistic Regression", "Naive Bayes", "Linear SVM"],
            index=0,
            key="model_selector"
        )
    
   
    text_input = st.text_area(
        "Enter email content:",
        height=200,
        placeholder="Paste your email content here...",
        key="email_input"
    )
    
    if st.button("Classify Email", key="classify_btn"):
        if not text_input.strip():
            st.warning("Please enter some text to classify.")
            return
        
        with st.spinner("Analyzing email..."):
           
            model = models[model_name]
            
           
            features = preprocess_text(text_input.lower(), vocabulary)
            prediction = model.predict([features])[0]
            is_spam = bool(prediction)
            
           
            if hasattr(model, 'predict_proba'):
                confidence = model.predict_proba([features])[0][prediction]
            elif hasattr(model, 'decision_function'):
                decision = model.decision_function([features])[0]
                confidence = 1 / (1 + np.exp(-abs(decision)))
            else:
                confidence = 1.0
            
          
            if is_spam:
                st.markdown('<div class="prediction-box spam"><h2>📨 SPAM DETECTED</h2></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="prediction-box not-spam"><h2>✅ NOT SPAM</h2></div>', unsafe_allow_html=True)
            
            st.markdown(f'<p class="confidence">Confidence: {confidence:.2%}</p>', unsafe_allow_html=True)
            
          
            with st.expander("See explanation"):
                st.markdown("### Why was this prediction made?")
                exp = get_model_explanation(model, text_input, vocabulary)
                
                st.markdown("#### Key features influencing the prediction:")
                for word, score in exp.as_list():
                    color = "negative" if score > 0 else "positive"
                    impact = "indicating SPAM" if score > 0 else "supporting NOT SPAM"
                    st.markdown(
                        f'<div class="feature {color}">'
                        f'• "{word}": {abs(score):.3f} ({impact})'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                
                st.info("""
                💡 How to interpret:
                - Green features suggest legitimate email content
                - Red features are typically associated with spam
                - The magnitude indicates how strongly each feature influenced the decision
                """)

    with tab2:
        st.markdown("### Model Performance Comparison")
        st.markdown("Compare accuracy, precision, recall, and F1-score across all models")
        
       
        metrics_data = []
        for name, model in models.items():
            cm, report, accuracy = calculate_metrics(model, X_test, y_test)
            
           
            metrics_data.append({
                'Model': name,
                'Accuracy': f"{accuracy:.4f}",
                'Precision (Spam)': f"{report['Spam']['precision']:.4f}",
                'Recall (Spam)': f"{report['Spam']['recall']:.4f}",
                'F1-Score (Spam)': f"{report['Spam']['f1-score']:.4f}",
                'Precision (Not Spam)': f"{report['Not Spam']['precision']:.4f}",
                'Recall (Not Spam)': f"{report['Not Spam']['recall']:.4f}",
                'F1-Score (Not Spam)': f"{report['Not Spam']['f1-score']:.4f}"
            })
            
          
            st.subheader(f"{name} - Confusion Matrix")
            fig = plot_confusion_matrix(cm)
            st.pyplot(fig)
            plt.close()
        
    
        st.subheader("Detailed Metrics Comparison")
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df.style.highlight_max(axis=0, subset=metrics_df.columns[1:]))
        
        st.markdown("""
        💡 **How to interpret the results:**
        - **Accuracy**: Overall correct predictions
        - **Precision**: Accuracy of positive predictions (How many predicted spams are actually spam?)
        - **Recall**: Proportion of actual positives identified (How many actual spams were caught?)
        - **F1-Score**: Harmonic mean of precision and recall
        
        The confusion matrix shows:
        - True Negatives (top-left): Correctly identified non-spam
        - False Positives (top-right): Non-spam wrongly marked as spam
        - False Negatives (bottom-left): Spam missed
        - True Positives (bottom-right): Correctly identified spam
        """)

if __name__ == "__main__":
    main()
