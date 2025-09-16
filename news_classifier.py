"""
Enhanced News Article Text Classifier
Improved version with better data handling, validation, and functionality
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os
from data_generator import NewsDataGenerator

class NewsClassifier:
    def __init__(self):
        self.models = {
            'naive_bayes': MultinomialNB(),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(random_state=42, n_estimators=100)
        }
        self.best_model = None
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        self.pipeline = None
        self.label_map = None
        
    def load_or_generate_data(self, data_file='news_dataset.csv', num_samples=1000):
        """Load existing dataset or generate new one"""
        try:
            if os.path.exists(data_file):
                print(f"Loading existing dataset from {data_file}")
                df = pd.read_csv(data_file)
            else:
                print(f"Generating new dataset with {num_samples} samples")
                generator = NewsDataGenerator()
                df = generator.save_dataset(data_file, num_samples)
            
            # Validate dataset
            if df.empty:
                raise ValueError("Dataset is empty")
            
            if 'Headline' not in df.columns or 'Category' not in df.columns:
                raise ValueError("Dataset must contain 'Headline' and 'Category' columns")
            
            # Remove any null values
            df = df.dropna()
            
            print(f"Dataset loaded successfully: {df.shape[0]} samples, {df['Category'].nunique()} categories")
            return df
            
        except Exception as e:
            print(f"Error loading/generating data: {e}")
            return None
    
    def prepare_data(self, df):
        """Prepare data for training"""
        try:
            # Encode categories
            df['Category_Label'] = df['Category'].astype('category').cat.codes
            self.label_map = dict(enumerate(df['Category'].astype('category').cat.categories))
            
            X = df['Headline']
            y = df['Category_Label']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            print(f"Error preparing data: {e}")
            return None, None, None, None
    
    def train_models(self, X_train, y_train):
        """Train multiple models and select the best one"""
        print("Training multiple models...")
        
        # Fit vectorizer on training data
        X_train_vec = self.vectorizer.fit_transform(X_train)
        
        best_score = 0
        best_model_name = None
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Cross-validation
            scores = cross_val_score(model, X_train_vec, y_train, cv=5, scoring='accuracy')
            avg_score = scores.mean()
            
            print(f"{name} - CV Accuracy: {avg_score:.4f} (+/- {scores.std() * 2:.4f})")
            
            if avg_score > best_score:
                best_score = avg_score
                best_model_name = name
                self.best_model = model
        
        print(f"\nBest model: {best_model_name} with CV accuracy: {best_score:.4f}")
        
        # Train the best model on full training set
        self.best_model.fit(X_train_vec, y_train)
        
        # Create pipeline for easy prediction
        self.pipeline = Pipeline([
            ('vectorizer', self.vectorizer),
            ('classifier', self.best_model)
        ])
        
        # Fit pipeline
        self.pipeline.fit(X_train, y_train)
        
        return best_model_name, best_score
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the trained model"""
        if self.pipeline is None:
            print("No trained model found. Please train first.")
            return None
        
        # Predictions
        y_pred = self.pipeline.predict(X_test)
        
        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {accuracy:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_map.values()))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.label_map.values(), 
                   yticklabels=self.label_map.values())
        plt.title("Confusion Matrix - News Article Classification")
        plt.xlabel("Predicted Category")
        plt.ylabel("Actual Category")
        plt.tight_layout()
        plt.show()
        
        return accuracy, cm
    
    def predict_single(self, headline):
        """Predict category for a single headline"""
        if self.pipeline is None:
            return "No trained model available"
        
        try:
            prediction = self.pipeline.predict([headline])[0]
            probability = self.pipeline.predict_proba([headline])[0]
            
            category = self.label_map[prediction]
            confidence = max(probability)
            
            return {
                'category': category,
                'confidence': confidence,
                'all_probabilities': {self.label_map[i]: prob for i, prob in enumerate(probability)}
            }
        except Exception as e:
            return f"Error making prediction: {e}"
    
    def save_model(self, filename='news_classifier_model.pkl'):
        """Save the trained model"""
        if self.pipeline is None:
            print("No trained model to save")
            return False
        
        try:
            model_data = {
                'pipeline': self.pipeline,
                'label_map': self.label_map,
                'vectorizer': self.vectorizer
            }
            joblib.dump(model_data, filename)
            print(f"Model saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self, filename='news_classifier_model.pkl'):
        """Load a trained model"""
        try:
            model_data = joblib.load(filename)
            self.pipeline = model_data['pipeline']
            self.label_map = model_data['label_map']
            self.vectorizer = model_data['vectorizer']
            print(f"Model loaded from {filename}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

def main():
    """Main function to demonstrate the classifier"""
    print("=== News Article Text Classifier ===\n")
    
    # Initialize classifier
    classifier = NewsClassifier()
    
    # Load or generate data
    df = classifier.load_or_generate_data(num_samples=1000)
    if df is None:
        print("Failed to load data. Exiting.")
        return
    
    print(f"\nDataset Info:")
    print(f"Shape: {df.shape}")
    print(f"Categories: {df['Category'].unique()}")
    print(f"Category distribution:\n{df['Category'].value_counts()}\n")
    
    # Prepare data
    X_train, X_test, y_train, y_test = classifier.prepare_data(df)
    if X_train is None:
        print("Failed to prepare data. Exiting.")
        return
    
    # Train models
    best_model, best_score = classifier.train_models(X_train, y_train)
    
    # Evaluate
    print(f"\n=== Model Evaluation ===")
    accuracy, cm = classifier.evaluate_model(X_test, y_test)
    
    # Save model
    classifier.save_model()
    
    # Test predictions
    print(f"\n=== Sample Predictions ===")
    test_headlines = [
        "Government announces new healthcare reforms",
        "Basketball team wins championship final",
        "AI breakthrough revolutionizes medical diagnosis",
        "Stock market surges after economic report",
        "Scientists discover new planet in distant galaxy"
    ]
    
    for headline in test_headlines:
        result = classifier.predict_single(headline)
        if isinstance(result, dict):
            print(f"Headline: '{headline}'")
            print(f"Predicted: {result['category']} (confidence: {result['confidence']:.3f})")
            print()

if __name__ == "__main__":
    main()
