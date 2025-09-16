"""
Flask Web Application for News Article Classification
Modern UI for the text classifier
"""

from flask import Flask, render_template, request, jsonify
import os
import sys
from news_classifier import NewsClassifier
import pandas as pd

app = Flask(__name__)

# Global classifier instance
classifier = None

def initialize_classifier():
    """Initialize and train the classifier if needed"""
    global classifier
    classifier = NewsClassifier()
    
    # Try to load existing model
    if os.path.exists('news_classifier_model.pkl'):
        print("Loading existing model...")
        if classifier.load_model():
            return True
    
    # If no model exists, train a new one
    print("Training new model...")
    df = classifier.load_or_generate_data(num_samples=1000)
    if df is not None:
        X_train, X_test, y_train, y_test = classifier.prepare_data(df)
        if X_train is not None:
            classifier.train_models(X_train, y_train)
            classifier.save_model()
            return True
    
    return False

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        headline = data.get('headline', '').strip()
        
        if not headline:
            return jsonify({'error': 'Please provide a headline'}), 400
        
        if classifier is None:
            return jsonify({'error': 'Classifier not initialized'}), 500
        
        result = classifier.predict_single(headline)
        
        if isinstance(result, dict):
            return jsonify({
                'success': True,
                'prediction': result['category'],
                'confidence': round(result['confidence'] * 100, 2),
                'all_probabilities': {k: round(v * 100, 2) for k, v in result['all_probabilities'].items()}
            })
        else:
            return jsonify({'error': str(result)}), 500
            
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """API endpoint for batch predictions"""
    try:
        data = request.get_json()
        headlines = data.get('headlines', [])
        
        if not headlines:
            return jsonify({'error': 'Please provide headlines'}), 400
        
        if classifier is None:
            return jsonify({'error': 'Classifier not initialized'}), 500
        
        results = []
        for headline in headlines:
            if headline.strip():
                result = classifier.predict_single(headline.strip())
                if isinstance(result, dict):
                    results.append({
                        'headline': headline,
                        'prediction': result['category'],
                        'confidence': round(result['confidence'] * 100, 2)
                    })
                else:
                    results.append({
                        'headline': headline,
                        'error': str(result)
                    })
        
        return jsonify({'success': True, 'results': results})
        
    except Exception as e:
        return jsonify({'error': f'Batch prediction failed: {str(e)}'}), 500

@app.route('/stats')
def stats():
    """Get model statistics"""
    try:
        if classifier is None or classifier.label_map is None:
            return jsonify({'error': 'Classifier not initialized'}), 500
        
        categories = list(classifier.label_map.values())
        
        # Load dataset for stats
        if os.path.exists('news_dataset.csv'):
            df = pd.read_csv('news_dataset.csv')
            category_counts = df['Category'].value_counts().to_dict()
        else:
            category_counts = {cat: 0 for cat in categories}
        
        return jsonify({
            'success': True,
            'categories': categories,
            'category_counts': category_counts,
            'total_samples': sum(category_counts.values())
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to get stats: {str(e)}'}), 500

if __name__ == '__main__':
    print("Initializing News Classifier...")
    if initialize_classifier():
        print("Classifier ready!")
        print("Starting Flask application...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to initialize classifier. Please check your setup.")
        sys.exit(1)
