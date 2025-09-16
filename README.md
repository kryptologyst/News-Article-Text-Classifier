# News Article Text Classifier

A machine learning project that automatically classifies news headlines into categories using Natural Language Processing and multiple classification algorithms.

## Project Overview

This project demonstrates text classification for news articles by:
- Generating a realistic mock dataset of news headlines
- Training multiple ML models (Naive Bayes, Logistic Regression, Random Forest)
- Providing both command-line and web interfaces
- Evaluating model performance with detailed metrics

## Features

- **Smart Data Generation**: Creates realistic news headlines across 7 categories
- **Multiple ML Models**: Compares Naive Bayes, Logistic Regression, and Random Forest
- **Modern Web UI**: Beautiful, responsive interface for real-time predictions
- **Batch Processing**: Classify multiple headlines simultaneously
- **Model Persistence**: Save and load trained models
- **Comprehensive Evaluation**: Detailed metrics and confusion matrices

## Categories

The classifier can identify these news categories:
- **Politics** - Government, elections, policy
- **Sports** - Games, tournaments, athletes
- **Technology** - AI, software, innovations
- **Finance** - Markets, economy, business
- **Science** - Research, discoveries, space
- **Health** - Medicine, wellness, studies
- **Entertainment** - Movies, music, celebrities

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd 0063_Text_classification_for_news_articles
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**

**Command Line Interface:**
```bash
python news_classifier.py
```

**Web Interface:**
```bash
python app.py
```
Then open http://localhost:5000 in your browser.

## Usage

### Command Line
```python
from news_classifier import NewsClassifier

# Initialize classifier
classifier = NewsClassifier()

# Load/generate data and train
df = classifier.load_or_generate_data(num_samples=1000)
X_train, X_test, y_train, y_test = classifier.prepare_data(df)
classifier.train_models(X_train, y_train)

# Make predictions
result = classifier.predict_single("Government announces new healthcare policy")
print(f"Category: {result['category']}")
print(f"Confidence: {result['confidence']:.2f}")
```

### Web Interface
1. Start the Flask app: `python app.py`
2. Open http://localhost:5000
3. Enter headlines for classification
4. View results with confidence scores and probability distributions

## Model Performance

The system automatically:
- Trains multiple models and selects the best performer
- Uses cross-validation for robust evaluation
- Provides detailed classification reports
- Generates confusion matrices for analysis

Typical performance metrics:
- **Accuracy**: 85-95% (depending on dataset size)
- **F1-Score**: High across all categories
- **Training Time**: < 30 seconds for 1000 samples

## Project Structure

```
├── news_classifier.py      # Main classifier class
├── data_generator.py       # Mock dataset generator
├── app.py                 # Flask web application
├── templates/
│   └── index.html         # Web interface
├── requirements.txt       # Dependencies
├── README.md             # This file
└── 0063.py              # Original simple version
```

## 🔧 Technical Details

### Text Processing
- **TF-IDF Vectorization** with n-grams (1,2)
- **Stop words removal** for English
- **Feature selection** (max 5000 features)
- **Min/Max document frequency** filtering

### Machine Learning
- **Multinomial Naive Bayes**: Fast, effective for text
- **Logistic Regression**: Linear classifier with regularization  
- **Random Forest**: Ensemble method for robustness
- **Cross-validation**: 5-fold CV for model selection
- **Stratified splitting**: Maintains class balance

### Web Framework
- **Flask**: Lightweight Python web framework
- **Bootstrap 5**: Modern, responsive UI components
- **AJAX**: Real-time predictions without page refresh
- **RESTful API**: JSON endpoints for integration

## UI Features

- **Gradient backgrounds** and modern design
- **Real-time predictions** with loading animations
- **Confidence visualization** with progress bars
- **Category badges** with color coding
- **Sample headlines** for quick testing
- **Batch processing** interface
- **Model statistics** display

## API Endpoints

- `GET /` - Main web interface
- `POST /predict` - Single headline classification
- `POST /batch_predict` - Multiple headline classification
- `GET /stats` - Model and dataset statistics

## Future Enhancements

- [ ] Add more news categories
- [ ] Implement deep learning models (BERT, etc.)
- [ ] Real-time news feed integration
- [ ] Sentiment analysis addition
- [ ] Multi-language support
- [ ] Model retraining interface
- [ ] Performance monitoring dashboard

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Built with scikit-learn for machine learning
- UI powered by Bootstrap and Font Awesome
- Inspired by real-world news classification needs

# News-Article-Text-Classifier
