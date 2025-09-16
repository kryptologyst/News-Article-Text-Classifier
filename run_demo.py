#!/usr/bin/env python3
"""
Demo script to showcase the News Classifier functionality
"""

from news_classifier import NewsClassifier
import os

def main():
    print("=" * 60)
    print("🗞️  NEWS ARTICLE CLASSIFIER DEMO")
    print("=" * 60)
    
    # Initialize classifier
    print("\n1. Initializing classifier...")
    classifier = NewsClassifier()
    
    # Generate/load data
    print("\n2. Loading dataset...")
    df = classifier.load_or_generate_data(num_samples=500)  # Smaller for demo
    
    if df is None:
        print("❌ Failed to load data")
        return
    
    print(f"✅ Dataset loaded: {df.shape[0]} samples")
    print(f"📊 Categories: {', '.join(df['Category'].unique())}")
    
    # Prepare and train
    print("\n3. Training models...")
    X_train, X_test, y_train, y_test = classifier.prepare_data(df)
    
    if X_train is None:
        print("❌ Failed to prepare data")
        return
    
    best_model, best_score = classifier.train_models(X_train, y_train)
    print(f"✅ Best model: {best_model} (CV Score: {best_score:.3f})")
    
    # Evaluate
    print("\n4. Evaluating model...")
    accuracy, _ = classifier.evaluate_model(X_test, y_test)
    print(f"✅ Test accuracy: {accuracy:.3f}")
    
    # Save model
    print("\n5. Saving model...")
    classifier.save_model()
    print("✅ Model saved successfully")
    
    # Demo predictions
    print("\n6. Demo Predictions:")
    print("-" * 40)
    
    test_headlines = [
        "President announces new economic stimulus package",
        "Local basketball team advances to championship finals", 
        "Scientists develop breakthrough AI medical diagnostic tool",
        "Stock market reaches record high amid positive earnings",
        "Researchers discover potentially habitable exoplanet",
        "New fitness app helps users track mental wellness",
        "Hollywood blockbuster breaks opening weekend records"
    ]
    
    for i, headline in enumerate(test_headlines, 1):
        result = classifier.predict_single(headline)
        if isinstance(result, dict):
            print(f"\n{i}. '{headline}'")
            print(f"   → {result['category']} ({result['confidence']:.1%} confidence)")
        else:
            print(f"\n{i}. Error: {result}")
    
    print("\n" + "=" * 60)
    print("🎉 Demo completed successfully!")
    print("💡 To run the web interface: python app.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
