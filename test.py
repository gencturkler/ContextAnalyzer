import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pickle
import os
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class BERTurkClassifier:
    def __init__(self, model_dir: str):
        """
        Initialize the BERTurk classifier with model, tokenizer, and label encoder.
        
        Args:
            model_dir: Path to directory containing the saved model, tokenizer, and label encoder
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Ensure model_dir is an absolute path without any problematic characters
        model_dir = os.path.abspath(model_dir)
        model_path = os.path.join(model_dir, "berturk_model")
        
        print(f"Loading model from: {model_path}")
        
        # Check if the model directory exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        
        # Load model components
        try:
            self.model = BertForSequenceClassification.from_pretrained(model_path)
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
        
        # Move model to appropriate device
        self.model = self.model.to(self.device)
        
        # Load label encoder
        label_encoder_path = os.path.join(model_dir, "label_encoder.pkl")
        if not os.path.exists(label_encoder_path):
            raise FileNotFoundError(f"Label encoder file not found: {label_encoder_path}")
            
        with open(label_encoder_path, "rb") as f:
            self.label_encoder = pickle.load(f)
        
        # Set model to evaluation mode
        self.model.eval()
    
    def predict(self, text: str, return_probs: bool = False) -> Dict:
        """
        Predict the category for a given text.
        
        Args:
            text: Input text to classify
            return_probs: If True, returns probabilities for all categories
            
        Returns:
            Dictionary containing predicted category and optionally probabilities
        """
        # Tokenize input text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128
        )
        
        # Move inputs to the same device as model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get predicted class
        logits = outputs.logits
        predicted_class_idx = torch.argmax(logits, dim=1).item()
        predicted_category = self.label_encoder.inverse_transform([predicted_class_idx])[0]
        
        result = {"predicted_category": predicted_category}
        
        if return_probs:
            # Convert logits to probabilities using softmax
            probs = torch.nn.functional.softmax(logits, dim=1).squeeze().cpu().numpy()
            
            # Create dictionary of category probabilities
            categories = self.label_encoder.classes_
            category_probs = {category: prob for category, prob in zip(categories, probs)}
            
            # Sort by probability (descending)
            sorted_probs = dict(sorted(category_probs.items(), key=lambda item: item[1], reverse=True))
            
            result["probabilities"] = sorted_probs
        
        return result
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        Predict categories for a batch of texts.
        
        Args:
            texts: List of texts to classify
            
        Returns:
            List of prediction dictionaries
        """
        return [self.predict(text) for text in texts]
    
    def evaluate(self, test_texts: List[str], test_labels: List[str]) -> Dict:
        """
        Evaluate the model on a test set.
        
        Args:
            test_texts: List of test texts
            test_labels: List of true labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Get predictions for all test texts
        predictions = []
        for text in test_texts:
            pred = self.predict(text)
            predictions.append(pred["predicted_category"])
        
        # Convert true labels from encoded form if needed
        if isinstance(test_labels[0], int):
            true_labels = self.label_encoder.inverse_transform(test_labels)
        else:
            true_labels = test_labels
        
        # Generate classification report
        report = classification_report(true_labels, predictions, output_dict=True)
        
        return {
            "classification_report": report,
            "accuracy": report["accuracy"],
            "predictions": predictions,
            "true_labels": true_labels
        }
    
    def analyze_dataset(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """
        Analyze a dataframe by adding prediction columns.
        
        Args:
            df: Input DataFrame
            text_column: Name of column containing text to classify
            
        Returns:
            DataFrame with added prediction columns
        """
        # Make predictions for each row
        predictions = []
        probabilities = []
        
        for text in df[text_column]:
            pred = self.predict(text, return_probs=True)
            predictions.append(pred["predicted_category"])
            probabilities.append(pred["probabilities"])
        
        # Add predictions to dataframe
        df["predicted_category"] = predictions
        df["prediction_probabilities"] = probabilities
        
        # Add top 3 predicted categories and their probabilities
        for i in range(1, 4):
            df[f"top_{i}_category"] = df["prediction_probabilities"].apply(lambda x: list(x.keys())[i-1])
            df[f"top_{i}_probability"] = df["prediction_probabilities"].apply(lambda x: list(x.values())[i-1])
        
        return df


def main():
    # Get the current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Navigate to the model directory using relative paths
    # Going up to parent_folder, then to my_bert_model
    MODEL_DIR = os.path.join(script_dir, "my_bert_model")
    
    print(f"Using model directory: {MODEL_DIR}")
    
    try:
        # Initialize classifier
        classifier = BERTurkClassifier(MODEL_DIR)
        
        # Example usage
        example_text = "Beyoğlu'nda bir binanın istinat duvarı çöktü. 📌Tedbir amacıyla 2 binanın boşaltılmasına karar verildi."
        
        # Single prediction
        prediction = classifier.predict(example_text, return_probs=True)
        print("\nExample Prediction:")
        print(f"Text: {example_text}")
        print(f"Predicted category: {prediction['predicted_category']}")
        print("Top 3 probabilities:")
        for category, prob in list(prediction['probabilities'].items())[:3]:
            print(f"  {category}: {prob:.4f}")
        
        # Batch prediction
        texts = [
            "Türkiye'de son dönemde yaşanan ekonomik gelişmeler piyasaları etkiledi.",
            "Galatasaray, Fenerbahçe derbisinde 3-2 kazandı.",
            "Bilim insanları yeni bir gezegen keşfetti."
        ]
        
        batch_predictions = classifier.predict_batch(texts)
        print("\nBatch Predictions:")
        for text, pred in zip(texts, batch_predictions):
            print(f"Text: {text[:50]}... | Predicted: {pred['predicted_category']}")
        
        # If you have test data saved (from the training script)
        test_data_path = os.path.join(MODEL_DIR, "test_data.pkl")
        if os.path.exists(test_data_path):
            with open(test_data_path, "rb") as f:
                test_data = pickle.load(f)
            
            evaluation = classifier.evaluate(test_data['texts'], test_data['labels'])
            print("\nEvaluation Results:")
            print(f"Accuracy: {evaluation['accuracy']:.2f}")
            print("\nClassification Report:")
            print(classification_report(evaluation['true_labels'], evaluation['predictions']))
        else:
            print(f"\nTest data not found at {test_data_path}")
        
        # If you want to analyze a CSV file
        # The CSV file is in the parent directory
        csv_path = os.path.join(script_dir, "modified_table1.csv")  
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            text_column = "Text"  # Change this to your text column name
            
            analyzed_df = classifier.analyze_dataset(df, text_column)
            output_path = os.path.join(script_dir, "analyzed_results.csv")
            analyzed_df.to_csv(output_path, index=False)
            print(f"\nAnalysis complete. Results saved to {output_path}")
        else:
            print(f"\nCSV file not found at {csv_path}")
            
    except Exception as e:
        print(f"Error in main function: {str(e)}")


if __name__ == "__main__":
    main()