# models.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier # Or use MultinomialNB, LogisticRegression from sklearn.naive_bayes, sklearn.linear_model
from sklearn.metrics import classification_report, accuracy_score
import joblib
import re
import nltk
from nltk.corpus import stopwords

# --- IMPORTANT: Download NLTK Stopwords if you haven't already ---
# Uncomment the line below and run this script once if you get a Resource NLTK error.
# nltk.download('stopwords')
# -----------------------------------------------------------------

class EmailClassifier:
    def __init__(self):
        """
        Initializes the EmailClassifier.
        vectorizer: Stores the TF-IDF vectorizer fitted on training data.
        model: Stores the trained classification model.
        """
        self.vectorizer = None
        self.model = None

    def load_data(self, file_path):
        """
        Loads the dataset from the given file path.
        This method assumes your dataset is structured with one column for
        the email content and another for its classification category.

        Args:
            file_path (str): The full path to your dataset file (e.g., 'data/combined_emails_with_natural_pii.csv').

        Returns:
            tuple: A tuple containing two lists:
                   - emails (list of str): The list of email contents.
                   - categories (list of str): The list of corresponding email categories.
        """
        df = None
        # --- Adapt this section based on your actual file format ---
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Please provide a .csv, .json, or .xlsx file.")
        # --------------------------------------------------------

        if df is None:
            raise ValueError("Failed to load DataFrame from the provided file.")

        # --- UPDATED COLUMN NAMES TO MATCH YOUR CSV ---
        # Based on your screenshot, the email content column is 'Email'
        # and the category column is 'Type'.
        
        try:
            emails = df['email'].tolist() # Column containing the email text - UPDATED
            categories = df['type'].tolist() # Column containing the category labels - UPDATED
        except KeyError as e:
            raise KeyError(f"Missing expected column in dataset: {e}. "
                           "Please ensure 'Email' and 'Type' columns exist in your CSV with correct casing.")

        print(f"Dataset loaded from {file_path}.")
        print(f"Found {len(emails)} emails with {len(set(categories))} unique categories.")
        return emails, categories

    def preprocess_text(self, text):
        """
        Applies standard text cleaning for classification.
        This function should be applied to masked emails.
        It keeps PII placeholders (e.g., [full_name]) intact.
        """
        text = str(text).lower() # Convert to string and lowercase
        
        # Remove special characters, numbers, and punctuation, but preserve content inside [] (PII placeholders)
        # This regex matches anything that's not a letter, whitespace, or a character inside square brackets.
        text = re.sub(r'[^a-zA-Z\s\[\]]+', '', text)
        
        text = re.sub(r'\s+', ' ', text).strip() # Remove extra spaces

        # Optional: Remove stopwords. This is generally beneficial for Traditional ML models.
        # For advanced Deep Learning/LLMs, stopword removal is often skipped.
        words = text.split()
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]
        text = ' '.join(words)
        
        return text

    def train(self, file_path_to_dataset):
        """
        Trains the email classification model.

        Args:
            file_path_to_dataset (str): Path to the dataset containing emails and categories.
                                        It's assumed that the emails in this dataset
                                        are either already masked or will be masked
                                        before preprocessing if the raw dataset is used.
        """
        # Load the raw emails and categories from the dataset
        raw_emails, categories = self.load_data(file_path_to_dataset)

        # --- IMPORTANT: PII Masking for Training Data ---
        # Your 'combined_emails_with_natural_pii.csv' contains ORIGINAL emails.
        # You MUST mask them here before training the classification model.
        from utils import mask_pii # Assuming mask_pii is in utils.py
        
        print("Applying PII masking to training data...")
        masked_emails = []
        for email in raw_emails:
            masked_email_string, _ = mask_pii(email) # Get just the masked string
            masked_emails.append(masked_email_string)
        print("PII masking applied to training data.")
        # -------------------------------------------------
        
        # Preprocess the (masked) emails
        processed_emails = [self.preprocess_text(email) for email in masked_emails]

        # Split data into training and testing sets
        # stratify=categories ensures that the proportion of categories is maintained in splits
        X_train, X_test, y_train, y_test = train_test_split(
            processed_emails, categories, test_size=0.2, random_state=42, stratify=categories
        )

        # 1. Text Vectorization (TF-IDF)
        # max_features limits the number of features (words) to consider
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2)) # Consider 1-gram and 2-gram (word pairs)
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        # 2. Choose and train a Traditional ML model (e.g., RandomForestClassifier)
        # You can swap this with MultinomialNB(), LogisticRegression(), SVC() etc.
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        # 'class_weight='balanced'' can help if your categories are imbalanced.

        print(f"Training {type(self.model).__name__}...")
        self.model.fit(X_train_vec, y_train)
        print("Training complete.")

        # 3. Evaluate the model on the test set
        y_pred = self.model.predict(X_test_vec)
        print("\n--- Model Evaluation Report ---")
        print(classification_report(y_test, y_pred))
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("-------------------------------")

    def predict(self, email_content: str) -> str:
        """
        Predicts the category of a single masked email.
        This `email_content` should already be masked by your PII masking utility
        before being passed to this prediction method.

        Args:
            email_content (str): A single masked email string.

        Returns:
            str: The predicted category.
        """
        if self.model is None or self.vectorizer is None:
            raise Exception("Model not trained or loaded. Call train() first or load a saved model.")
        
        # 1. Preprocess the input email using the SAME preprocessing logic as training
        processed_email = self.preprocess_text(email_content)

        # 2. Vectorize the input email using the SAME vectorizer fitted during training
        # .transform expects an iterable, so we pass [processed_email]
        email_vec = self.vectorizer.transform([processed_email])
        
        # 3. Make prediction
        prediction = self.model.predict(email_vec)[0] # [0] because predict returns an array
        return str(prediction) # Ensure the predicted category is returned as a string

    def save_model(self, model_path="email_classifier_model.joblib", vectorizer_path="tfidf_vectorizer.joblib"):
        """
        Saves the trained model and vectorizer to disk.

        Args:
            model_path (str): Path to save the trained classification model.
            vectorizer_path (str): Path to save the fitted TF-IDF vectorizer.
        """
        if self.model is None or self.vectorizer is None:
            raise Exception("No model or vectorizer to save. Train or load first.")
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        print(f"Model saved to {model_path} and vectorizer to {vectorizer_path}")

    def load_model(self, model_path="email_classifier_model.joblib", vectorizer_path="tfidf_vectorizer.joblib"):
        """
        Loads a trained model and vectorizer from disk.

        Args:
            model_path (str): Path to the saved classification model.
            vectorizer_path (str): Path to the saved TF-IDF vectorizer.
        """
        try:
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            print(f"Model loaded from {model_path} and vectorizer from {vectorizer_path}")
        except FileNotFoundError:
            print(f"Model files not found. Please ensure {model_path} and {vectorizer_path} exist "
                  "and that you have trained and saved a model first.")
            self.model = None
            self.vectorizer = None # Ensure they are None if loading fails

# --- Example Usage (for initial training and testing) ---
# This block runs only when models.py is executed directly.
if __name__ == "__main__":
    classifier = EmailClassifier()
    
    # --- IMPORTANT: Set the correct path to your 'combined_emails_with_natural_pii.csv' file ---
    # Adjust this path based on where you placed your CSV file.
    # E.g., if it's in a 'data' folder next to models.py:
    # dataset_file_path = 'data/combined_emails_with_natural_pii.csv'
    # If it's in the same directory as models.py:
    dataset_file_path = 'combined_emails_with_natural_pii.csv'
    # -----------------------------------------------------------------------------------------

    try:
        # Call the train method to load data, preprocess, train, and evaluate
        print("\n--- Starting Model Training ---")
        classifier.train(dataset_file_path)
        print("--- Model Training Finished ---")

        # Save the trained model and vectorizer for later use by your API
        classifier.save_model()

        # --- Test the prediction with a sample masked email ---
        print("\n--- Testing Prediction ---")
        # This sample email should be masked before passing to predict
        sample_raw_email = "Hello, my name is [full_name], and my email is [email]. I have an [incident] regarding my service. My phone number is [phone_number]."
        
        # Mask the sample email using the utils.py function
        from utils import mask_pii
        sample_masked_email, _ = mask_pii(sample_raw_email)

        predicted_category = classifier.predict(sample_masked_email)
        print(f"Sample Masked Email: '{sample_masked_email}'")
        print(f"Predicted Category: {predicted_category}")
        print("--- Prediction Test Finished ---")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure your dataset path is correct and column names in load_data match your CSV.")