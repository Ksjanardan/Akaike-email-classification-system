# email-classification-system

**1. Project Overview**

This project implements a FastAPI for email classification and PII masking, utilizing machine learning models. Containerized with Docker, the API is deployed on Hugging Face Spaces, offering public access and interactive testing. It efficiently classifies emails, masks sensitive data, and facilitates demasking.
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
**2. Problem Statement**

In today's digital landscape, organizations face challenges managing vast volumes of incoming emails, needing to categorize them efficiently while simultaneously protecting sensitive personal identifiable information (PII) to ensure privacy compliance. This project addresses these two key problems by developing an automated system capable of classifying emails into predefined categories and accurately detecting and masking PII within their content.
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
**3. Model Details**

Overall Approach: Employs a two-pronged machine learning strategy for distinct tasks.
Email Classification:
Feature Extraction: Uses a TF-IDF Vectorizer to transform text data into numerical features, capturing the importance of words within the email corpus.
Classifier: Feeds vectorized data into a supervised machine learning classifier (e.g., Logistic Regression or a similar scikit-learn model, based on your training choice) trained on categorized email datasets.
PII Detection:
Library Used: Utilizes the spaCy library.
Specific Model: Leverages spaCy's pre-trained statistical model, en_core_web_sm, for Named Entity Recognition (NER).
Functionality: Identifies various types of PII, including names, phone numbers, email addresses, and locations.
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
**4. System Pipeline**

The system operates as a robust API-driven pipeline. Upon receiving an email through the FastAPI endpoint, the raw text undergoes preprocessing. For classification, the preprocessed text is passed through the pre-trained TF-IDF vectorizer and then fed into the loaded classification model to predict its category. For PII handling, spaCy's NER capabilities identify PII entities, which are then masked or demasked as per the request. The entire application is containerized using Docker for consistent environment setup and deployed on Hugging Face Spaces, making it publicly accessible and scalable.
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
**5. PII Detection & Masking**

PII detection is performed using spaCy's Named Entity Recognition (NER) module, which is trained to recognize various entity types including common PII categories (e.g., PERSON, GPE for locations, ORG for organizations, and patterns for phone numbers/emails). Once identified, the detected PII is then masked within the text using a custom masking strategy, typically replacing the sensitive information with placeholder tokens (e.g., [PERSON], [EMAIL]) to protect privacy. The API also includes a demasking endpoint that reverses this process, reconstructing the original text if the masked tokens and their corresponding original values are provided.
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
