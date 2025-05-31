#  Secure Email Management System with PII Masking

## 1. Project Overview
This project implements a FastAPI for email classification and PII masking, utilizing machine learning models. Containerized with Docker, the API is deployed on Hugging Face Spaces, offering public access and interactive testing. It efficiently classifies emails, masks sensitive data, and facilitates demasking.

---

## 2. Problem Statement
Given an incoming email text, the system should:

-Accurately classify the email into one of several predefined categories (e.g., Sales, HR, Marketing).
-Detect and mask any personally identifiable information (PII) present within the email content.
-Provide functionality to demask previously masked PII using a provided mapping.
-Return all processing results (e.g., masked text, classification, extracted PII) in a well-defined JSON format.

---

## 3. Model Details
Overall Approach: Employs a two-pronged machine learning strategy for distinct tasks.
**Email Classification:**

**Feature Extraction:** Uses a TF-IDF Vectorizer to transform text into numerical features, capturing word importance.

**Classifier:** Employs a supervised machine learning classifier (e.g., Logistic Regression or similar scikit-learn model, based on your training choice) trained on categorized email datasets.

**PII Detection:**

**Library Used:** Utilizes the spaCy library.

**Specific Model:** Leverages spaCy's pre-trained statistical model, en_core_web_sm, for Named Entity Recognition (NER).

**Functionality:** Identifies various PII types including names, phone numbers, email addresses, and locations.

---

## 4. System Pipeline
-**1. Input:** Accepts JSON payload containing the email text (e.g., in a content field) for processing.
-**2. Text Preprocessing:** The raw input text undergoes cleaning, including lowercasing, removal of punctuation, special characters, and digits, and elimination of common stopwords.
-**3. PII Detection & Masking:** Utilizes spaCy's NER model to identify and then mask sensitive PII (e.g., names, phone numbers) within the cleaned text using designated placeholders.
-**4. Text Vectorization:** For classification purposes, the (potentially masked) text is transformed into numerical features using the pre-trained TF-IDF Vectorizer.
-**5. Email Classification:** The vectorized text is fed into the trained machine learning model to predict the email's category.
-**6. Output Generation:** Assembles the processed results, including the masked text, identified PII mapping (if applicable), and the predicted email category.
-**7. Output:** Returns a structured JSON response containing the processed information.

## 5. PII Detection & Masking
Types of PII detected include:
- Phone Numbers
- Email Addresses
- Aadhar Numbers
- Dates
- Expiry Numbers

Each entity is replaced with a corresponding placeholder like `[phone]`, `[aadhar_num]`, etc., and recorded in a list with position, classification, and original text.

---

## 6. API Endpoints
### `GET /`
Health check endpoint.
**Response:**
```json
{"message":"Welcome to the Email Classification and PII Masking API!"}
```

### `POST /classify`
Accepts email content and returns classification with PII masking.
**Request Body:**
```json
{
  "subject": "Important: KYC Update",
  "body": "Please update your Aadhar number 1234-5678-9123 in the system."
}
```

**Response:**
```json
{
  "detail": [
    {
      "type": "missing",
      "loc": [
        "body",
        "masked_email"
      ],
      "msg": "Field required",
      "input": {
        "subject": "Important: KYC Update",
        "body": "Please update your Aadhar number 1234-5678-9123 in the system."
      }
    },
    {
      "type": "missing",
      "loc": [
        "body",
        "masked_entities"
      ],
      "msg": "Field required",
      "input": {
        "subject": "Important: KYC Update",
        "body": "Please update your Aadhar number 1234-5678-9123 in the system."
      }
    }
  ]
}
```

---

## 7. Sample Testing via `curl`
curl -X 'POST' \
  'https://janardhanks537-email-pii-classifier.hf.space/classify' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "subject": "Important: KYC Update",
  "body": "Please update your Aadhar number 1234-5678-9123 in the system."
}'

---

## 8. Deployment & Source Code Links
- **Hugging Face Deployment**: [https://github.com/Ksjanardan/email-classification-system/](https://github.com/Ksjanardan/email-classification-system/)
- **GitHub Repository**: [https://github.com/Ksjanardan/email-classification-system/](https://github.com/Ksjanardan/email-classification-system/)

---

## 9. Conclusion
In conclusion, this project successfully developed and deployed a robust FastAPI-based system for automated email classification and Personally Identifiable Information (PII) masking. By integrating machine learning models (TF-IDF and a classifier) with spaCy's advanced Named Entity Recognition capabilities, the solution effectively addresses critical challenges in email management and data privacy. Containerization with Docker and deployment on Hugging Face Spaces ensured a scalable, accessible, and easily maintainable API, providing a practical tool for efficient email processing and enhanced data security

---

### Designed and executed by : **JANARDHAN K S**

