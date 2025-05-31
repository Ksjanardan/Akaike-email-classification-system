# Email Classification System for Support Team

## 1. Project Overview
This project is developed as part of the internship assignment for Akaike Technologies. The goal is to build an API-based system that classifies support team emails into categories and masks personally identifiable information (PII) in the email content.

---

## 2. Problem Statement
Given an incoming email with a subject and body, the system should:
- Detect and mask any personally identifiable information (PII)
- Classify the email into predefined categories (e.g., complaint, query, return, etc.)
- Return the results in a strictly defined JSON format

---

## 3. Model Details
- **Model Used**: Logistic Regression (or specify your actual model)
- **Vectorizer**: TfidfVectorizer
- **Training Data**: Custom labeled dataset with email samples

---

## 4. System Pipeline
1. **Input**: Accepts JSON with `subject` and `body`
2. **Combine Text**: Concatenates subject and body into one string
3. **PII Masking**: Detects and replaces PII using regular expressions and named entity recognition (NER)
4. **Text Cleaning**: Lowercasing, removing stopwords, special characters, etc.
5. **Vectorization**: Transforms text using TfidfVectorizer
6. **Classification**: Predicts the category using the trained model
7. **Output**: Returns a structured JSON response

---

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
{"message": "🚀 Email Classifier API is running!"}
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
  "input_email_body": "Important: KYC Update Please update your Aadhar number 1234-5678-9123 in the system.",
  "list_of_masked_entities": [
    {
      "position": [55, 59],
      "classification": "expiry_no",
      "entity": "1234"
    },
    {
      "position": [55, 69],
      "classification": "aadhar_num",
      "entity": "1234-5678-9123"
    }
  ],
  "masked_email": "Important: KYC Update Please update your Aadhar number [expiry_no][aadhar_num] in the system.",
  "category_of_the_email": "complaint"
}
```

---

## 7. Sample Testing via `curl`
```bash
curl -X POST "https://sahanavaidya-akaike-email-classification-system.hf.space/classify" \
  -H "Content-Type: application/json" \
  -d "{\"subject\": \"Important: KYC Update\", \"body\": \"Please update your Aadhar number 1234-5678-9123 in the system.\"}"
```

---

## 8. Deployment & Source Code Links
- **Hugging Face Deployment**: [https://sahanavaidya-akaike-email-classification-system.hf.space](https://sahanavaidya-akaike-email-classification-system.hf.space)
- **GitHub Repository**: [https://github.com/YOUR_USERNAME/akaike-email-classifier](https://github.com/YOUR_USERNAME/akaike-email-classifier)

---

## 9. Conclusion
This system is a lightweight and production-ready solution for automating email triaging and PII protection in customer support communications. The API strictly adheres to the format required by Akaike Technologies for automated evaluation.

---

Prepared by: **Sahana Vaidya**

