from utils import mask_pii

# --- Sample Emails for Testing PII Masking ---
sample_emails = [
    "Dear John Doe, your new email is john.doe@example.com. Please call me at +91-9876543210 or 011-12345678. Your DOB is 15/03/1990. Your Aadhar number is 1234 5678 9012. Your card is 1234-5678-9012-3456, CVV 123, expires 12/25.",
    "Hello Sarah Connor, my contact is sarah.c@company.org. Phone: 555-123-4567. My birth date is April 20, 1985. The Aadhar is 9876-5432-1098. Card: 9999 8888 7777 6666, CVV 456, exp 01/2026.",
    "No PII in this email, just a regular message about a meeting.",
    "Meeting with Alex Johnson on 05/01/2025. His email is alex@work.net. His phone is 123-456-7890. Card ending 1111 2222 3333 4444. CVV 789, expiry 06/27. Aadhar 0000-1111-2222. DOB: 12-25-1999."
]

print("--- Testing PII Masking ---")
for i, email in enumerate(sample_emails):
    print(f"\nOriginal Email {i+1}:")
    print(email)

    masked_email, masked_entities = mask_pii(email)

    print(f"\nMasked Email {i+1}:")
    print(masked_email)

    print(f"\nMasked Entities {i+1}:")
    for entity in masked_entities:
        print(f"  - Original: '{entity['entity']}', Classification: '{entity['classification']}', Position: {entity['position']}")
    print("-" * 50)