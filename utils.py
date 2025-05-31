# utils.py

import re
import spacy
from typing import List, Dict, Tuple

# Load SpaCy English model once globally when the utility file is loaded
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("SpaCy model 'en_core_web_sm' not found. Please run 'python -m spacy download en_core_web_sm'")
    nlp = None # Handle case where model is not loaded

def mask_pii(email_content: str) -> Tuple[str, List[Dict]]:
    """
    Detects and masks PII entities in the email content.

    Args:
        email_content (str): The original email content.

    Returns:
        Tuple[str, List[Dict]]: A tuple containing:
            - masked_email (str): The email content with PII masked.
            - masked_entities (List[Dict]): A list of dictionaries, each describing a masked entity.
                                            Each dictionary contains 'position', 'classification', and 'entity'.
                                            The 'position' here refers to the start and end in the *original* string.
    """
    if not email_content:
        return "", []

    # List to store detected entities before consolidation
    # Each entry will be a dict: {'start': int, 'end': int, 'text': str, 'classification': str, 'priority': int}
    # Higher priority means it should override lower priority overlaps
    # Priorities: Credit Card (7) > Aadhar (6) > Email (5) > DOB (regex-based) (4) > Expiry (with keywords, 3) > Phone (2) > CVV (1) > Full Name (0)
    raw_detected_entities = []

    # Helper function to add a detected entity with its priority
    def add_entity(start, end, text, classification, priority):
        raw_detected_entities.append({
            'start': start,
            'end': end,
            'text': text,
            'classification': classification,
            'priority': priority
        })

    # --- PII DETECTION LOGIC ---

    # 1. Credit/Debit Card Number Detection (Priority 7)
    credit_card_pattern = r'\b(?:\d[ -]*?){13,19}\b'
    for match in re.finditer(credit_card_pattern, email_content):
        add_entity(match.start(), match.end(), match.group(0), 'credit_debit_no', 7)

    # 2. Aadhar Card Number Detection (Priority 6)
    aadhar_pattern = r'\b\d{4}[ -]?\d{4}[ -]?\d{4}\b'
    for match in re.finditer(aadhar_pattern, email_content):
        add_entity(match.start(), match.end(), match.group(0), 'aadhar_num', 6)

    # 3. Email Address Detection (Priority 5)
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    for match in re.finditer(email_pattern, email_content):
        add_entity(match.start(), match.end(), match.group(0), 'email', 5)
        
    # 4. Date of Birth Detection (Regex-based, Priority 4)
    dob_pattern = r'\b(?:' \
                      r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|' \
                      r'\d{4}[/-]\d{1,2}[/-]\d{1,2}|' \
                      r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2},?\s+\d{4}' \
                      r')\b'
    for match in re.finditer(dob_pattern, email_content, re.IGNORECASE):
        # Context check for DOB keywords
        context_start = max(0, match.start() - 50)
        context_end = min(len(email_content), match.end() + 50)
        context = email_content[context_start:context_end].lower()
        has_dob_keywords = any(keyword in context for keyword in ["dob", "date of birth", "born on", "birth date"])
        if has_dob_keywords:
            add_entity(match.start(), match.end(), match.group(0), 'dob', 4)

    # 5. Card Expiry Number Detection (MM/YY or MM/YYYY) (Priority 3)
    # Reverted to stricter pattern and added context check
    expiry_pattern = r'\b(0[1-9]|1[0-2])\/?([0-9]{2}|[0-9]{4})\b'
    for match in re.finditer(expiry_pattern, email_content):
        # Context check for expiry keywords
        context_start = max(0, match.start() - 20) # Look in a smaller window for expiry
        context_end = min(len(email_content), match.end() + 20)
        context = email_content[context_start:context_end].lower()
        has_expiry_keywords = any(keyword in context for keyword in ["exp", "expires", "expiry", "valid thru", "valid until", "expiration"])
        if has_expiry_keywords:
            add_entity(match.start(), match.end(), match.group(0), 'card_exp_date', 3) # Corrected classification name

    # 6. Phone Number Detection (Priority 2)
    # Refined to prevent matching 4-digit years like '2025'
    phone_pattern = r'\b(?:\+?\d{1,3}[-.\s]?)?(?:\d{3}[-.\s]?){2,}\d{4}\b|\b(?:0|91)?[6789]\d{9}\b'
    for match in re.finditer(phone_pattern, email_content):
        add_entity(match.start(), match.end(), match.group(0), 'phone_number', 2)

    # 7. CVV Number Detection (3 or 4 digits) (Priority 1)
    cvv_pattern = r'\b\d{3,4}\b'
    for match in re.finditer(cvv_pattern, email_content):
        add_entity(match.start(), match.end(), match.group(0), 'cvv_no', 1)

    # 8. IP Address Detection
    ip_address_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    for match in re.finditer(ip_address_pattern, email_content):
        # Basic validation for IP address ranges (0-255)
        parts = [int(p) for p in match.group(0).split('.')]
        if all(0 <= p <= 255 for p in parts):
            add_entity(match.start(), match.end(), match.group(0), 'ip_address', 8) # Assign a high priority

    # 9. Passport Number (common formats - placeholder)
    passport_pattern = r'\b[A-Z]{1,2}\d{7,8}\b|\b\d{7,8}[A-Z]{1,2}\b' # Example: A1234567 or 1234567A
    for match in re.finditer(passport_pattern, email_content):
        add_entity(match.start(), match.end(), match.group(0), 'passport_num', 9) # Assign a high priority

    # 10. Driver's License Number (example common patterns - placeholder)
    # These vary heavily by region, so this is a generic example
    driver_license_pattern = r'\b[A-Z]{2}\d{13}\b|\b\d{15}\b|\b[A-Z]\d{7}[A-Z]\b'
    for match in re.finditer(driver_license_pattern, email_content):
        add_entity(match.start(), match.end(), match.group(0), 'driver_license_num', 10) # Assign a very high priority

    # 11. Full Name Detection (using SpaCy, Priority 0 for PERSON entities)
    if nlp:
        doc = nlp(email_content)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                # Basic filtering for common non-name entities that SpaCy might misclassify
                if (' ' in ent.text.strip() and # Ensure it's not a single word
                    not re.search(r'\d', ent.text) and # No digits in name
                    not re.search(r'[^\w\s\'-]', ent.text) and # Only letters, spaces, hyphens, apostrophes
                    len(ent.text.strip()) > 3 # Not too short
                   ):
                    add_entity(ent.start_char, ent.end_char, ent.text, 'full_name', 0)
            # SpaCy DATE entities - primarily used to reinforce DOB if keywords are present
            elif ent.label_ == "DATE":
                date_text = ent.text.strip()
                # Check if SpaCy's DATE entity matches our DOB regex patterns
                is_dob_pattern_spacy = re.match(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', date_text) or \
                                       re.match(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2},?\s+\d{4}\b', date_text)

                if is_dob_pattern_spacy:
                    context_start = max(0, ent.start_char - 50)
                    context_end = min(len(email_content), ent.end_char + 50)
                    context = email_content[context_start:context_end].lower()

                    has_dob_keywords_spacy = any(keyword in context for keyword in ["dob", "date of birth", "born on", "birth date"])

                    if has_dob_keywords_spacy:
                        add_entity(ent.start_char, ent.end_char, ent.text, 'dob', 4)

    else:
        print("Skipping SpaCy detection as model 'en_core_web_sm' was not loaded.")

    # --- Consolidation of Detected Entities ---
    # Sort by start position, then by priority (descending) to prioritize higher priority entities in overlaps
    raw_detected_entities.sort(key=lambda x: (x['start'], -x['priority']))

    final_unique_entities = []
    
    for current_entity in raw_detected_entities:
        is_covered_by_existing_higher_priority = False
        for final_entity in final_unique_entities:
            # Check if current entity is fully covered by an existing higher or equal priority entity
            if (final_entity['start'] <= current_entity['start'] and final_entity['end'] >= current_entity['end']) and \
               (final_entity['priority'] >= current_entity['priority']):
                is_covered_by_existing_higher_priority = True
                break
            
            # Check for overlap where current entity is higher priority and should replace/merge
            if current_entity['priority'] > final_entity['priority'] and \
               max(current_entity['start'], final_entity['start']) < min(current_entity['end'], final_entity['end']):
                # If there's an overlap and current_entity has higher priority, remove the existing one
                # and allow current_entity to be added or merged later.
                # For simplicity in this consolidation, we'll just remove the lower priority overlap.
                # More complex logic might merge ranges or split.
                # For this exercise, we are prioritizing by completely containing or by starting earlier with higher priority.
                pass # This is handled by sorting and the logic below.

        if not is_covered_by_existing_higher_priority:
            # Add entity if not covered, and then remove any lower-priority overlaps it now covers
            entities_to_remove_indices = []
            for i, final_entity in enumerate(final_unique_entities):
                # If current_entity completely contains a final_entity with lower priority
                if current_entity['priority'] > final_entity['priority'] and \
                   (current_entity['start'] <= final_entity['start'] and current_entity['end'] >= final_entity['end']):
                    entities_to_remove_indices.append(i)
                # If current_entity overlaps and has higher priority
                elif current_entity['priority'] > final_entity['priority'] and \
                     max(current_entity['start'], final_entity['start']) < min(current_entity['end'], final_entity['end']):
                    entities_to_remove_indices.append(i)
            
            for index in sorted(entities_to_remove_indices, reverse=True):
                del final_unique_entities[index]
            
            final_unique_entities.append(current_entity)

    # --- Masking Logic ---
    # Sort by start position in reverse order to mask from end to beginning
    # This prevents position shifts when characters are replaced
    final_unique_entities.sort(key=lambda x: x['start'], reverse=True)

    masked_email_chars = list(email_content)
    final_masked_entities_output = [] # This list will store details for demasking

    for entity_info in final_unique_entities:
        start_idx = entity_info['start']
        end_idx = entity_info['end']
        original_text = entity_info['text']
        classification = entity_info['classification']

        placeholder = f"[{classification}]"

        # Replace the original text with the placeholder in the character list
        masked_email_chars[start_idx:end_idx] = list(placeholder)

        # Store the original entity and its classification for demasking
        final_masked_entities_output.append({
            "position": [start_idx, end_idx], # Original position in the string
            "classification": classification,
            "entity": original_text
        })

    masked_email = "".join(masked_email_chars)

    # Sort the output entities by their original position for consistent demasking later
    final_masked_entities_output.sort(key=lambda x: x['position'][0])

    return masked_email, final_masked_entities_output


def demask_pii(masked_email: str, masked_entities: List[Dict]) -> str:
    """
    Demasks PII in the email content using the stored masked entities.

    Args:
        masked_email (str): The email content with PII masked (containing placeholders like '[full_name]').
        masked_entities (List[Dict]): A list of dictionaries, as returned by mask_pii,
                                       each describing a masked entity with its original 'position',
                                       'classification', and 'entity' (original text).

    Returns:
        str: The email content with PII demasked.
    """
    if not masked_email:
        return ""

    demasked_email_chars = list(masked_email)
    
    # Sort entities by their *current* position in the masked string in reverse order
    # to handle replacement without shifting indices.
    # We need to find the placeholder and replace it with the original entity.
    
    # This approach needs to dynamically find placeholders because their positions
    # will shift if previous replacements change length.
    # A more robust demasking might re-parse the masked string for placeholders.
    # However, for this simplified demasking, we rely on the original position
    # and the knowledge that placeholders are relatively consistent.

    # A more reliable demasking strategy when placeholders vary in length
    # is to rebuild the string.
    
    # Let's use a simpler, more robust regex-based placeholder replacement for demasking
    # that doesn't rely on original character indices after potential length changes.
    
    current_demasked_email = masked_email
    
    # Sort entities by position in ascending order for demasking from left to right.
    # This is important when placeholders might vary in length from original text.
    # If the original position was used, it would be incorrect due to shifts.
    # So, we will find the placeholder pattern and replace it.
    
    # Create a mapping from classification placeholder to original entity
    placeholder_to_original = {}
    for entity_info in masked_entities:
        classification = entity_info['classification']
        original_entity = entity_info['entity']
        placeholder = f"[{classification}]"
        
        # To handle cases where multiple instances of the same classification might exist,
        # and we need to replace them sequentially.
        # This requires more complex logic if positions are not exact.
        # For simplicity, we assume one-to-one replacement of *found placeholders*.
        # The stored masked_entities typically have the *original* positions.
        
        # A simple approach is to iterate through the original entities and
        # replace the first occurrence of their placeholder with their original text.
        # This relies on the original `mask_pii` capturing distinct entities.
        
        # To ensure correct demasking, we'll try to match the *first* occurrence
        # of a specific placeholder and replace it with its corresponding entity.
        # This is more robust if a single mask_pii call produced the masked_email.

        # The `masked_entities` list has the *original* text and classification.
        # We need to find the `[classification]` placeholder in the `masked_email`
        # and replace it with `original_text`.
        
        # Use re.sub with a counter or iterate to replace one by one if multiple same placeholders
        # A more straightforward way is to iterate through `masked_entities_output`
        # and use regex to find and replace the *first* instance of its placeholder type.
        
        # The safest approach is to build the original string piece by piece using the positions.
        # But since we altered `masked_email_chars` with `list(placeholder)`, the length changes.
        # So, the original position information is for the original string, not the masked string.
        
        # Let's adjust the demasking logic to work with the structure returned by `mask_pii`.
        # `masked_email` is the string with placeholders.
        # `masked_entities` contains `original_text` and `original_position`.
        
        # The key is to match the placeholder pattern `[classification]` in the `masked_email`
        # and replace it with the correct `entity` value from `masked_entities`.
        # To handle multiple instances of the same placeholder, we need to be careful.

        # Let's try to reconstruct the string piece by piece.
        # We need to know the *current* positions of the placeholders in the masked string.
        # Since `mask_pii` replaced with `f"[{classification}]"`, these placeholders are consistent.
        
        # We can create a list of parts of the string and insert the original entities.
        
        # This requires `masked_entities` to be sorted by original position.
        # However, the string `masked_email` already has shifted positions.
        # The simplest way is to replace placeholder by placeholder.

        # A common way to demask is to re-tokenize the masked email for placeholders
        # and then map them back using the stored original values.
        
        # The simplest way to implement demask_pii based on the `mask_pii` output:
        # Create a list of the masked email's characters.
        # Iterate through `masked_entities` sorted by ORIGINAL position.
        # This makes it hard as lengths change.
        
        # A better approach: Recreate the original string step by step.
        # We'll use the original positions, not the positions in the masked string.
        # We need a mechanism to track where the placeholder was *originally*.
        # The `masked_entities` output from `mask_pii` already contains original `position`.

    demasked_email = list(masked_email) # Start with characters of the masked email

    # We need to iterate through entities sorted by their *current* position in the masked email.
    # This is tricky because the lengths of placeholders can be different from original entities.
    # Example: "My name is John Doe." -> "My name is [full_name]."
    # If we replace `[full_name]` with `John Doe`, the string length changes.
    # Future replacements would be off if we use original indices.

    # A more robust demasking logic is to:
    # 1. Iterate through the `masked_email` string.
    # 2. When a placeholder `[classification]` is found, find the corresponding `original_text`
    #    from `masked_entities` (which has the original text and its original position).
    # 3. Replace the placeholder.

    # To handle multiple identical placeholders (e.g., two [phone_number] masks),
    # we need to ensure we map the correct original entity to the correct placeholder.
    # The `masked_entities` list from `mask_pii` already maintains order by original position.
    
    # Let's make `demask_pii` iterate through the original `masked_entities`
    # and find/replace the first occurrence of the specific placeholder in the `current_demasked_email`.
    
    current_demasked_email = masked_email
    
    # Sort entities by their *original* position for consistent demasking
    # (assuming `mask_pii` also sorted its output this way).
    # The `mask_pii` function already sorts `final_masked_entities_output` by position[0].
    sorted_masked_entities = sorted(masked_entities, key=lambda x: x['position'][0])

    # We need to apply replacements sequentially to avoid index issues.
    # Build a list of tuples: (start, end, replacement_text) for the *masked* string.
    # This is difficult because the `position` in `masked_entities` is based on the *original* string.
    
    # The easiest approach for this challenge is to use `re.sub` with a callback.
    # This will replace specific placeholders with corresponding original entities.
    
    # Create a mapping from placeholder type to a list of original entities
    # This assumes that the `masked_entities` list is correctly ordered
    # to match the order of placeholders in the masked string.
    # `mask_pii` returns `final_masked_entities_output` sorted by `position[0]`.
    
    # Let's use a regex to find all placeholders and then use a counter to map them.
    
    # Store original entities in a dictionary mapped by placeholder type and an index.
    # Example: {'[full_name]': ['John Doe', 'Jane Smith'], '[email]': ['john@example.com']}
    
    entity_map_by_type = {}
    for entity_info in sorted_masked_entities:
        placeholder_key = f"[{entity_info['classification']}]"
        if placeholder_key not in entity_map_by_type:
            entity_map_by_type[placeholder_key] = []
        entity_map_by_type[placeholder_key].append(entity_info['entity'])

    def replacement_callback(match):
        placeholder = match.group(0)
        if placeholder in entity_map_by_type and entity_map_by_type[placeholder]:
            # Pop the first entity from the list to handle sequential replacements
            return entity_map_by_type[placeholder].pop(0)
        return placeholder # Should not happen if data is consistent

    # The regex to find any placeholder like [classification]
    placeholder_pattern = r'\[[a-zA-Z_]+\]'
    
    # Apply replacements. re.sub will find all matches and apply the callback.
    # This is the most robust way to handle demasking without managing shifting indices manually.
    demasked_email_content = re.sub(placeholder_pattern, replacement_callback, masked_email)
    
    return demasked_email_content