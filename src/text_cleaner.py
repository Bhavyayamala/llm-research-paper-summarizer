import re

def clean_text(text):

    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)

    # Remove citations like [1], [25]
    text = re.sub(r'\[\d+\]', '', text)

    # Remove extra special characters
    text = re.sub(r'[^a-zA-Z0-9.,!?():;\-\s]', '', text)

    return text.strip()