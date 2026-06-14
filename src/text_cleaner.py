import re

def clean_text(text):

    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'[^a-zA-Z0-9.,!?():;\-\s]', '', text)

    return text.strip()


def remove_references(text):

    pattern = r"References[\s\S]*"

    cleaned_text = re.sub(pattern, "", text)

    return cleaned_text