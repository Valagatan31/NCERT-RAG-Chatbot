import re

# Read raw text
with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

#  Remove repeated headers/footers
text = re.sub(r"Reprint 20\d{2}-\d{2}", "", text)
text = re.sub(r"THE LIVING WORLD", "", text)

#  Merge lines within sentences
# Replace newline followed by lowercase letter with space
text = re.sub(r"\n(?=[a-z])", " ", text)

# Replace multiple newlines (paragraph breaks) with double newline
text = re.sub(r"\n{2,}", "\n\n", text)

# Normalize spaces
text = re.sub(r" {2,}", " ", text)

# remove page numbers (numbers alone in line)
text = re.sub(r"^\d+\s*$", "", text, flags=re.MULTILINE)

#  Strip leading/trailing spaces
text = text.strip()

#  Save cleaned text
with open("cleaned_data.txt", "w", encoding="utf-8") as f:
    f.write(text)

print("Text cleaned and saved as cleaned_data.txt")
