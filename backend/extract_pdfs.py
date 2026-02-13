import os
from PyPDF2 import PdfReader

# Folder paths
biology_folder = r"C:\Users\Admin\Desktop\curio\voice_assistence_python\backend\biology"
chemistry_folder = r"C:\Users\Admin\Desktop\curio\voice_assistence_python\backend\chemistry"
output_file = r"C:\Users\Admin\Desktop\curio\voice_assistence_python\backend\data.txt"

# Combine both folders
folders = [biology_folder, chemistry_folder]

# Create or clear data.txt
with open(output_file, "w", encoding="utf-8") as outfile:
    for folder in folders:
        for filename in os.listdir(folder):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(folder, filename)
                print(f"Extracting text from: {pdf_path}")

                try:
                    reader = PdfReader(pdf_path)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() or ""
                    
                    # Write to file with header
                    outfile.write(f"\n\n===== {filename} =====\n")
                    outfile.write(text.strip())
                except Exception as e:
                    print(f"Error reading {filename}: {e}")

print(f"\nAll text saved to: {output_file}")
