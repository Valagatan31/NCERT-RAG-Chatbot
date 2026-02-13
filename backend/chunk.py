from langchain.text_splitter import RecursiveCharacterTextSplitter

# Read your data
with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Split text into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " "]
)

chunks = splitter.split_text(text)
print(f"Total Chunks: {len(chunks)}")

#  Save chunks
with open("chunks.txt", "w", encoding="utf-8") as f:
    for i, chunk in enumerate(chunks):
        f.write(f"===== Chunk {i+1} =====\n{chunk}\n\n")
