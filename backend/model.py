from huggingface_hub import snapshot_download

# Local folder where the model will be saved
local_dir = r"C:\Users\Admin\Desktop\hugging_face\openai-whisper-large"

# Download the Whisper Large model from Hugging Face
snapshot_download(
    repo_id="openai/whisper-large",
    local_dir=local_dir,
    local_dir_use_symlinks=False  
)

print("Whisper Large model downloaded successfully to:", local_dir)
