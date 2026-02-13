import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Paths
original_model_path = r"C:\Users\Admin\Desktop\hugging_face\nllb-600M-offline"
save_path = r"C:\Users\Admin\Desktop\hugging_face\nllb-600M-int8"

print(" Loading original model...")
model = AutoModelForSeq2SeqLM.from_pretrained(original_model_path)

print(" Applying dynamic quantization...")
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},  # Only linear layers quantized
    dtype=torch.qint8
)

#  Important: Instead of save_pretrained(), save state_dict only:
print(" Saving quantized weights...")
torch.save(quantized_model.state_dict(), f"{save_path}/pytorch_model.bin")

#  Save tokenizer + config for HuggingFace
print(" Saving tokenizer and configs...")
tokenizer = AutoTokenizer.from_pretrained(original_model_path)
tokenizer.save_pretrained(save_path)

model.config.save_pretrained(save_path)

print(" Quantized model saved successfully at:", save_path)
