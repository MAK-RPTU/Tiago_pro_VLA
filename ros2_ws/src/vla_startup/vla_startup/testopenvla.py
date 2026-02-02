from transformers import AutoModelForVision2Seq
import torch

model = AutoModelForVision2Seq.from_pretrained(
    "/opt/pal/alum/share/hf_models/openvla-7b",
    attn_implementation="eager",
    trust_remote_code=True,
    local_files_only=True
)

print("Model loaded OK")