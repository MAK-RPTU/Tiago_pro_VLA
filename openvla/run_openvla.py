import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
import os
import sys

MODEL_ID = "/home/ubuntu/workspace/tiago_pro/share/hf_models/openvla-7b"  # 로컬 경로

IMAGE_PATH = "./images/green_apple.png"

def main():
    try:
        print("Starting OpenVLA script...", flush=True)
        print(f"PyTorch version: {torch.__version__}", flush=True)
        print(f"CUDA available: {torch.cuda.is_available()}", flush=True)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        print(f"Using device: {device}, torch_dtype: {dtype}", flush=True)

        print("Loading processor (local)...", flush=True)
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        print("Processor loaded successfully.", flush=True)

        print("Loading model (local, may take a while the first time)...", flush=True)
        vla = AutoModelForVision2Seq.from_pretrained(
            MODEL_ID,
            dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            attn_implementation="eager",  # Disable SDPA to avoid _supports_sdpa attribute error
            trust_remote_code=True,
        )
        print("Model loaded successfully.", flush=True)

        image_path = IMAGE_PATH
        if not os.path.exists(image_path):
            print(f"ERROR: Image file '{image_path}' not found!", flush=True)
            sys.exit(1)

        print(f"Loading image from {image_path}...", flush=True)
        image = Image.open(image_path).convert("RGB")

        system_prompt = (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions."
        )
        instruction = "pick up the apple"
        prompt = f"{system_prompt} USER: What action should the robot take to {instruction}? ASSISTANT:"

        print("Processing inputs...", flush=True)
        inputs = processor(prompt, image, return_tensors="pt").to(device)
        if device == "cuda":
            inputs = {k: (v.to(dtype) if v.dtype.is_floating_point else v) for k, v in inputs.items()}

        print("Predicting action...", flush=True)
        with torch.no_grad():
            #action = vla.predict_action(**inputs, do_sample=False)
            action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

        print("Action:", action, flush=True)
        try:
            print("shape:", action.shape, flush=True)
        except:
            pass
        print("Done.", flush=True)

    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
