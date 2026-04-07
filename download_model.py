"""
download_model.py — Download GGUF models from Hugging Face.

Usage:
    python download_model.py --model TheBloke/Llama-2-7B-chat-GGUF --filename llama-2-7b-chat.Q4_K_M.gguf
    python download_model.py --list  # Show popular models
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

# Popular GGUF models
POPULAR_MODELS = {
    "llama2-7b": {
        "repo_id": "TheBloke/Llama-2-7B-chat-GGUF",
        "filename": "llama-2-7b-chat.Q4_K_M.gguf",
        "description": "Llama-2-7B (Q4_K_M) - 8GB VRAM friendly"
    },
    "mistral-7b": {
        "repo_id": "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
        "filename": "mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        "description": "Mistral-7B (Q4_K_M)"
    },
    "neural-chat-7b": {
        "repo_id": "TheBloke/neural-chat-7B-v3-2-GGUF",
        "filename": "neural-chat-7b-v3-2.Q4_K_M.gguf",
        "description": "Neural-Chat-7B v3.2 (Q4_K_M)"
    },
}

def list_models():
    """Print available models."""
    print("\n📦 Popular GGUF Models:\n")
    for key, info in POPULAR_MODELS.items():
        print(f"  {key:20} {info['description']}")
    print()

def download_model(repo_id: str, filename: str, output_name: str | None = None):
    """Download a model from Hugging Face."""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    token = os.getenv("HF_TOKEN")
    
    output_name = output_name or filename
    output_path = models_dir / output_name
    
    print(f"⏳ Downloading {filename}...")
    print(f"   From: {repo_id}")
    print(f"   To: {output_path}")
    print(f"   Auth: {'HF_TOKEN detected' if token else 'anonymous request'}")
    if not token:
        print("   Tip: set HF_TOKEN or run `hf auth login` for better large-file download reliability.")
    
    try:
        downloaded_path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=models_dir,
                token=token,
            )
        )

        if downloaded_path != output_path:
            if output_path.exists():
                output_path.unlink()
            downloaded_path.replace(output_path)

        print(f"✅ Successfully downloaded to: {output_path}")
        return str(output_path)
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Download GGUF models from Hugging Face"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List popular models"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Preset model key (e.g., 'llama2-7b') or full repo_id"
    )
    parser.add_argument(
        "--filename",
        type=str,
        help="Model filename in the repo"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Custom output filename (optional)"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_models()
        return
    
    # Use preset model
    if args.model and args.model in POPULAR_MODELS:
        info = POPULAR_MODELS[args.model]
        repo_id = info["repo_id"]
        filename = info["filename"]
        print(f"📥 Using preset: {args.model}")
    elif args.model and args.filename:
        repo_id = args.model
        filename = args.filename
    else:
        parser.print_help()
        list_models()
        return
    
    download_model(repo_id, filename, args.output)

if __name__ == "__main__":
    main()
