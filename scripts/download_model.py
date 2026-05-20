import os
import sys
from huggingface_hub import snapshot_download

def main():
    # Define model repository ID
    repo_id = "google/gemma-4-E2B-it"
    
    # Resolve the destination path: "../models/gemma-4-E2B-it" relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_dir = os.path.abspath(os.path.join(script_dir, "..", "models", "gemma-4-E2B-it"))
    
    print(f"Downloading model '{repo_id}' from Hugging Face...")
    print(f"Target local directory: {local_dir}")
    
    try:
        # Download the model snapshot
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,  # Download the actual files, not symlinks
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"],  # Optional: skip unused formats if any
        )
        print("\nDownload completed successfully!")
    except Exception as e:
        print(f"\nError downloading model: {e}", file=sys.stderr)
        print("Please ensure that you have run 'huggingface-cli login' if this model requires authentication,", file=sys.stderr)
        print("and that your internet connection is stable.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
