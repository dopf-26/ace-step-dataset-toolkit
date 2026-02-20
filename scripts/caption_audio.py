import os
import sys
import json
import warnings
import subprocess
import tempfile

# Suppress known model warnings
warnings.filterwarnings("ignore", message=".*Unrecognized keys.*rope_scaling.*")
warnings.filterwarnings("ignore", message=".*Qwen2_5OmniToken2WavModel.*")
warnings.filterwarnings("ignore", message=".*load_in_4bit.*load_in_8bit.*deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, module="bitsandbytes")

MODEL_REPO = "ACE-Step/acestep-captioner"
SUPPORTED_AUDIO_FORMATS = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac', '.wma']
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cli_config.json")

# ─────────────────────────────────────────────────────────────
# Edit your prompt here. The model output is saved as-is into
# a single _caption.txt sidecar file. build_dataset.py reads
# it and merges it with your Mixxx CSV to build the final JSON.
# ─────────────────────────────────────────────────────────────
CAPTION_PROMPT = (
    "You are a professional music metadata tagger preparing training data for ACE-Step. "
    "Listen carefully to this audio track and write a detailed description. "
    "Cover: specific instrumentation (name every instrument you hear — e.g. bowed cello, "
    "fortepiano, alto saxophone — not generic terms like strings or brass), "
    "whether vocals are present (gender, register, timbre) or confirm instrumental, "
    "recording and production character (e.g. concert hall reverb, dry studio, close-mic'd), "
    "mood and emotional character, and how the track develops from start to finish. "
    "Write 3 to 5 sentences. Start with A or An. "
    "Genre, BPM, key, and time signature are handled separately — do not include them."
)


def load_cli_config():
    if not os.path.exists(CONFIG_PATH):
        return {}
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"[WARNING] Failed to read config: {e}")
        return {}

def save_cli_config(config):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"[WARNING] Failed to save config: {e}")

def lazy_imports():
    global torch, Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, process_mm_info
    import torch
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    from qwen_omni_utils import process_mm_info

def list_gpus():
    """List GPUs using CUDA's device enumeration (not nvidia-smi order)"""
    try:
        import torch
        if not torch.cuda.is_available():
            return []
        gpus = []
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            gpus.append((i, None, name))
        return gpus
    except Exception:
        return []

def choose_gpu():
    print("\n" + "="*60)
    print("GPU SELECTION")
    print("="*60)
    gpus = list_gpus()
    if not gpus:
        print("[WARNING] No GPUs detected. Using default device.")
        return "default"
    print("Available GPUs:")
    for index, uuid, name in gpus:
        print(f"  {index}: {name}")
    print("  A: Use all GPUs")
    gpu_indices = {str(index) for index, _, _ in gpus}
    while True:
        choice = input("Select GPU index or A for all: ").strip().lower()
        if choice == "a":
            return "all"
        if choice in gpu_indices:
            return choice
        print("[ERROR] Invalid choice.")

def apply_gpu_choice(gpu_choice):
    if gpu_choice in ("all", "default"):
        return "auto", gpu_choice
    # Return specific device to prevent model from using all GPUs
    device_map = f"cuda:{gpu_choice}"
    print(f"[INFO] Using GPU {gpu_choice}")
    return device_map, str(gpu_choice)

def get_batch_size(default_value=1):
    print("\n" + "="*60)
    print("BATCH SIZE")
    print("="*60)
    print(f"Press Enter for default ({default_value}).")
    while True:
        choice = input("Batch size: ").strip()
        if not choice:
            return default_value
        try:
            v = int(choice)
            if v >= 1:
                return v
        except ValueError:
            pass
        print("[ERROR] Must be a positive integer.")

def get_precision_choice():
    print("\n" + "="*60)
    print("MODEL PRECISION")
    print("="*60)
    print("1. FP16  - Best quality  (~15-20GB VRAM)")
    print("2. 8-bit - Good quality  (~10-15GB VRAM)")
    print("3. 4-bit - Lower quality (~7-10GB VRAM)")
    print("="*60)
    while True:
        choice = input("Enter choice (1/2/3): ").strip()
        if choice == "1": return "fp16"
        if choice == "2": return "8bit"
        if choice == "3": return "4bit"
        print("[ERROR] Enter 1, 2, or 3.")

def load_model(model_repo, device_map, precision):
    print(f"\n[INFO] Loading model in {precision.upper()} precision...")
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"[INFO] GPU:  {torch.cuda.get_device_name(0)}")
        print(f"[INFO] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("[WARNING] CUDA not available — will run on CPU (very slow).")
    print()

    # Set torch_dtype based on precision
    if precision == "fp16":
        torch_dtype = torch.float16
    elif precision in ("8bit", "4bit"):
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    load_kwargs = {
        "device_map": device_map,
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
        # Use sdpa (scaled dot product attention) instead of flash_attention_2
        # because this model has components that require fp32
        "attn_implementation": "sdpa",
    }

    if precision == "8bit":
        load_kwargs["load_in_8bit"] = True
    elif precision == "4bit":
        load_kwargs["load_in_4bit"] = True

    print(f"[INFO] Using SDPA attention (compatible with model requirements).")

    try:
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_repo, **load_kwargs)
        model.disable_talker()
        print("[INFO] Talker disabled (text-only mode).")
        processor = Qwen2_5OmniProcessor.from_pretrained(model_repo, trust_remote_code=True)
        print("[SUCCESS] Model loaded.\n")
        return model, processor
    except Exception as e:
        print(f"\n[ERROR] Failed to load model: {e}")
        sys.exit(1)


def truncate_audio_to_duration(audio_path, max_duration_seconds):
    """
    Truncate audio file to a maximum duration using librosa.
    If max_duration_seconds is 0 or None, returns original path unchanged.
    Otherwise creates a temp file with truncated audio and returns its path.
    """
    if not max_duration_seconds or max_duration_seconds <= 0:
        return audio_path
    
    try:
        import librosa
        import soundfile as sf
        
        # Load audio truncated to max duration
        y, sr = librosa.load(audio_path, sr=None, mono=False, duration=max_duration_seconds)
        
        # Create temp file in system temp directory
        file_ext = os.path.splitext(audio_path)[1]
        temp_fd, temp_path = tempfile.mkstemp(suffix=file_ext)
        os.close(temp_fd)
        
        # Save truncated audio (handle mono/stereo)
        if y.ndim > 1:
            sf.write(temp_path, y.T, sr)
        else:
            sf.write(temp_path, y, sr)
        
        return temp_path
    
    except ImportError:
        print(f"[WARNING] librosa/soundfile not available, using full audio file")
        return audio_path
    except Exception as e:
        print(f"[WARNING] Failed to truncate {os.path.basename(audio_path)}: {e}")
        return audio_path


def extract_reply(full_text):
    """Strip conversation template boilerplate, return just the model's reply."""
    if "assistant\n" in full_text:
        return full_text.split("assistant\n")[-1].strip()
    if "assistant" in full_text:
        return full_text.split("assistant")[-1].strip()
    return full_text.strip()


def main():
    print("\n" + "="*60)
    print("ACE-STEP AUDIO CAPTIONER")
    print("="*60 + "\n")

    # Get base folder from command-line argument
    if len(sys.argv) > 1:
        base_folder = sys.argv[1]
    else:
        base_folder = input("Base folder path (drag & drop works): ").strip().replace('"', '').replace("'", '')
    
    if not os.path.isdir(base_folder):
        print(f"[ERROR] Not a valid folder: {base_folder}")
        sys.exit(1)
    
    # Audio folder is always audio/ subfolder
    audio_folder = os.path.join(base_folder, "audio")
    if not os.path.isdir(audio_folder):
        print(f"[ERROR] No 'audio' subfolder found in: {base_folder}")
        sys.exit(1)
    
    print(f"[INFO] Using project folder: {base_folder}")
    print(f"[INFO] Audio folder: {audio_folder}\n")

    config = load_cli_config()

    # GPU
    gpu_choice = config.get("gpu_choice")
    if gpu_choice is None:
        gpu_choice = choose_gpu()
        config["gpu_choice"] = gpu_choice
        save_cli_config(config)
    else:
        print(f"[INFO] Using saved GPU choice: {gpu_choice}")

    device_map, resolved = apply_gpu_choice(str(gpu_choice))
    if resolved != str(gpu_choice) and resolved not in ("all", "default"):
        config["gpu_choice"] = resolved
        save_cli_config(config)

    lazy_imports()

    if resolved not in ("all", "default"):
        if torch.cuda.is_available():
            print(f"[INFO] Active GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("[WARNING] CUDA still not detected — check drivers.")
    elif resolved == "all":
        print("[INFO] Using all available GPUs.")

    # Batch size
    batch_size = config.get("batch_size")
    if batch_size is None:
        batch_size = get_batch_size()
        config["batch_size"] = batch_size
        save_cli_config(config)
    else:
        print(f"[INFO] Batch size: {batch_size}")

    # Precision
    precision = config.get("precision")
    if precision is None:
        precision = get_precision_choice()
        config["precision"] = precision
        save_cli_config(config)
    else:
        print(f"[INFO] Precision: {precision}")

    model, processor = load_model(MODEL_REPO, device_map, precision)
    print(f"[INFO] Prompt: {CAPTION_PROMPT[:80]}...")

    audio_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(audio_folder)
        for f in files
        if os.path.splitext(f)[1].lower() in SUPPORTED_AUDIO_FORMATS
    ]

    if not audio_files:
        print(f"[WARNING] No audio files found in {audio_folder}")
        sys.exit(1)

    # Skip already captioned
    pending = [af for af in audio_files
               if not os.path.exists(os.path.splitext(af)[0] + "_caption.txt")]
    skipped = len(audio_files) - len(pending)

    print(f"\n[INFO] Found {len(audio_files)} audio file(s).")
    if skipped:
        print(f"[INFO] Skipping {skipped} already-captioned file(s).")
    print(f"[INFO] To process: {len(pending)}")

    if not pending:
        print("[INFO] All files already captioned.")
    else:
        # Ask for maximum audio duration
        print("\n" + "="*60)
        print("AUDIO DURATION LIMIT")
        print("="*60)
        max_duration = input("Max audio length in seconds (0 = use whole file): ").strip()
        try:
            max_duration = int(max_duration) if max_duration else 0
        except ValueError:
            print("[WARNING] Invalid input, using whole files.")
            max_duration = 0
        
        if max_duration > 0:
            print(f"[INFO] Will truncate audio to {max_duration} second(s) to save memory.")
        else:
            print("[INFO] Using full audio files.")
        print("="*60)
        success = error = 0

        for batch_start in range(0, len(pending), batch_size):
            batch = pending[batch_start:batch_start + batch_size]
            print(f"\n[Batch {batch_start+1}-{batch_start+len(batch)} of {len(pending)}]")
            for f in batch:
                print(f"  {os.path.basename(f)}")

            # Truncate audio files if needed
            truncated_batch = []
            truncated_paths = []
            try:
                for af in batch:
                    truncated_path = truncate_audio_to_duration(af, max_duration)
                    truncated_batch.append(af)  # Keep original for caption saving
                    truncated_paths.append(truncated_path)  # Use truncated for model
                
                conversations = [
                    [
                        {"role": "system", "content": [{"type": "text", "text": (
                            "You are Qwen, a virtual human developed by the Qwen Team, "
                            "Alibaba Group, capable of perceiving auditory and visual inputs, "
                            "as well as generating text and speech."
                        )}]},
                        {"role": "user", "content": [
                            {"type": "audio", "audio": trunc_af},
                            {"type": "text", "text": CAPTION_PROMPT},
                        ]},
                    ]
                    for trunc_af in truncated_paths
                ]

                text_input = processor.apply_chat_template(
                    conversations, add_generation_prompt=True, tokenize=False
                )
                audios, images, videos = process_mm_info(conversations, use_audio_in_video=False)
                inputs = processor(
                    text=text_input, audio=audios, images=images, videos=videos,
                    return_tensors="pt", padding=True, use_audio_in_video=False
                ).to(model.device).to(model.dtype)

                output_ids = model.generate(
                    **inputs, use_audio_in_video=False, return_audio=False, max_new_tokens=512
                )
                full_texts = processor.batch_decode(
                    output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )

                for af, trunc_af, full_text in zip(truncated_batch, truncated_paths, full_texts):
                    caption = extract_reply(full_text)
                    base = os.path.splitext(af)[0]
                    out_path = base + "_caption.txt"
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write(caption)
                    print(f"  [OK] {os.path.basename(out_path)}")
                    print(f"       {caption[:120]}{'...' if len(caption) > 120 else ''}")
                    success += 1
                    
                    # Clean up temp file if it was created
                    if trunc_af != af and os.path.exists(trunc_af):
                        try:
                            os.remove(trunc_af)
                        except Exception:
                            pass

            except Exception as e:
                print(f"  [ERROR] Batch failed: {e}")
                import traceback; traceback.print_exc()
                error += len(batch)
                # Clean up any temp files on error
                for orig_af, trunc_af in zip(truncated_batch, truncated_paths):
                    if trunc_af != orig_af and os.path.exists(trunc_af):
                        try:
                            os.remove(trunc_af)
                        except Exception:
                            pass

        print("\n" + "="*60)
        print(f"Done. Success: {success}  Failed: {error}")
        print("="*60)

if __name__ == "__main__":
    main()
