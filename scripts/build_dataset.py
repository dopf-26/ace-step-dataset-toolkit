#!/usr/bin/env python3
"""
Build Dataset JSON from captioned audio files + Mixxx CSV.
Output format matches the official ACE-Step Gradio app exactly.

Workflow:
  1. Run caption_audio.py on your audio folder  -> creates _caption.txt per file
  2. Export Mixxx library: Library > Export to CSV
  3. Run this script -> merges everything into dataset.json

Field sources:
  caption                    <- _caption.txt (AI-written, free-form)
  bpm, keyscale, genre,
  artist, title, album,
  duration                   <- Mixxx CSV
  timesignature              <- librosa beat analysis (estimated)
  lyrics/is_instrumental     <- detected from caption text or user input
  custom_tag                 <- user-supplied dataset name
"""

import os
import sys
import csv
import io
import json
import hashlib
import re
from datetime import datetime
from pathlib import Path

SUPPORTED_AUDIO_FORMATS = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac', '.wma'}


# ── ID generation ─────────────────────────────────────────────────────────────

def make_id(audio_path):
    """Generate a short hex ID from the filename, matching ACE-Step's format."""
    return hashlib.md5(Path(audio_path).name.encode()).hexdigest()[:8]


# ── Mixxx CSV ─────────────────────────────────────────────────────────────────

# Column name mappings: English -> German
MIXXX_COLUMNS = {
    'Location': 'Speicherort',
    'Key': 'Tonart',
    'Duration': 'Dauer',
    'Artist': 'Interpret',
    'Title': 'Titel',
    'Album': 'Album',
    'Genre': 'Genre',
    'BPM': 'BPM',
}

def get_mixxx_field(row, field_name):
    """Get a field from Mixxx row, trying both English and German column names."""
    # Try English first
    if field_name in row and row[field_name]:
        return row[field_name]
    # Try German equivalent
    german_name = MIXXX_COLUMNS.get(field_name, field_name)
    if german_name in row and row[german_name]:
        return row[german_name]
    return ''

def load_mixxx_csv(csv_path):
    with open(csv_path, 'r', encoding='utf-8') as f:
        raw = f.read()
    raw = raw.replace('\r\r\n', '\n').replace('\r\n', '\n').replace('\r', '\n')
    reader = csv.DictReader(io.StringIO(raw))
    records = {}
    for row in reader:
        location = get_mixxx_field(row, 'Location').strip()
        if location:
            records[Path(location).stem.lower()] = row
    print(f"[INFO] Loaded {len(records)} track(s) from Mixxx CSV.")
    return records


def duration_to_seconds(duration_str):
    """'M:SS' or 'H:MM:SS' -> int seconds."""
    parts = duration_str.strip().split(':')
    try:
        parts = [int(p) for p in parts]
        if len(parts) == 2: return parts[0] * 60 + parts[1]
        if len(parts) == 3: return parts[0] * 3600 + parts[1] * 60 + parts[2]
    except ValueError:
        pass
    return None


def format_keyscale(mixxx_key):
    """
    Convert Mixxx key to ACE-Step keyscale format.
    Mixxx gives 'B♭' or 'Gm' — ACE-Step wants 'Bb major' or 'G minor'.
    If two keyscales are present separated by '/', keep only the second one.
    Special characters are normalized: '♭' -> 'b', '♯' -> '#'
    """
    key = mixxx_key.strip()
    if not key:
        return ""
    # If there are two keyscales separated by '/', keep only the second one
    if '/' in key:
        key = key.split('/')[-1].strip()
    # Replace special characters with ASCII equivalents
    key = key.replace('♭', 'b').replace('♯', '#')
    # If already formatted with minor/major, return as-is
    if 'minor' in key.lower() or 'major' in key.lower():
        return key
    # Otherwise, apply transformation based on suffix
    if key.endswith('m'):
        # Minor key: 'Gm' -> 'G minor'
        return key[:-1] + ' minor'
    else:
        # Major key: 'Bb' -> 'Bb major'
        return key + ' major'


def clean_bpm(bpm_str):
    """Return BPM as int (matching ACE-Step format)."""
    try:
        return int(round(float(bpm_str)))
    except (ValueError, TypeError):
        return None


# ── Time signature detection ──────────────────────────────────────────────────

def get_audio_duration(audio_path):
    """Get the duration of an audio file in seconds."""
    try:
        import librosa
        duration = librosa.get_duration(path=str(audio_path))
        return int(round(duration))
    except ImportError:
        return None
    except Exception:
        return None


def detect_time_signature(audio_path):
    """
    Estimate time signature using librosa beat tracking + autocorrelation.
    Returns "2", "3", or "4" as a string (matching ACE-Step format), or None.
    Accuracy ~75-80% on classical. Flagged in metadata as estimated.
    Note: Only loads first 90 seconds for analysis (not for duration calculation).
    """
    try:
        import librosa
        import numpy as np

        # Only load first 90 seconds for time signature analysis
        y, sr = librosa.load(str(audio_path), sr=None, mono=True, duration=90)
        _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

        if len(beat_frames) < 12:
            return None

        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        s = onset_env[beat_frames].astype(float)
        s = (s - s.mean()) / (s.std() + 1e-8)

        def autocorr(arr, lag):
            return float(np.corrcoef(arr[:-lag], arr[lag:])[0, 1])

        c2 = autocorr(s, 2)
        c3 = autocorr(s, 3)
        c4 = autocorr(s, 4)

        best = max(c2, c3, c4)
        if best == c4: return "4"
        if best == c3: return "3"
        return "2"

    except ImportError:
        return None
    except Exception:
        return None


# ── Caption & lyrics ──────────────────────────────────────────────────────────

def read_caption(audio_path):
    caption_path = Path(str(Path(audio_path).with_suffix('')) + '_caption.txt')
    if caption_path.exists():
        try:
            return caption_path.read_text(encoding='utf-8').strip()
        except Exception:
            pass
    return None


def read_lyrics(audio_path):
    """Read a _lyrics.txt sidecar if present (from download-lyrics.py)."""
    lyrics_path = Path(str(Path(audio_path).with_suffix('')) + '_lyrics.txt')
    if lyrics_path.exists():
        try:
            return lyrics_path.read_text(encoding='utf-8').strip()
        except Exception:
            pass
    return None


def detect_instrumental(caption, lyrics):
    """
    Determine is_instrumental from caption text and lyrics content.
    Returns True if the track appears to be instrumental.
    """
    if lyrics and lyrics.strip() and lyrics.strip() != "[Instrumental]":
        return False
    if caption:
        instrumental_hints = ['instrumental', 'no vocal', 'no vocals',
                               'without vocal', 'without vocals']
        vocal_hints = ['vocalist', 'singer', 'singing', 'vocals', 'voice',
                       'soprano', 'alto', 'tenor', 'baritone', 'choir', 'chorus']
        caption_lower = caption.lower()
        if any(h in caption_lower for h in instrumental_hints):
            return True
        if any(h in caption_lower for h in vocal_hints):
            return False
    return True  # default to instrumental if uncertain


# ── Sample builder ────────────────────────────────────────────────────────────

def build_sample(audio_path, mixxx_row, custom_tag, detect_timesig,
                 timesig_estimated_flags, manual_timesig=None, force_instrumental=False, output_base=None, language="unknown", default_genre=""):
    audio_path = Path(audio_path)

    caption  = read_caption(audio_path) or ""
    lyrics   = read_lyrics(audio_path)
    
    # Force instrumental if specified, otherwise detect
    if force_instrumental:
        is_instr = True
    else:
        is_instr = detect_instrumental(caption, lyrics)

    # Lyrics fields — match ACE-Step exactly
    if is_instr:
        lyrics_field = "[Instrumental]"
    else:
        lyrics_field = lyrics or ""

    # Calculate relative path if output_base is provided
    if output_base:
        try:
            relative_path = audio_path.relative_to(output_base)
            audio_path_str = str(relative_path).replace('\\', '/')
        except ValueError:
            # Fallback to absolute if relative path calculation fails
            audio_path_str = str(audio_path)
    else:
        audio_path_str = str(audio_path)

    sample = {
        "id":               make_id(audio_path),
        "audio_path":       audio_path_str,
        "filename":         audio_path.name,
        "caption":          caption,
        "genre":            default_genre,
        "lyrics":           lyrics_field,
        "raw_lyrics":       "",
        "formatted_lyrics": "",
        "bpm":              None,
        "keyscale":         "",
        "timesignature":    "",
        "duration":         None,
        "language":         language,
        "is_instrumental":  is_instr,
        "custom_tag":       custom_tag,
        "labeled":          True,
        "prompt_override":  None,
    }

    if mixxx_row:
        bpm      = clean_bpm(get_mixxx_field(mixxx_row, 'BPM'))
        keyscale = format_keyscale(get_mixxx_field(mixxx_row, 'Key'))
        genre    = get_mixxx_field(mixxx_row, 'Genre').strip().lower()
        duration = duration_to_seconds(get_mixxx_field(mixxx_row, 'Duration'))

        if bpm:      sample['bpm']      = bpm
        if keyscale: sample['keyscale'] = keyscale
        if genre:    sample['genre']    = genre
        if duration: sample['duration'] = duration
    
    # If duration not available from Mixxx, calculate from audio file
    if not sample['duration']:
        sample['duration'] = get_audio_duration(audio_path)

    if detect_timesig:
        ts = detect_time_signature(audio_path)
        if ts:
            sample['timesignature'] = ts
            timesig_estimated_flags.append(audio_path.name)
    elif manual_timesig:
        sample['timesignature'] = manual_timesig

    return sample


# ── UI helpers ────────────────────────────────────────────────────────────────

def ask_yes_no(prompt, default=True):
    tag = "Y/n" if default else "y/N"
    while True:
        val = input(f"{prompt} [{tag}]: ").strip().lower()
        if not val:            return default
        if val in ('y','yes'): return True
        if val in ('n','no'):  return False
        print("  Enter y or n.")

def get_tag_position():
    """Ask user for tag_position choice."""
    print("\n" + "="*60)
    print("TAG POSITION")
    print("="*60)
    print("How should the custom_tag relate to captions when training?")
    print("(This is metadata only — not applied to data, used by downstream tools)")
    print("  1. replace  - Custom tag replaces the caption")
    print("  2. prepend  - Custom tag is prepended to the caption")
    print("  3. append   - Custom tag is appended to the caption")
    print("="*60)
    while True:
        choice = input("Enter choice (1/2/3): ").strip()
        if choice == "1": return "replace"
        if choice == "2": return "prepend"
        if choice == "3": return "append"
        print("  Enter 1, 2, or 3.")

def get_input(prompt, default=None):
    suffix = f" (default: {default})" if default else ""
    while True:
        val = input(f"{prompt}{suffix}: ").strip().replace('"','').replace("'",'')
        if val:     return val
        if default: return default
        print("  Please enter a value.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("BUILD DATASET JSON  (ACE-Step format)")
    print("="*60 + "\n")

    # Input folder from command-line argument or user input
    if len(sys.argv) > 1:
        input_folder = Path(sys.argv[1]).resolve()
    else:
        input_folder = Path(get_input("Input folder path")).resolve()
    
    if not input_folder.is_dir():
        print(f"[ERROR] Not a valid folder: {input_folder}")
        sys.exit(1)

    # Audio subfolder
    audio_folder = input_folder / "audio"
    if not audio_folder.is_dir():
        print(f"[ERROR] No 'audio' subfolder found in {input_folder}")
        sys.exit(1)

    audio_files = sorted(
        f for f in audio_folder.iterdir()
        if f.suffix.lower() in SUPPORTED_AUDIO_FORMATS
    )
    if not audio_files:
        print(f"[ERROR] No audio files found in {audio_folder}")
        sys.exit(1)
    
    print(f"[INFO] Using project folder: {input_folder}")
    print(f"[INFO] Found {len(audio_files)} audio file(s) in {audio_folder.name}/\n")

    # Dataset name / custom tag
    dataset_name = get_input("Dataset name / custom tag (e.g. quart3t, my-lora)")
    dataset_name = dataset_name.replace(' ', '-')
    
    # Check if all tracks are instrumental
    print()
    all_instrumental = ask_yes_no("Is this dataset purely instrumental (no vocals)?", default=False)
    if all_instrumental:
        print("[INFO] All tracks will be marked as instrumental.")
        language = "unknown"  # Instrumental tracks don't have a language
    else:
        language = get_input("Enter language tag (e.g., en, es, de, fr, it, pt)", default="en")
        print(f"[INFO] Setting language to: {language}")

    # Auto-detect Mixxx CSV
    print()
    mixxx_lookup = {}
    csv_path = input_folder / "metadata.csv"
    if csv_path.is_file():
        print(f"[INFO] Found metadata.csv, loading...")
        mixxx_lookup = load_mixxx_csv(csv_path)
    else:
        print(f"[INFO] No metadata.csv found in {input_folder.name}/, skipping Mixxx metadata.")

    # Time signature
    print()
    detect_timesig = ask_yes_no(
        "Detect time signature via librosa? (~75-80% accurate, flagged as estimated)",
        default=True
    )
    
    manual_timesig = None
    if not detect_timesig:
        while True:
            ts_input = input("Enter manual time signature for all tracks (2, 3, 4, or 6): ").strip()
            if ts_input in ('2', '3', '4', '6'):
                manual_timesig = ts_input
                print(f"[INFO] Will set time signature to {manual_timesig}/4 for all tracks.")
                break
            elif ts_input == '':
                print("[INFO] Skipping time signature.")
                break
            else:
                print("  Please enter 2, 3, 4, 6, or leave empty to skip.")

    # Genre for all tracks
    print()
    default_genre = get_input("Genre for all tracks (e.g., classical, electronic, jazz)", default="")
    if default_genre:
        print(f"[INFO] Will set genre to: {default_genre}")
    else:
        print("[INFO] Skipping genre.")

    # Tag position
    tag_position = get_tag_position()
    print(f"[INFO] Tag position: {tag_position}")

    # Output path
    output_path = input_folder / f"{dataset_name}.json"
    print(f"\n[INFO] Output will be saved as: {output_path.name}")

    # Build samples
    print("\n" + "="*60)
    print("BUILDING DATASET")
    print("="*60)

    samples = []
    timesig_estimated = []
    stats = {"captioned": 0, "no_caption": 0, "matched": 0, "unmatched": 0, "instrumental": 0}

    for i, audio_path in enumerate(audio_files, start=1):
        print(f"  [{i:3d}/{len(audio_files)}] {audio_path.name}", end=" ", flush=True)

        mixxx_row = mixxx_lookup.get(audio_path.stem.lower())
        if mixxx_lookup:
            if mixxx_row:
                stats["matched"] += 1
            else:
                stats["unmatched"] += 1
                print(f"\n    [WARNING] No Mixxx match", end=" ")

        sample = build_sample(
            audio_path, mixxx_row, dataset_name,
            detect_timesig, timesig_estimated, manual_timesig, all_instrumental, input_folder, language, default_genre
        )
        samples.append(sample)

        if sample["caption"]:
            stats["captioned"] += 1
        else:
            stats["no_caption"] += 1
            print(f"\n    [WARNING] No caption file", end=" ")

        if sample["is_instrumental"]:
            stats["instrumental"] += 1

        info_parts = []
        if sample.get("bpm"):       info_parts.append(f"BPM:{sample['bpm']}")
        if sample.get("keyscale"):  info_parts.append(sample['keyscale'])
        if sample.get("timesignature"): info_parts.append(f"{sample['timesignature']}/4")
        if info_parts:
            print(f"-> {', '.join(info_parts)}", end="")
        print()

    # Top-level metadata block (matches ACE-Step format)
    dataset = {
        "metadata": {
            "name":             f"{dataset_name}_dataset",
            "custom_tag":       dataset_name,
            "tag_position":     tag_position,
            "created_at":       datetime.now().isoformat(),
            "num_samples":      len(samples),
            "all_instrumental": all_instrumental,
            "genre_ratio":      30,
        },
        "samples": samples,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    # Summary
    print("\n" + "="*60)
    print("DONE")
    print("="*60)
    print(f"Total samples:       {len(samples)}")
    print(f"With captions:       {stats['captioned']}")
    if stats['no_caption']:
        print(f"Missing captions:    {stats['no_caption']}  <- run caption_audio.py first")
    if mixxx_lookup:
        print(f"Mixxx matches:       {stats['matched']}")
        if stats['unmatched']:
            print(f"Unmatched to CSV:    {stats['unmatched']}  <- check filenames match")
    print(f"Instrumental:        {stats['instrumental']}/{len(samples)}")
    if detect_timesig and timesig_estimated:
        print(f"Time sig detected:   {len(timesig_estimated)} (estimated, ~75-80% accurate)")
    print(f"\nOutput: {output_path}")
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INFO] Cancelled.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)
