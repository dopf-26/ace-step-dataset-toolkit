import os
import sys
import re
import json
import lyricsgenius
from tinytag import TinyTag

# --- CONFIGURATION ---
CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cli_config.json")
# ---------------------

def load_config():
    """Load configuration from cli_config.json"""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_config(config):
    """Save configuration to cli_config.json"""
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

def get_genius_token():
    """Get Genius API token from config or prompt user"""
    config = load_config()
    
    if 'genius_api_token' in config and config['genius_api_token']:
        return config['genius_api_token']
    
    print("Genius API token not found in configuration.")
    print("You can get a free token at: https://genius.com/api-clients")
    token = input("Enter your Genius API token: ").strip()
    
    if token:
        config['genius_api_token'] = token
        save_config(config)
        print("Token saved to cli_config.json\n")
    
    return token

def clean_name(text):
    """Removes common suffixes that mess up search results."""
    if not text: return ""
    # Remove things like (Remastered), [Official Video], feat. Artist, etc.
    text = re.sub(r'[\(\[].*?[\)\]]', '', text) 
    text = re.sub(r'(?i)\s(feat|ft|with|prod)\.?\s.*', '', text)
    return text.strip()

def download_lyrics_for_folder():
    token = get_genius_token()
    
    if not token:
        print("Error: Genius API token is required to download lyrics.")
        return

    genius = lyricsgenius.Genius(token)
    genius.verbose = False 
    
    # Get base folder from command-line argument
    if len(sys.argv) > 1:
        base_folder = sys.argv[1].strip().replace('"', '').replace("'", "")
    else:
        base_folder = input("Enter the path to your project folder: ").strip().replace('"', '').replace("'", "")

    if not os.path.isdir(base_folder):
        print(f"Error: Path '{base_folder}' not found.")
        return

    # Audio folder is always audio/ subfolder
    folder_path = os.path.join(base_folder, "audio")
    if not os.path.isdir(folder_path):
        print(f"Error: No 'audio' subfolder found in '{base_folder}'.")
        return

    audio_extensions = ('.mp3', '.m4a', '.flac', '.wav', '.ogg', '.wma')
    print(f"\nScanning folder: {folder_path}\n")

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(audio_extensions):
            if filename.startswith("NOLYRICS_"):
                continue

            file_path = os.path.join(folder_path, filename)
            base_name, extension = os.path.splitext(filename)
            txt_path = os.path.join(folder_path, f"{base_name}_lyrics.txt")

            if os.path.exists(txt_path):
                print(f" -> Skipping: {filename} (Already exists)")
                continue
            
            try:
                tag = TinyTag.get(file_path)
                artist = clean_name(tag.artist) if tag.artist else ""
                title = clean_name(tag.title) if tag.title else base_name
                
                print(f"Searching: {artist} - {title}...")
                
                # Attempt 1: Specific search with artist filter
                song = genius.search_song(title, artist)

                # Attempt 2: If failed, try a "Broad Search" (better for missing punctuation)
                if not song:
                    print(f"    - Specific search failed. Trying broad fuzzy search...")
                    song = genius.search_song(f"{artist} {title}")

                if song:
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write(song.lyrics)
                    print(f"    SUCCESS: Saved {base_name}_lyrics.txt")
                else:
                    new_filename = f"NOLYRICS_{filename}"
                    os.rename(file_path, os.path.join(folder_path, new_filename))
                    print(f"    NOT FOUND: Renamed to {new_filename}")

            except Exception as e:
                print(f"    ERROR processing {filename}: {e}")

    print("\nTask complete.")

if __name__ == "__main__":
    download_lyrics_for_folder()