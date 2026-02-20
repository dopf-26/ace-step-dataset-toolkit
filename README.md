# ACE-STEP Dataset Toolkit

A comprehensive CLI toolkit for preparing audio datasets for the ACE-STEP music generation model. This toolkit automates the process of audio captioning, lyric downloading, and dataset configuration.

## Features

- **Automated Audio Captioning**: Uses [ace-step-captioner](https://github.com/ace-step/ace-step-captioner) to generate descriptions for audio files
- **Lyric Downloading**: Integrates [genius-api](https://docs.genius.com/) to automatically fetch lyrics for your tracks
- **Metadata Integration**: Seamlessly combines key and BPM data from [Mixxx](https://www.mixxx.org/) DJ software
- **Smart Config Generation**: Automatically detects captions, lyrics, and BPM/key metadata, generating a training-ready configuration file
- **Windows Optimized**: Built with Windows support and streamlined setup

## Requirements

- **Windows OS**
- **[uv](https://github.com/astral-sh/uv)** - Fast Python package manager and installer
- **[Mixxx](https://www.mixxx.org/)** - For BPM and key detection (manual export)
- **[ace-step-captioner](https://github.com/ace-step/ace-step-captioner)** - Audio captioning
- **[genius-api](https://docs.genius.com/)** - Lyrics API access

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/dopf-26/ace-step-dataset-toolkit
   cd ace-step-dataset-toolkit
   ```

2. **Run the installation script**:
   ```bash
   install.bat
   ```
   This will create a uv virtual environment and install all necessary Python packages.

3. **Launch the toolkit**:
   ```bash
   run_toolkit.bat
   ```

4. **Setup Settings**:
   When running the individual steps you will be asked to provide a genius-api to download lyrics and setup the proper settings like cuda-selection, ace-step-captioner quantization and so on. To reset these settings delete the cli_config.json in the project folder.

## Dataset Structure

Organize your audio files in the following directory structure for the toolkit to work optimally:

```
your_dataset/
├── metadata.csv          (Exported from Mixxx)
└── audio/
    ├── track1.wav
    ├── track2.wav
    └── ...
```

### Metadata File

**metadata.csv** should be exported directly from Mixxx and contain BPM and key information for your tracks.
Import your audio files into mixxx, right click and analyze to get the BPM and Key.
Add them to a new playlist and then export the playlist as metadata.csv into your base folder.

### Generated Files

After running the toolkit, the following files will be created:

```
your_dataset/
├── metadata.csv
├── triggerword.json           (Training-ready configuration)
└── audio/
    ├── track1.wav
    ├── track1_caption.txt
    ├── track1_lyrics.txt
    └── ...
```

## Workflow

The toolkit operates in three main steps:

### Step 1: Prepare Metadata
- Export your Mixxx project and ensure `metadata.csv` is in your dataset's base folder
- Place all audio files in the `audio/` subfolder

### Step 2: Generate Captions & Lyrics
- **Captioning**: Audio files are automatically captioned using ace-step-captioner
- **Lyrics**: Track information is used to download lyrics via genius-api
- Both captions and lyrics are saved in the dataset audio subfolder
- I cant stress this enough: MANUALLY EDIT your lyrics and make sure that they fit 100% to the audio!

### Step 3: Generate Config
- The toolkit automatically detects all captions, lyrics, and metadata
- Combines BPM and key information from `metadata.csv`
- Generates `triggerword.json` - a complete, training-ready configuration file
- This file is placed in your dataset's base folder and ready for model training

## Configuration File

The generated `config.json` includes:

- Audio file paths and metadata
- Track captions (from ace-step-captioner)
- Lyrics (from genius-api)
- Musical metadata (BPM, key from Mixxx)
- All information formatted for direct use with ACE-STEP training

## Usage

```bash
run_toolkit.bat
```

Follow the on-screen prompts to:
1. Select your dataset folder
2. Generate captions and lyrics
3. Create the final training configuration

## Troubleshooting

- **Genius API Issues**: Ensure you have a valid Genius API token configured
- **Mixxx Metadata**: Verify `metadata.csv` contains the correct track information and is set to either english or german
- **Audio Files**: Confirm all audio files are in the `audio/` subfolder
- **Missing Dependencies**: Run `install.bat` again to ensure all packages are installed

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

This project is part of the ACE-STEP ecosystem. See the LICENSE file for details.

## References

- [ACE-STEP](https://github.com/ace-step)
- [ace-step-captioner](https://github.com/ace-step/ace-step-captioner)
- [Genius API](https://docs.genius.com/)
- [Mixxx Documentation](https://mixxx.org/manual/)
- [uv Documentation](https://docs.astral.sh/uv/)
- Thanks to mmoalem for further improving the code and adding mixxx to the mix!
