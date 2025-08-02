# Audio Processing Examples

This directory contains examples for audio processing, speech recognition, and text-to-speech.

## Examples

### Text-to-Speech
- `kokoro_tts.py` - Kokoro TTS (Text-to-Speech) example

### Voice Activity Detection & Speech Processing
- `vad_ultravox_kokoro_streaming.py` - Complete speech-to-speech pipeline with VAD, Ultravox ASR, and Kokoro TTS

## Running the Examples

Make sure you have the ML dependencies installed:
```bash
pip install -e ".[ml]"
```

Start the remote service if using remote execution:
```bash
cd ../remote_service
docker-compose up
```

Run an example:
```bash
python vad_ultravox_kokoro_streaming.py
```

## Audio Assets

Sample audio files are available in `../assets/audio/`

## Generated Outputs

Generated TTS outputs are saved to `../assets/generated/`