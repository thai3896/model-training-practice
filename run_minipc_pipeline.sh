#!/bin/bash
set -e

print_header() {
    echo ""
    echo "==============================================================="
    echo "$1"
    echo "==============================================================="
}

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
MINIPC_USER="root"
MINIPC_HOST="minipc-ubuntu"

# IMPORTANT: Change this to wherever the project lives on your Mini PC!
MINIPC_PROJECT_DIR="/root/thai/git/model-training-practice"
# ---------------------------------------------------------

# Cache your SSH key password so it doesn't prompt you 3 times!
echo "🔑 Checking SSH Agent..."
ssh-add ~/.ssh/id_rsa 2>/dev/null || echo "Please enter your password once to cache it."

print_header "1. PREPARING REMOTE DIRECTORY"
echo "Logging into ${MINIPC_HOST} to pull latest scripts from Git..."
ssh ${MINIPC_USER}@${MINIPC_HOST} << EOF
    set -e
    cd ${MINIPC_PROJECT_DIR}
    echo "📥 Pulling latest Python scripts from Git..."
    git pull
    
    # Ensure the dataset folder exists before we try to sync files into it!
    mkdir -p voice_lab/dataset
EOF

print_header "2. SYNCING FILES TO MINI PC"
echo "Securely copying scripts and audio files directly to the Mini PC over the network..."
rsync -avz --progress ./voice_lab/ ${MINIPC_USER}@${MINIPC_HOST}:${MINIPC_PROJECT_DIR}/voice_lab/

print_header "3. EXECUTING PIPELINE ON MINI PC"
echo "Logging in to start the AI training..."
ssh ${MINIPC_USER}@${MINIPC_HOST} << EOF
    set -e
    cd ${MINIPC_PROJECT_DIR}/voice_lab
    
    echo "📦 Installing system FFmpeg & eSpeak for PyTorch audio decoding..."
    apt-get update -qq && apt-get install -y ffmpeg espeak-ng
    
    echo "⚙️ Setting up Python 3.10 Environment (Coqui TTS requires < 3.12)..."
    # Install 'uv' if not present
    if ! command -v uv &> /dev/null; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
        # Use \$HOME so it resolves on the Mini PC, not on your Mac!
        source \$HOME/.local/bin/env || source \$HOME/.cargo/env
    fi
    
    # Create a Python 3.10 virtual environment
    if [ ! -d "${MINIPC_PROJECT_DIR}/tts_env" ]; then
        uv venv --python 3.10 ${MINIPC_PROJECT_DIR}/tts_env
    fi
    
    source ${MINIPC_PROJECT_DIR}/tts_env/bin/activate
    
    echo "📦 Installing PyTorch, Coqui TTS, and dependencies..."
    # We must pin setuptools < 70 because newer versions completely removed pkg_resources!
    uv pip install torch torchaudio TTS "setuptools<70.0.0" torchcodec soundfile
    
    echo "🧹 Freeing up GPU Memory (killing ghost processes from earlier tests)..."
    pkill -9 -f "fine_tune_voice.py" || true
    pkill -9 -f "test_my_voice.py" || true
    
    echo "🧠 STARTING AI TRAINING SCRIPT (with Auto-Restart enabled)..."
    while true; do
        python3 fine_tune_voice.py
        EXIT_CODE=\$?
        
        if [ \$EXIT_CODE -eq 0 ]; then
            echo "✅ Training completed successfully!"
            break
        elif [ \$EXIT_CODE -eq 130 ]; then
            echo "🛑 Training manually cancelled by user (Ctrl+C). Stopping."
            break
        else
            echo "⚠️ Training crashed unexpectedly (Exit Code: \$EXIT_CODE)!"
            echo "🧹 Cleaning up GPU and auto-restarting in 5 seconds to resume progress..."
            pkill -9 -f "fine_tune_voice.py" || true
            sleep 5
        fi
    done
    
    echo "🎤 STARTING INFERENCE TEST..."
    python3 test_my_voice.py
EOF

print_header "4. SYNCING RESULT BACK TO MAC"
echo "Pulling the AI-generated audio back to your local machine..."
rsync -avz --progress ${MINIPC_USER}@${MINIPC_HOST}:${MINIPC_PROJECT_DIR}/voice_lab/presentation_test.wav ./voice_lab/public/presentation_test.wav || echo "No test file generated yet."

echo "==============================================================="
echo "✅ PIPELINE FINISHED! Check the Web UI to hear your AI voice."
echo "==============================================================="
