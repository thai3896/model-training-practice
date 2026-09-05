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
MINIPC_PROJECT_DIR="/root/model-training-practice"
# ---------------------------------------------------------

print_header "1. SYNCING DATASET (BYPASSING GIT)"
echo "Securely copying audio files directly to the Mini PC over the network..."
# rsync only syncs the changes, so it's lightning fast!
rsync -avz --progress ./voice_lab/dataset/ ${MINIPC_USER}@${MINIPC_HOST}:${MINIPC_PROJECT_DIR}/voice_lab/dataset/

print_header "2. EXECUTING PIPELINE ON MINI PC"
echo "Logging into ${MINIPC_HOST} to pull scripts and run training..."

ssh ${MINIPC_USER}@${MINIPC_HOST} << EOF
    set -e
    cd ${MINIPC_PROJECT_DIR}
    
    echo "📥 Pulling latest Python scripts from Git..."
    git pull
    
    cd voice_lab
    
    # NOTE: If you use a specific virtual environment on your Mini PC (like conda or venv), 
    # add the activation command here (e.g., 'source myenv/bin/activate')
    
    echo "🧠 STARTING AI TRAINING SCRIPT..."
    python3 fine_tune_voice.py
    
    echo "🎤 STARTING INFERENCE TEST..."
    python3 test_my_voice.py
EOF

print_header "✅ ALL DONE"
echo "The pipeline finished successfully on the Mini PC!"
echo "You can pull the generated 'presentation_test.wav' back to your Mac using:"
echo "scp ${MINIPC_USER}@${MINIPC_HOST}:${MINIPC_PROJECT_DIR}/voice_lab/presentation_test.wav ./"
