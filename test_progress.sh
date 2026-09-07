#!/bin/bash

# Configuration
MINIPC_USER="root"
MINIPC_HOST="minipc-ubuntu"
MINIPC_PROJECT_DIR="/root/thai/git/model-training-practice"

echo "==============================================================="
echo "🎤 MID-TRAINING CHECKER"
echo "==============================================================="
echo "This will safely test your AI's current progress WITHOUT stopping the training loop."

# Push the absolute latest inference script to the Mini PC (in case we updated it while training)
rsync -avz ./voice_lab/test_my_voice.py ${MINIPC_USER}@${MINIPC_HOST}:${MINIPC_PROJECT_DIR}/voice_lab/test_my_voice.py >/dev/null

# Run the inference script on the Mini PC
ssh ${MINIPC_USER}@${MINIPC_HOST} "cd ${MINIPC_PROJECT_DIR}/voice_lab && source ../tts_env/bin/activate && python3 test_my_voice.py"

echo ""
echo "📥 Syncing latest audio and training metadata back to Mac..."
rsync -avz --progress ${MINIPC_USER}@${MINIPC_HOST}:${MINIPC_PROJECT_DIR}/voice_lab/presentation_test.wav ./voice_lab/public/presentation_test.wav
rsync -avz --progress ${MINIPC_USER}@${MINIPC_HOST}:${MINIPC_PROJECT_DIR}/voice_lab/latest_step.txt ./voice_lab/latest_step.txt 2>/dev/null || true

echo ""
echo "✅ DONE! Refresh your Web UI to hear the current state of your AI."
