#!/bin/bash

MINIPC_USER="root"
MINIPC_HOST="minipc-ubuntu"
MINIPC_PROJECT_DIR="/root/thai/git/model-training-practice"

echo "==============================================================="
echo "🚀 DEMO DATASET GENERATOR"
echo "==============================================================="
echo "This script will use a pre-trained AI to instantly generate all 100 audio files."

echo ""
echo "📤 1. Syncing latest script and python generator to Mini PC..."
rsync -avz ./voice_lab/recording_script.txt ${MINIPC_USER}@${MINIPC_HOST}:${MINIPC_PROJECT_DIR}/voice_lab/recording_script.txt
rsync -avz ./generate_demo_audio.py ${MINIPC_USER}@${MINIPC_HOST}:${MINIPC_PROJECT_DIR}/generate_demo_audio.py

echo ""
echo "🧠 2. Running Neural Synthesis on the GPU..."
ssh ${MINIPC_USER}@${MINIPC_HOST} "cd ${MINIPC_PROJECT_DIR} && source tts_env/bin/activate && python3 generate_demo_audio.py"

echo ""
echo "📥 3. Pulling the 100 audio files back to your Mac..."
rsync -avz ${MINIPC_USER}@${MINIPC_HOST}:${MINIPC_PROJECT_DIR}/voice_lab/dataset/ ./voice_lab/dataset/

echo ""
echo "✅ ALL DONE! Refresh your Web UI, and you will see all 100 sentences are magically recorded!"
echo "You can now run ./run_minipc_pipeline.sh to test a full training run!"
