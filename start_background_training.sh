#!/bin/bash

MINIPC_HOST="minipc-ubuntu"
MINIPC_USER="root"
MINIPC_PROJECT_DIR="/root/thai/git/model-training-practice"

echo "=========================================================="
echo "🚀 DEPLOYING MASSIVE BACKGROUND TRAINING TO MINI PC"
echo "=========================================================="

# 1. Sync the updated Python code to the Mini PC
echo "🔄 Syncing latest code..."
rsync -avz --exclude 'tts_train_output' --exclude 'dataset/phoneme_cache' ./voice_lab/ ${MINIPC_USER}@${MINIPC_HOST}:${MINIPC_PROJECT_DIR}/voice_lab/

# 2. Create the fault-tolerant crash-loop script DIRECTLY on the Mini PC
echo "⚙️  Building fault-tolerant loop on Mini PC..."
ssh ${MINIPC_USER}@${MINIPC_HOST} "cat << 'EOF' > ${MINIPC_PROJECT_DIR}/voice_lab/minipc_loop.sh
#!/bin/bash
while true; do
    echo \"[$(date)] Starting training...\"
    source ../tts_env/bin/activate
    python3 fine_tune_voice.py
    EXIT_CODE=\$?
    if [ \$EXIT_CODE -eq 0 ]; then
        echo \"✅ Training completed successfully!\"
        break
    else
        echo \"⚠️ Crash detected (Code \$EXIT_CODE). Cleaning up zombie processes...\"
        pkill -9 -f \"fine_tune_voice.py\"
        sleep 5
        echo \"🔄 Restarting training...\"
    fi
done
EOF"

ssh ${MINIPC_USER}@${MINIPC_HOST} "chmod +x ${MINIPC_PROJECT_DIR}/voice_lab/minipc_loop.sh"

# 3. Stop any existing runs to prevent conflicts
echo "🧹 Stopping any existing training processes..."
ssh ${MINIPC_USER}@${MINIPC_HOST} "pkill -9 -f 'minipc_loop.sh' || true"
ssh ${MINIPC_USER}@${MINIPC_HOST} "pkill -9 -f 'fine_tune_voice.py' || true"

# 4. Launch the script completely detached using nohup
echo "🔥 Igniting the engine in detached mode..."
ssh ${MINIPC_USER}@${MINIPC_HOST} "cd ${MINIPC_PROJECT_DIR}/voice_lab && nohup ./minipc_loop.sh > massive_training.log 2>&1 &"

echo "=========================================================="
echo "✅ SUCCESS! Training is now running completely independently on the Mini PC."
echo ""
echo "You can completely close your Mac, pack it up, and go to work."
echo "The Mini PC will crash-loop indefinitely until it hits 10,000 epochs."
echo ""
echo "To watch the live logs at any time, just run this command:"
echo "ssh root@minipc-ubuntu 'tail -f /root/thai/git/model-training-practice/voice_lab/massive_training.log'"
echo "=========================================================="
