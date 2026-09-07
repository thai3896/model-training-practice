#!/bin/bash

echo "==============================================================="
echo "⏱️ AI AUTO-CHECKER DAEMON"
echo "==============================================================="
echo "This script will generate a new preview every 3 minutes."
LAST_TESTED_STEP="0"

while true; do
    # Check if the training script is actually running on the Mini PC
    IS_RUNNING=$(ssh root@minipc-ubuntu "pgrep -f 'fine_tune_voice.py' >/dev/null && echo 'RUNNING' || echo 'STOPPED'")
    
    if [ "$IS_RUNNING" == "STOPPED" ]; then
        echo "🏁 Training appears to have finished! Shutting down auto-checker."
        echo "Your main pipeline script will generate the absolute final audio."
        break
    fi
    
    echo "==============================================================="
    echo "⏳ [$(date +'%I:%M %p')] Checking for new training checkpoints..."
    
    # Fast query to the Mini PC to find the most recently modified .pth file by timestamp, then extract its step number
    CURRENT_STEP=$(ssh root@minipc-ubuntu "find /root/thai/git/model-training-practice/voice_lab/tts_train_output -name '*_[0-9]*.pth' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -n 1 | grep -oE '[0-9]+\.pth' | grep -oE '[0-9]+'" 2>/dev/null)
    
    if [ -z "$CURRENT_STEP" ]; then
        CURRENT_STEP="unknown"
    fi
    
    if [ "$CURRENT_STEP" == "unknown" ]; then
        echo "🛑 No checkpoints exist yet. The AI is still warming up and hasn't saved its first step."
        echo "⏭️ Waiting 3 minutes for the next check..."
    elif [ "$CURRENT_STEP" == "$LAST_TESTED_STEP" ]; then
        echo "🛑 Still at Step $CURRENT_STEP. The AI hasn't saved a new checkpoint yet."
        echo "⏭️ Skipping audio generation to save GPU resources and keep the UI clean."
    else
        echo "🚀 New checkpoint found! (Step $CURRENT_STEP). Triggering audio generation..."
        ./test_progress.sh
        
        LAST_TESTED_STEP=$CURRENT_STEP
        
        # Save a historical copy for the UI, including the step number
        TIMESTAMP=$(date +'%H%M')
        cp ./voice_lab/public/presentation_test.wav "./voice_lab/public/preview_${TIMESTAMP}_step${CURRENT_STEP}.wav" 2>/dev/null || true
        echo "✅ Preview saved for Step $CURRENT_STEP."
    fi
    
    echo ""
    echo "⏳ Waiting 3 minutes for the next check..."
    echo "(Refresh your Web UI to hear the latest version)"
    sleep 180
done
