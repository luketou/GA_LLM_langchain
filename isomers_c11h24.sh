#!/usr/bin/bash

#PBS -l select=1:ncpus=4:ngpus=1
#PBS -l place=vscatter:shared
#PBS -q gpu
#PBS -j oe
#PBS -N isomer_c11h24_job

# 記錄開始時間
START_TIME=$SECONDS

# 清除環境變數避免衝突
unset LD_PRELOAD
unset BNB_CUDA_VERSION 
unset LD_LIBRARY_PATH

# 設置正確的 CUDA 相關環境變數
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 設置 Conda 環境
source /home/luketou/miniconda3/bin/activate agent_predictor

# 確保使用正確的系統庫版本（避免 GLIBCXX 不兼容問題）
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH


# 設定工作目錄
cd /home/luketou/LLM_AI_agent/Agent_predictor/GA_LLM_langchain

# 匯入 .env 中的環境變數（包含 API 金鑰）
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

echo "=========================================================="
echo "Starting on : $(date)"
echo "Running on node : $(hostname)"
echo "Current directory : $(pwd)"
echo "Current job ID : $PBS_JOBID"
echo "GPU Device: $CUDA_VISIBLE_DEVICES"
echo "=========================================================="

rm -rf log/isomer_c11h24.log
mkdir -p log
touch log/isomer_c11h24.log
python main.py --task isomer_c11h24  2>&1 | tee -a log/isomer_c11h24.log

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo '=========================================================='
echo "任務完成，耗時 $(($ELAPSED_TIME/60)) 分鐘 $(($ELAPSED_TIME%60)) 秒"
echo "Job Ended at $(date)"
echo '=========================================================='

conda deactivate
