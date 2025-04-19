#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=sae_training
#SBATCH --mem=32GB
#SBATCH --time=01:05:00 
#SBATCH --output=sae_training_%j.out
#SBATCH --error=sae_training_%j.err

module load cuda/11.8

#  python sae_training_script.py --checkpoint_path=sae_model.pt --time_limit=300

python rnn_experiments/train_rnn.py



# Check if training is complete
if grep -q "Training complete!" sae_training_${SLURM_JOB_ID}.out; then
    echo "Training has completed successfully."
    # compressing model to reduce transfer size
    tar -czf final_model.tar.gz shakespeare_rnn.pt
    echo "Model compressed and ready for transfer at: ~/projects/sae_training/final_model.tar.gz"

fi
