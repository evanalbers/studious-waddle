#!/usr/bin/env python
import os
import subprocess
import argparse

def submit_training_job():
    # Create the job script
    job_script = """#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=sae_training
#SBATCH --mem=4GB
#SBATCH --time=1:05:00  
#SBATCH --output=sae_training_%j.out
#SBATCH --error=sae_training_%j.err

python sae_training_script.py --checkpoint_path=sae_model.pt --time_limit=3600

# Check if training is complete
if grep -q "Training complete!" sae_training_${SLURM_JOB_ID}.out; then
    echo "Training has completed successfully."
    # compressing model to reduce transfer size
    tar -czf final_model.tar.gz sae_model.pt
    echo "Model compressed and ready for transfer at: ~/projects/sae_training/final_model.tar.gz"

else
    # Submit another job
    sbatch job_script.sh
fi
"""
    
    # Write job script to file
    with open("job_script.sh", "w") as f:
        f.write(job_script)
    
    # Make executable
    os.chmod("job_script.sh", 0o755)
    
    # Submit the job
    subprocess.run(["sbatch", "job_script.sh"])
    print("Job submitted!")

if __name__ == "__main__":
    submit_training_job()