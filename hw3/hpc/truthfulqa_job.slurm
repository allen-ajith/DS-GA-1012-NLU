#!/bin/bash

# Please adjust these settings according to your needs.
#SBATCH --account=ds_ga_1012-2025sp
#SBATCH --partition=n1s8-v100-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=6:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=truthfulqa
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aa12938@nyu.edu
#SBATCH --output="truthfulqa.out"


module purge
cd path/to/your/code
OVERLAY_FILE=path/to/your/singularity/overlay/file
SINGULARITY_IMAGE=path/to/your/singularity/image/file
singularity exec --nv \
	    --overlay $OVERLAY_FILE $SINGULARITY_IMAGE \
	    /bin/bash -c "source /ext3/env.sh; bash hpc/run_truthfulqa_evaluation.sh"