## Repository to reproduce issues with dask+smac
This repo contains a dummy script which uses dask+smac for multi-node and multi-gpu search. Create an environment and install the requirements with 

```pip install -r requirements.txt```

# Run search on a single worker
Note you may need to adjust the node names for you cluster/node names. The script below will run smac without dask (ie a single worker setting). The single worker is allocated 4 gpus. Use the slurm command below to launch your job. 

```sbatch job_single.sh```

# Run search on a multiple workers
Note you may need to adjust the node names for you cluster/node names. This script will run smac with dask (ie a multiple worker setting) with 8 workers. First start the dask scheduler.

```sbatch scheduler_dpn.sh ```

Start the dask workers. One run of the script below starts a single worker. Launch the script below with sbatch 8 times to start 8 workers. Each worker uses 8 gpus

```sbatch workers_dpn.sh ```

Finally start the job script for the distributed search

```sbatch job.sh ```


