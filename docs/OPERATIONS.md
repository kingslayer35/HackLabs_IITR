# GPU and local inference operations

No GPU job is launched by repository preparation. On the target machine, activate the pinned environment and use the required screen/Slurm pattern:

```bash
screen -dmS termmap_submission
screen -S termmap_submission -X stuff $'srun --gres=gpu:1 --cpus-per-task=6 --mem=48G --pty sh -i\n'
squeue -u "$USER"
```

After the allocation is running, use the actual checkout path inside that screen. First run `bash slurm/run_indexes.sh --smoke`; after its smoke STATUS is DONE, run `bash slurm/run_indexes.sh; exit`. Prediction uses the same smoke-first procedure automatically through `bash run.sh input.csv output.csv`. Training has separate `run_pretrain.sh` and `run_posttrain.sh`; select their training config before running. They are not part of preparing or evaluating the submission snapshot.

Every wrapper prints host/GPU, enforces a Slurm allocation, sets seed/determinism variables, and records DONE or FAILED beneath configured runtime runs. `; exit` releases the allocated shell when full work ends. Do not hold idle GPUs or cancel someone else's job.

Use `tail`, STATUS files, `squeue` and `screen -ls` to monitor. No Slurm controller, background scorer or shared environment in the research project is edited by these scripts.

Transformers keeps the base/adapter resident inside the prediction process. Optional vLLM runs a separate local-only server and batches candidate requests. More workers cannot remove token/model compute; compare actual throughput and footprint before claiming lower latency. Backend changes require their own provenance and acceptance checks.
