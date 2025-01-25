# Parallel machine learning using PyTorch
## Start
- Book fours cores in NEF cluster
- Enter to the directory worked previously
- Update the directory `git pull` or `git clone https://gitlab.inria.fr/chxu/pytorch_exercice.git`
- Enter into the "distributed" directory
- Activate your conda environment
- Execute the code `torchrun --nnodes=1 --nproc-per-node=4 training.py --batch_size 8000000`

## Tasks
0. Read training.py code, answering the following questions.
1. What is the model we train? 
2. `ddp_model = DDP(model)` means what? How it works to achieve parallel machine learning?
**Hints:**  DDP wraps lower-level distributed communication details and provides a clean API as if it were a local model. Gradient synchronization communications take place during the backward pass and overlap with the backward computation. When the backward() returns, param.grad already contains the synchronized gradient tensor.
3. Why the printed gradients are the same for every worker?
4. The command `torchrun --nnodes=1 --nproc-per-node=4 training.py --batch_size 8000000` will assign how many samples for each processor?
5. Modify the --nproc-per-node values from 1 and 4, what is the observation for the computation time?
6. Test smaller --batch_size (e.g., 8000), what is your observation?

## Exercice
1. Create a new file training_manuel.py to recode the Toy example without usage of *torch.nn.parallel.DistributedDataParallel*. Compare the performance with DDP.
2. Create a new file training_realist.py to make the previous machine learning task (classify the facial image) to be parallel. 
**Hints:** Personalize the dataloader for every worker. 
