from utils.args import parse_args
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import time

from torch.nn.parallel import DistributedDataParallel as DDP

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def run(rank, size, args):
    print(f"Start running basic DDP example on rank {rank}.")
    model = ToyModel()
    ddp_model = DDP(model)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    sample_size = args.batch_size//size
    print(f'Experiment with sample size: {sample_size}')
    st = time.time()
    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(sample_size, 10))
    labels = torch.randn(sample_size, 5)
    loss_fn(outputs, labels).backward()
    et = time.time()
    print(f'Execution time: {et-st} seconds')

    print(f"rank {rank}: with gradients {model.net1.weight.grad}\n")
    optimizer.step()
    dist.destroy_process_group()
    print(f"Finished running basic DDP example on rank {rank}.")

if __name__ == "__main__":
    dist.init_process_group("gloo", init_method="env://")
    size = dist.get_world_size()
    rank = dist.get_rank()
    args = parse_args()
    run(rank, size, args)