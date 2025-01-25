import torch
import torch.distributed as dist

def run(rank, size):
    tensor = torch.tensor(rank+1)
    if rank == 0: tensor_old = tensor.clone()
    group = dist.new_group([0,1,2,3])
    print(f"I am {rank} of {size} with a tensor {tensor}")
    if rank == 0:
        print("**********\nStarting Communication\n************")
    dist.reduce(tensor=tensor, dst=0, op=dist.ReduceOp.SUM, group = group)
    if rank == 0: tensor = tensor-tensor_old
    if rank == 0: print('Rank ', rank, ' has data ', tensor.item())

if __name__ == "__main__":
    dist.init_process_group("gloo", init_method="env://")
    size = dist.get_world_size()
    rank = dist.get_rank()
    run(rank, size)
