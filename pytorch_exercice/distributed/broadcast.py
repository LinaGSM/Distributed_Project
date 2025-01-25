import torch
import torch.distributed as dist

def run(rank, size):
    tensor = torch.tensor(rank)
    group = dist.new_group([0, 1, 2])
    # print(f"I am {rank} of {size} with a tensor {tensor}")

    if rank == 0: print("**********\nStarting Communication\n************")
    dist.broadcast(tensor=tensor, src=0, group=group)
    print('Rank ', rank, ' has data ', tensor)


if __name__ == "__main__":
    dist.init_process_group("gloo", init_method="env://")
    size = dist.get_world_size()
    rank = dist.get_rank()
    run(rank, size)
