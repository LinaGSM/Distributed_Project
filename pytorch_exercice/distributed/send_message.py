import torch
import torch.distributed as dist

def run(rank, size):
    tensor = torch.tensor(rank)
    print(f"I am {rank} of {size} with a tensor {tensor}")
    if rank == 0:
        print("**********\nStarting Communication\n************")
        # Sending the tensor to process 1
        dist.send(tensor=tensor, dst=1)
        print('Rank', rank, 'sends', tensor, 'to process 1')
    else:
        # Receive tensor from process 0
        if rank == 1:
            dist.recv(tensor=tensor, src=0)
            print('Rank', rank, 'receives', tensor, 'from process 0')

    if rank==0 or rank==1: print('Rank ', rank, ' has data ', tensor)


if __name__ == "__main__":
    dist.init_process_group("gloo", init_method="env://")
    size = dist.get_world_size()
    rank = dist.get_rank()
    run(rank, size)
