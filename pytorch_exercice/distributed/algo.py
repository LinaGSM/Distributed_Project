import torch
import torch.distributed as dist

def run(rank, size):
    tensor = torch.tensor(1)
    iterations = 10
    group = [i for i in range(size)]
    for i in zip(range(iterations)):
        # Step 1
        if rank == 0:
            print(f"Iter {i}: Rank 0 broadcasts to the group {group}")
            tensor_old = tensor.clone()
        groupe = dist.new_group(group)
        dist.broadcast(tensor=tensor, src=0, group=groupe)

        # Step 2
        if rank in group and rank != 0:
            tensor += 1

        # Step 3
        dist.reduce(tensor=tensor, dst=0, op=dist.ReduceOp.SUM, group=groupe)
        if rank == 0:
            tensor -= tensor_old
            tensor = tensor/(size-1) + tensor_old

    if rank == 0: print(f"The final value of Rank {0} is {tensor}")


if __name__ == "__main__":
    dist.init_process_group("gloo", init_method="env://")
    size = dist.get_world_size()
    rank = dist.get_rank()
    run(rank, size)