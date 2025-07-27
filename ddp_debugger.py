import os
import code
import builtins
import torch.distributed as dist

def ddp_debug(local_vars=None, allowed_ranks=[0], barrier=True):
    """
    Safe interactive debugging in a DDP setup.

    Parameters:
    - local_vars: dictionary of local variables (e.g. locals()).
    - allowed_ranks: list of ranks allowed to enter REPL.
    - barrier: whether to call dist.barrier() before and after.
    """
    if not dist.is_initialized():
        print("DDP not initialized, opening interactive session.")
        code.interact(local=local_vars or globals())
        return

    rank = dist.get_rank()

    if barrier:
        dist.barrier()

    if rank in allowed_ranks:
        print(f"Rank {rank} entering interactive mode. Others are blocked.")
        code.interact(local=local_vars or globals())

    if barrier:
        dist.barrier()

    if rank not in allowed_ranks:
        print(f"Rank {rank} passed ddp_debug barrier.")
