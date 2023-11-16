import torch
import torch_xla.core.xla_model as xm
import torch.distributed as dist
import torch_xla.distributed.xla_backend
import os

def rprint(txt):
    rank = os.environ.get("RANK", "unk")
    if int(rank) == 0:
        print(f"{rank}: {txt}", flush=True)

dist.init_process_group('xla')
rprint("before 1st rendezvous")
xm.rendezvous('first')

device = xm.xla_device()
for c in range(1000000):
    ones = torch.ones((2, 3))
    xones = ones.to(device)
    result = xm.all_reduce('sum', xones)
    xm.mark_step()
    result_cpu = result.cpu()
    expected = torch.ones((2, 3)) * int(os.environ.get("WORLD_SIZE", 0))
    rprint(f"result: {c}: {result}  result.size(): {result.size()}")
    assert torch.all(result_cpu == expected), f'ERROR: {result_cpu} != {expected}'

rprint("before final rendezvous")
xm.rendezvous('last')
