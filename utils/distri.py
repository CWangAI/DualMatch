'''Semi-Supervised tricks'''

import torch
import torch.nn as nn


class DistributionAlignment(nn.Module):
    '''Distribution Alignment is imported from SimMatch'''

    def __init__(self, args=None):
        super().__init__()
        if args.DA:
            self.DA_len = 32
            self.register_buffer("DA_queue", torch.zeros(self.DA_len, args.num_classes, dtype=torch.float))
            self.register_buffer("DA_ptr", torch.zeros(1, dtype=torch.long))
            self.local_rank = args.local_rank

    @torch.no_grad()
    def distribution_alignment(self, probs):
        probs_bt_mean = probs.mean(0)
        ptr = int(self.DA_ptr)
        if self.local_rank == -1:
            self.DA_queue[ptr] = probs_bt_mean
        else:
            torch.distributed.all_reduce(probs_bt_mean)
            self.DA_queue[ptr] = probs_bt_mean / torch.distributed.get_world_size()  
        self.DA_ptr[0] = (ptr + 1) % self.DA_len
        probs = probs / self.DA_queue.mean(0)
        probs = probs / probs.sum(dim=1, keepdim=True)
        return probs.detach()