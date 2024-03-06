import torch
import torch.nn as nn
import torch.nn.functional as F

class Dinov2SSL(nn.Module):
    def __init__(self, encoder, projector, queue_size=65536, m=0.99, T=0.07, mlp=False):
        super(Dinov2SSL, self).__init__()
        self.encoder = encoder
        self.projector = projector
        self.queue_size = queue_size
        self.m = m
        self.T = T
        self.mlp = mlp

        self.register_buffer("queue", torch.randn(self.queue_size, self.projector.output_dim))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # Replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr:ptr + batch_size, :] = keys.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddpm(self, x, mean=0.5, std=0.5):
        # Random permutation
        idx_shuffle = torch.randperm(x.shape[0])

        # Random scaling
        r = torch.randn(x.shape[0], 1, 1, 1).to(x.device)
        r = mean + std * torch.sigmoid(r)

        return x[idx_shuffle] * r

    def forward(self, im_q, im_k):
        q = self.encoder(im_q)
        q = self.projector(q)
        q = F.normalize(q, dim=1)

        # Compute key features
        with torch.no_grad():
            im_k = self._batch_shuffle_ddpm(im_k)  # iBot
            k = self.encoder(im_k)
            k = self.projector(k)
            k = F.normalize(k, dim=1)

        # MoCo
        l_moco = torch.einsum('nc,nc->n', [q, k]).mean()

        # Compute logits
        l_pos = torch.einsum('nc,nc->n', [q, self.queue.clone().detach()])
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach().T])

        # KoleoGANs
        l_kol = - ((l_pos - l_neg.detach().mean(dim=1)).exp().mean() + 1).log()

        # Sinkhorn divergence
        Q = torch.exp(l_pos / self.T).unsqueeze(1)
        l_sink = torch.sum(Q * (torch.log(Q / torch.exp(l_neg / self.T).unsqueeze(2)) - self.T), dim=(1, 2)).mean()

        # iBot loss
        loss = l_moco + l_kol + l_sink

        # Update queue
        self._dequeue_and_enqueue(k)

        return loss