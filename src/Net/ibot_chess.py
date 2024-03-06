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
    

class ChessSSL(nn.Module):
    def __init__(self, encoder, projector, masking_ratio=0.15, temperature=0.07, base_temperature=0.07, sinkhorn_iterations=3):
        super(ChessSSL, self).__init__()
        self.encoder = encoder
        self.projector = projector
        self.masking_ratio = masking_ratio
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.sinkhorn_iterations = sinkhorn_iterations

    def masked_chessboard(self, x):
        mask = torch.rand_like(x[:, :, 0]) < self.masking_ratio
        x_masked = x.clone()
        x_masked[mask] = 0
        return x_masked, mask

    def forward(self, x1, x2):
        # Mask the chessboard positions
        x1_masked, mask1 = self.masked_chessboard(x1)
        x2_masked, mask2 = self.masked_chessboard(x2)

        # Encode the masked positions
        z1 = self.encoder(x1_masked)
        z2 = self.encoder(x2_masked)

        # Project the representations
        p1 = self.projector(z1)
        p2 = self.projector(z2)

        # Compute the cosine similarity matrix
        sim_matrix = torch.einsum('nc,mc->nm', F.normalize(p1), F.normalize(p2)) / self.temperature

        # Compute the Sinkhorn divergence loss
        loss_sink = self.sinkhorn_divergence(sim_matrix)

        # Compute the masked reconstruction loss
        loss_rec = self.reconstruction_loss(z1, x1, mask1) + self.reconstruction_loss(z2, x2, mask2)

        # Compute the final loss
        loss = loss_sink + loss_rec

        return loss

    def sinkhorn_divergence(self, sim_matrix):
        Q = torch.exp(sim_matrix / self.temperature).T
        Q /= Q.sum()

        K = torch.exp(sim_matrix / self.base_temperature)

        for _ in range(self.sinkhorn_iterations):
            Q /= torch.einsum('ab,b->a', K, Q.sum(dim=0))
            Q /= Q.sum()

        loss = torch.einsum('ab,ab->', Q, torch.log(Q) - sim_matrix / self.temperature)

        return loss

    def reconstruction_loss(self, z, x, mask):
        x_rec = self.projector(z, inverse=True)
        loss = F.mse_loss(x_rec[mask], x[mask])
        return loss