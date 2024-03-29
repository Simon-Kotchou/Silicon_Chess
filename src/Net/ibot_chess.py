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
    
class Dinov2ChessSSL(nn.Module):
    def __init__(self, encoder, projector, predictor, queue_size=65536, m=0.99, T=0.07, num_prototypes=1024, ema_decay=0.99):
        super(Dinov2ChessSSL, self).__init__()
        self.encoder = encoder
        self.projector = projector
        self.predictor = predictor
        self.queue_size = queue_size
        self.m = m
        self.T = T
        self.num_prototypes = num_prototypes
        self.ema_decay = ema_decay

        self.register_buffer("queue", torch.randn(self.queue_size, self.projector.output_dim))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.prototypes = nn.Linear(projector.output_dim, num_prototypes, bias=False)
        self.ema_prototypes = nn.Linear(projector.output_dim, num_prototypes, bias=False)
        self.ema_prototypes.weight.data.copy_(self.prototypes.weight.data)
        self.ema_prototypes.requires_grad_(False)

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
    def _batch_shuffle_ddpm(self, x):
        # Random permutation
        idx_shuffle = torch.randperm(x.shape[0])

        # Random masking
        mask = torch.rand_like(x) < 0.15
        x_masked = x.clone()
        x_masked[mask] = 0

        return x_masked[idx_shuffle]

    def forward(self, x1, x2):
        # Encode the chessboard positions
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)

        # Project the representations
        p1 = self.projector(z1)
        p2 = self.projector(z2)

        # Compute the generative loss (MSE)
        x1_rec = self.predictor(p1)
        x2_rec = self.predictor(p2)
        loss_gen = F.mse_loss(x1_rec, x1) + F.mse_loss(x2_rec, x2)

        # Compute the discriminative loss (contrastive)
        logits_12 = torch.einsum('nc,mc->nm', F.normalize(p1), F.normalize(p2)) / self.T
        logits_21 = logits_12.T
        labels = torch.arange(logits_12.size(0), dtype=torch.long, device=logits_12.device)
        loss_dis = 0.5 * (F.cross_entropy(logits_12, labels) + F.cross_entropy(logits_21, labels))

        # Compute the dynamics loss (anticipation)
        z1_next = self.encoder(x2)
        z2_next = self.encoder(x1)
        p1_next = self.projector(z1_next)
        p2_next = self.projector(z2_next)
        loss_dyn = F.mse_loss(p1_next, p2) + F.mse_loss(p2_next, p1)

        # Compute the prototypical loss
        prototypes = F.normalize(self.prototypes.weight)
        ema_prototypes = F.normalize(self.ema_prototypes.weight)
        logits_proto_p1 = torch.einsum('nc,mc->nm', F.normalize(p1), prototypes)
        logits_proto_p2 = torch.einsum('nc,mc->nm', F.normalize(p2), prototypes)
        logits_ema_proto_p1 = torch.einsum('nc,mc->nm', F.normalize(p1.detach()), ema_prototypes)
        logits_ema_proto_p2 = torch.einsum('nc,mc->nm', F.normalize(p2.detach()), ema_prototypes)
        loss_proto = F.cross_entropy(logits_proto_p1, logits_ema_proto_p2.max(dim=1)[1]) + \
                     F.cross_entropy(logits_proto_p2, logits_ema_proto_p1.max(dim=1)[1])

        # Update the EMA prototypes
        self._update_ema_prototypes()

        # Compute key features with DDPM masking
        with torch.no_grad():
            x1_masked = self._batch_shuffle_ddpm(x1)
            x2_masked = self._batch_shuffle_ddpm(x2)
            k1 = self.encoder(x1_masked)
            k2 = self.encoder(x2_masked)
            k1 = self.projector(k1)
            k2 = self.projector(k2)
            k1 = F.normalize(k1, dim=1)
            k2 = F.normalize(k2, dim=1)

        # Compute the MoCo loss
        l_pos = torch.einsum('nc,nc->n', [p1, k2]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [p1, self.queue.clone().detach()])
        logits_moco = torch.cat([l_pos, l_neg], dim=1) / self.T
        labels_moco = torch.zeros(logits_moco.shape[0], dtype=torch.long, device=logits_moco.device)
        loss_moco = F.cross_entropy(logits_moco, labels_moco)

        # Update the queue
        self._dequeue_and_enqueue(k1)

        # Compute the final loss
        loss = loss_gen + loss_dis + loss_dyn + loss_proto + loss_moco

        return loss

    @torch.no_grad()
    def _update_ema_prototypes(self):
        self.ema_prototypes.weight.data.mul_(self.ema_decay).add_(
            self.prototypes.weight.data, alpha=1 - self.ema_decay)
        

class IJEPA(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim, 
                 num_heads, enc_layers, pred_layers):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.context_encoder = ViTEncoder(img_size, patch_size, embed_dim, num_heads, enc_layers)
        self.target_encoder = ViTEncoder(img_size, patch_size, embed_dim, num_heads, enc_layers)
        self.predictor = ViTPredictor(embed_dim, num_heads, pred_layers)
        
    def forward(self, x):
        # Sample masks
        context_masks, target_masks = multi_block_masking(self.img_size, self.patch_size)
        context_masks = context_masks.to(x.device).unsqueeze(0).repeat(x.shape[0], 1).view(x.shape[0], 1, self.img_size, self.img_size)
        target_masks = target_masks.to(x.device).unsqueeze(1).repeat(x.shape[0], 1, 1).view(x.shape[0], -1, self.img_size, self.img_size)

        # Mask images
        context_imgs = x * context_masks
        target_imgs = x.unsqueeze(1) * target_masks

        # Encode
        context_reps = self.context_encoder(context_imgs)
        with torch.no_grad():
            self.target_encoder.eval()
            target_reps = self.target_encoder(target_imgs.view(-1, 3, self.img_size, self.img_size)).view(x.shape[0], -1, self.img_size**2 // self.patch_size**2, context_reps.shape[-1])

        # Get position embeddings for target blocks
        target_pos = torch.nonzero(target_masks.flatten(2))[:, 2].view(x.shape[0], -1) 

        # Predict targets
        pred_reps = self.predictor(context_reps, target_pos)

        # Compute L2 loss
        loss = torch.mean(torch.stack([torch.mean((pred_reps[i] - target_reps[i])**2) for i in range(x.shape[0])]))

        return loss
    
    def update_target_encoder(self, momentum=0.996):
        with torch.no_grad():
            for param_q, param_k in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
                param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)

def multi_block_masking(img_size, patch_size, num_targets=4, context_scale=(0.85, 1.0), target_scale=(0.15, 0.2), target_aspect_ratio=(0.75, 1.5)):
    num_patches = (img_size // patch_size) ** 2
    context_mask = torch.zeros(num_patches)
    target_masks = []

    # Sample context block
    context_size = torch.rand(1) * (context_scale[1] - context_scale[0]) + context_scale[0]
    context_patches = int(num_patches * context_size)
    context_idx = torch.randperm(num_patches)[:context_patches]
    context_mask[context_idx] = 1

    # Sample target blocks
    for _ in range(num_targets):
        target_size = torch.rand(1) * (target_scale[1] - target_scale[0]) + target_scale[0]
        aspect_ratio = torch.rand(1) * (target_aspect_ratio[1] - target_aspect_ratio[0]) + target_aspect_ratio[0]
        target_patches = int(num_patches * target_size)
        target_h = int(torch.sqrt(target_patches / aspect_ratio))
        target_w = int(target_patches // target_h)
        target_idx = torch.randint(0, num_patches, (target_h * target_w,))
        target_mask = torch.zeros(num_patches)
        target_mask[target_idx] = 1
        target_masks.append(target_mask)
        
        # Remove overlapping patches from context
        context_mask[target_idx] = 0

    target_masks = torch.stack(target_masks)
    return context_mask, target_masks