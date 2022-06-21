import torch



def permute_augmentation(batch_x,p_corrupt=0.05):
    mask = torch.full_like(batch_x,p_corrupt, device=batch_x.device, dtype=batch_x.dtype)
    mask = mask.bernoulli()
    x_corrupt = torch.stack([batch_x[torch.randperm(batch_x.shape[0]), i]
                             for i in range(batch_x.shape[1])], dim=1).to(batch_x.device)
    batch_corrupt = (mask * x_corrupt) + ((1 - mask) * batch_x)
    return batch_corrupt