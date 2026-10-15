import torch
import torch.nn.functional as F

def contrastive_loss(embeddings: torch.Tensor, temperature: float) -> float:
    """
    Compute the exact NT-Xent / InfoNCE contrastive loss.
    """
    # 1. Correct L2 normalization along the feature dimension
    # (Replaces the deprecated .norm method securely)
    embeddings = F.normalize(embeddings, p=2, dim=-1)

    # 2. Slice into even (Aug_A) and odd (Aug_B) views
    z1 = embeddings[0::2]  # Shape: (N, d)
    z2 = embeddings[1::2]  # Shape: (N, d)
    N = z1.shape[0]

    # 3. Compute ALL pairwise combinations scaled by temperature
    sim_12 = (z1 @ z2.T) / temperature  # Compares A to B (Positives on diagonal)
    sim_21 = (z2 @ z1.T) / temperature  # Compares B to A (Positives on diagonal)
    sim_11 = (z1 @ z1.T) / temperature  # Compares A to other A's (Negatives)
    sim_22 = (z2 @ z2.T) / temperature  # Compares B to other B's (Negatives)

    # 4. We cannot compare an image to itself (e.g. Dog A to Dog A)
    # So we force the main diagonal of sim_11 and sim_22 to a massive negative number.
    # When PyTorch calculates exp(-1e9), it becomes exactly 0.
    self_mask = torch.eye(N, device=embeddings.device, dtype=torch.bool)
    sim_11 = sim_11.masked_fill(self_mask, -1e9)
    sim_22 = sim_22.masked_fill(self_mask, -1e9)

    # 5. Assemble the true InfoNCE logit rows
    # Row i needs to evaluate its positive match in z2 against ALL other z2s and z1s
    logits_z1 = torch.cat([sim_12, sim_11], dim=1)  # Shape: (N, 2N)
    logits_z2 = torch.cat([sim_21, sim_22], dim=1)  # Shape: (N, 2N)
    
    # Stacking them gives us all 2N rows required by SimCLR
    logits = torch.cat([logits_z1, logits_z2], dim=0)  # Shape: (2N, 2N)

    # 6. Targets point to the column index of the true match.
    # Because sim_12 and sim_21 are placed first, the true index is always 'i'
    targets = torch.arange(N, device=embeddings.device)
    targets = torch.cat([targets, targets], dim=0)  # Shape: (2N,)

    # 7. Compute the standard Cross Entropy Loss over the full grid
    loss = F.cross_entropy(logits, targets)
    
    return loss.item()
