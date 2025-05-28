import torch
import torch.nn.functional as F

def brier_score(probs, targets):
    """
    probs: tensor of shape [N, num_classes], predicted probabilities
    targets: tensor of shape [N], true class indices
    """
    one_hot = F.one_hot(targets, num_classes=probs.size(1)).float()
    return torch.mean((probs - one_hot) ** 2)
