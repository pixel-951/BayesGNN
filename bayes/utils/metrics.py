import torch
import torch.nn.functional as F

def brier_score(probs, targets):
    """
    probs: tensor of shape [N, num_classes], predicted probabilities
    targets: tensor of shape [N], true class indices
    """
    one_hot = F.one_hot(targets, num_classes=probs.size(1)).float()
    return torch.mean((probs - one_hot) ** 2)


def brier_over_under(probs, targets, n_bins=15):
    num_classes = probs.size(1)
    one_hot = F.one_hot(targets, num_classes=num_classes).float()


    confidences, predictions = torch.max(probs, dim=1)
    accuracies = predictions.eq(targets).float()

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    over_conf, under_conf = 0.0, 0.0
    total_samples = len(probs)

    for i in range(n_bins):
        lower, upper = bin_boundaries[i], bin_boundaries[i+1]
        in_bin = (confidences > lower) & (confidences <= upper)
        bin_size = in_bin.sum().item()

        if bin_size == 0:
            continue

        bin_conf = confidences[in_bin].mean().item()
        bin_acc = accuracies[in_bin].mean().item()
        gap_sq = (bin_conf - bin_acc) ** 2
        weight = bin_size / total_samples

        if bin_conf > bin_acc:
            over_conf += weight * gap_sq
        elif bin_conf < bin_acc:
            under_conf += weight * gap_sq

    return over_conf, under_conf





def expected_calibration_error(probs, targets, n_bins=10):
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = predictions.eq(targets).float()

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece_over = ece_under = 0.0

    for i in range(n_bins):
        lower, upper = bin_boundaries[i], bin_boundaries[i+1]
        in_bin = (confidences > lower) & (confidences <= upper)
        bin_size = in_bin.sum().item()

        if bin_size == 0:
            continue

        bin_conf = confidences[in_bin].mean().item()
        bin_acc = accuracies[in_bin].mean().item()
        gap = abs(bin_conf - bin_acc)
        weight = bin_size / len(probs)

        if bin_conf > bin_acc:  # Overconfidence
            ece_over += weight * gap
        elif bin_conf < bin_acc:  # Underconfidence
            ece_under += weight * gap

    total_ece = ece_over + ece_under
    return total_ece, ece_over, ece_under