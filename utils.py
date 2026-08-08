import torch
import torch.nn as nn
import numpy as np

# Hard-to-separate emotion pairs per dataset (label indices)
HARD_PAIRS = {
    'IEMOCAP': [(0, 4), (1, 5), (2, 5), (3, 5)],   # hap/exc, sad/fru, neu/fru, ang/fru
    'IEMOCAP4': [(0, 2), (1, 3)],                    # hap/neu, sad/ang
    'MELD': [(0, 4), (0, 3), (5, 6)],               # neu/joy, neu/sad, dis/ang
    'CMUMOSEI7': [(0, 1), (2, 3), (4, 5)],          # adjacent sentiment levels
}


def within_pair_error_analysis(labels, preds, dataset):
    """Print within-pair error rates for hard emotion pairs."""
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    pairs = HARD_PAIRS.get(dataset, [])
    if not pairs:
        print(f"No hard pairs defined for dataset '{dataset}'.")
        return

    print('\n── Within-pair error rate analysis ──')
    for a, b in pairs:
        mask = np.isin(labels, [a, b])
        n = mask.sum()
        if n == 0:
            continue
        err = np.mean(
            [(yt == a and yp == b) or (yt == b and yp == a)
             for yt, yp in zip(labels[mask], preds[mask])]
        )
        print(f"  pair {a}-{b} | n={n:4d}  within-pair err rate={err:.3f}")


def batch_to_all_tva(feature_t, feature_v, feature_a, lengths, no_cuda):

    node_feature_t, node_feature_v, node_feature_a = [], [], []
    batch_size = feature_t.size(1)

    for j in range(batch_size):
        node_feature_t.append(feature_t[:lengths[j], j, :])
        node_feature_v.append(feature_v[:lengths[j], j, :])
        node_feature_a.append(feature_a[:lengths[j], j, :])

    node_feature_t = torch.cat(node_feature_t, dim=0)
    node_feature_v = torch.cat(node_feature_v, dim=0)
    node_feature_a = torch.cat(node_feature_a, dim=0)

    if not no_cuda:
        node_feature_t = node_feature_t.cuda()
        node_feature_v = node_feature_v.cuda()
        node_feature_a = node_feature_a.cuda()

    return node_feature_t, node_feature_v, node_feature_a


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params:
        num: int, the number of loss
        x: multi-task loss
    Examples:
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """

    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i]**
                               2) * loss + torch.log(1 + self.params[i]**2)
        return loss_sum
