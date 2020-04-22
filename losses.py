import numpy as np
import torch 
import torch.nn as nn

def get_distance(x):
    _x = x.detach()
    sim = torch.matmul(_x, _x.t())
    dist = 2 - 2*sim
    dist += torch.eye(dist.shape[0]).to(dist.device)   # maybe dist += torch.eye(dist.shape[0]).to(dist.device)*1e-8
    dist = dist.sqrt()
    return dist

class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    Args:
        margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3, mutual_flag=False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.mutual = mutual_flag

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        # inputs = 1. * inputs / (torch.norm(inputs, 2, dim=-1, keepdim=True).expand_as(inputs) + 1e-12)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        if self.mutual:
            return loss, dist
        return loss

class NPairLoss(nn.Module):
    """
    N-Pair loss
    Sohn, Kihyuk. "Improved Deep Metric Learning with Multi-class N-pair Loss Objective," Advances in Neural Information
    Processing Systems. 2016.
    http://papers.nips.cc/paper/6199-improved-deep-metric-learning-with-multi-class-n-pair-loss-objective
    """

    def __init__(self, l2_reg=0.02):
        super(NPairLoss, self).__init__()
        self.l2_reg = l2_reg

    def forward(self, embeddings, target):
        n_pairs, n_negatives = self.get_n_pairs(target)

        if embeddings.is_cuda:
            n_pairs = n_pairs.cuda()
            n_negatives = n_negatives.cuda()

        anchors = embeddings[n_pairs[:, 0]]    # (n, embedding_size)
        positives = embeddings[n_pairs[:, 1]]  # (n, embedding_size)
        negatives = embeddings[n_negatives]    # (n, n-1, embedding_size)

        losses = self.n_pair_loss(anchors, positives, negatives) \
            + self.l2_reg * self.l2_loss(anchors, positives)

        return losses

    @staticmethod
    def get_n_pairs(labels):
        """
        Get index of n-pairs and n-negatives
        :param labels: label vector of mini-batch
        :return: A tuple of n_pairs (n, 2)
                        and n_negatives (n, n-1)
        """
        labels = labels.cpu().data.numpy()
        n_pairs = []

        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            anchor, positive = np.random.choice(label_indices, 2, replace=False)
            n_pairs.append([anchor, positive])

        n_pairs = np.array(n_pairs)

        n_negatives = []
        for i in range(len(n_pairs)):
            negative = np.concatenate([n_pairs[:i, 1], n_pairs[i+1:, 1]])
            n_negatives.append(negative)

        n_negatives = np.array(n_negatives)

        return torch.LongTensor(n_pairs), torch.LongTensor(n_negatives)

    @staticmethod
    def n_pair_loss(anchors, positives, negatives):
        """
        Calculates N-Pair loss
        :param anchors: A torch.Tensor, (n, embedding_size)
        :param positives: A torch.Tensor, (n, embedding_size)
        :param negatives: A torch.Tensor, (n, n-1, embedding_size)
        :return: A scalar
        """
        anchors = torch.unsqueeze(anchors, dim=1)  # (n, 1, embedding_size)
        positives = torch.unsqueeze(positives, dim=1)  # (n, 1, embedding_size)

        x = torch.matmul(anchors, (negatives - positives).transpose(1, 2))  # (n, 1, n-1)
        x = torch.sum(torch.exp(x), 2)  # (n, 1)
        loss = torch.mean(torch.log(1+x))
        return loss

    @staticmethod
    def l2_loss(anchors, positives):
        """
        Calculates L2 norm regularization loss
        :param anchors: A torch.Tensor, (n, embedding_size)
        :param positives: A torch.Tensor, (n, embedding_size)
        :return: A scalar
        """
        return torch.sum(anchors ** 2 + positives ** 2) / anchors.shape[0]

class MarginLoss(nn.Module):
    def __init__(self, margin=0.2, weight=None,beta = 1.2, batch_axis=0):
        super(MarginLoss, self).__init__()
        self._margin = margin
        self.beta = beta

    def forward(self, anchors, positives, negatives, a_indices=None):
        beta = self.beta

        d_ap = torch.sqrt(torch.sum((positives - anchors)**2, dim=1) +1e-8)
        d_an = torch.sqrt(torch.sum((negatives - anchors)**2, dim=1) +1e-8)

        pos_loss = torch.clamp(d_ap - beta + self._margin, min=0.0)
        neg_loss = torch.clamp(beta - d_an + self._margin, min=0.0)

        pair_cnt = int(torch.sum((pos_loss > 0.0) + (neg_loss > 0.0)))

        loss = torch.sum(pos_loss + neg_loss) / pair_cnt
        return loss


class DistanceWeightedSampling(nn.Module):
    '''
    parameters
    ----------
    batch_k: int
        number of images per class
    Inputs:
        data: input tensor with shapeee (batch_size, edbed_dim)
            Here we assume the consecutive batch_k examples are of the same class.
            For example, if batch_k = 5, the first 5 examples belong to the same class,
            6th-10th examples belong to another class, etc.
    Outputs:
        a_indices: indicess of anchors
        x[a_indices]
        x[p_indices]
        x[n_indices]
        xxx
    '''

    def __init__(self, batch_k, cutoff=0.5, nonzero_loss_cutoff=1.4, normalize =False):
        super(DistanceWeightedSampling,self).__init__()
        self.batch_k = batch_k
        self.cutoff = cutoff
        self.nonzero_loss_cutoff = nonzero_loss_cutoff
        self.normalize = normalize

    def forward(self, x):
        import ipdb; ipdb.set_trace()
        k = self.batch_k
        n, d = x.shape
        distance = get_distance(x)
        distance = distance.clamp(min=self.cutoff)
        log_weights = ((2.0 - float(d)) * distance.log() - (float(d-3)/2)*torch.log(torch.clamp(1.0 - 0.25*(distance*distance), min=1e-8)))

        if self.normalize:
            log_weights = (log_weights - log_weights.min()) / (log_weights.max() - log_weights.min() + 1e-8)

        weights = torch.exp(log_weights - torch.max(log_weights))

        if x.device != weights.device:
            weights = weights.to(x.device)

        mask = torch.ones_like(weights)
        for i in range(0,n,k):
            mask[i:i+k, i:i+k] = 0

        mask_uniform_probs = mask.double() *(1.0/(n-k))

        weights = weights*mask*((distance < self.nonzero_loss_cutoff).float()) + 1e-8
        weights_sum = torch.sum(weights, dim=1, keepdim=True)
        weights = weights / weights_sum

        a_indices = []
        p_indices = []
        n_indices = []

        np_weights = weights.cpu().numpy()
        for i in range(n):
            block_idx = i // k

            if weights_sum[i] != 0:
                n_indices +=  np.random.choice(n, k-1, p=np_weights[i]).tolist()
            else:
                n_indices +=  np.random.choice(n, k-1, p=mask_uniform_probs[i]).tolist()
            for j in range(block_idx * k, (block_idx + 1)*k):
                if j != i:
                    a_indices.append(i)
                    p_indices.append(j)

        return  a_indices, x[a_indices], x[p_indices], x[n_indices], x
