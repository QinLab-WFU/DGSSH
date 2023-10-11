"""Define functions to create the triplet loss with online triplet mining."""

import tensorflow as tf
import numpy as np
import torch


def _pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.
    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = torch.matmul(embeddings, embeddings.t())

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = torch.diag(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = torch.unsqueeze(square_norm, 1) - 2.0 * dot_product + torch.unsqueeze(square_norm, 0)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = torch.max(distances, torch.tensor([0.0]).cuda())

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = (torch.eq(distances, 0.0)).float()
        distances = distances + mask * 1e-16

        distances = torch.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (torch.sub(1.0, mask))

    return distances


def _get_anchor_positive_triplet_mask(labels):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check that i and j are distinct
    indices_equal = torch.eye(labels.shape[0]).cuda().byte()
    indices_not_equal = torch.tensor([1.0]).cuda() - indices_equal
    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1))

    # Combine the two masks
    mask = torch.mul(indices_not_equal, labels_equal)

    return mask


def _get_anchor_negative_triplet_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1))

    mask = torch.logical_not(labels_equal)

    return mask


def _get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = torch.eye(labels.shape[0]).cuda().byte()
    indices_not_equal = torch.tensor([1.0]).cuda() - indices_equal
    i_not_equal_j = torch.unsqueeze(indices_not_equal, 2)
    i_not_equal_k = torch.unsqueeze(indices_not_equal, 1)
    j_not_equal_k = torch.unsqueeze(indices_not_equal, 0)

    distinct_indices = torch.mul(torch.mul(i_not_equal_j, i_not_equal_k), j_not_equal_k)

    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1)).float()
    i_equal_j = torch.unsqueeze(label_equal, 2)
    i_equal_k = torch.unsqueeze(label_equal, 1)
    valid_labels = torch.mul(i_equal_j, torch.tensor([1.0]).cuda() - i_equal_k)

    # Combine the two masks
    mask = torch.mul(distinct_indices, valid_labels)
    return mask


def batch_all_triplet_loss(labels, embeddings, margin=100, squared=False):
    """Build the triplet loss over a batch of embeddings.
    We generate all the valid triplets and average the loss over the positive ones.
    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    # shape (batch_size, batch_size, 1)
    anchor_positive_dist = torch.unsqueeze(pairwise_dist, 2)
    assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)
    # shape (batch_size, 1, batch_size)
    anchor_negative_dist = torch.unsqueeze(pairwise_dist, 1)
    assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = _get_triplet_mask(labels)
    mask = mask.float()
    triplet_loss = torch.mul(mask, triplet_loss)

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = torch.max(triplet_loss, torch.tensor([0.0]).cuda())

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = torch.gt(triplet_loss, 1e-16).float()
    num_positive_triplets = torch.sum(valid_triplets)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = torch.sum(triplet_loss) / (num_positive_triplets + 1e-16)

    return triplet_loss


def batch_hard_triplet_loss(labels, embeddings, margin=10, squared=False):
    """Build the triplet loss over a batch of embeddings.
    For each anchor, we get the hardest positive and hardest negative to form a triplet.
    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
    mask_anchor_positive = mask_anchor_positive.float()

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = torch.multiply(mask_anchor_positive, pairwise_dist)

    # shape (batch_size, 1)
    hardest_positive_dist, _ = torch.max(anchor_positive_dist, dim=1, keepdim=True)
    hard_positive_indices = torch.argmax(anchor_positive_dist, dim=1)
    # torch.summary.scalar("hardest_positive_dist", torch.mean(hardest_positive_dist))

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels)
    mask_anchor_negative = (mask_anchor_negative).float()

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist, _ = torch.max(pairwise_dist, dim=1, keepdim=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # shape (batch_size,)
    hardest_negative_dist, _ = torch.min(anchor_negative_dist, dim=1, keepdim=True)
    # tf.summary.scalar("hardest_negative_dist", tf.reduce_mean(hardest_negative_dist))

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    triplet_loss = torch.maximum(hardest_positive_dist - hardest_negative_dist + margin, torch.tensor([0.0]).cuda())

    # Get final mean triplet loss
    triplet_loss = torch.mean(triplet_loss)

    return triplet_loss


def corrective_triplet_loss(labels, embeddings, gamma=1, margin=10, squared=False):
    """Build the apdaptive triplet loss over a batch of embeddings.
    We generate all the valid triplets and average the loss over the positive ones and add it to the distribution shift.
    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        lamdbda_:trade-off parameter
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        adaptive_triplet_loss: scalar tensor containing the triplet loss (L = L_triplet + λ ∗ L_match)
    """

    # Get the pairwise distance matrix

    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
    mask_anchor_positive = mask_anchor_positive.float()

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = torch.multiply(mask_anchor_positive, pairwise_dist)

    # shape (batch_size, 1)
    hardest_positive_dist, _ = torch.max(anchor_positive_dist, dim=1, keepdim=True)
    hard_positive_indices = torch.argmax(anchor_positive_dist, dim=1)

    # torch.summary.scalar("hardest_positive_dist", torch.reduce_mean(hardest_positive_dist))

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels)
    mask_anchor_negative = (mask_anchor_negative).float()

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    # max_anchor_negative_dist = torch.max(pairwise_dist, dim=1, keepdim=True)
    max_anchor_negative_dist, _ = torch.max(pairwise_dist, dim=1, keepdim=True)

    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # shape (batch_size,)
    hardest_negative_dist, _ = torch.min(anchor_negative_dist, dim=1, keepdim=True)
    hard_negative_indices = torch.argmin(anchor_negative_dist, dim=1)
    # torch.summary.scalar("hardest_negative_dist", torch.reduce_mean(hardest_negative_dist))

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    triplet_loss = torch.maximum(hardest_positive_dist - hardest_negative_dist + margin, torch.tensor([0.0]).cuda())

    # Get final mean triplet loss
    L_triplet = torch.mean(triplet_loss)

    # Embeding dict stores the mean of all embeddings of each instance
    embedding_dict = dict()
    for i in range(len(labels)):
        index = int(labels[i].cpu().numpy())
        if index not in embedding_dict:
            embedding_dict[index] = [embeddings[i]]
        else:
            embedding_dict[index].append(embeddings[i])
    # Taking mean of the embeddings in embedding_dict
    for label in embedding_dict:
        embedding_dict[label] = torch.mean(torch.stack(embedding_dict[label]), dim=0)
    # L_match_dict stores the mean of embeddings of the instances choosen in the triplet selections
    L_match_dict = dict()
    # Adding instances from hard positives
    for i in hard_positive_indices.cpu().numpy():
        index = int(labels[i].cpu().numpy())
        if index not in L_match_dict:
            L_match_dict[index] = [embeddings[i]]
        else:
            L_match_dict[index].append(embeddings[i])

    # Adding instances from hard negatives
    for i in hard_negative_indices.cpu().numpy():
        index = int(labels[i].cpu().numpy())
        if index not in L_match_dict:
            L_match_dict[index] = [embeddings[i]]
        else:
            L_match_dict[index].append(embeddings[i])

    # Taking mean of the embeddings in L_match_dict
    for label in L_match_dict:
        L_match_dict[label] = torch.mean(torch.stack(L_match_dict[label]), dim=0)
    # Find L Match using sum of l2 norm of L_triplet - L_match_dict
    L_constraint = []
    for ind in L_match_dict:
        l2_norm = np.linalg.norm((embedding_dict[ind] - L_match_dict[ind]).cpu().detach().numpy(), ord=2)
        L_constraint.append(l2_norm)
    L_constraint = np.sum(L_constraint)

    # Calculate triplet loss, triplet loss = L_triplet + λ ∗ L_match
    triplet_loss = L_triplet + (gamma * L_constraint)

    return triplet_loss
