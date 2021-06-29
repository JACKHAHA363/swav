import torch
from pandas import DataFrame
from tqdm import tqdm
import numpy as np
from scipy.sparse import csr_matrix
import random

@torch.no_grad()
def get_LDA_mat(features, labels, centroids, imgs_per_chunk=1000):
    """
    :param features: [bsz, feat_size]
    :param labels: [bsz]
    :param centroids: [nb_labels, feat_size]
    :return: LDA scores [nb_labels, nb_labels]
    """
    # Compute all intra-cluster
    intra_cluster = torch.zeros(len(centroids)).to(features.device)
    for class_id in tqdm(range(len(centroids))):
        class_features = features[labels == class_id]
        intra_dist_mean = 0
        count = 0
        for idx in range(0, len(class_features), imgs_per_chunk):
            batch_embs = class_features[idx: min((idx + imgs_per_chunk), len(class_features))]
            cosine_dist = 1 - torch.mm(batch_embs, class_features.t())
            intra_dist_mean += cosine_dist.sum().item()
            count += batch_embs.shape[0] * class_features.shape[0]
        intra_dist_mean /= count
        intra_cluster[class_id] = intra_dist_mean

    intra_cluster_mat = intra_cluster[:, None] + intra_cluster[None, :]
    inter_cluster_mat = 1 - torch.mm(centroids, centroids.t())
    LDA = inter_cluster_mat / intra_cluster_mat
    return LDA


@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, ks, T, num_classes=1000, imgs_per_chunk=100, leave_one_out=False):
    if not isinstance(ks, list):
        ks = [ks]
    top1_corrects = {k: torch.zeros(num_classes).to(train_labels.device) for k in ks}
    top5_corrects = {k: torch.zeros(num_classes).to(train_labels.device) for k in ks}
    totals = 1e-18 + torch.zeros(num_classes).to(train_labels.device)
    train_features = train_features.t()
    num_test_images = test_labels.shape[0]
    largest_k = max(ks)
    if leave_one_out:
        largest_k += 1
    retrieval_one_hot = torch.zeros(largest_k, num_classes).to(train_labels.device)
    for idx in tqdm(range(0, num_test_images, imgs_per_chunk)):
        # get the features for test images
        features = test_features[
                   idx : min((idx + imgs_per_chunk), num_test_images), :
                   ]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(largest_k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * largest_k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        for k in ks:
            start = 0 if not leave_one_out else 1
            probs = torch.sum(
                torch.mul(
                    retrieval_one_hot.view(batch_size, -1, num_classes)[:, start:start+k],
                    distances_transform.view(batch_size, -1, 1)[:, start:start+k],
                ),
                1,
            )
            _, predictions = probs.sort(1, True)

            # find the predictions that match the target
            correct = predictions.eq(targets.data.view(-1, 1))
            for label, is_correct in zip(targets, correct.narrow(1,0,1).sum(-1)):
                top1_corrects[k][label] += is_correct

            for label, is_correct in zip(targets, correct.narrow(1,0,min(5, num_classes)).sum(-1)):
                top5_corrects[k][label] += is_correct

        for label in targets:
            totals[label] += 1
    results = []
    for k in ks:
        top1 = (top1_corrects[k].sum() * 100. / totals.sum()).item()
        top5 = (top5_corrects[k].sum() * 100. / totals.sum()).item()
        top1_bal = ((top1_corrects[k] / totals).mean() * 100).item()
        top5_bal = ((top5_corrects[k] / totals).mean() * 100).item()
        results.append([k, top1, top5, top1_bal, top5_bal])
    df = DataFrame(results, columns=['k', 'top1', 'top5', 'top1_bal', 'top5_bal'])
    return df


def get_indices_sparse(data):
    cols = np.arange(data.size)
    M = csr_matrix((cols, (data.ravel(), cols)), shape=(int(data.max()) + 1, data.size))
    return [np.unravel_index(row.data, data.shape) for row in M]


@torch.no_grad()
def get_assignments(embs, centroids, imgs_per_chunk=1000):
    """
    :param embs: [nb_img, size]
    :param centroids: [cent_size, size]
    :param imgs_per_chunk:  batch size
    :return: [nb_img]
    """
    nb_imgs = embs.shape[0]
    assignments = []
    for idx in range(0, nb_imgs, imgs_per_chunk):
        batch_embs = embs[idx: min((idx + imgs_per_chunk), nb_imgs)]
        dot_product = torch.mm(batch_embs, centroids.t())
        _, assignment = dot_product.max(dim=1)
        assignments.append(assignment)
    return torch.cat(assignments, dim=-1)


def balanced_cluster_memory(embs, K, kmeans_iter):
    cluster_size = int(embs.shape[0] / K)
    device = embs.device
    random_idx = torch.randperm(len(embs))[:K]
    assert len(random_idx) >= K, "please reduce the number of centroids"
    centroids = embs[random_idx]
    for n_iter in tqdm(range(kmeans_iter + 1)):
        # finish
        if n_iter == kmeans_iter:
            break
        for c_idx in range(0, K, 100):
            end_idx = min(c_idx + 100, K)
            dot_product = torch.mm(centroids[c_idx: end_idx], embs.t())
            _, indices = dot_product.topk(cluster_size, largest=True, sorted=True)
            centroids[c_idx: end_idx] = embs[indices].mean(1)

        # normalize centroids
        centroids = torch.nn.functional.normalize(centroids, dim=1)
    return centroids


def cluster_memory(embs, K, kmeans_iter):
    device = embs.device
    random_idx = torch.randperm(len(embs))[:K]
    assert len(random_idx) >= K, "please reduce the number of centroids"
    centroids = embs[random_idx]
    for n_iter in tqdm(range(kmeans_iter + 1)):
        # E step
        assignments = get_assignments(embs, centroids)

        # finish
        if n_iter == kmeans_iter:
            break

        # M step
        where_helper = get_indices_sparse(assignments.cpu().numpy())
        counts = torch.zeros(K).to(device, non_blocking=True).int()
        emb_sums = torch.zeros(K, embs.shape[1]).to(device, non_blocking=True)
        for k in range(len(where_helper)):
            if len(where_helper[k][0]) > 0:
                emb_sums[k] = torch.sum(
                    embs[where_helper[k][0]],
                    dim=0,
                )
                counts[k] = len(where_helper[k][0])
        mask = counts > 0
        centroids[mask] = emb_sums[mask] / counts[mask].unsqueeze(1)
        gini_coef = gini(counts.cpu().numpy())
        print('Gini Coef for cluster numbers:', gini_coef)

        # normalize centroids
        centroids = torch.nn.functional.normalize(centroids, dim=1)
    return centroids


def gini(x):
    mad = np.abs(np.subtract.outer(x, x)).mean()
    rmad = mad/np.mean(x)
    # Gini coefficient
    g = 0.5 * rmad
    return g


def collect_knn_results(train_features, train_labels, test_features, test_labels,
                        nb_knn, temperature, train_ratio=0.2, with_train=True):
    print("Features are ready!\nStart the k-NN classification.")
    nb_classes = (train_labels.max()).item() + 1
    test_result = knn_classifier(train_features, train_labels,
                                 test_features, test_labels, nb_knn,
                                 temperature, num_classes=nb_classes)
    if with_train:
        train_ids = [i for i in range(train_labels.shape[0])]
        random.shuffle(train_ids)
        train_ids = train_ids[:int(train_ratio * len(train_ids))]
        train_result = knn_classifier(train_features, train_labels,
                                      train_features[train_ids],
                                      train_labels[train_ids],
                                      nb_knn, temperature,
                                      num_classes=nb_classes,
                                      leave_one_out=True)
        test_result = test_result.set_index('k')
        train_result = train_result.set_index('k')
        test_result['train_top1'] = train_result['top1']
        test_result['train_top5'] = train_result['top5']
    return test_result

