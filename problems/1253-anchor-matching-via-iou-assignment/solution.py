import numpy as np

def match_anchors(anchors, gt_boxes, pos_threshold=0.5, neg_threshold=0.4):
    anchors = np.asarray(anchors, dtype=np.float64).reshape(-1, 4)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
    N, M = anchors.shape[0], gt_boxes.shape[0]

    # Edge cases
    if N == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    if M == 0:
        return np.zeros(N, dtype=np.int64), np.full(N, -1, dtype=np.int64)

    # 1. IoU table (N, M)
    a = anchors[:, None, :]
    g = gt_boxes[None, :, :]

    w = np.clip(np.minimum(a[..., 2], g[..., 2]) - np.maximum(a[..., 0], g[..., 0]), 0.0, None)
    h = np.clip(np.minimum(a[..., 3], g[..., 3]) - np.maximum(a[..., 1], g[..., 1]), 0.0, None)
    inter = w * h

    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_g = (g[..., 2] - g[..., 0]) * (g[..., 3] - g[..., 1])
    union = area_a + area_g - inter

    iou = np.where(union > 0, inter / np.where(union > 0, union, 1.0), 0.0)

    # 2. Best GT per anchor
    max_iou = iou.max(axis=1)
    best_gt = iou.argmax(axis=1)

    # 3. Threshold labels
    labels = np.full(N, -1, dtype=np.int64)
    labels[max_iou < neg_threshold] = 0
    pos = max_iou >= pos_threshold
    labels[pos] = 1

    matched_gt = np.full(N, -1, dtype=np.int64)
    matched_gt[pos] = best_gt[pos]

    # 4. Force match, GTs in order so later j wins
    for j in range(M):
        i = iou[:, j].argmax()
        labels[i] = 1
        matched_gt[i] = j

    return labels, matched_gt