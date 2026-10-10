import numpy as np


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter

    return float(inter / union) if union > 0 else 0.0

def mean_average_precision(preds, gts, iou_thresh=0.5):
    aps = []
    classes = {c for c, _ in gts}
    for c in classes:
        gt_boxes = [b for cid, b in gts if cid == c]
        num_gt = len(gt_boxes)
        used = [False]*num_gt

        pred_boxes = [(b,score) for cid, score, b in preds if cid == c]
        pred_boxes.sort(key = lambda x: -x[1])
        p, r = [] , []
        tp, fp = 0, 0
        for pred_box, sc in pred_boxes:
            best, best_idx = 0, None
            for i, gt_box in enumerate(gt_boxes):
                if used[i]:
                    continue
                iou_boxes = iou(pred_box, gt_box)

                if iou_boxes > best:
                    best = iou_boxes
                    best_idx = i 

            if best >= iou_thresh and best_idx is not None:
                used[best_idx] = True 
                tp += 1
            else:
                fp += 1

            p.append(tp / (tp + fp))
            r.append(tp / num_gt)

        for i in range(len(p) - 2, -1, -1):
            p[i] = max(p[i+1], p[i])

        ap, prev_r = 0.0, 0.0
        for rec, prec in zip(r, p):
            ap += (rec - prev_r) * prec
            prev_r = rec
        aps.append(ap)

    return round(sum(aps) / len(aps), 4)
    
