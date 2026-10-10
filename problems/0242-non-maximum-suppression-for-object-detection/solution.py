import torch
from typing import List, Union

def non_maximum_suppression(boxes, scores, iou_threshold) -> Union[List[int], int]:
    # validation
    if not isinstance(boxes, torch.Tensor) or not isinstance(scores, torch.Tensor):
        return -1
    if boxes.dim() != 2 or boxes.shape[1] != 4:
        return -1
    if scores.dim() != 1 or scores.shape[0] != boxes.shape[0]:
        return -1
    if isinstance(iou_threshold, bool) or not isinstance(iou_threshold, (int, float)):
        return -1
    if not 0 <= iou_threshold <= 1:
        return -1
    if boxes.shape[0] == 0:
        return []
    if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all():
        return -1
    if (boxes[:, 2] < boxes[:, 0]).any() or (boxes[:, 3] < boxes[:, 1]).any():
        return -1

    boxes = boxes.double()
    x1, y1, x2, y2 = boxes.unbind(1)
    areas = (x2 - x1) * (y2 - y1)

    order = torch.argsort(scores, descending=True, stable=True)
    keep = []

    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        rest = order[1:]
        if rest.numel() == 0:
            break

        xx1 = torch.maximum(x1[i], x1[rest])
        yy1 = torch.maximum(y1[i], y1[rest])
        xx2 = torch.minimum(x2[i], x2[rest])
        yy2 = torch.minimum(y2[i], y2[rest])
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        union = areas[i] + areas[rest] - inter
        ious = torch.where(union > 0, inter / union, torch.zeros_like(inter))

        order = rest[ious <= iou_threshold]

    return keep