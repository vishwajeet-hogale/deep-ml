import math
import torch

def bilinear_resize(image, new_height: int, new_width: int) -> list:
    img = torch.as_tensor(image, dtype=torch.float64)
    H, W = img.shape[:2]
    out_image = torch.empty((new_height, new_width) + img.shape[2:], dtype=torch.float64)

    scale_h = new_height / H
    scale_w = new_width / W

    for i in range(new_height):
        for j in range(new_width):
            y, x = i / scale_h, j / scale_w

            y1, x1 = math.floor(y), math.floor(x)
            y2, x2 = min(y1 + 1, H - 1), min(x1 + 1, W - 1)
            dy, dx = y - y1, x - x1

            top = img[y1, x1] * (1 - dx) + img[y1, x2] * dx
            bot = img[y2, x1] * (1 - dx) + img[y2, x2] * dx

            out_image[i, j] = top * (1 - dy) + bot * dy

    return torch.round(out_image, decimals=2).tolist()