"""数据读取与单图推理共用的尺寸和文本处理。"""


def resize_shape(height, width, short_side, max_size):
    scale = min(short_side / min(height, width), max_size / max(height, width))
    return int(round(height * scale)), int(round(width * scale))


def normalize_caption(caption):
    caption = caption.strip().lower()
    return caption if caption.endswith(".") else caption + " ."
