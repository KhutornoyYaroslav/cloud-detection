import numpy as np


def image_to_tiles(image: np.array, tile_size: int, tile_pad: int = 0) -> np.array:
    if not image.ndim in [2, 3]:
        raise ValueError("Expected a 2D or 3D array as input")

    h, w = image.shape[0:2] 

    # prepare expanded frame
    ts = tile_size - 2 * tile_pad
    assert ts > 0, "Tile padding is bigger than tile size itself"

    rows = h // ts + (1 if h % ts else 0)
    cols = w // ts + (1 if w % ts else 0)

    # pad image
    pads = [(tile_pad, rows * ts - h + tile_pad), (tile_pad, cols * ts - w + tile_pad)]
    if image.ndim == 3:
        pads += [(0, 0)]
    padded_image = np.pad(image, pads, mode='constant', constant_values=0)

    # split image to tiles
    tiles = []
    for y in range(0, rows * ts, ts):
        for x in range(0, cols * ts, ts):
            tile = padded_image[y:y + tile_size, x:x + tile_size]
            tiles.append(tile)

    return np.stack(tiles, 0) # (n, h, w, c)
