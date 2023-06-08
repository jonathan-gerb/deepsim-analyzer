import cv2
import numpy as np

def feature_map_to_colormap(test_head):
    minmaxed = (test_head - test_head.min()) / (test_head.max() - test_head.min())
    minmaxed *= 255
    minmaxed = minmaxed.astype(np.uint8)
    heatmap = cv2.applyColorMap(minmaxed, cv2.COLORMAP_INFERNO)
    return heatmap, minmaxed


def overlay_transparent(background, overlay, x, y):
    # taken from https://stackoverflow.com/questions/40895785/using-opencv-to-overlay-transparent-image-onto-another-image

    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255
            ],
            axis = 2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

    return background

def overlay_heatmap(original_image, heatmap, minmaxed, transparancy=0.7):

    # add empty alpha channel
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2RGBA)

    # only take top 50% of probability mass
    mask = np.zeros_like(minmaxed, dtype=np.float32)
    mask[minmaxed >= np.mean(minmaxed)] = 1
    mask *= 255

    # make overlay transparant
    mask[mask == 255] *= transparancy
    mask = mask.astype(np.uint8)

    # add alpha channel to heatmap image
    heatmap[:,:, 3] = mask

    # mix heatmap with original image
    overlayed = overlay_transparent(np.array(original_image), heatmap, 0, 0)
    return overlayed
