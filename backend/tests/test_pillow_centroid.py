from PIL import Image, ImageDraw
from backend.app import detect_red_pillow_coords


def test_pillow_centroid_on_ring():
    # create an image with a red ring (rim brighter than center)
    w, h = 200, 200
    img = Image.new('RGB', (w, h), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    cx, cy = 100, 100
    r_out = 40
    r_in = 20
    # draw outer ring with strong red (continuous outline)
    draw.ellipse((cx - r_out, cy - r_out, cx + r_out, cy + r_out), outline=(255, 0, 0), width=6)
    # draw inner area with dull red to simulate belly
    draw.ellipse((cx - r_in, cy - r_in, cx + r_in, cy + r_in), fill=(120, 30, 30))

    detected = detect_red_pillow_coords(img, red_min=120, score_min=20, min_pixels=50)
    assert detected[0] is not None and detected[1] is not None
    dcx, dcy = detected[0], detected[1]
    # should be close to actual center
    assert abs(dcx - cx) <= 3
    assert abs(dcy - cy) <= 3
