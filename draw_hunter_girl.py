"""Draws a 96x96 pixel-art hunter girl portrait and saves it scaled up 7x."""
from PIL import Image, ImageDraw

W = H = 96
SCALE = 7

# Palette
BG_TOP = (22, 30, 34)
BG_BOT = (40, 52, 50)
FOG = (58, 72, 70)
HAIR = (24, 22, 28)
HAIR_MID = (44, 42, 52)
HAIR_HI = (82, 82, 98)
HAIR_EDGE = (10, 8, 12)
SKIN = (250, 238, 232)
SKIN_SHADE = (232, 214, 210)
SKIN_DARK = (206, 182, 182)
LIP = (214, 160, 160)
LASH = (16, 14, 18)
IRIS = (22, 20, 26)
IRIS_HI = (96, 96, 110)
SCLERA = (244, 242, 244)
CLOAK = (50, 58, 44)
CLOAK_HI = (72, 82, 60)
CLOAK_DARK = (30, 36, 28)
LEATHER = (92, 60, 40)
LEATHER_HI = (124, 84, 56)
BRASS = (196, 156, 84)
BRASS_HI = (240, 210, 140)
SHIRT = (150, 138, 120)

img = Image.new("RGB", (W, H))
px = img.load()
for y in range(H):
    t = y / (H - 1)
    for x in range(W):
        px[x, y] = tuple(int(BG_TOP[i] * (1 - t) + BG_BOT[i] * t) for i in range(3))
d = ImageDraw.Draw(img)

# Misty background: faint tree trunks and fog
for tx, w in ((6, 4), (82, 5), (90, 3), (14, 2)):
    d.rectangle((tx, 0, tx + w, H), fill=(30, 40, 40))
d.ellipse((-20, 70, 40, 100), fill=FOG)
d.ellipse((60, 74, 120, 104), fill=FOG)

# --- Hair behind the head ---
d.ellipse((26, 12, 70, 56), fill=HAIR)         # skull
d.polygon([(27, 40), (25, 70), (30, 72), (32, 46)], fill=HAIR)   # left back lock
d.polygon([(69, 40), (72, 76), (66, 74), (64, 46)], fill=HAIR)   # right back lock

# --- Cloak: hood pooled behind the shoulders ---
d.polygon([(8, 96), (14, 80), (26, 70), (40, 68), (56, 68), (70, 70),
           (82, 80), (88, 96)], fill=CLOAK)
d.polygon([(18, 78), (30, 70), (40, 72), (30, 80)], fill=CLOAK_HI)    # hood fold
d.polygon([(66, 72), (78, 78), (70, 82)], fill=CLOAK_HI)
d.line((14, 80, 26, 72), fill=CLOAK_DARK)
d.line((82, 80, 70, 72), fill=CLOAK_DARK)

# Hood bunched up behind the neck
d.polygon([(26, 74), (30, 64), (40, 60), (56, 60), (66, 64), (70, 74), (56, 70), (40, 70)], fill=CLOAK_DARK)
d.line((31, 66, 40, 62), fill=CLOAK)
d.line((65, 66, 56, 62), fill=CLOAK)

# Neck
d.polygon([(43, 60), (53, 60), (54, 76), (42, 76)], fill=SKIN_SHADE)
d.polygon([(42, 60), (54, 60), (54, 66), (48, 69), (42, 66)], fill=SKIN_DARK)

# Shirt V and cloak front panels
d.polygon([(40, 76), (56, 76), (48, 88)], fill=SHIRT)
d.polygon([(22, 96), (30, 80), (40, 74), (48, 90), (46, 96)], fill=CLOAK)
d.polygon([(74, 96), (66, 80), (56, 74), (48, 90), (50, 96)], fill=CLOAK)
d.line((40, 74, 48, 90), fill=CLOAK_DARK)
d.line((56, 74, 48, 90), fill=CLOAK_DARK)
d.line((31, 82, 38, 76), fill=CLOAK_HI)
d.line((65, 82, 58, 76), fill=CLOAK_HI)
d.line((28, 88, 34, 96), fill=CLOAK_DARK)
d.line((68, 88, 62, 96), fill=CLOAK_DARK)

# Leather strap across the chest and brass clasp
d.polygon([(30, 96), (64, 74), (68, 77), (36, 96)], fill=LEATHER)
d.line((32, 96, 65, 75), fill=LEATHER_HI)
d.rectangle((40, 86, 44, 90), fill=BRASS)
d.point((41, 87), fill=BRASS_HI)
# Cloak clasp at the throat
d.line((41, 77, 55, 77), fill=LEATHER)
d.ellipse((45, 75, 51, 81), fill=BRASS)
d.ellipse((47, 77, 49, 79), fill=BRASS_HI)
d.point((47, 76), fill=BRASS_HI)

# --- Face ---
d.polygon([(32, 34), (64, 34), (65, 48), (63, 56), (58, 63), (52, 67),
           (48, 68), (44, 67), (38, 63), (33, 56), (31, 48)], fill=SKIN)
d.line((33, 56, 38, 63), fill=SKIN_SHADE)
d.line((63, 56, 58, 63), fill=SKIN_SHADE)
d.line((38, 63, 44, 67), fill=SKIN_SHADE)
d.line((58, 63, 52, 67), fill=SKIN_SHADE)
d.rectangle((32, 34, 64, 41), fill=SKIN_SHADE)    # shadow under bangs

# Half-lidded eyes (blank, calm)
def eye(x0, flip):
    # 11px wide, centred on y=50
    xs = (lambda i: x0 + 10 - i) if flip else (lambda i: x0 + i)
    # sclera + iris (only lower part visible under heavy lid)
    for i in range(1, 10):
        d.line((xs(i), 49, xs(i), 52), fill=SCLERA)
    for i in range(3, 8):
        d.line((xs(i), 49, xs(i), 52), fill=IRIS)
    # heavy upper lid: flat, slightly drooping at the outer corner
    for i in range(0, 11):
        d.point((xs(i), 48), fill=LASH)
    for i in range(1, 10):
        d.point((xs(i), 47), fill=LASH)
    d.point((xs(10), 49), fill=LASH)
    d.point((xs(11), 49), fill=LASH)
    d.point((xs(11), 47), fill=LASH)       # outer lash flick
    d.point((xs(12), 46), fill=LASH)
    # crease line above the lid
    for i in range(2, 9):
        d.point((xs(i), 45), fill=SKIN_DARK)
    # lower lash line, soft
    for i in range(2, 10):
        d.point((xs(i), 53), fill=SKIN_DARK)
    d.point((xs(8), 54), fill=SKIN_DARK)
    d.point((xs(5), 54), fill=SKIN_DARK)

eye(34, flip=True)
eye(51, flip=False)

# Nose: tiny shadow; mouth: small flat neutral line
d.point((48, 57), fill=SKIN_DARK)
d.point((47, 58), fill=SKIN_DARK)
d.line((46, 62, 50, 62), fill=LIP)
d.point((48, 63), fill=SKIN_SHADE)

# --- Front hair ---
# Crown of the head
d.ellipse((26, 8, 70, 36), fill=HAIR)

# Messy bangs: many separate tapered strands of uneven length
# (root_x, tip_x, tip_y, half_width)
strands = [
    (29, 27, 48, 3), (32, 31, 43, 3), (35, 34, 46, 2), (37, 40, 41, 2),
    (40, 38, 45, 2), (43, 45, 43, 2), (45, 47, 48, 2), (48, 46, 40, 2),
    (50, 52, 46, 2), (53, 51, 42, 2), (55, 58, 45, 2), (58, 56, 41, 2),
    (60, 62, 46, 3), (63, 64, 42, 3), (67, 69, 49, 3),
]
for rx, tx, ty, w in strands:
    d.polygon([(rx - w, 26), (rx + w, 26), (tx, ty)], fill=HAIR)
# Thin lines inside the fringe to separate the strands
for rx, tx, ty, w in strands[::2]:
    d.line((rx, 28, (rx + tx) // 2, (28 + ty) // 2), fill=HAIR_MID)

# Stray tufts poking out of the silhouette
for poly in (
    [(27, 20), (29, 16), (21, 13)],
    [(69, 20), (67, 16), (75, 14)],
    [(36, 11), (39, 9), (32, 5)],
    [(60, 11), (57, 9), (63, 4)],
    [(26, 28), (26, 32), (19, 34)],
    [(70, 28), (70, 32), (77, 35)],
):
    d.polygon(poly, fill=HAIR)

# Slim side locks, a little jagged
d.polygon([(27, 34), (31, 36), (31, 50), (29, 62), (27, 54)], fill=HAIR)
d.polygon([(28, 52), (30, 56), (26, 64)], fill=HAIR)
d.polygon([(69, 34), (65, 36), (65, 50), (67, 60), (69, 54)], fill=HAIR)
d.polygon([(68, 52), (66, 56), (70, 66)], fill=HAIR)

# Ahoge: two strands curling up from the crown
d.line([(44, 9), (45, 5), (48, 2), (53, 1)], fill=HAIR)
d.line([(47, 9), (48, 6), (51, 3), (55, 3)], fill=HAIR)

# Hair sheen
for pts in (
    [(36, 14), (42, 12), (52, 12), (58, 14)],
    [(31, 18), (34, 16)],
):
    d.line(pts, fill=HAIR_HI)

# Dark edge where hair meets background, for a crisp sprite look
src = img.copy().load()
hair_set = {HAIR, HAIR_MID, HAIR_HI}
for y in range(H):
    for x in range(W):
        if src[x, y] in hair_set:
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H and src[nx, ny] not in hair_set \
                        and src[nx, ny] not in (SKIN, SKIN_SHADE, SKIN_DARK, LASH):
                    px[x, y] = HAIR_EDGE
                    break

img.resize((W * SCALE, H * SCALE), Image.NEAREST).save("hunter_girl.png")
img.save("hunter_girl_96x96.png")
print("saved")
