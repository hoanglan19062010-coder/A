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
IRIS = (40, 38, 46)
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

# --- Hair behind the head (bun + back volume) ---
d.ellipse((18, 6, 46, 32), fill=HAIR)          # bun
d.ellipse((26, 12, 70, 56), fill=HAIR)         # skull
d.polygon([(28, 40), (24, 72), (34, 74), (36, 44)], fill=HAIR)   # left back lock
d.polygon([(68, 40), (72, 70), (62, 72), (60, 44)], fill=HAIR)   # right back lock

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
d.rectangle((32, 34, 64, 44), fill=SKIN_SHADE)    # shadow under bangs

# Half-lidded eyes (blank, calm)
def eye(x0, flip):
    # 11px wide, centred on y=50
    xs = (lambda i: x0 + 10 - i) if flip else (lambda i: x0 + i)
    # sclera + iris (only lower part visible under heavy lid)
    for i in range(1, 10):
        d.line((xs(i), 49, xs(i), 52), fill=SCLERA)
    for i in range(3, 8):
        d.line((xs(i), 49, xs(i), 52), fill=IRIS)
    d.point((xs(5), 51), fill=IRIS_HI)
    d.point((xs(4), 52), fill=IRIS_HI)
    d.point((xs(6), 49), fill=(250, 250, 255))   # tiny catchlight
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
# Crown
d.polygon([(28, 34), (30, 22), (38, 14), (48, 11), (58, 13), (66, 22), (68, 34)], fill=HAIR)
# Long straight bangs, slight side part
bangs = [
    (31, 48, 28, 36), (34, 45, 30, 39), (37, 44, 33, 42), (40, 45, 36, 45),
    (43, 43, 39, 48), (46, 45, 42, 51), (49, 43, 46, 54), (52, 45, 50, 57),
    (55, 44, 53, 60), (58, 45, 56, 63), (61, 44, 59, 66), (65, 48, 62, 68),
]
for tx, ty, bl, br in bangs:
    d.polygon([(bl, 30), (br, 30), (tx, ty)], fill=HAIR)
# Face-framing side strands
d.polygon([(28, 32), (34, 36), (33, 52), (31, 66), (28, 74), (27, 56)], fill=HAIR)
d.polygon([(68, 32), (63, 36), (64, 52), (66, 64), (69, 80), (70, 56)], fill=HAIR)
d.line((69, 60, 72, 82), fill=HAIR)            # long thin strand
d.line((70, 60, 71, 76), fill=HAIR)

# Messy flyaways from the bun
for poly in (
    [(24, 10), (28, 8), (22, 2), (18, 0)],          # up-left spike
    [(20, 16), (20, 20), (10, 17), (5, 19)],        # left spike
    [(22, 24), (25, 27), (14, 32), (11, 38)],       # down-left spike
    [(64, 20), (66, 24), (75, 16), (80, 13)],       # right spike
):
    d.polygon(poly, fill=HAIR)

# Hair sheen
for pts in (
    [(40, 15), (48, 13), (56, 15)],
    [(38, 17), (42, 16)],
    [(24, 12), (30, 10)],
    [(22, 16), (26, 22)],
):
    d.line(pts, fill=HAIR_HI)
for x in (36, 44, 52, 60):
    d.line((x, 20, x - 1, 32), fill=HAIR_MID)
d.line((66, 40, 67, 60), fill=HAIR_MID)
d.line((30, 40, 29, 60), fill=HAIR_MID)
d.line((24, 22, 30, 24), fill=HAIR_MID)

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
