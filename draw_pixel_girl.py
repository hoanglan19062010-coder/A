"""Draws a 64x64 pixel-art anime girl portrait and saves it scaled up 10x."""
from PIL import Image, ImageDraw

W = H = 64
SCALE = 10

# Palette
BG_TOP = (28, 30, 44)
BG_BOT = (44, 40, 52)
GLOW = (120, 140, 160)
GLOW2 = (80, 96, 118)
HAIR = (236, 168, 104)
HAIR_LIGHT = (250, 204, 140)
HAIR_DARK = (196, 120, 70)
HAIR_SHADOW = (150, 84, 52)
SKIN = (252, 222, 196)
SKIN_SHADE = (236, 188, 160)
SKIN_DARK = (212, 156, 132)
OUTLINE = (70, 40, 36)
EYE_DARK = (20, 70, 76)
EYE = (40, 150, 150)
EYE_LIGHT = (120, 220, 200)
WHITE = (255, 255, 255)
BLUSH = (246, 170, 160)
COLLAR = (30, 70, 110)
COLLAR_LIGHT = (60, 120, 170)
GOLD = (220, 170, 70)
ARMOR = (200, 206, 214)
ARMOR_LIGHT = (240, 244, 248)
ARMOR_DARK = (130, 138, 152)
CREST = (30, 76, 90)
CAPE = (24, 66, 76)
CAPE_LIGHT = (40, 98, 108)

img = Image.new("RGB", (W, H))
px = img.load()

# Background gradient + soft window glow on the left
for y in range(H):
    t = y / (H - 1)
    c = tuple(int(BG_TOP[i] * (1 - t) + BG_BOT[i] * t) for i in range(3))
    for x in range(W):
        px[x, y] = c
d = ImageDraw.Draw(img)
d.ellipse((-16, 28, 14, 64), fill=(40, 46, 62))
d.ellipse((-12, 32, 10, 60), fill=(56, 66, 86))
d.ellipse((-8, 36, 6, 56), fill=GLOW2)
d.ellipse((-5, 40, 3, 52), fill=GLOW)

# Cape / shoulders
d.polygon([(4, 64), (10, 54), (22, 50), (42, 50), (54, 54), (60, 64)], fill=CAPE)
d.polygon([(8, 64), (12, 57), (20, 54), (20, 64)], fill=CAPE_LIGHT)
d.line((6, 62, 20, 55), fill=GOLD)
d.line((58, 62, 44, 55), fill=GOLD)

# Back of hair (bob)
d.polygon([(14, 16), (20, 7), (32, 4), (44, 7), (50, 16), (52, 30),
           (51, 44), (46, 46), (18, 46), (13, 44), (12, 30)], fill=HAIR_DARK)

# Neck
d.rectangle((28, 42, 36, 52), fill=SKIN_SHADE)
d.rectangle((28, 42, 36, 45), fill=SKIN_DARK)

# Collar
d.polygon([(22, 48), (28, 46), (32, 50), (36, 46), (42, 48), (44, 54), (20, 54)], fill=COLLAR)
d.line((23, 49, 28, 47), fill=COLLAR_LIGHT)
d.line((26, 47, 37, 47), fill=GOLD)
d.point([(28, 47), (36, 47)], fill=GOLD)

# Armor chest plate with crescent crest
d.polygon([(20, 54), (44, 54), (46, 64), (18, 64)], fill=ARMOR)
d.line((21, 55, 43, 55), fill=ARMOR_LIGHT)
d.line((20, 54, 18, 64), fill=ARMOR_DARK)
d.line((44, 54, 46, 64), fill=ARMOR_DARK)
d.ellipse((27, 56, 37, 66), fill=CREST)
d.ellipse((30, 55, 39, 64), fill=ARMOR)
for sx, ex in ((22, 27), (42, 37)):
    d.line((sx, 58, ex, 61), fill=CREST)
    d.line((sx, 62, ex, 63), fill=CREST)
d.point([(19, 60), (45, 60)], fill=ARMOR_DARK)

# Face
d.polygon([(19, 20), (45, 20), (45, 32), (42, 39), (36, 44), (32, 46),
           (28, 44), (22, 39), (19, 32)], fill=SKIN)
# Cheek shading under the bangs
d.polygon([(19, 20), (45, 20), (45, 24), (19, 24)], fill=SKIN_SHADE)
d.line((22, 39, 28, 44), fill=SKIN_SHADE)
d.line((42, 39, 36, 44), fill=SKIN_SHADE)

# Eyes (left and right)
def eye(x0, flip):
    # x0 = left edge of 7px-wide eye, rows 27..32
    d.rectangle((x0, 27, x0 + 6, 33), fill=WHITE)
    d.rectangle((x0 + 1, 27, x0 + 5, 33), fill=EYE)
    d.rectangle((x0 + 1, 27, x0 + 5, 28), fill=EYE_DARK)
    d.rectangle((x0 + 2, 29, x0 + 4, 31), fill=EYE_DARK)
    d.line((x0 + 2, 32, x0 + 4, 32), fill=EYE_LIGHT)
    d.line((x0 + 2, 33, x0 + 4, 33), fill=EYE_LIGHT)
    hx = x0 + 4 if flip else x0 + 2
    d.rectangle((hx, 28, hx, 29), fill=WHITE)
    d.point((x0 + 6 if flip else x0, 31), fill=EYE_LIGHT)
    # upper lash line
    d.line((x0 - 1, 26, x0 + 7, 26), fill=OUTLINE)
    if flip:
        d.point((x0 + 7, 27), fill=OUTLINE)
    else:
        d.point((x0 - 1, 27), fill=OUTLINE)
    d.line((x0 + 1, 34, x0 + 5, 34), fill=SKIN_DARK)
    # brow
    
eye(22, flip=False)
eye(35, flip=True)

# Nose, blush, smile
d.point((32, 36), fill=SKIN_DARK)
d.point((31, 37), fill=SKIN_DARK)
for bx in (21, 39):
    d.line((bx, 36, bx + 3, 36), fill=BLUSH)
d.line((31, 40, 33, 40), fill=(170, 96, 86))
d.point([(30, 39), (34, 39)], fill=(170, 96, 86))

# Side locks framing the face
d.polygon([(13, 18), (20, 18), (22, 30), (21, 44), (18, 47), (15, 40), (13, 30)], fill=HAIR)
d.polygon([(51, 18), (44, 18), (42, 30), (43, 44), (46, 47), (49, 40), (51, 30)], fill=HAIR)
d.line((19, 22, 20, 42), fill=HAIR_DARK)
d.line((45, 22, 44, 42), fill=HAIR_DARK)
d.line((15, 24, 16, 38), fill=HAIR_LIGHT)

# Top of head
d.polygon([(15, 18), (18, 10), (24, 6), (32, 4), (40, 6), (46, 10), (49, 18)], fill=HAIR)
d.line((22, 8, 30, 6), fill=HAIR_LIGHT)
d.line((20, 10, 24, 8), fill=HAIR_LIGHT)

# Spiky bangs: (tip_x, tip_y, base_left, base_right)
spikes = [
    (18, 26, 15, 22), (22, 28, 18, 27), (25, 25, 22, 29),
    (28, 27, 25, 31), (31, 24, 28, 33), (34, 26, 31, 36),
    (37, 27, 34, 39), (40, 25, 37, 42), (43, 28, 40, 46), (46, 26, 43, 49),
]
for tx, ty, bl, br in spikes:
    d.polygon([(bl, 17), (br, 17), (tx, ty)], fill=HAIR)
    d.line((bl, 17, tx, ty), fill=HAIR_DARK)
    d.point((tx, ty), fill=HAIR_SHADOW)
# Strands with a highlight
d.line((26, 18, 28, 24), fill=HAIR_LIGHT)
d.line((36, 18, 37, 23), fill=HAIR_LIGHT)
# Ahoge (the iconic stray strand)
d.line((32, 4, 34, 1), fill=HAIR)
d.line((34, 1, 37, 2), fill=HAIR)

# Outline pass: darken pixels next to background for crisp sprite edges
fig = {HAIR, HAIR_LIGHT, HAIR_DARK, SKIN, SKIN_SHADE, SKIN_DARK, BLUSH}
src = img.copy().load()
def is_bg(c):
    return c not in fig and c not in (WHITE, EYE, EYE_DARK, EYE_LIGHT, OUTLINE)
for y in range(H):
    for x in range(W):
        if src[x, y] in (HAIR, HAIR_LIGHT, HAIR_DARK):
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H and is_bg(src[nx, ny]) and src[nx, ny] not in (SKIN,):
                    px[x, y] = HAIR_SHADOW
                    break

img.resize((W * SCALE, H * SCALE), Image.NEAREST).save("pixel_anime_girl.png")
img.save("pixel_anime_girl_64x64.png")
print("saved")
