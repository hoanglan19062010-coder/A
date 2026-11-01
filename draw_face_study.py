"""Pixel-art anime face construction study in 4 steps (2x2 sheet).

Proportions follow the classic anime head guide:
- cranium circle, jaw tapering to a soft point
- eye line at half the head height, nose and mouth splitting the lower half
- ears between eye line and nose, neck about a third of the head width
"""
import math
from PIL import Image, ImageDraw, ImageFont

PW, PH = 80, 100      # panel size in pixels
SCALE = 5
BG = (250, 248, 244)
GUIDE_RED = (240, 150, 150)
GUIDE_BLUE = (140, 180, 230)
GUIDE_FAINT = (225, 222, 228)
LINE = (58, 40, 50)
SKIN = (252, 234, 224)
SKIN_SHADE = (236, 200, 192)
HAIR = (78, 62, 92)
HAIR_SHADE = (54, 42, 66)
HAIR_HI = (130, 112, 150)
IRIS = [(30, 40, 80), (40, 60, 110), (60, 90, 150), (90, 130, 190), (120, 170, 220)]
WHITE = (255, 255, 255)
CLOTH = (60, 70, 90)

CX = 40
TOP, CHIN = 14, 74
EYE_Y = (TOP + CHIN) // 2          # 44: half of the head
NOSE_Y = 57
MOUTH_Y = 64


def arc(cx, cy, r, a0, a1, n=24):
    return [(round(cx + r * math.cos(math.radians(a))), round(cy + r * math.sin(math.radians(a))))
            for a in [a0 + (a1 - a0) * k / n for k in range(n + 1)]]


FACE = arc(CX, 36, 20, 180, 360) + [(60, 44), (60, 53), (57, 62), (51, 69), (44, 73), (40, 74),
                                    (36, 73), (29, 69), (23, 62), (20, 53), (20, 44)]
EAR_L = [(21, 45), (17, 46), (16, 50), (18, 55), (21, 57)]
EAR_R = [(2 * CX - x, y) for x, y in EAR_L]
NECK = [(35, 66), (45, 66), (46, 81), (34, 81)]

BACK_HAIR = [(18, 22), (14, 30), (14, 36), (11, 44), (12, 52), (9, 62), (10, 70), (7, 80),
             (8, 86), (4, 96), (11, 92), (14, 99), (19, 93), (24, 98), (28, 90), (40, 88),
             (52, 90), (56, 98), (61, 93), (66, 99), (69, 92), (76, 96), (72, 86), (73, 80),
             (70, 70), (71, 62), (68, 52), (69, 44), (66, 36), (66, 30), (62, 22)]
CROWN = (17, 11, 63, 50)
# bangs clumps from a part slightly right of centre: (root_l, root_r, tip_x, tip_y)
BANGS = [(20, 27, 22, 47), (25, 32, 28, 41), (30, 37, 33, 44), (35, 41, 37, 40),
         (39, 45, 41, 46), (43, 49, 46, 42), (47, 54, 51, 44), (52, 58, 55, 40),
         (56, 61, 59, 47)]
SIDE_L = [(19, 30), (24, 36), (24, 50), (22, 62), (20, 70), (22, 80), (19, 76), (16, 83), (16, 70), (17, 56), (16, 44)]
SIDE_R = [(2 * CX - x, y) for x, y in SIDE_L]


def clump(rl, rr, tx, ty, ox, oy):
    mid = (24 + ty) // 2
    return [(ox + rl, oy + 22), (ox + rr, oy + 22), (ox + rr - 1, oy + mid),
            (ox + tx, oy + ty), (ox + rl + 1, oy + mid)]


def off(pts, ox, oy):
    return [(x + ox, y + oy) for x, y in pts]


def eye(d, x0, oy, flip, color):
    """12px wide anime eye. x0 = outer corner of the left eye (mirrored when flip)."""
    xs = (lambda i: x0 + 11 - i) if flip else (lambda i: x0 + i)
    p = lambda i, y, c: d.point((xs(i), oy + y), fill=c)
    top = EYE_Y + 1
    if color:
        for y in range(top + 1, top + 9):
            for i in range(1, 11):
                p(i, y, WHITE)
        for row, y in enumerate(range(top + 1, top + 9)):
            c = IRIS[min(row * len(IRIS) // 8, len(IRIS) - 1)]
            lo, hi = (4, 8) if y in (top + 1, top + 8) else (3, 9)
            for i in range(lo, hi + 1):
                p(i, y, c)
        for y in range(top + 2, top + 6):
            for i in range(5, 8):
                p(i, y, IRIS[0])
        p(4, top + 2, WHITE)
        p(4, top + 3, WHITE)
        p(8, top + 6, (200, 225, 250))
    else:
        # line-art iris
        for y in range(top + 2, top + 7):
            p(3, y, LINE)
            p(9, y, LINE)
        for i in range(5, 8):
            p(i, top + 8, LINE)
        p(4, top + 7, LINE)
        p(8, top + 7, LINE)
        for y in range(top + 3, top + 6):
            for i in range(5, 8):
                p(i, y, LINE)
        p(5, top + 3, BG if not color else WHITE)
    # thick upper lash, heaviest at outer corner
    for i in range(2, 10):
        p(i, top - 1, LINE)
    for i in range(0, 12):
        p(i, top, LINE)
    p(-1, top, LINE)
    p(0, top + 1, LINE)
    p(-1, top + 2, LINE)
    p(1, top + 1, LINE)
    # lower lash
    for i in range(3, 9):
        p(i, top + 10, LINE if not color else SKIN_SHADE)
    p(1, top + 9, LINE)
    # brow
    for i in range(2, 10):
        p(i, top - 5 + (1 if i < 4 else 0), LINE if not color else HAIR_SHADE)


def features(d, ox, oy, color):
    eye(d, ox + 23, oy, False, color)
    eye(d, ox + 46, oy, True, color)
    d.point((ox + CX, oy + NOSE_Y), fill=LINE if not color else SKIN_SHADE)
    d.point((ox + CX - 1, oy + NOSE_Y + 1), fill=LINE if not color else (210, 160, 150))
    m = LINE if not color else (190, 110, 110)
    d.line((ox + CX - 2, oy + MOUTH_Y, ox + CX + 2, oy + MOUTH_Y), fill=m)
    d.point((ox + CX - 3, oy + MOUTH_Y - 1), fill=m)
    d.point((ox + CX + 3, oy + MOUTH_Y - 1), fill=m)


def guides(d, ox, oy, faint):
    red = GUIDE_FAINT if faint else GUIDE_RED
    blue = GUIDE_FAINT if faint else GUIDE_BLUE
    d.ellipse((ox + 20, oy + 16, ox + 60, oy + 56), outline=blue)
    d.line((ox + CX, oy + 8, ox + CX, oy + 92), fill=red)
    for y in (EYE_Y, NOSE_Y, MOUTH_Y):
        d.line((ox + 12, oy + y, ox + 68, oy + y), fill=blue)
    d.line((ox + 12, oy + TOP, ox + 68, oy + TOP), fill=red)
    d.line((ox + 12, oy + CHIN, ox + 68, oy + CHIN), fill=red)


def panel(img, ox, oy, step):
    d = ImageDraw.Draw(img)
    if step == 1:
        guides(d, ox, oy, faint=False)
        d.polygon(off(FACE, ox, oy), outline=GUIDE_BLUE)
        d.polygon(off(NECK, ox, oy), outline=GUIDE_BLUE)
        return
    if step == 4:
        d.polygon(off(BACK_HAIR, ox, oy), fill=HAIR_SHADE, outline=LINE)
        d.polygon([(ox + 2, oy + 100), (ox + 8, oy + 88), (ox + 34, oy + 80), (ox + 46, oy + 80),
                   (ox + 72, oy + 88), (ox + 78, oy + 100)], fill=CLOTH, outline=LINE)
    else:
        guides(d, ox, oy, faint=True)
    fill = SKIN if step == 4 else BG
    d.polygon(off(NECK, ox, oy), fill=SKIN_SHADE if step == 4 else None, outline=LINE)
    if step < 4:
        d.line((ox + 34, oy + 80, ox + 8, oy + 89), fill=LINE)
        d.line((ox + 46, oy + 80, ox + 72, oy + 89), fill=LINE)
    d.polygon(off(EAR_L, ox, oy), fill=fill, outline=LINE)
    d.polygon(off(EAR_R, ox, oy), fill=fill, outline=LINE)
    d.polygon(off(FACE, ox, oy), fill=fill, outline=LINE)
    if step == 4:
        d.line((ox + 28, oy + 66, ox + 40, oy + 72), fill=SKIN)  # keep jaw clean
        for c in BANGS:  # shadow of bangs on forehead
            d.polygon([(x, y + 2) for x, y in clump(*c, ox, oy)], fill=SKIN_SHADE)
        for x in (26, 54):
            d.line((ox + x - 2, oy + 57, ox + x + 1, oy + 57), fill=(248, 196, 196))  # blush
    features(d, ox, oy, color=(step == 4))
    if step >= 3:
        hf = HAIR if step == 4 else BG
        if step == 3:
            d.polygon(off(BACK_HAIR, ox, oy), outline=LINE)
            d.polygon(off(FACE, ox, oy), fill=BG, outline=LINE)
            d.polygon(off(EAR_L, ox, oy), fill=BG, outline=LINE)
            d.polygon(off(EAR_R, ox, oy), fill=BG, outline=LINE)
            features(d, ox, oy, color=False)
        crown = [(ox + 17, oy + 32)]
        for k in range(19):
            a = math.radians(180 + k * 10)
            r = 1.0 + (0.07 if k % 2 else 0.0)
            crown.append((round(ox + 40 + 23 * r * math.cos(a)), round(oy + 31 + 19 * r * math.sin(a))))
        crown.append((ox + 63, oy + 32))
        d.polygon(crown, fill=hf, outline=LINE)
        # loose flyaway strands on top
        d.line((ox + 42, oy + 12, ox + 41, oy + 9, ox + 43, oy + 7, ox + 46, oy + 8), fill=LINE)
        d.rectangle((ox + 18, oy + 22, ox + 62, oy + 31), fill=hf)
        d.polygon(off(SIDE_L, ox, oy), fill=hf, outline=LINE)
        d.polygon(off(SIDE_R, ox, oy), fill=hf, outline=LINE)
        for c in BANGS:
            d.polygon(clump(*c, ox, oy), fill=hf)
        for c in BANGS:
            pts = clump(*c, ox, oy)
            d.line(pts[2:] + [pts[4]], fill=LINE)
        # hair part and flow lines from the crown
        d.line((ox + 44, oy + 12, ox + 44, oy + 22), fill=LINE)
        for x in (30, 36, 52, 56):
            d.line((ox + 44, oy + 14, ox + x, oy + 30), fill=HAIR_SHADE if step == 4 else GUIDE_FAINT)
        if step == 4:
            d.arc((ox + 22, oy + 15, ox + 58, oy + 40), 200, 330, fill=HAIR_HI)
            d.arc((ox + 24, oy + 17, ox + 56, oy + 40), 215, 315, fill=HAIR_HI)
            for pts in ([(14, 40), (12, 60), (9, 82)], [(66, 40), (68, 60), (71, 82)],
                        [(18, 56), (17, 74)], [(62, 56), (63, 74)]):
                d.line([(ox + x, oy + y) for x, y in pts], fill=HAIR)
            for pts in ([(20, 40), (20, 62), (18, 74)], [(60, 40), (60, 62), (62, 74)]):
                d.line([(ox + x, oy + y) for x, y in pts], fill=HAIR_HI)


def main():
    sheet = Image.new("RGB", (PW * 2, PH * 2), BG)
    for k, (ox, oy) in enumerate(((0, 0), (PW, 0), (0, PH), (PW, PH)), start=1):
        panel(sheet, ox, oy, k)
    big = sheet.resize((PW * 2 * SCALE, PH * 2 * SCALE), Image.NEAREST)

    # add a title bar and step captions
    out = Image.new("RGB", (big.width, big.height + 70), BG)
    out.paste(big, (0, 70))
    d = ImageDraw.Draw(out)
    title = ImageFont.load_default(size=30)
    small = ImageFont.load_default(size=22)
    d.text((20, 18), "Anime face construction - pixel study", fill=LINE, font=title)
    labels = ["1  Guides: circle, jaw, eye line at 1/2",
              "2  Features: eyes, nose, mouth, ears",
              "3  Hair: crown, bangs, side locks",
              "4  Color + shading"]
    for k, (ox, oy) in enumerate(((0, 0), (PW, 0), (0, PH), (PW, PH))):
        d.text((ox * SCALE + 12, 70 + oy * SCALE + 8), labels[k], fill=(170, 90, 100), font=small)
    d.line((PW * SCALE, 70, PW * SCALE, out.height), fill=(210, 205, 210), width=2)
    d.line((0, 70 + PH * SCALE, out.width, 70 + PH * SCALE), fill=(210, 205, 210), width=2)
    out.save("face_study.png")
    print("saved")


main()
