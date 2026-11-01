"""Pixel-art anime face construction study in 4 steps (2x2 sheet), hi-res version.

Geometry is laid out on an 80x100 grid and multiplied by K, so the art has K times
more pixels in each direction for finer lines, eyes and hair detail.

Proportions follow the classic anime head guide:
- cranium circle, jaw tapering to a soft rounded chin
- eye line at half the head height, nose and mouth splitting the lower half
- ears between eye line and nose, short slim neck
"""
import math
from PIL import Image, ImageDraw, ImageFont

K = 2                      # detail multiplier
PW, PH = 80 * K, 100 * K   # panel size in pixels
SCALE = 3                  # final display upscale
BG = (250, 248, 244)
GUIDE_RED = (240, 150, 150)
GUIDE_BLUE = (140, 180, 230)
GUIDE_FAINT = (226, 223, 229)
LINE = (58, 40, 50)
SKIN = (252, 236, 226)
SKIN_SHADE = (240, 208, 198)
SKIN_SHADE2 = (226, 186, 178)
BLUSH = (248, 190, 190)
LIP = (196, 112, 114)
HAIR = (78, 62, 92)
HAIR_SHADE = (54, 42, 66)
HAIR_DEEP = (38, 30, 48)
HAIR_MID = (100, 82, 118)
HAIR_HI = (146, 126, 170)
HAIR_SHINE = (190, 176, 210)
IRIS = [(24, 30, 66), (30, 40, 82), (38, 54, 104), (50, 74, 130), (66, 100, 160),
        (88, 132, 192), (118, 168, 220), (150, 196, 236)]
PUPIL = (16, 18, 40)
WHITE = (255, 255, 255)
SCLERA_SHADE = (222, 220, 236)
CLOTH = (60, 70, 90)
CLOTH_HI = (82, 94, 118)

CX = 40
TOP, CHIN = 14, 72
EYE_Y = (TOP + CHIN) // 2          # 43: half the head; eyes sit just below it
NOSE_Y = 60                          # halfway between eye centre and chin
MOUTH_Y = 66                         # halfway between nose and chin


def arc(cx, cy, r, a0, a1, n=48):
    return [(cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a)))
            for a in [a0 + (a1 - a0) * k / n for k in range(n + 1)]]


def mirror(pts):
    return [(2 * CX - x, y) for x, y in pts]


# bigger cranium; round cheeks tapering to a softly pointed chin
FACE = arc(CX, 36, 21, 180, 360) + [(61, 44), (60.8, 50), (59.8, 55.5), (57.5, 60.5), (54, 65),
                                    (49, 68.8), (43.5, 71.4), (40, 72.4), (36.5, 71.4), (31, 68.8),
                                    (26, 65), (22.5, 60.5), (20.2, 55.5), (19.2, 50), (19, 44)]
EAR_L = [(19.5, 45), (17, 45.5), (16, 48.5), (16.3, 53), (17.5, 57.5), (19.3, 60), (20, 60)]   # eye line to nose
EAR_R = mirror(EAR_L)
NECK = [(35.5, 64), (44.5, 64), (45.5, 81), (34.5, 81)]
SHOULDERS = [(1, 100), (6, 89), (20, 84), (34.5, 80), (45.5, 80), (60, 84), (74, 89), (79, 100)]

BACK_HAIR = [(18, 22), (14, 30), (14, 36), (11, 44), (12, 52), (9, 62), (10, 70), (7, 80),
             (8, 86), (4, 96), (11, 92), (14, 99), (19, 93), (24, 98), (28, 90), (40, 88),
             (52, 90), (56, 98), (61, 93), (66, 99), (69, 92), (76, 96), (72, 86), (73, 80),
             (70, 70), (71, 62), (68, 52), (69, 44), (66, 36), (66, 30), (62, 22)]
# bangs clumps: (root_l, root_r, tip_x, tip_y) - part slightly right of centre
BANGS = [(19, 25, 21, 47), (23, 29, 25.5, 42), (27, 33, 29, 45.5), (31, 36.5, 33, 41),
         (34.5, 40, 37, 44), (38.5, 44, 40.5, 46.5), (42.5, 47.5, 45, 42), (46, 51.5, 49, 45),
         (50, 55.5, 52.5, 40.5), (53.5, 58.5, 56, 44.5), (57, 62, 59.5, 47)]
SIDE_L = [(19, 30), (24, 36), (24.5, 50), (22.5, 62), (20.5, 70), (22, 80), (19, 76.5),
          (16.5, 83), (16, 70), (17, 56), (16, 44)]
SIDE_R = mirror(SIDE_L)


class Pen:
    """Draws on the hi-res panel using 80x100 grid coordinates."""

    def __init__(self, d, ox, oy):
        self.d, self.ox, self.oy = d, ox, oy

    def P(self, pts):
        return [(round(self.ox + x * K), round(self.oy + y * K)) for x, y in pts]

    def poly(self, pts, fill=None, outline=None):
        self.d.polygon(self.P(pts), fill=fill, outline=outline)

    def line(self, pts, fill, width=1):
        self.d.line(self.P(pts), fill=fill, width=width)

    def ellipse(self, box, fill=None, outline=None):
        (x0, y0), (x1, y1) = self.P([box[:2], box[2:]])
        self.d.ellipse((x0, y0, x1, y1), fill=fill, outline=outline)

    def px(self, x, y, c):
        """Raw hi-res pixel relative to panel origin."""
        self.d.point((self.ox + x, self.oy + y), fill=c)


def clump(rl, rr, tx, ty):
    ty -= 3.5          # let the tips rest just above the lashes
    mid = (24 + ty) / 2
    return [(rl, 22), (rr, 22), (rr - 0.6, mid), (tx, ty), (rl + 0.6, mid)]


def eye(pen, cx, flip, color):
    """Hi-res anime eye. cx is the eye centre on the 80-grid."""
    s = -1 if flip else 1                      # +1: outer corner on the left
    X = lambda dx: round(pen.ox + (cx * K) + s * dx)
    Y = lambda dy: round(pen.oy + (EYE_Y + 1) * K + dy)
    d = pen.d
    w = 13                                      # half-width in hi-res pixels
    h_top, h_bot = 0, 20
    # eye opening shape: flat-ish top, rounded bottom
    def inside(dx, dy):
        if dy < 1 or dy > h_bot:
            return False
        t = dy / h_bot
        half = w * math.sqrt(max(0.0, 1 - max(0.0, (t - 0.45) / 0.55) ** 2))
        return -w + (1 if dy < 3 else 0) <= dx <= half
    icx, icy, irx, iry = 1, 10.5, 7.5, 10.5     # iris ellipse
    for dy in range(1, h_bot + 1):
        for dx in range(-w, w + 1):
            if not inside(dx, dy):
                continue
            in_iris = ((dx - icx) / irx) ** 2 + ((dy - icy) / iry) ** 2 <= 1
            if color:
                c = SCLERA_SHADE if dy <= 3 else WHITE
                if in_iris:
                    c = IRIS[min(int(dy / h_bot * len(IRIS)), len(IRIS) - 1)]
                    if ((dx - icx) / irx) ** 2 + ((dy - icy) / iry) ** 2 > 0.78:
                        c = IRIS[0]                             # dark rim
                    if ((dx - icx) / 3.6) ** 2 + ((dy - 9) / 5.5) ** 2 <= 1:
                        c = PUPIL
                    if dy <= 4:
                        c = IRIS[0]                             # lid shadow
            else:
                c = BG
                if in_iris and ((dx - icx) / irx) ** 2 + ((dy - icy) / iry) ** 2 > 0.72:
                    c = LINE
                if ((dx - icx) / 3.6) ** 2 + ((dy - 9) / 5.5) ** 2 <= 1:
                    c = LINE
            d.point((X(dx), Y(dy)), fill=c)
    # highlights
    hl = WHITE
    for dx, dy in ((-2, 5), (-3, 5), (-2, 6), (-3, 6), (-2, 7), (-3, 7), (-1, 6), (-4, 6)):
        d.point((X(dx), Y(dy)), fill=hl)
    if color:
        for dx, dy in ((3, 16), (4, 16), (3, 17), (5, 16)):
            d.point((X(dx), Y(dy)), fill=(210, 230, 250))
    # thick upper lash, arched, heaviest and flicked at the outer corner
    for dx in range(-w - 1, w + 2):
        arch = round(2.2 * (1 - (dx / (w + 1)) ** 2))
        thick = 4 if dx < 3 else 3
        for t in range(-1, thick - 1):
            d.point((X(dx), Y(-arch + t)), fill=LINE)
    for dx, dy in ((-w - 2, 1), (-w - 3, 2), (-w - 2, 2), (-w - 4, 3), (-w - 1, 3), (-w, 4)):
        d.point((X(dx), Y(dy)), fill=LINE)
    d.point((X(w + 1), Y(2)), fill=LINE)
    # double eyelid crease
    for dx in range(-w + 3, w - 2):
        arch = round(2 * (1 - (dx / w) ** 2))
        d.point((X(dx), Y(-4 - arch)), fill=SKIN_SHADE2 if color else LINE)
    # lower lash, soft
    for dx in range(-w + 3, w - 3):
        d.point((X(dx), Y(h_bot + 1)), fill=SKIN_SHADE2 if color else LINE)
    for dx, dy in ((-w + 1, h_bot - 2), (-w + 2, h_bot - 1), (-w + 2, h_bot), (-w + 3, h_bot)):
        d.point((X(dx), Y(dy)), fill=LINE)
    # brow
    for dx in range(-w + 2, w - 1):
        y = -11 + (1 if dx < -w + 5 else 0)
        d.point((X(dx), Y(y)), fill=HAIR_SHADE if color else LINE)


def features(pen, color):
    eye(pen, CX - 12.5, False, color)
    eye(pen, CX + 12.5, True, color)
    nose = SKIN_SHADE2 if color else LINE
    pen.line([(CX + 0.5, NOSE_Y - 1), (CX, NOSE_Y)], fill=nose)
    m = LIP if color else LINE
    pen.line([(CX - 2.5, MOUTH_Y - 0.5), (CX - 1.5, MOUTH_Y), (CX + 1.5, MOUTH_Y), (CX + 2.5, MOUTH_Y - 0.5)], fill=m)
    if color:
        pen.line([(CX - 1, MOUTH_Y + 1), (CX + 1, MOUTH_Y + 1)], fill=(236, 196, 190))


def guides(pen, faint):
    red = GUIDE_FAINT if faint else GUIDE_RED
    blue = GUIDE_FAINT if faint else GUIDE_BLUE
    pen.ellipse((19, 15, 61, 57), outline=blue)
    pen.line([(19, 44), (CX, CHIN), (61, 44)], fill=blue)   # two lines meeting at the chin
    pen.line([(CX, 8), (CX, 92)], fill=red)
    for y in (EYE_Y, NOSE_Y, MOUTH_Y):
        pen.line([(12, y), (68, y)], fill=blue)
    pen.line([(12, TOP), (68, TOP)], fill=red)
    pen.line([(12, CHIN), (68, CHIN)], fill=red)


def crown_shape():
    pts = [(16.5, 32)]
    for k in range(37):
        a = math.radians(180 + k * 5)
        r = 1.0 + (0.045 if k % 2 else 0.0)
        pts.append((40 + 23.5 * r * math.cos(a), 31 + 19.5 * r * math.sin(a)))
    pts.append((63.5, 32))
    return pts


def hair_front(pen, step):
    color = step == 4
    hf = HAIR if color else BG
    pen.poly(crown_shape(), fill=hf, outline=LINE)
    pen.poly([(18, 22), (62, 22), (62.5, 31), (17.5, 31)], fill=hf)
    pen.poly(SIDE_L, fill=hf, outline=LINE)
    pen.poly(SIDE_R, fill=hf, outline=LINE)
    for c in BANGS:
        pen.poly(clump(*c), fill=hf)
    if color:
        # shade toward the tips of each clump, then a lighter stripe
        for rl, rr, tx, ty in BANGS:
            pen.poly([((rl + rr) / 2 - 1.2, 34), ((rl + rr) / 2 + 1.2, 34), (tx, ty - 3)], fill=HAIR_SHADE)
            pen.line([((rl + rr) / 2 - 0.5, 25), ((rl + rr) / 2 + (tx - (rl + rr) / 2) * 0.5, (25 + ty) / 2)], fill=HAIR_MID)
    for c in BANGS:
        pts = clump(*c)
        pen.line(pts[2:] + [pts[4]], fill=LINE)
    # flyaway curl and strand flow from the part
    pen.line([(42, 12), (41, 9), (42.5, 7), (45.5, 7.5)], fill=LINE)
    flow = HAIR_SHADE if color else GUIDE_FAINT
    for x in (24, 29, 34, 50, 55, 60):
        pen.line([(44, 13), ((44 + x) / 2, 18), (x, 28)], fill=flow)
    pen.line([(44, 10.5), (44, 22)], fill=LINE)
    if color:
        # angel-ring shine, broken into strands
        for k, (a0, a1) in enumerate(((198, 228), (234, 262), (270, 300), (306, 336))):
            pen.d.arc(tuple(pen.P([(22, 14), (58, 40)])[0] + pen.P([(22, 14), (58, 40)])[1]),
                      a0, a1, fill=HAIR_HI, width=2)
            pen.d.arc(tuple(pen.P([(23, 15.5), (57, 40)])[0] + pen.P([(23, 15.5), (57, 40)])[1]),
                      a0 + 4, a1 - 6, fill=HAIR_SHINE)
        # strands on side locks
        for pts in ([(20, 38), (21.5, 52), (20, 66), (19.5, 76)], [(18, 46), (18, 62), (17.5, 74)]):
            pen.line(pts, fill=HAIR_MID)
            pen.line(mirror(pts), fill=HAIR_MID)


def panel(img, ox, oy, step):
    pen = Pen(ImageDraw.Draw(img), ox, oy)
    if step == 1:
        guides(pen, faint=False)
        pen.poly(FACE, outline=GUIDE_BLUE)
        pen.poly(NECK, outline=GUIDE_BLUE)
        pen.line(SHOULDERS[1:-1], fill=GUIDE_BLUE)
        return
    color = step == 4
    if not color:
        guides(pen, faint=True)
    # back hair + clothes
    if step >= 3:
        pen.poly(BACK_HAIR, fill=HAIR_SHADE if color else BG, outline=LINE)
        if color:
            for pts in ([(14, 40), (12, 60), (9, 84)], [(17, 50), (15, 70), (13, 92)],
                        [(11.5, 64), (9, 90)]):
                pen.line(pts, fill=HAIR)
                pen.line(mirror(pts), fill=HAIR)
            pen.poly([(22, 60), (58, 60), (56, 88), (24, 88)], fill=HAIR_DEEP)
        else:
            for pts in ([(14, 40), (12, 60), (9, 84)], [(17, 50), (15, 70), (13, 92)]):
                pen.line(pts, fill=GUIDE_FAINT)
                pen.line(mirror(pts), fill=GUIDE_FAINT)
    if color:
        pen.poly(SHOULDERS, fill=CLOTH, outline=LINE)
        pen.line([(8, 90), (22, 85.5), (34.5, 81.5)], fill=CLOTH_HI)
        pen.line(mirror([(8, 90), (22, 85.5), (34.5, 81.5)]), fill=CLOTH_HI)
    else:
        pen.line(SHOULDERS[1:-1], fill=LINE)
    pen.poly(NECK, fill=SKIN if color else BG, outline=LINE)
    if color:
        pen.poly([(35.5, 64), (44.5, 64), (44.8, 70), (40, 73.5), (35.2, 70)], fill=SKIN_SHADE)
        pen.poly([(34.5, 79), (45.5, 79), (45.5, 81), (34.5, 81)], fill=SKIN_SHADE)
    skin = SKIN if color else BG
    pen.poly(EAR_L, fill=skin, outline=LINE)
    pen.poly(EAR_R, fill=skin, outline=LINE)
    if color:
        pen.line([(17.8, 48), (17.6, 54)], fill=SKIN_SHADE2)
        pen.line(mirror([(17.8, 48), (17.6, 54)]), fill=SKIN_SHADE2)
    pen.poly(FACE, fill=skin, outline=LINE)
    if color:
        for c in BANGS:  # bangs cast a soft shadow on the forehead
            rl, rr, tx, ty = c
            pen.poly([(rl, 23), (rr, 23), (rr - 0.6, (26 + ty) / 2 + 0.5), (tx, ty - 0.5), (rl + 0.6, (26 + ty) / 2 + 0.5)],
                     fill=SKIN_SHADE)
        # soft cheek shade along the jaw and blush
        pen.line([(23, 60), (27, 64.8), (32, 68.6), (37, 71)], fill=SKIN_SHADE)
        pen.line(mirror([(23, 60), (27, 64.8), (32, 68.6), (37, 71)]), fill=SKIN_SHADE)
        for x in (CX - 13, CX + 13):
            pen.ellipse((x - 3.5, 57, x + 3.5, 59.5), fill=BLUSH)
            for k in range(3):
                pen.line([(x - 2 + k * 1.6, 57.6), (x - 2.8 + k * 1.6, 58.8)], fill=(236, 160, 164))
    features(pen, color)
    if step >= 3:
        hair_front(pen, step)


def main():
    sheet = Image.new("RGB", (PW * 2, PH * 2), BG)
    for k, (ox, oy) in enumerate(((0, 0), (PW, 0), (0, PH), (PW, PH)), start=1):
        panel(sheet, ox, oy, k)
    sheet.save("face_study_raw.png")
    big = sheet.resize((PW * 2 * SCALE, PH * 2 * SCALE), Image.NEAREST)

    bar = 80
    out = Image.new("RGB", (big.width, big.height + bar), BG)
    out.paste(big, (0, bar))
    d = ImageDraw.Draw(out)
    d.text((24, 22), "Anime face construction - pixel study", fill=LINE, font=ImageFont.load_default(size=36))
    small = ImageFont.load_default(size=26)
    labels = ["1  Guides: circle, chin lines, eye line",
              "2  Features: eyes, nose, mouth, ears",
              "3  Hair: crown, bangs, side locks",
              "4  Color + shading"]
    for k, (ox, oy) in enumerate(((0, 0), (PW, 0), (0, PH), (PW, PH))):
        d.text((ox * SCALE + 14, bar + oy * SCALE + 10), labels[k], fill=(170, 90, 100), font=small)
    d.line((PW * SCALE, bar, PW * SCALE, out.height), fill=(210, 205, 210), width=2)
    d.line((0, bar + PH * SCALE, out.width, bar + PH * SCALE), fill=(210, 205, 210), width=2)
    out.save("face_study.png")
    print("saved", out.size)


main()
