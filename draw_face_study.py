"""Pixel-art anime face construction study in 4 steps (2x2 sheet), hi-res version.

Geometry is laid out on an 80x100 grid and multiplied by K, so the art has K times
more pixels in each direction for finer lines, eyes and hair detail.

Proportions follow the classic anime head guide:
- cranium circle, jaw tapering to a soft rounded chin
- eye line at half the head height, nose and mouth splitting the lower half
- ears between eye line and nose, short slim neck
"""
import math
import random
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
HAIR_HI = (156, 136, 184)
HAIR_SHINE = (214, 202, 232)
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

def smooth(ctrl, n=8, closed=True):
    """Catmull-Rom spline through control points -> dense smooth outline."""
    pts = list(ctrl)
    m = len(pts)
    out = []
    rng = range(m) if closed else range(m - 1)
    for i in rng:
        p0 = pts[(i - 1) % m] if closed or i > 0 else pts[0]
        p1 = pts[i]
        p2 = pts[(i + 1) % m]
        p3 = pts[(i + 2) % m] if closed or i + 2 < m else pts[-1]
        for k in range(n):
            t = k / n
            t2, t3 = t * t, t * t * t
            out.append(tuple(0.5 * ((2 * p1[j]) + (-p0[j] + p2[j]) * t
                                    + (2 * p0[j] - 5 * p1[j] + 4 * p2[j] - p3[j]) * t2
                                    + (-p0[j] + 3 * p1[j] - 3 * p2[j] + p3[j]) * t3) for j in range(2)))
    if not closed:
        out.append(pts[-1])
    return out


# Long back hair: smooth, full volume, ending in a few big soft clumps
BACK_HAIR = smooth([(40, 13), (51, 14.5), (59, 19), (63.5, 27), (65, 38), (67, 50), (70, 64), (73, 78),
                    (76, 92), (70, 88), (67, 97), (61, 89), (56, 95), (52, 86), (40, 86),
                    (28, 86), (24, 95), (19, 89), (13, 97), (10, 88), (4, 92),
                    (7, 78), (10, 64), (13, 50), (15, 38), (16.5, 27), (21, 19), (29, 14.5)])
# Volume crown: smooth dome a bit wider than the skull
# hair hugging the cranium circle (centre 40,36 r 21) with ~2.5px of thickness
CROWN = [(40 + 23.5 * math.cos(math.radians(a)), 35.5 + 23 * math.sin(math.radians(a)))
         for a in range(180, 361, 4)] + [(63.5, 34), (40, 31), (16.5, 34)]
# Bangs: soft curved clumps (root_l, root_r, tip_x, tip_y), part slightly right of centre
BANGS = [(20.5, 25, 20, 47), (22, 30, 25, 43), (27, 35, 30.5, 46), (32, 39, 35, 42.5),
         (36, 43, 39.5, 46.5), (41, 47, 44, 43), (45, 52, 48.5, 46), (49.5, 56, 53, 42.5),
         (54, 60.5, 56.5, 46), (57.5, 59.5, 61, 47.5)]
# Side locks: gentle S-curve that frames the face and flips softly at the end
SIDE_L = smooth([(16, 28), (22, 31), (24.5, 42), (23.5, 54), (21.5, 66), (22.5, 76), (20, 84),
                 (17.5, 77), (16.5, 66), (17.5, 54), (16, 42)])
SIDE_R = mirror(SIDE_L)


class Pen:
    """Draws on the hi-res panel using 80x100 grid coordinates."""

    def __init__(self, d, ox, oy, img=None):
        self.d, self.ox, self.oy, self.img = d, ox, oy, img

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
    """A curved, tapering clump of bangs (bulges slightly toward its tip side)."""
    lean = (tx - (rl + rr) / 2) * 0.35
    left = smooth([(rl, 22), (rl + 0.5 + lean, (22 + ty) / 2), (tx, ty)], n=6, closed=False)
    right = smooth([(tx, ty), (rr - 0.3 + lean, (22 + ty) / 2), (rr, 22)], n=6, closed=False)
    return left + right[1:]


def eye(pen, cx, flip, color):
    """Hi-res anime eye. cx is the eye centre on the 80-grid."""
    s = -1 if flip else 1                      # +1: outer corner on the left
    X = lambda dx: round(pen.ox + (cx * K) + s * dx)
    Y = lambda dy: round(pen.oy + (EYE_Y + 1) * K + dy)
    d = pen.d
    w = 11                                      # half-width in hi-res pixels
    h_top, h_bot = 0, 17
    # eye opening shape: flat-ish top, rounded bottom
    def inside(dx, dy):
        if dy < 1 or dy > h_bot:
            return False
        t = dy / h_bot
        half = w * math.sqrt(max(0.0, 1 - max(0.0, (t - 0.45) / 0.55) ** 2))
        return -w + (1 if dy < 3 else 0) <= dx <= half
    icx, icy, irx, iry = 1, 9, 6.5, 9           # iris ellipse
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
                    if ((dx - icx) / 3.1) ** 2 + ((dy - 7.5) / 4.6) ** 2 <= 1:
                        c = PUPIL
                    if dy <= 4:
                        c = IRIS[0]                             # lid shadow
            else:
                c = BG
                if in_iris and ((dx - icx) / irx) ** 2 + ((dy - icy) / iry) ** 2 > 0.72:
                    c = LINE
                if ((dx - icx) / 3.1) ** 2 + ((dy - 7.5) / 4.6) ** 2 <= 1:
                    c = LINE
            d.point((X(dx), Y(dy)), fill=c)
    # highlights
    hl = WHITE
    for dx, dy in ((-2, 4), (-3, 4), (-2, 5), (-3, 5), (-2, 6), (-3, 6), (-1, 5)):
        d.point((X(dx), Y(dy)), fill=hl)
    if color:
        for dx, dy in ((3, 13), (4, 13), (3, 14)):
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
        if pen.img.getpixel((X(dx), Y(-4 - arch))) in (SKIN, SKIN_SHADE, BG):
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
    eye(pen, CX - 10.5, False, color)
    eye(pen, CX + 10.5, True, color)
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


def hair_front(pen, step):
    color = step == 4
    hf = HAIR if color else BG
    pen.poly(SIDE_L, fill=hf, outline=LINE)
    pen.poly(SIDE_R, fill=hf, outline=LINE)
    pen.poly(CROWN, fill=hf, outline=LINE)
    pen.poly([(20.5, 25), (59.5, 25), (63, 32), (17, 32)], fill=hf)
    for c in BANGS:
        pen.poly(clump(*c), fill=hf)
    if color:
        # each clump: darker core toward the tip, lighter rim on one side = rounded volume
        for rl, rr, tx, ty in BANGS:
            cx = (rl + rr) / 2
            y0 = (24 + ty) / 2
            pen.poly(smooth([(cx - 0.6 + (tx - cx) * 0.4, y0), (cx + (tx - cx) * 0.7, (y0 + ty) / 2 + 1), (tx, ty - 1.2)],
                            n=5, closed=False) + [(cx + 1 + (tx - cx) * 0.4, y0)], fill=HAIR_SHADE)
            pen.line(smooth([(rl + 1.4, 23.5), (rl + 1.8 + (tx - cx) * 0.3, (23 + ty) / 2), (tx - 0.4, ty - 3)],
                            n=5, closed=False), fill=HAIR_MID)
        # big soft volume shadows where the crown curves away
        # side locks: shaded inner half, highlight curve following the S
        pen.line(smooth([(21, 36), (22.5, 48), (21, 62), (20.5, 74)], n=6, closed=False), fill=HAIR_SHADE)
        pen.line(smooth([(18, 40), (19.2, 52), (18.6, 64), (19.2, 76)], n=6, closed=False), fill=HAIR_MID)
        pen.line(mirror(smooth([(21, 36), (22.5, 48), (21, 62), (20.5, 74)], n=6, closed=False)), fill=HAIR_SHADE)
        pen.line(mirror(smooth([(18, 40), (19.2, 52), (18.6, 64), (19.2, 76)], n=6, closed=False)), fill=HAIR_MID)
    for c in BANGS:
        pts = clump(*c)
        pen.line(pts[len(pts) // 3:], fill=LINE)
    # strand flow from the part, curving with the dome
    flow = HAIR_SHADE if color else GUIDE_FAINT
    for x in (21, 27, 33, 54, 59, 62):
        side = -1 if x < 44 else 1
        mid = (44 + side * 6 + x) / 2
        pen.line(smooth([(44 + side * 0.5, 14), (mid, 16 + abs(x - 44) * 0.25), (x, 29)], n=6, closed=False), fill=flow)
    pen.line([(44, 12.5), (44, 19)], fill=LINE)          # the part
    pen.line(smooth([(43, 13), (42, 10), (43.5, 8), (46.5, 8.8)], n=5, closed=False), fill=LINE)   # ahoge


HAIR_RAMP = [HAIR_DEEP, HAIR_SHADE, HAIR, HAIR_MID, HAIR_HI, HAIR_SHINE]
STRAND_LINES = (HAIR_DEEP, HAIR_SHADE, LINE)


def hair_light(pen, img):
    """Hair shading in the style of the 'add glow' technique:
    1. light at the top, darker toward the ends (banded gradient)
    2. a glow band across the hair...
    3. ...cut by the strand lines ("eraser")...
    4. ...and smudged into streaks along the strands ("blur + finger tip")."""
    ox, oy = pen.ox, pen.oy
    W, H = PW, PH
    px = img.load()
    hair = {c: i for i, c in enumerate(HAIR_RAMP)}

    # 1. gradient: shift each hair pixel along the ramp by height
    for y in range(H):
        gy = y / K
        for x in range(W):
            checker = (x + y) % 2 == 0
            if gy < 15 or (gy < 18 and checker):
                shift = 1
            elif gy < 60 or (gy < 66 and checker):
                shift = 0
            else:
                shift = -1
            if shift == 0:
                continue
            c = px[ox + x, oy + y]
            if c in hair and c != HAIR_DEEP:
                px[ox + x, oy + y] = HAIR_RAMP[max(0, min(len(HAIR_RAMP) - 2, hair[c] + shift))]

    rng = random.Random(7)

    def band(center_y, x_range, half=2.5):
        """Glow band whose centre follows center_y(gx); streaks up and down per column."""
        for x in range(*x_range):
            gx = x / K
            cy = center_y(gx) * K
            up = rng.choice((2, 3, 3, 4, 5, 6))
            down = rng.choice((3, 4, 6, 7, 9, 11))
            if rng.random() < 0.18:          # some columns barely shine
                up, down = 1, 2
            for y in range(int(cy - half - up), int(cy + half + down) + 1):
                if not (0 <= y < H):
                    continue
                c = px[ox + x, oy + y]
                if c not in hair or c in STRAND_LINES:
                    continue                 # strand lines cut the glow
                dy = y - cy
                if abs(dy) <= half * 0.5:
                    nc = HAIR_SHINE
                elif abs(dy) <= half:
                    nc = HAIR_HI
                elif (dy < 0 and -dy - half <= up * 0.5) or (dy > 0 and dy - half <= down * 0.45):
                    nc = HAIR_HI
                else:
                    nc = HAIR_MID
                if hair[nc] > hair[c]:
                    px[ox + x, oy + y] = nc

    # glow ring around the crown, curving down toward the sides of the head
    band(lambda gx: 23 + ((gx - 40) / 23) ** 2 * 7, (int(17 * K), int(63 * K)), half=3)
    # softer second glow on the long hair lengths
    band(lambda gx: 54 + ((gx - 40) / 30) ** 2 * 3, (int(4 * K), int(76 * K)), half=1.5)


def panel(img, ox, oy, step):
    pen = Pen(ImageDraw.Draw(img), ox, oy, img)
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
        strands = ([(15, 40), (13.5, 56), (11.5, 72), (9.5, 88)], [(17.5, 46), (16, 62), (15, 78), (14, 94)],
                   [(13, 60), (11, 75), (8.5, 88)], [(19, 62), (19, 78), (20, 92)])
        if color:
            pen.poly(smooth([(22, 56), (58, 56), (55, 86), (25, 86)], n=4), fill=HAIR_DEEP)
            for k, pts in enumerate(strands):
                c = HAIR_MID if k == 1 else HAIR
                pen.line(smooth(pts, n=6, closed=False), fill=c)
                pen.line(mirror(smooth(pts, n=6, closed=False)), fill=c)
        else:
            for pts in strands[:2]:
                pen.line(smooth(pts, n=6, closed=False), fill=GUIDE_FAINT)
                pen.line(mirror(smooth(pts, n=6, closed=False)), fill=GUIDE_FAINT)
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
            pen.poly([(x, y + 1.5) for x, y in clump(*c)], fill=SKIN_SHADE)
        # soft cheek shade along the jaw and blush
        pen.line([(23, 60), (27, 64.8), (32, 68.6), (37, 71)], fill=SKIN_SHADE)
        pen.line(mirror([(23, 60), (27, 64.8), (32, 68.6), (37, 71)]), fill=SKIN_SHADE)
        for x in (CX - 13, CX + 13):
            pen.ellipse((x - 3.5, 57, x + 3.5, 59.5), fill=BLUSH)
            for k in range(3):
                pen.line([(x - 2 + k * 1.6, 57.6), (x - 2.8 + k * 1.6, 58.8)], fill=(236, 160, 164))
    if step >= 3:
        hair_front(pen, step)
        if color:
            hair_light(pen, img)
    features(pen, color)       # eyes and brows sit on top of the bangs


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
        x, y = ox * SCALE + 14, bar + oy * SCALE + 10
        l, t, r, bt = d.textbbox((x, y), labels[k], font=small)
        d.rectangle((l - 6, t - 4, r + 6, bt + 4), fill=BG)
        d.text((x, y), labels[k], fill=(170, 90, 100), font=small)
    d.line((PW * SCALE, bar, PW * SCALE, out.height), fill=(210, 205, 210), width=2)
    d.line((0, bar + PH * SCALE, out.width, bar + PH * SCALE), fill=(210, 205, 210), width=2)
    out.save("face_study.png")
    print("saved", out.size)


main()
