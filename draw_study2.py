"""Anime face construction study, rebuilt carefully step by step.

Canvas 400 x 500 (1x), drawn with coverage anti-aliasing. Proportions follow the manga book:
  - cranium circle (centre (200,190), r 105); two lines from the cheeks meeting at a soft chin
  - eye line at the vertical middle of the head, eyes just below it, one eye-width apart
  - nose halfway between eyes and chin, mouth halfway between nose and chin
  - ears from the eye line to the nose; short neck a little narrower than the jaw
Hair: a volume ring above the skull, a fringe of overlapping clumps with soft pointed tips,
side locks framing the face, long back hair. One light (upper left), two-tone cel shading,
an angel ring broken by the strands.

STAGE selects what is drawn: 1 guides, 2 features, 3 hair line art, 4 colour + shading.
"""
import math
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 400, 500
SS = 4
STAGE = 4

# ------------------------------------------------------------------ palette
PAPER = (250, 248, 244)
GUIDE_RED = (236, 150, 150)
GUIDE_BLUE = (140, 180, 230)
GUIDE_FAINT = (222, 220, 226)
LINE = (62, 42, 56)
LINE_SOFT = (150, 110, 118)
SKIN = (254, 237, 228)
SKIN_SH = (238, 206, 202)
SKIN_SH2 = (220, 170, 170)
BLUSH = (250, 196, 198)
BLUSH_LINE = (236, 160, 166)
LIP = (206, 118, 122)
LIP_HI = (255, 226, 220)
NOSE = (196, 142, 142)
HAIR = (86, 68, 112)
HAIR_SH = (58, 44, 82)
HAIR_DEEP = (38, 28, 56)
HAIR_MID = (118, 98, 150)
HAIR_HI = (158, 138, 192)
HAIR_SHINE = (210, 196, 232)
HAIR_LINE = (30, 20, 40)
BROW = (54, 40, 66)
SCLERA = (252, 251, 253)
SCLERA_SH = (210, 208, 224)
IRIS = [(26, 40, 92), (34, 62, 128), (46, 94, 168), (70, 130, 200), (104, 170, 224), (150, 208, 240), (190, 232, 246)]
PUPIL = (14, 18, 40)
LASH = (40, 28, 42)
LASH_LOW = (150, 104, 110)
IRIS_GLINT = (236, 246, 252)
TOP = (56, 78, 98)
TOP_SH = (40, 58, 76)
TOP_HI = (78, 104, 126)
TOP_LINE = (28, 36, 48)

CX = 200


def mix(a, b, t):
    return tuple(a[i] * (1 - t) + b[i] * t for i in range(3))


def mirror(pts):
    return [(2 * CX - x, y) for x, y in pts]


def smooth(ctrl, n=8, closed=True):
    pts = list(ctrl)
    m = len(pts)
    if m < 3:
        return pts
    out = []
    for i in (range(m) if closed else range(m - 1)):
        p0 = pts[(i - 1) % m] if closed or i > 0 else pts[0]
        p1, p2 = pts[i], pts[(i + 1) % m]
        p3 = pts[(i + 2) % m] if closed or i + 2 < m else pts[-1]
        for k in range(n):
            t = k / n
            t2, t3 = t * t, t * t * t
            out.append(tuple(0.5 * (2 * p1[j] + (-p0[j] + p2[j]) * t + (2 * p0[j] - 5 * p1[j] + 4 * p2[j] - p3[j]) * t2
                                    + (-p0[j] + 3 * p1[j] - 3 * p2[j] + p3[j]) * t3) for j in range(2)))
    if not closed:
        out.append(pts[-1])
    return out


def band(pts, widths, side=1):
    n = len(pts)
    ws = widths if isinstance(widths, (list, tuple)) else [widths] * n
    if len(ws) != n:
        ws = [ws[0] + (ws[-1] - ws[0]) * i / max(1, n - 1) for i in range(n)]
    off = []
    for i, (x, y) in enumerate(pts):
        xa, ya = pts[max(0, i - 1)]
        xb, yb = pts[min(n - 1, i + 1)]
        tx, ty = xb - xa, yb - ya
        L = math.hypot(tx, ty) or 1.0
        off.append((x - ty / L * ws[i] * side, y + tx / L * ws[i] * side))
    return list(pts) + off[::-1]


def strip(centre, w0, w1, pw=1.0):
    """Closed strip around a centre line, w0 wide at the start and w1 at the end."""
    n = len(centre)
    ws = [w0 + (w1 - w0) * (i / (n - 1)) ** pw for i in range(n)]
    up = band(centre, [w / 2 for w in ws], 1)[n:][::-1]
    dn = band(centre, [w / 2 for w in ws], -1)[n:][::-1]
    return up + dn[::-1]


def ellipse(cx, cy, rx, ry, n=36):
    return [(cx + rx * math.cos(2 * math.pi * k / n), cy + ry * math.sin(2 * math.pi * k / n)) for k in range(n)]


# ------------------------------------------------------------------ canvas
img = np.zeros((H, W, 3), np.float32)


def raster(pts, width=None, closed=True):
    out = np.zeros((H, W), np.float32)
    if len(pts) < 2:
        return out
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    pad = (width or 0) + 2
    x0, x1 = max(0, int(min(xs) - pad)), min(W, int(max(xs) + pad) + 1)
    y0, y1 = max(0, int(min(ys) - pad)), min(H, int(max(ys) + pad) + 1)
    if x1 <= x0 or y1 <= y0:
        return out
    bw, bh = x1 - x0, y1 - y0
    big = Image.new("L", (bw * SS, bh * SS), 0)
    d = ImageDraw.Draw(big)
    q = [((x - x0) * SS, (y - y0) * SS) for x, y in pts]
    if width is None:
        if len(q) >= 3:
            d.polygon(q, fill=255)
    else:
        if closed:
            q = q + q[:1]
        d.line(q, fill=255, width=max(1, round(width * SS)), joint="curve")
    out[y0:y1, x0:x1] = np.asarray(big.resize((bw, bh), Image.BOX), np.float32) / 255
    return out


def colour(c):
    c = np.asarray(c, np.float32)
    return np.broadcast_to(c, img.shape) if c.ndim == 1 else c


def paint(a, c):
    col = colour(c)
    full = a >= 0.55
    mid = (a >= 0.36) & ~full
    low = (a >= 0.18) & (a < 0.36)
    img[full] = col[full]
    img[mid] = img[mid] * 0.4 + col[mid] * 0.6
    img[low] = img[low] * 0.68 + col[low] * 0.32


def fill(pts, c, clip=None, thr=0.5):
    m = raster(pts) >= thr
    if clip is not None:
        m &= clip
    img[m] = colour(c)[m]
    return m


def stroke(pts, c, width=1.0, closed=False, clip=None):
    a = raster(pts, width, closed)
    if clip is not None:
        a = a * clip
    paint(a, c)


def shift(M, dx, dy):
    out = np.zeros_like(M)
    h, w = M.shape
    ys, yd = slice(max(0, dy), min(h, h + dy)), slice(max(0, -dy), min(h, h - dy))
    xs, xd = slice(max(0, dx), min(w, w + dx)), slice(max(0, -dx), min(w, w - dx))
    out[ys, xs] = M[yd, xd]
    return out


def C(colour_value):
    """Flat colour in stage 4, paper in the line-art stages."""
    return colour_value if STAGE >= 4 else PAPER


# ------------------------------------------------------------------ design (canvas px)
HEAD_C, HEAD_R = (200, 190), 105
TOP_Y, CHIN_Y = 85, 357
EYE_Y, NOSE_Y, MOUTH_Y = 228, 287, 320
FACE_L = [(200, 357), (182, 354), (163, 345), (143, 328), (124, 303), (108, 272), (98, 240), (95, 206), (98, 176),
          (107, 146), (123, 120), (146, 100), (172, 88), (200, 85)]
FACE = FACE_L[:-1] + mirror(FACE_L)[::-1][:-1]
JAW_L = FACE_L[:7]
EAR_L = [(98, 228), (88, 232), (83, 248), (86, 268), (96, 283), (105, 286)]
EAR_IN_L = [(94, 242), (91, 256), (95, 270)]
NECK = [(174, 330), (226, 330), (229, 430), (171, 430)]
NECKLINE = [(156, 404), (178, 416), (200, 420), (222, 416), (244, 404)]
SHOULDER_L = [(-10, 498), (22, 462), (72, 444), (128, 426), (156, 404)]

# hair
CROWN_C, CROWN_R = (200, 196), 120
SIDE_L = [(96, 200), (86, 236), (78, 290), (76, 350), (80, 400), (74, 440), (84, 464), (100, 442), (104, 400), (106, 350),
          (108, 296), (114, 240), (118, 204)]
BACK_L = [(80, 196), (66, 240), (58, 300), (54, 360), (58, 420), (48, 470), (66, 498), (84, 474), (100, 498), (118, 472),
          (138, 498), (160, 470), (184, 496), (200, 474)]
# fringe lower edge, right to left: notches (n) and soft tips (t), x then y
FRINGE = [("n", 308, 206), ("t", 286, 220), ("n", 266, 192), ("t", 248, 214), ("n", 230, 190), ("t", 215, 234),
          ("n", 197, 194), ("t", 181, 216), ("n", 164, 190), ("t", 148, 222), ("n", 130, 192), ("t", 112, 222),
          ("n", 92, 208)]


def fringe_edge():
    """Soft pointed tips: each side of a tip bows inward a little."""
    pts = []
    for i, (kind, x, y) in enumerate(FRINGE):
        if kind == "n":
            pts.append((x, y))
            continue
        nx0, ny0 = FRINGE[i - 1][1:]
        nx1, ny1 = FRINGE[i + 1][1:]
        pts += smooth([(nx0, ny0), (nx0 * 0.35 + x * 0.65, ny0 + (y - ny0) * 0.45), (x - 2.8, y - 9), (x, y)], 6, closed=False)[1:]
        pts += smooth([(x, y), (x + 2.8, y - 9), (nx1 * 0.35 + x * 0.65, ny1 + (y - ny1) * 0.45), (nx1, ny1)], 6, closed=False)[1:-1]
    return pts


def crown_arc(a0=182, a1=358, r=None):
    r0 = r or CROWN_R
    pts = []
    for a in range(a0, a1 + 1, 2):
        t = math.radians(a)
        rr = r0 + 2.2 * math.sin(math.radians((a - 182) * 2.6)) + 1.2 * math.sin(math.radians((a - 182) * 5.3 + 40))
        pts.append((CROWN_C[0] + rr * math.cos(t), CROWN_C[1] + rr * math.sin(t)))
    return pts


def front_mass():
    """The hair mass over the forehead: crown arc on top, the fringe edge below."""
    return crown_arc() + [(319, 204)] + fringe_edge() + [(81, 204)]


# ------------------------------------------------------------------ parts
def guides(faint):
    red = GUIDE_FAINT if faint else GUIDE_RED
    blue = GUIDE_FAINT if faint else GUIDE_BLUE
    stroke(ellipse(*HEAD_C, HEAD_R, HEAD_R, 72), blue, 1.0, closed=True)
    stroke([(98, 240), (200, 357), (302, 240)], blue, 1.0)
    stroke([(200, 40), (200, 490)], red, 1.0)
    for y in (EYE_Y, NOSE_Y, MOUTH_Y):
        stroke([(70, y), (330, y)], blue, 1.0)
    for y in (TOP_Y, CHIN_Y):
        stroke([(70, y), (330, y)], red, 1.0)


def eye(cx, cy, w, h, side, colour_on):
    """Big anime eye. side = +1 when the outer corner is on the viewer's right."""
    def X(u):
        return cx + side * u * w / 2
    cin, cout = 0.10 * h, -0.14 * h                     # inner corner a little lower than the outer

    def base(u):
        return cin + (cout - cin) * (u + 1) / 2

    def top(u):
        return base(u) - 0.72 * h * max(0.0, 1 - ((u - 0.08) / 1.08) ** 2) ** 0.55

    def bot(u):
        return base(u) + 0.44 * h * max(0.0, 1 - (u / 1.02) ** 2) ** 0.8
    us = np.linspace(-1, 1, 48)
    opening = [(X(u), cy + top(u)) for u in us] + [(X(u), cy + bot(u)) for u in us][::-1]
    M = raster(opening) >= 0.5
    if colour_on:
        img[M] = SCLERA
        lid = raster([(X(u), cy + top(u)) for u in us] + [(X(u), cy + top(u) + 0.13 * h) for u in us][::-1]) >= 0.5
        img[lid & M] = SCLERA_SH
    else:
        img[M] = PAPER
    # iris: tall ellipse, its top tucked under the lid
    icx, icy = cx + side * 0.02 * w, cy + 0.06 * h
    rx, ry = 0.33 * w, 0.60 * h
    ys, xs = np.nonzero(M)
    du = (xs + 0.5 - icx) / rx
    dv = (ys + 0.5 - icy) / ry
    e = du * du + dv * dv
    inside = e <= 1
    if colour_on:
        k = np.clip(((dv + 1) / 2 * len(IRIS)).astype(int), 0, len(IRIS) - 1)
        c = np.asarray(IRIS, np.float32)[k]
        c[e > 0.82] = IRIS[0]
        c[(ys + 0.5 - cy) < (np.array([top(u) for u in (xs + 0.5 - cx) / (side * w / 2)]) + 0.11 * h)] = IRIS[0]
        pupil = (du / 0.3) ** 2 + ((dv + 0.06) / 0.4) ** 2 <= 1
        c[pupil] = PUPIL
        img[ys[inside], xs[inside]] = c[inside]
        # highlights: light from the upper left, both eyes the same side
        big = ((xs + 0.5 - (icx - 0.13 * w)) / (0.12 * w)) ** 2 + ((ys + 0.5 - (icy - 0.24 * h)) / (0.11 * h)) ** 2 <= 1
        small = ((xs + 0.5 - (icx + 0.13 * w)) / (0.06 * w)) ** 2 + ((ys + 0.5 - (icy + 0.26 * h)) / (0.05 * h)) ** 2 <= 1
        img[ys[big & inside], xs[big & inside]] = (255, 255, 255)
        img[ys[small & inside], xs[small & inside]] = IRIS_GLINT
    else:
        rim = inside & (e > 0.86)
        img[ys[rim], xs[rim]] = LINE
        pupil = (du / 0.3) ** 2 + ((dv + 0.06) / 0.4) ** 2 <= 1
        img[ys[pupil], xs[pupil]] = LINE
        big = ((xs + 0.5 - (icx - 0.13 * w)) / (0.12 * w)) ** 2 + ((ys + 0.5 - (icy - 0.24 * h)) / (0.11 * h)) ** 2 <= 1
        img[ys[big & pupil], xs[big & pupil]] = PAPER
    # upper lash: thick band, heaviest toward the outer corner, tapering at the inner end, with a flick
    def th(u):
        t = 0.05 * h + 0.10 * h * min(1.0, max(0.0, (u + 0.5) / 1.3))
        if u < -0.6:
            t *= max(0.35, (u + 1.1) / 0.5)
        return t
    us2 = np.linspace(-1.06, 1.06, 56)
    lash = [(X(u), cy + top(u) - th(u)) for u in us2] + [(X(u), cy + top(u) + 0.05 * h) for u in us2][::-1]
    fill(lash, LASH, thr=0.45)
    fill([(X(0.94), cy + top(0.94) - th(0.94)), (X(1.24), cy + top(1.0) - 0.26 * h), (X(1.06), cy + top(1.0) + 0.02 * h)], LASH, thr=0.45)
    # lower lid: thin soft line, with two tiny lashes at the outer corner
    stroke([(X(u), cy + bot(u) + 0.07 * h) for u in np.linspace(-0.55, 0.98, 20)], LASH_LOW, 1.0)
    for u0, du0 in ((0.72, 0.1), (0.9, 0.12)):
        stroke([(X(u0), cy + bot(u0) + 0.06 * h), (X(u0 + du0), cy + bot(u0) + 0.2 * h)], LASH, 1.1)
    # crease
    stroke([(X(u), cy + top(u) - th(u) - 0.12 * h) for u in np.linspace(-0.5, 0.75, 16)], SKIN_SH2 if colour_on else LINE_SOFT, 0.9)


def brow(cx, side, colour_on):
    pts = [(cx + side * 38, 196), (cx + side * 18, 191), (cx - side * 6, 192), (cx - side * 26, 198)]
    centre = smooth(pts, 8, closed=False)
    fill(strip(centre[::-1], 1.2, 3.0, 0.8), BROW if colour_on else LINE)


def features(colour_on):
    skin = C(SKIN)
    # ears (behind the face)
    for ear, inner in ((EAR_L, EAR_IN_L), (mirror(EAR_L), mirror(EAR_IN_L))):
        ep = smooth(ear, 8, closed=False)
        fill(ep + [ear[-1], ear[0]], skin)
        stroke(ep, LINE, 1.3)
        stroke(smooth(inner, 6, closed=False), LINE_SOFT if colour_on else LINE, 1.0)
    face = smooth(FACE, 8)
    MF = fill(face, skin)
    stroke(face, LINE, 1.4, closed=True)
    if STAGE < 3:
        for side in (1, -1):
            eye(200 + side * 55, EYE_Y + 4, 58, 40, side, colour_on)
            brow(200 + side * 56, side, colour_on)
    stroke([(203, 282), (201, 290)], NOSE if colour_on else LINE, 1.3)
    if colour_on:
        fill(smooth([(196, 288), (203, 291), (199, 294)], 4), SKIN_SH2)
    stroke(smooth([(189, 318), (200, 322), (211, 318)], 8, closed=False), LIP if colour_on else LINE, 1.4)
    if colour_on:
        stroke(smooth([(194, 327), (200, 329), (206, 327)], 6, closed=False), LIP_HI, 1.1)
    return MF


def body(colour_on):
    neck = smooth(NECK, 4)
    MN = fill(neck, C(SKIN))
    stroke([NECK[0], NECK[3]], LINE, 1.3)
    stroke([NECK[1], NECK[2]], LINE, 1.3)
    top_edge = smooth(SHOULDER_L + NECKLINE[1:-1] + mirror(SHOULDER_L)[::-1], 8, closed=False)
    MT = fill(top_edge + [(W + 10, H + 10), (-10, H + 10)], C(TOP))
    stroke(top_edge, TOP_LINE if colour_on else LINE, 1.4)
    return MN, MT


def hair(colour_on, MF):
    hair_line = HAIR_LINE if colour_on else LINE
    back = smooth(BACK_L + mirror(BACK_L)[::-1], 8)
    MB = fill(back, C(HAIR_SH))
    if colour_on:
        for pts in ([(72, 250), (64, 330), (62, 420)], [(100, 300), (104, 400), (102, 470)]):
            stroke(smooth(pts, 6, closed=False), HAIR_DEEP, 1.0, clip=MB)
            stroke(smooth(mirror(pts), 6, closed=False), HAIR_DEEP, 1.0, clip=MB)
    stroke(back, hair_line, 1.4, closed=True)
    return MB


def front_hair(colour_on, MF):
    hf = C(HAIR)
    masks = np.zeros((H, W), bool)
    mass = front_mass()
    fringe = raster(mass) >= 0.5
    locks = [smooth(SIDE_L, 8), smooth(mirror(SIDE_L), 8)]
    if colour_on:
        sh = shift(fringe, 3, 7) & MF & ~fringe
        img[sh] = SKIN_SH
    for lk in locks:
        masks |= fill(lk, hf)
    masks |= fill(mass, hf)
    return masks, mass, locks


def shade_hair(masks, mass, locks):
    for x0 in (160, 222, 248):                           # strands falling from the crown
        stroke(smooth([(x0, 104), (x0 + (x0 - 200) * 0.05, 126), (x0 + (x0 - 200) * 0.1, 150)], 6, closed=False), HAIR_SH, 0.9, clip=masks)
    for xc in (132, 168, 196, 232, 268):
        stroke(smooth([(xc - 1, 112), (xc, 134), (xc + 1 + (xc - 200) * 0.04, 160)], 6, closed=False), HAIR_SH, 0.9, clip=masks)
    # tips of the fringe darker: a soft triangle rising from each tip
    for i, (kind, x, y) in enumerate(FRINGE):
        if kind != "t":
            continue
        nx0, nx1 = FRINGE[i - 1][1], FRINGE[i + 1][1]
        fill(smooth([(nx0 * 0.45 + x * 0.55, y - 22), (nx1 * 0.45 + x * 0.55, y - 22), (x + 1, y - 2), (x - 1, y - 2)], 4),
             mix(HAIR, HAIR_SH, 0.6), masks)
    # inner side of the locks darker, a highlight line on the outer side
    for lk, side in zip(locks, (1, -1)):
        inner = smooth([(200 - side * 86, 210), (200 - side * 92, 284), (200 - side * 94, 344), (200 - side * 92, 420)], 6, closed=False)
        fill(band(inner, 7, -side), HAIR_SH, masks)
        stroke(smooth([(200 - side * 103, 240), (200 - side * 106, 330), (200 - side * 101, 410)], 6, closed=False), HAIR_MID, 0.9, clip=masks)
    # angel ring: a thin glow that follows the crown, cut by the strand lines and smudged
    # up and down into streaks of different lengths (the glow / eraser / blur method)
    ring = smooth([(100, 164), (126, 140), (164, 126), (200, 122), (236, 126), (274, 140), (300, 164)], 8, closed=False)
    rx = np.array([p[0] for p in ring])
    ry = np.array([p[1] for p in ring])
    rng = random.Random(5)
    ramp = [HAIR_DEEP, HAIR_SH, HAIR, HAIR_MID, HAIR_HI, HAIR_SHINE]
    idx = {c: i for i, c in enumerate(ramp)}
    run = 0
    for x in range(102, 298):
        cy = float(np.interp(x, rx, ry))
        prof = 1.0 - 0.7 * abs((x - 200) / 98) ** 1.5
        if run <= 0:
            run = rng.choice((1, 2, 2, 3, 3, 4))
            up = rng.choice((1, 2, 2, 3, 4)) * prof
            down = rng.choice((2, 3, 4, 6, 8)) * prof
            if rng.random() < 0.2:
                up, down = 0.5, 1.0
        run -= 1
        half = (0.6 + 2.0 * prof)
        for y in range(int(cy - half - up), int(cy + half + down) + 1):
            if not (0 <= y < H) or not masks[y, x]:
                continue
            cur = tuple(int(v) for v in img[y, x])
            if cur not in idx or cur in (HAIR_SH, HAIR_DEEP):
                continue                      # strand lines cut the glow
            dy = y - cy
            if abs(dy) <= half * 0.45:
                nc = HAIR_SHINE
            elif abs(dy) <= half:
                nc = HAIR_HI
            elif (dy < 0 and -dy - half <= up * 0.5) or (dy > 0 and dy - half <= down * 0.45):
                nc = HAIR_HI
            else:
                nc = HAIR_MID
            if idx[nc] > idx[cur]:
                img[y, x] = nc


def hair_lines(colour_on, mass, locks):
    hair_line = HAIR_LINE if colour_on else LINE
    sep = HAIR_SH if colour_on else LINE
    stroke(mass, hair_line, 1.3, closed=True)
    below = (np.arange(H)[:, None] > 214).astype(np.float32) * np.ones((1, W), np.float32)
    for lk in locks:
        stroke(lk, hair_line, 1.3, closed=True, clip=below)
    # short separation lines rising from the notches between the tips
    for kind, x, y in FRINGE[1:-1]:
        if kind == "n":
            stroke(smooth([(x, y), (x + (x - 200) * 0.02, y - 18), (x + (x - 200) * 0.06, y - 40)], 6, closed=False), sep, 1.0)
    # strands falling from the crown
    for x0 in (158, 186, 222, 246):
        stroke(smooth([(x0, 104), (x0 + (x0 - 200) * 0.05, 126), (x0 + (x0 - 200) * 0.1, 150)], 6, closed=False), sep, 0.8)


def draw():
    img[:] = PAPER
    if STAGE == 1:
        guides(False)
        stroke(smooth(FACE, 8), GUIDE_BLUE, 1.0, closed=True)
        stroke(smooth(NECK, 4), GUIDE_BLUE, 1.0, closed=True)
        stroke(smooth(SHOULDER_L + NECKLINE[1:-1] + mirror(SHOULDER_L)[::-1], 8, closed=False), GUIDE_BLUE, 1.0)
        return
    colour_on = STAGE >= 4
    if not colour_on:
        guides(True)
    MB = hair(colour_on, None) if STAGE >= 3 else None
    MN, MT = body(colour_on)
    if colour_on:
        fill(band(smooth(NECKLINE, 6, closed=False), 5, 1), TOP_SH, MT)
        stroke(smooth(NECKLINE, 6, closed=False), TOP_HI, 1.2)
    MF = features(colour_on)
    if colour_on:
        # neck shadow under the chin, soft side shading on the face
        jaw = smooth(JAW_L[::-1] + mirror(JAW_L)[1:], 8, closed=False)
        fill(band(jaw, [8, 12, 15, 18, 15, 12, 8], 1), SKIN_SH2, MN)
        fill(band(jaw, 4, -1), SKIN_SH, MF)
        for side in (1, -1):
            fill(band(smooth([(200 - side * 97, 236), (200 - side * 104, 266), (200 - side * 117, 296), (200 - side * 134, 321)], 6, closed=False), 6, side), SKIN_SH, MF)
            bx = 200 - side * 50
            fill(ellipse(bx, 270, 15, 6), BLUSH, MF)
            for k in (-1, 0, 1):
                stroke([(bx + k * 7 - 2.5, 267), (bx + k * 7 + 2.5, 273)], BLUSH_LINE, 1.0, clip=MF)
    if STAGE >= 3:
        masks, mass, locks = front_hair(colour_on, MF)
        if colour_on:
            shade_hair(masks, mass, locks)
            for side in (1, -1):   # locks throw a soft shadow on the cheeks
                lm = raster(locks[0 if side == 1 else 1]) >= 0.5
                img[shift(lm, -side * 4, 3) & MF & ~masks] = SKIN_SH
        hair_lines(colour_on, mass, locks)
        # eyes sit on top of the fringe; brows show faintly where the fringe covers them
        for side in (1, -1):
            eye(200 + side * 55, EYE_Y + 4, 58, 40, side, colour_on)
            pts = [(200 + side * 56 + side * 38, 196), (200 + side * 56 + side * 18, 191), (200 + side * 56 - side * 6, 192), (200 + side * 56 - side * 26, 198)]
            m = raster(strip(smooth(pts, 8, closed=False)[::-1], 1.2, 3.0, 0.8)) >= 0.5
            col = np.asarray(BROW if colour_on else LINE, np.float32)
            img[m & ~masks] = col
            img[m & masks] = img[m & masks] * 0.5 + col * 0.5


def render(stage):
    global STAGE
    STAGE = stage
    draw()
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def sheet(scale=2):
    labels = ["1  Guides: circle, chin lines, eye line", "2  Features: eyes, nose, mouth, ears",
              "3  Hair: one mass, pointed fringe, locks", "4  Color + shading"]
    panels = [render(k) for k in (1, 2, 3, 4)]
    bar = 48 * scale
    out = Image.new("RGB", (W * 2 * scale, H * 2 * scale + bar), PAPER)
    d = ImageDraw.Draw(out)
    d.text((16 * scale, 12 * scale), "Anime face construction - pixel study", fill=LINE, font=ImageFont.load_default(size=20 * scale))
    small = ImageFont.load_default(size=13 * scale)
    for k, im in enumerate(panels):
        ox, oy = (k % 2) * W * scale, bar + (k // 2) * H * scale
        out.paste(im.resize((W * scale, H * scale), Image.NEAREST), (ox, oy))
        x, y = ox + 8 * scale, oy + 6 * scale
        l, t, r, b = d.textbbox((x, y), labels[k], font=small)
        d.rectangle((l - 4, t - 3, r + 4, b + 3), fill=PAPER)
        d.text((x, y), labels[k], fill=(170, 90, 100), font=small)
    d.line((W * scale, bar, W * scale, out.height), fill=(214, 208, 212), width=2)
    d.line((0, bar + H * scale, out.width, bar + H * scale), fill=(214, 208, 212), width=2)
    return out, panels[3]


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "sheet":
        out, final = sheet()
        out.save("face_study.png")
        final.save("face_study_portrait_raw.png")
        final.resize((W * 3, H * 3), Image.NEAREST).save("face_study_portrait.png")
        print("saved sheet", out.size)
    else:
        stage = int(sys.argv[1]) if len(sys.argv) > 1 else 4
        im = render(stage)
        im.save(f"study2_stage{stage}.png")
        print("saved stage", stage)
