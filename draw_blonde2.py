"""Blonde girl, light-brown eyes, chin lifted and looking up: drawn in 2D on the pose traced from
the reference study (head angle, feature placement, neck and collar), with my own character on it:
loose long blonde hair with swept bangs, no elf ears, plain collar and top.

Shapes are given in the reference's pixel grid (509 x 602) and mapped to the canvas by P().
Everything is painted with coverage-based anti-aliasing (raster) so the lines stay smooth.
"""
import math
import random
import numpy as np
from PIL import Image, ImageDraw

W, H = 340, 402
SCALE = 3
SS = 4
S = 509 / W                     # reference px per canvas px

# ----------------------------------------------------------------------------- palette
BG = (196, 222, 236)
BG2 = (214, 232, 240)
LINE = (142, 92, 82)            # brown line on skin
LINE_SOFT = (190, 140, 126)
SKIN = (254, 236, 226)
SKIN_SH = (240, 204, 198)
SKIN_SH2 = (222, 172, 172)
BLUSH = (248, 196, 196)
LIP = (210, 128, 128)
MOUTH_IN = (150, 80, 90)
NOSE = (206, 150, 140)
HAIR = (242, 210, 134)
HAIR_SH = (210, 166, 96)
HAIR_DEEP = (168, 122, 70)
HAIR_HI = (252, 238, 194)
HAIR_SHINE = (255, 251, 234)
HAIR_LINE = (176, 128, 70)
BROW = (162, 116, 70)
SCLERA = (252, 250, 250)
SCLERA_SH = (222, 216, 226)
IRIS = [(84, 50, 32), (112, 68, 38), (146, 94, 46), (178, 122, 58), (206, 150, 72), (226, 176, 92)]
PUPIL = (46, 28, 22)
LASH = (66, 40, 38)
LASH_LOW = (170, 116, 106)
COLLAR = (240, 234, 228)
COLLAR_SH = (208, 198, 202)
COLLAR_BAND = (58, 50, 62)
COLLAR_LINE = (120, 100, 106)
TOP = (72, 78, 108)
TOP_SH = (50, 54, 80)
TOP_HI = (98, 106, 138)
TOP_LINE = (36, 38, 56)


def P(x, y):
    return (x / S, y / S)


def PP(pts):
    return [P(x, y) for x, y in pts]


def mix(a, b, t):
    return tuple(a[i] * (1 - t) + b[i] * t for i in range(3))


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


def tapered(centre, w0, w1, pw=1.0):
    """Closed strip around a centre line, w0 wide at the start, w1 at the end (curved taper)."""
    n = len(centre)
    ws = [w0 + (w1 - w0) * (i / (n - 1)) ** pw for i in range(n)]
    up = band(centre, [w / 2 for w in ws], 1)[n:][::-1]
    dn = band(centre, [w / 2 for w in ws], -1)[n:][::-1]
    return up + dn[::-1]


# ----------------------------------------------------------------------------- canvas
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


# ----------------------------------------------------------------------------- shapes (reference px)
FACE = [(146, 150), (134, 196), (126, 238), (126, 274), (136, 302), (158, 324), (200, 341), (250, 347), (300, 344),
        (330, 323), (349, 302), (344, 262), (339, 216), (330, 170), (318, 136), (298, 110), (262, 96), (222, 94),
        (186, 104), (160, 124)]
JAW = [(158, 324), (200, 341), (250, 349), (300, 345), (330, 323), (349, 302)]
NECK = [(254, 340), (328, 326), (322, 404), (246, 406)]
COLLAR_TOP = [(232, 405), (258, 397), (286, 394), (312, 397), (336, 405)]
COLLAR_BOT = [(234, 444), (260, 438), (288, 436), (318, 439), (346, 450)]
TOP_EDGE = [(30, 602), (56, 546), (118, 504), (190, 468), (234, 444), (260, 438), (288, 436), (318, 439), (346, 450),
            (400, 478), (452, 516), (486, 562), (496, 602)]
# the whole hair silhouette (behind everything)
BACK_HAIR = [(120, 200), (100, 262), (96, 332), (106, 420), (98, 520), (108, 602), (452, 602), (466, 520), (480, 420),
             (480, 330), (464, 250), (452, 190), (444, 150), (424, 114), (394, 86), (350, 60), (300, 50), (250, 50),
             (200, 64), (160, 94), (130, 134)]
# hair masses in front (drawn in this order). Each: closed outline, smooth.
SWEEP_L = [(224, 72), (196, 82), (166, 106), (144, 142), (132, 182), (128, 214), (142, 212), (154, 184), (166, 150),
           (184, 122), (206, 104), (226, 94)]
SWEEP_L2 = [(216, 92), (204, 118), (196, 148), (182, 170), (176, 168), (184, 142), (192, 112), (206, 94)]
SWEEP_R = [(230, 72), (282, 78), (334, 104), (364, 150), (372, 200), (362, 250), (340, 296), (322, 306), (334, 262),
           (342, 218), (332, 170), (306, 136), (272, 108), (240, 92)]
SWEEP_R2 = [(244, 68), (300, 66), (356, 80), (402, 112), (436, 160), (448, 204), (428, 196), (404, 150), (370, 118),
            (330, 94), (286, 86), (250, 84)]
LOCK_R = [(352, 222), (376, 300), (392, 390), (398, 470), (386, 540), (378, 590), (404, 560), (420, 470), (428, 380),
          (420, 300), (404, 236), (388, 190)]
LOCK_R2 = [(344, 300), (356, 380), (360, 440), (352, 500), (368, 470), (372, 400), (364, 330)]
LOCK_L = [(128, 210), (112, 280), (108, 360), (116, 440), (104, 520), (112, 580), (134, 560), (146, 470), (158, 380),
          (160, 300), (152, 240), (146, 208)]
EAR_L = [(160, 304), (148, 306), (140, 320), (146, 336), (158, 338), (164, 326)]
EAR_R = [(346, 290), (360, 284), (370, 296), (368, 314), (356, 324), (346, 316)]
STRAYS = [[(300, 112), (318, 160), (322, 220), (318, 282)], [(150, 156), (140, 200), (136, 246)],
          [(236, 84), (246, 112), (250, 136)]]


def eye(outer, inner, h, up):
    """Almond eye between two corners; h = height (ref px); up = screen direction of 'up' for the
    face. The iris sits tucked under the upper lid (she is looking up); light from the upper left."""
    ox, oy = P(*outer)
    ix, iy = P(*inner)
    h = h / S
    cx, cy = (ox + ix) / 2, (oy + iy) / 2
    ux, uy = ix - ox, iy - oy
    L = math.hypot(ux, uy)
    ux, uy = ux / L, uy / L
    vx, vy = -uy, ux
    if vx * up[0] + vy * up[1] > 0:
        vx, vy = -vx, -vy
    half = L / 2

    def pt(t, v):
        return (cx + ux * t * half + vx * v, cy + uy * t * half + vy * v)

    def topv(t):
        return -0.58 * h * max(0.0, 1 - t * t) ** 0.65 - 0.06 * h * (t + 1)

    def botv(t):
        return 0.42 * h * max(0.0, 1 - t * t) ** 0.9
    ts = np.linspace(-1, 1, 40)
    opening = [pt(t, topv(t)) for t in ts] + [pt(t, botv(t)) for t in ts][::-1]
    M = raster(opening) >= 0.5
    img[M] = SCLERA
    lidsh = raster([pt(t, topv(t)) for t in ts] + [pt(t, topv(t) + 0.16 * h) for t in ts][::-1]) >= 0.5
    img[lidsh & M] = SCLERA_SH
    icx, icy = pt(-0.1, -0.1 * h)
    rx, ry = 0.36 * L, 0.6 * h
    ys, xs = np.nonzero(M)
    dx, dy = xs + 0.5 - icx, ys + 0.5 - icy
    du = (dx * ux + dy * uy) / rx
    dv = (dx * vx + dy * vy) / ry
    e = du * du + dv * dv
    k = np.clip(((dv + 1) / 2 * len(IRIS)).astype(int), 0, len(IRIS) - 1)
    c = np.asarray(IRIS, np.float32)[k]
    c[e > 0.8] = IRIS[0]
    pupil = (du / 0.4) ** 2 + ((dv + 0.05) / 0.48) ** 2 <= 1
    c[pupil] = PUPIL
    sel = e <= 1
    img[ys[sel], xs[sel]] = c[sel]
    for (du0, dv0, r0, r1, col) in ((-0.42, -0.46, 0.2, 0.22, (255, 255, 255)), (0.42, 0.55, 0.13, 0.1, (250, 232, 200))):
        hm = (((du - du0) / r0) ** 2 + ((dv - dv0) / r1) ** 2 <= 1) & sel
        img[ys[hm], xs[hm]] = col
    th = lambda t: (0.12 + 0.17 * float(np.clip((0.3 - t) / 1.3, 0, 1))) * h * (1 if t < 0.7 else max(0.35, (1.05 - t) / 0.35))
    ts2 = np.linspace(-1.08, 1.02, 44)
    lash = [pt(t, topv(t) - th(t)) for t in ts2] + [pt(t, topv(t) + 0.05 * h) for t in ts2][::-1]
    fill(lash, LASH, thr=0.45)
    fill([pt(-0.9, -0.32 * h), pt(-1.42, -0.62 * h), pt(-1.05, -0.12 * h)], LASH, thr=0.45)
    stroke([pt(t, botv(t) + 0.08 * h) for t in np.linspace(-0.9, 0.92, 20)], LASH_LOW, 0.9)
    stroke([pt(t, botv(t) + 0.08 * h) for t in np.linspace(-0.95, -0.55, 8)], LASH, 1.0)
    stroke([pt(t, topv(t) - th(t) - 0.14 * h) for t in np.linspace(-0.55, 0.6, 14)], SKIN_SH2, 0.8)


def hair_mass(ctrl, shade_side=None, line=True):
    poly = smooth(PP(ctrl), 7)
    m = fill(poly, HAIR)
    if line:
        stroke(poly, HAIR_LINE, 1.0, closed=True)
    return m, poly


def draw():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    g = np.clip(1 - np.hypot(xx - 0.2 * W, yy - 0.15 * H) / (0.9 * W), 0, 1)[..., None]
    img[:] = np.asarray(BG, np.float32) * (1 - 0.6 * g) + np.asarray(BG2, np.float32) * 0.6 * g

    # back hair: one big smooth shape, a little darker (it is behind everything)
    back = smooth(PP(BACK_HAIR), 6)
    MB = fill(back, HAIR_SH)
    for pts in ([(140, 240), (128, 330), (124, 430), (116, 540)], [(446, 220), (462, 330), (466, 440), (458, 560)],
                [(170, 300), (160, 400), (150, 520)], [(420, 300), (430, 400), (436, 520)]):
        stroke(smooth(PP(pts), 6, closed=False), HAIR_DEEP, 1.0, clip=MB)
    stroke(back, HAIR_LINE, 1.1, closed=True)

    # neck, top, collar
    neck = smooth(PP(NECK), 4)
    MN = fill(neck, SKIN)
    topm = fill(smooth(PP(TOP_EDGE), 6) + [(W + 5, H + 5), (-5, H + 5)], TOP)
    fill(band(smooth(PP(TOP_EDGE[:4]), 6, closed=False), 16 / S, 1), TOP_HI, topm)
    fill(band(smooth(PP(TOP_EDGE[8:]), 6, closed=False), 14 / S, 1), TOP_SH, topm)
    collar = smooth(PP(COLLAR_TOP), 6, closed=False) + smooth(PP(COLLAR_BOT), 6, closed=False)[::-1]
    MC = fill(collar, COLLAR)
    fill(band(smooth(PP(COLLAR_BOT), 6, closed=False), 7 / S, -1), COLLAR_SH, MC)
    fill(band(smooth(PP(COLLAR_TOP), 6, closed=False), 5 / S, 1), COLLAR_BAND, MC)
    fill(band(smooth(PP(COLLAR_BOT), 6, closed=False), 9 / S, 1), TOP_SH, topm)
    stroke(smooth(PP(COLLAR_TOP), 6, closed=False), COLLAR_LINE, 1.0)
    stroke(smooth(PP(COLLAR_BOT), 6, closed=False), TOP_LINE, 1.0)
    stroke(smooth(PP(TOP_EDGE[:5]), 6, closed=False), TOP_LINE, 1.1)
    stroke(smooth(PP(TOP_EDGE[8:]), 6, closed=False), TOP_LINE, 1.1)
    stroke(PP([NECK[0], NECK[3]]), LINE_SOFT, 1.0)
    stroke(PP([NECK[1], NECK[2]]), LINE_SOFT, 1.0)
    # soft shadow on the neck under the jaw
    jaw = smooth(PP(JAW), 6, closed=False)
    for wdt, col in ((38, SKIN_SH), (27, mix(SKIN_SH, SKIN_SH2, 0.5)), (16, SKIN_SH2)):
        fill(band(jaw, [wdt * 0.7 / S, wdt / S, wdt * 1.05 / S, wdt / S, wdt * 0.75 / S, wdt * 0.5 / S], 1), col, MN)

    # face
    face = smooth(PP(FACE), 6)
    MF = fill(face, SKIN)
    fill(band(smooth(PP([(349, 302), (344, 262), (339, 216), (330, 170)]), 6, closed=False), 9 / S, 1), SKIN_SH, MF)
    stroke(smooth(PP(FACE[:6]), 6, closed=False), LINE, 1.1)
    stroke(smooth(PP(FACE[9:16]), 6, closed=False), LINE, 1.1)
    stroke(jaw, LINE_SOFT, 0.95)
    fill(band(jaw, [8 / S, 12 / S, 14 / S, 12 / S, 8 / S, 5 / S], -1), SKIN_SH, MF)   # jaw turning under

    for ear, inner in ((EAR_L, [(155, 312), (147, 320), (151, 332)]), (EAR_R, [(354, 294), (362, 302), (358, 316)])):
        ep = smooth(PP(ear), 6)
        me = fill(ep, SKIN)
        fill(band(smooth(PP(inner), 6, closed=False), 4 / S, 1), SKIN_SH, me)
        stroke(smooth(PP(inner), 6, closed=False), LINE_SOFT, 0.9)
        stroke(ep, LINE_SOFT, 1.0, closed=True)
    # hair in front: temple sweeps, the strand across the near cheek, long locks
    masses = [SWEEP_R2, LOCK_R, LOCK_R2, LOCK_L, SWEEP_L, SWEEP_L2, SWEEP_R]
    hair_all = np.zeros((H, W), bool)
    polys = [smooth(PP(c), 7) for c in masses]
    front = np.zeros((H, W), bool)
    for poly in polys:
        front |= raster(poly) >= 0.5
    sh = shift(front, 3, 5) & MF & ~front                      # hair shadow on the skin
    img[sh] = SKIN_SH
    for poly in polys:
        m = fill(poly, HAIR)
        hair_all |= m
    # shading inside the masses: a darker lower part and a few strand lines
    for ctrl, inner in ((SWEEP_L, [(170, 150), (156, 180), (146, 206)]), (SWEEP_R, [(320, 170), (340, 220), (334, 272)]),
                        (SWEEP_R2, [(360, 110), (400, 136), (424, 176)]), (LOCK_R, [(392, 300), (404, 400), (406, 520)]),
                        (LOCK_L, [(140, 300), (138, 400), (130, 520)])):
        stroke(smooth(PP(inner), 6, closed=False), HAIR_SH, 1.0, clip=hair_all)
    for ctrl, tipshade in ((SWEEP_L, [(150, 150), (134, 186), (130, 214), (142, 212), (154, 184), (166, 150)]),
                           (SWEEP_R, [(350, 200), (362, 250), (340, 296), (322, 306), (334, 262), (342, 218)]),
                           (LOCK_R, [(394, 420), (398, 470), (386, 540), (378, 590), (404, 560), (420, 470), (416, 420)]),
                           (LOCK_L, [(112, 420), (116, 440), (104, 520), (112, 580), (134, 560), (146, 470), (140, 420)])):
        fill(smooth(PP(tipshade), 6), HAIR_SH, hair_all)
    for poly in polys:
        stroke(poly, HAIR_LINE, 1.0, closed=True)
    for x1, y1 in ((190, 90), (256, 84), (298, 86), (206, 100)):          # lines fanning out from the part
        stroke(PP([(226, 76), (x1, y1)]), HAIR_SH, 0.8, clip=hair_all)
    for pts in STRAYS:
        c = smooth(PP(pts), 8, closed=False)
        poly = tapered(c, 7 / S, 1.5 / S, 0.9)
        m = fill(poly, HAIR)
        stroke(poly, HAIR_LINE, 0.8, closed=True)
        hair_all |= m
    # angel ring across the top, broken by the strands
    ring = smooth(PP([(140, 150), (170, 112), (220, 86), (280, 76), (340, 84), (396, 110), (430, 150)]), 8, closed=False)
    rim = raster(band(ring, 6 / S, 1) + band(ring, 6 / S, -1)[len(ring):]) >= 0.5
    base_hair = hair_all & (np.abs(img - np.asarray(HAIR, np.float32)).sum(-1) < 6)
    img[rim & base_hair] = HAIR_HI
    core = raster(band(ring, 2.5 / S, 1) + band(ring, 2.5 / S, -1)[len(ring):]) >= 0.5
    img[core & base_hair] = HAIR_SHINE
    rng = random.Random(4)
    for k in range(0, len(ring), 4):
        x, y = ring[k]
        if rng.random() < 0.7:
            stroke([(x - 0.4, y - 4 - rng.random() * 4), (x + 0.8, y + 5 + rng.random() * 6)], HAIR, 0.9, clip=hair_all)

    # features
    eye((140, 166), (186, 194), 36, (0.18, -1))
    eye((306, 148), (247, 174), 44, (0.18, -1))
    stroke(smooth(PP([(150, 151), (168, 150), (192, 163)]), 6, closed=False), BROW, 1.6)
    stroke(smooth(PP([(250, 134), (276, 125), (306, 128)]), 6, closed=False), BROW, 1.8)
    stroke(PP([(205, 212), (212, 219)]), NOSE, 1.1)
    stroke(smooth(PP([(221, 223), (240, 229), (261, 225)]), 6, closed=False), LIP, 1.2)
    fill(smooth(PP([(228, 229), (252, 229), (248, 237), (234, 237)]), 4), MOUTH_IN)
    for cx_, cy_ in ((166, 232), (314, 224)):
        fill(smooth(PP([(cx_ - 17, cy_ - 6), (cx_ + 17, cy_ - 6), (cx_ + 17, cy_ + 6), (cx_ - 17, cy_ + 6)]), 6), BLUSH, MF & ~hair_all)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


if __name__ == "__main__":
    im = draw()
    im.save("blonde_girl2_raw.png")
    im.resize((W * SCALE, H * SCALE), Image.NEAREST).save("blonde_girl2.png")
    print("saved", im.size)
