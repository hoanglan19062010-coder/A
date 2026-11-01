"""Pixel-art anime face construction study (2x2 sheets): front view and a 15-degree head turn.

Geometry is laid out on an 80x100 grid and multiplied by K, so the art has K times more
pixels in each direction.

Front-view proportions:
- cranium circle; two lines from the cheeks meet at a soft chin
- eye line at half the head height, eyes sit just below it
- nose halfway between eyes and chin, mouth halfway between nose and chin
- ears from the eye line down to the nose

Turning the head:
the head is modelled as stacked horizontal slices - circles for the skull, ellipses for the
jaw that move forward and shrink toward the chin. Every point of the front-view design gets
a depth from that model, the head is rotated around its vertical axis, and the point is
drawn where it lands. Cheek and jaw outlines are rebuilt from the rotated slices, so the far
cheek fills out and the near jaw slims down, while the skull circle stays the same.
The body and the long back hair keep facing front.
"""
import math
import random
from PIL import Image, ImageDraw, ImageFont

K = 4                      # detail multiplier: pixels per grid square
PW, PH = 80 * K, 100 * K   # panel size in pixels
SCALE = 3                  # final display upscale
E = K / 2                  # pixels per "eye unit" (the eye and shine were first tuned at K = 2)
OUTLINE_W = 0.85 * E       # outline stroke width in pixels
BG = (250, 248, 244)
PAPER_HAIR = (249, 247, 243)   # hair fill in the line-art steps: looks like paper, but brows/creases can tell it apart
GUIDE_RED = (240, 150, 150)
GUIDE_BLUE = (140, 180, 230)
GUIDE_FAINT = (226, 223, 229)
LINE = (58, 40, 50)
SKIN = (252, 236, 226)
SKIN_SHADE = (240, 208, 198)
SKIN_SHADE2 = (226, 186, 178)
BLUSH = (248, 190, 190)
BLUSH_LINE = (236, 160, 164)
LIP = (196, 112, 114)
LIP_HI = (236, 196, 190)
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
IRIS_GLINT = (210, 230, 250)
CLOTH = (60, 70, 90)
CLOTH_HI = (82, 94, 118)

CX = 40
TOP, CHIN = 14, 72
CHIN_Y = 72.4
EYE_Y = (TOP + CHIN) // 2          # 43: half the head; eyes sit just below it
NOSE_Y = 60                          # halfway between eye centre and chin
MOUTH_Y = 66                         # halfway between nose and chin

TURN = 0.0   # head yaw in degrees; positive turns the face toward the viewer's left


def cos_sin():
    t = math.radians(TURN)
    return math.cos(t), math.sin(t)


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


def mirror(pts):
    return [(2 * CX - x, y) for x, y in pts]


def table(rows, y):
    """Piecewise-linear lookup in [(y, value), ...] sorted by y."""
    if y <= rows[0][0]:
        return rows[0][1]
    for (y0, v0), (y1, v1) in zip(rows, rows[1:]):
        if y <= y1:
            return v0 + (v1 - v0) * (y - y0) / (y1 - y0)
    return rows[-1][1]


def ellipse_pts(l, t, r, b, n=28):
    cx, cy, rx, ry = (l + r) / 2, (t + b) / 2, (r - l) / 2, (b - t) / 2
    return [(cx + rx * math.cos(2 * math.pi * k / n), cy + ry * math.sin(2 * math.pi * k / n)) for k in range(n)]


# ------------------------------------------------------------------ 3D head model
# Front-view half-width of the lower face: round cheeks tapering to a softly pointed chin.
_jaw = smooth([(21, 44), (20.8, 50), (19.8, 55.5), (17.5, 60.5), (14, 65), (9, 68.8), (3.5, 71.4),
               (0, CHIN_Y)], n=10, closed=False)
JAW_TABLE = []
for _a, _y in _jaw:
    if not JAW_TABLE or _y > JAW_TABLE[-1][0]:
        JAW_TABLE.append((_y, max(0.0, _a)))
# How far forward each jaw slice sits, and how far the front of the face reaches.
ZC_TABLE = [(44, 0), (50, 1), (56, 3.5), (62, 7), (68, 11), (CHIN_Y, 15)]
FRONT_TABLE = [(44, 21), (50, 20.5), (56, 20), (62, 19.5), (68, 18.5), (CHIN_Y, 17)]


def head_slice(y):
    """Slice of the head at height y: (half-width a, forward offset zc, half-depth b)."""
    if y < 36:
        a = math.sqrt(max(0.0, 21 ** 2 - (y - 36) ** 2))
        return a, 0.0, a
    if y <= 44:
        return 21.0, 0.0, 21.0
    a = table(JAW_TABLE, y)
    zc = table(ZC_TABLE, y)
    return a, zc, max(0.5, table(FRONT_TABLE, y) - zc)


def face_z(x, y):
    """Depth (toward the viewer) of the face/skull surface under front-view point (x, y)."""
    a, zc, b = head_slice(y)
    r = abs(x - CX)
    if a < 1e-6:
        return zc + b if r < 1e-6 else zc - min(r * 1.5, 8.0)
    if r <= a:
        return zc + b * math.sqrt(1 - (r / a) ** 2)
    return zc - min((r - a) * 1.5, 8.0)


def shell_z(x, y):
    """Depth of the hair shell that hugs the skull."""
    r2 = 23.5 ** 2 - (x - CX) ** 2 - (y - 35.5) ** 2
    return math.sqrt(r2) if r2 > 0 else 0.0


def bang_z(x, y):
    """Bangs grow from the shell and then lie just in front of the face."""
    return shell_z(x, y) if y <= 30 else max(shell_z(x, y), face_z(x, y) + 1.5)


def lock_z(x, y):
    """Side locks hang beside the cheeks, angled outward."""
    return max(-4.0, min(12.0, 9 - 0.8 * (abs(x - CX) - 16)))


def turn_x(x, z):
    """Rotate a point (front-view x, depth z) around the head's vertical axis."""
    c, s = cos_sin()
    return CX + (x - CX) * c - z * s


def warp(pts, zf):
    return [(turn_x(x, zf(x, y)), y) for x, y in pts]


SIL_YS = [15 + i / K for i in range(int((CHIN_Y - 15) * K) + 1)] + [CHIN_Y]


def silhouette(y):
    """Leftmost / rightmost x (relative to CX) of the rotated head slice at height y."""
    a, zc, b = head_slice(y)
    c, s = cos_sin()
    xs = [a * math.cos(t) * c - (zc + b * math.sin(t)) * s
          for t in (2 * math.pi * k / 120 for k in range(120))]
    lo, hi = min(xs), max(xs)
    # anime touch on the far side: small dip at the temple, cheekbone bump below the eye
    f = TURN / 15
    adj = 0.55 * math.exp(-((y - 44) / 2.5) ** 2) - 0.5 * math.exp(-((y - 52) / 3) ** 2)
    if f > 0:
        lo += adj * f
    elif f < 0:
        hi += adj * f
    return lo, hi


def face_outline():
    left, right = [], []
    for y in SIL_YS:
        lo, hi = silhouette(y)
        left.append((CX + lo, y))
        right.append((CX + hi, y))
    return right + left[::-1]


def center_line():
    return [(turn_x(CX, face_z(CX, y)), y) for y in SIL_YS]


def chin_point():
    return turn_x(CX, face_z(CX, CHIN_Y)), CHIN_Y


EAR_L = [(19.5, 45), (17, 45.5), (16, 48.5), (16.3, 53), (17.5, 57.5), (19.3, 60), (20, 60)]
EAR_SHADE_L = [(17.8, 48), (17.6, 54)]


def ear_map(pts, side):
    """Ears stay glued to the head outline; the near ear opens up, the far one squeezes."""
    s = abs(cos_sin()[1])
    near = TURN != 0 and (side > 0) == (TURN > 0)
    g = 1 + 1.2 * s if near else 1 - 1.5 * s
    out = []
    for x, y in pts:
        a = head_slice(y)[0]
        lo, hi = silhouette(y)
        edge = hi if side > 0 else lo
        out.append((CX + edge + side * (abs(x - CX) - a) * g, y))
    return out


NECK = [(35.5, 64), (44.5, 64), (45.5, 81), (34.5, 81)]
SHOULDERS = [(1, 100), (6, 89), (20, 84), (34.5, 80), (45.5, 80), (60, 84), (74, 89), (79, 100)]


def neck_shift(y):
    """The top of the neck turns a little with the head; the base stays with the body."""
    return -1.6 * (TURN / 15) * max(0.0, min(1.0, (81 - y) / 17))


def neck_warp(pts):
    return [(x + neck_shift(y), y) for x, y in pts]


# ------------------------------------------------------------------ hair shapes (front view)
# Long back hair: smooth, full volume, ending in a few big soft clumps
BACK_HAIR = smooth([(40, 13), (51, 14.5), (59, 19), (63.5, 27), (65, 38), (67, 50), (70, 64), (73, 78),
                    (76, 92), (70, 88), (67, 97), (61, 89), (56, 95), (52, 86), (40, 86),
                    (28, 86), (24, 95), (19, 89), (13, 97), (10, 88), (4, 92),
                    (7, 78), (10, 64), (13, 50), (15, 38), (16.5, 27), (21, 19), (29, 14.5)])
BACK_STRANDS = ([(15, 40), (13.5, 56), (11.5, 72), (9.5, 88)], [(17.5, 46), (16, 62), (15, 78), (14, 94)],
                [(13, 60), (11, 75), (8.5, 88)], [(19, 62), (19, 78), (20, 92)])
# hair hugging the cranium circle (centre 40,36 r 21) with ~2.5px of thickness
CROWN_ARC = [(40 + 23.5 * math.cos(math.radians(a)), 35.5 + 23 * math.sin(math.radians(a)))
             for a in range(180, 361, 4)]
# Bangs: soft curved clumps (root_l, root_r, tip_x, tip_y), part slightly right of centre
BANGS = [(20.5, 25, 20, 47), (22, 30, 25, 43), (27, 35, 30.5, 46), (32, 39, 35, 42.5),
         (36, 43, 39.5, 46.5), (41, 47, 44, 43), (45, 52, 48.5, 46), (49.5, 56, 53, 42.5),
         (54, 60.5, 56.5, 46), (57.5, 59.5, 61, 47.5)]
# A second, darker layer of bangs behind the first, peeking out under the short clumps
BANGS_BACK = [(23, 28, 26.5, 46.8), (33, 38, 36.2, 46.2), (42, 47, 45.4, 47), (50.5, 55.5, 53.8, 46.2)]
# Side locks: gentle S-curve that frames the face and flips softly at the end
SIDE_L = smooth([(16, 28), (22, 31), (24.5, 42), (23.5, 54), (21.5, 66), (22.5, 76), (20, 84),
                 (17.5, 77), (16.5, 66), (17.5, 54), (16, 42)])
SIDE_R = mirror(SIDE_L)
LOCK_SHADE = [(21, 36), (22.5, 48), (21, 62), (20.5, 74)]
LOCK_HI = [(18, 40), (19.2, 52), (18.6, 64), (19.2, 76)]
LOCK_EDGE = [(22.7, 31), (25.2, 42), (24.2, 54), (22.2, 66)]      # just inside each lock, on the cheek
# Strand flow lines from the part: (end x, end y)
FLOW = [(21, 29), (27, 29), (33, 29), (54, 29), (59, 29), (62, 29),
        (24.5, 25), (30, 25.5), (51, 25.5), (57, 25)]


def crown_poly():
    # the outline of a ball doesn't change when it turns; only the hairline swings round
    return CROWN_ARC + [(63.5, 34), (turn_x(40, shell_z(40, 31)), 31), (16.5, 34)]


class Pen:
    """Draws on the hi-res panel using 80x100 grid coordinates."""

    def __init__(self, d, ox, oy, img):
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

    def _clip(self, P, fill, only, draw):
        xs, ys = [p[0] for p in P], [p[1] for p in P]
        x0, x1 = max(0, int(min(xs)) - 3), min(self.img.width - 1, int(max(xs)) + 3)
        y0, y1 = max(0, int(min(ys)) - 3), min(self.img.height - 1, int(max(ys)) + 3)
        if x1 < x0 or y1 < y0:
            return
        mask = Image.new("1", (x1 - x0 + 1, y1 - y0 + 1), 0)
        draw(ImageDraw.Draw(mask), [(x - x0, y - y0) for x, y in P])
        m, px = mask.load(), self.img.load()
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                if m[x - x0, y - y0] and px[x, y] in only:
                    px[x, y] = fill

    def poly_clip(self, pts, fill, only):
        """Fill a polygon, but only over pixels that already have one of the `only` colours."""
        P = self.P(pts)
        if len(P) >= 3:
            self._clip(P, fill, only, lambda md, q: md.polygon(q, fill=1))

    def line_clip(self, pts, fill, only, width=1):
        P = self.P(pts)
        if len(P) >= 2:
            self._clip(P, fill, only, lambda md, q: md.line(q, fill=1, width=width, joint="curve"))

    def aa_line(self, pts, color, width=None, closed=False, ss=4):
        """Outline with pixel-art anti-aliasing: full colour where the stroke covers a pixel
        well, one of two in-between tones on the stair steps, nothing elsewhere."""
        width = OUTLINE_W if width is None else width
        P = [(self.ox + x * K, self.oy + y * K) for x, y in pts]
        if closed:
            P = P + [P[0]]
        if len(P) < 2:
            return
        xs, ys = [p[0] for p in P], [p[1] for p in P]
        pad = int(width) + 3
        x0, x1 = max(0, int(min(xs)) - pad), min(self.img.width - 1, int(max(xs)) + pad)
        y0, y1 = max(0, int(min(ys)) - pad), min(self.img.height - 1, int(max(ys)) + pad)
        if x1 < x0 or y1 < y0:
            return
        bw, bh = x1 - x0 + 1, y1 - y0 + 1
        big = Image.new("L", (bw * ss, bh * ss), 0)
        ImageDraw.Draw(big).line([((x - x0 + 0.5) * ss, (y - y0 + 0.5) * ss) for x, y in P],
                                 fill=255, width=max(1, round(width * ss)), joint="curve")
        cov = big.resize((bw, bh), Image.BOX).load()
        px = self.img.load()
        for y in range(bh):
            for x in range(bw):
                a = cov[x, y] / 255
                if a < 0.2:
                    continue
                if a >= 0.55:
                    px[x0 + x, y0 + y] = color
                else:
                    t = 0.33 if a < 0.38 else 0.6
                    c0 = px[x0 + x, y0 + y]
                    px[x0 + x, y0 + y] = tuple(round(c0[i] * (1 - t) + color[i] * t) for i in range(3))


def clump(rl, rr, tx, ty):
    """A curved, tapering clump of bangs (bulges slightly toward its tip side)."""
    lean = (tx - (rl + rr) / 2) * 0.35
    left = smooth([(rl, 22), (rl + 0.5 + lean, (22 + ty) / 2), (tx, ty)], n=6, closed=False)
    right = smooth([(tx, ty), (rr - 0.3 + lean, (22 + ty) / 2), (rr, 22)], n=6, closed=False)
    return left + right[1:]


def bang_clumps(lst):
    """Bangs after the turn: roots ride on the hair shell, tips on the face."""
    return [(turn_x(rl, bang_z(rl, 22)), turn_x(rr, bang_z(rr, 22)), turn_x(tx, bang_z(tx, ty)), ty)
            for rl, rr, tx, ty in lst]


# ------------------------------------------------------------------ face features
def eye(pen, cx, flip, color, scale=1.0, gaze=0.0):
    """Anime eye. Shapes are measured in 'eye units' (E pixels each) and filled pixel by
    pixel, so the eye stays smooth at any resolution. scale squeezes it sideways (far eye on
    a turned head); gaze slides the iris left/right (eye units)."""
    s = -1 if flip else 1                       # +1: outer corner on the left
    ecx = pen.ox + cx * K                       # eye centre (screen px)
    top = pen.oy + (EYE_Y + 1) * K              # top of the opening (screen px)
    d, px = pen.d, pen.img.load()
    SX = lambda u: ecx + s * u * E              # local u (+ toward the nose) -> screen x
    SY = lambda v: top + v * E
    w, h = 11 * scale, 17                       # half-width and height of the opening
    iris_c = s * scale + gaze                   # iris centre, screen direction, from the eye centre
    irx, iry, icy = 6.5 * scale, 9, 9
    prx, pry, pcy = 3.1 * scale, 4.6, 7.5

    def inside(u, v):
        """Eye opening: flat-ish top, rounded bottom toward the inner corner."""
        if v < 1 or v > h:
            return False
        t = v / h
        half = w * math.sqrt(max(0.0, 1 - max(0.0, (t - 0.45) / 0.55) ** 2))
        return -w + (1 if v < 3 else 0) <= u <= half

    def frange(a, b, step=0.25):
        n = max(1, int((b - a) / step))
        return [a + (b - a) * i / n for i in range(n + 1)]

    # white, iris, pupil
    xa, xb = int(ecx - (w + 2) * E) - 1, int(ecx + (w + 2) * E) + 1
    for py in range(int(top), int(top + (h + 1) * E) + 1):
        v = (py - top) / E
        for x in range(xa, xb + 1):
            sx = (x - ecx) / E
            if not inside(s * sx, v):
                continue
            dx = sx - iris_c
            e = (dx / irx) ** 2 + ((v - icy) / iry) ** 2
            pupil = (dx / prx) ** 2 + ((v - pcy) / pry) ** 2 <= 1
            if color:
                c = SCLERA_SHADE if v <= 3 else WHITE
                if e <= 1:
                    i = min(int(v / h * len(IRIS)), len(IRIS) - 1)
                    if v > icy + 1 and int((math.atan2(v - icy, dx) + math.pi) * 26 / (2 * math.pi)) % 2:
                        i = max(0, i - 1)              # fine radial texture in the lower iris
                    c = IRIS[i]
                    if e > 0.8:
                        c = IRIS[0]                    # dark rim
                    if pupil:
                        c = PUPIL
                    if v <= 4:
                        c = IRIS[0]                    # shadow of the lid
            else:
                c = BG
                if 0.76 < e <= 1 or pupil:
                    c = LINE
            px[x, py] = c

    def blob(cu, cv, ru, rv, c, clip=True):
        """Filled ellipse in screen-direction eye units; clipped to the opening by default."""
        for py in range(int(SY(cv - rv)) - 1, int(SY(cv + rv)) + 2):
            v = (py - top) / E
            for x in range(int(ecx + (cu - ru) * E) - 1, int(ecx + (cu + ru) * E) + 2):
                sx = (x - ecx) / E
                if ((sx - cu) / ru) ** 2 + ((v - cv) / rv) ** 2 <= 1 and (not clip or inside(s * sx, v)):
                    px[x, py] = c

    ks = max(scale, 0.8)
    blob(iris_c - 3.3 * ks, 5.1, 1.35 * ks, 1.6, WHITE)          # highlight: light from the upper left
    if color:
        blob(iris_c + 2.6 * ks, 13.6, 0.95 * ks, 0.62, IRIS_GLINT)

    # thick upper lash: an arched band, heaviest at the outer corner, tapering at the inner end
    def arch(u):
        return 2.2 * (1 - (u / (w + 1)) ** 2)

    def thick(u):
        t = 3 + max(0.0, min(1.0, (3 - u) / 4))
        if u > w - 1:
            t *= max(0.35, (w + 1.5 - u) / 2.5)
        return t

    us = frange(-w - 1.5, w + 1.5)
    d.polygon([(SX(u), SY(-arch(u) - 1.5)) for u in us] +
              [(SX(u), SY(-arch(u) - 1.5 + thick(u))) for u in reversed(us)], fill=LINE)
    wing = [(-w - 1.4, -1.0), (-w - 4.8, 2.9), (-w - 4.1, 3.6), (-w - 1.7, 3.0),
            (-w - 0.4, 4.7), (-w + 0.6, 4.2), (-w + 0.6, 1.5)]
    d.polygon([(SX(u), SY(v)) for u, v in wing], fill=LINE)          # outer flick
    blob(s * (w + 1), 2, 0.6, 0.6, LINE, clip=False)                  # inner corner
    # double eyelid crease (only on skin, not over hair)
    crease = [(SX(u), SY(-4 - 2 * (1 - (u / w) ** 2))) for u in frange(-w + 3, w - 3)]
    pen._clip(crease, SKIN_SHADE2 if color else LINE, {SKIN, SKIN_SHADE, BG},
              lambda md, q: md.line(q, fill=1, width=1))
    # lower lash, soft, with a short dark stroke at the outer end
    d.line([(SX(u), SY(h + 0.9)) for u in (-w + 3, w - 4)], fill=SKIN_SHADE2 if color else LINE, width=1)
    d.polygon([(SX(u), SY(v)) for u, v in ((-w + 0.6, h - 2.6), (-w + 1.6, h - 2.9), (-w + 3.8, h + 0.3),
                                           (-w + 2.5, h + 0.7))], fill=LINE)
    # brow: thin in the middle of its length, a touch lower at the outer end
    ub = frange(-w + 2, w - 2)
    mid, half_len = (ub[0] + ub[-1]) / 2, (ub[-1] - ub[0]) / 2
    bv = lambda u: -11 + max(0.0, min(1.0, (-w + 5 - u) / 3))
    bt = lambda u: 0.3 + 0.32 * (1 - ((u - mid) / half_len) ** 2)
    brow = [(SX(u), SY(bv(u) - bt(u))) for u in ub] + [(SX(u), SY(bv(u) + bt(u))) for u in reversed(ub)]
    pen._clip(brow, HAIR_SHADE if color else LINE, {SKIN, SKIN_SHADE, BG},
              lambda md, q: md.polygon(q, fill=1))


def eye_layout():
    """Where each eye lands after the turn: (centre x, width scale) for left and right."""
    ye = EYE_Y + 1 + 4.25                # vertical middle of the eye opening
    out = []
    for side in (-1, 1):
        c0 = CX + side * 10.5
        xo, xi = c0 + side * 5.5, c0 - side * 5.5          # outer and inner corners
        po, pin = turn_x(xo, face_z(xo, ye)), turn_x(xi, face_z(xi, ye))
        out.append(((po + pin) / 2, min(1.04, abs(po - pin) / 11)))
    return out


def features(pen, color):
    gaze = 1.5 * TURN / 15          # eyes glance back toward the viewer
    (lx, ls), (rx, rs) = eye_layout()
    eye(pen, lx, False, color, ls, gaze)
    eye(pen, rx, True, color, rs, gaze)
    nose = SKIN_SHADE2 if color else LINE
    if TURN == 0:
        pen.aa_line([(CX + 0.5, NOSE_Y - 1), (CX, NOSE_Y)], nose, width=0.6 * E)
    else:
        # the nose sticks out, so its tip slides past the centre line
        tip = (turn_x(CX, face_z(CX, NOSE_Y) + 2.6), NOSE_Y)
        top = (turn_x(CX + 0.4, face_z(CX, NOSE_Y - 2.2) + 1.4), NOSE_Y - 2.2)
        pen.aa_line([top, tip], nose, width=0.6 * E)
        if color:
            pen.line([(tip[0] + 0.6, NOSE_Y + 0.6), (tip[0] + 1.6, NOSE_Y + 0.6)], fill=SKIN_SHADE)
    mouth = [(CX - 2.5, MOUTH_Y - 0.5), (CX - 1.5, MOUTH_Y), (CX + 1.5, MOUTH_Y), (CX + 2.5, MOUTH_Y - 0.5)]
    pen.aa_line(warp(mouth, face_z), LIP if color else LINE, width=0.75 * E)
    if color:
        pen.line(warp([(CX - 1, MOUTH_Y + 1), (CX + 1, MOUTH_Y + 1)], face_z), fill=LIP_HI)


def jaw_shade(side):
    """Soft shade line just inside the jaw outline."""
    pts = []
    for y, inset in ((60, 0.7), (64.8, 1.2), (68.6, 1.25), (71, 1.6)):
        lo, hi = silhouette(y)
        pts.append((CX + (lo + inset if side < 0 else hi - inset), y))
    return smooth(pts, n=4, closed=False)


def face_shading(pen):
    only = {SKIN}
    for c in bang_clumps(BANGS) + bang_clumps(BANGS_BACK):     # bangs cast a soft shadow
        pen.poly_clip([(x, y + 1.5) for x, y in clump(*c)], SKIN_SHADE, only)
    for side in (-1, 1):
        pen.line_clip(jaw_shade(side), SKIN_SHADE, only, width=round(E))
    edge = smooth(LOCK_EDGE, n=6, closed=False)                 # side locks shade the cheeks
    pen.line_clip(warp(edge, lock_z), SKIN_SHADE, only, width=round(E))
    pen.line_clip(warp(mirror(edge), lock_z), SKIN_SHADE, only, width=round(E))
    if TURN:
        # the side of the face turning away from the light falls into shadow
        f = abs(TURN) / 15
        outer, inner = [], []
        for i in range(53):
            y = 45 + i * 0.5
            lo, hi = silhouette(y)
            w = (1.9 - 0.9 * (y - 45) / 26) * f
            if TURN > 0:
                outer.append((CX + hi + 0.6, y))
                inner.append((CX + hi - w, y))
            else:
                outer.append((CX + lo - 0.6, y))
                inner.append((CX + lo + w, y))
        pen.poly_clip(outer + inner[::-1], SKIN_SHADE, only)
    for side in (-1, 1):                                        # blush follows the cheek
        bx = CX + side * 13
        z = lambda x: face_z(x, 58.25)
        l, r = turn_x(bx - 3.5, z(bx - 3.5)), turn_x(bx + 3.5, z(bx + 3.5))
        pen.poly_clip(ellipse_pts(l, 57, r, 59.5), BLUSH, {SKIN, SKIN_SHADE})
        for k in range(3):
            x0 = bx - 2 + k * 1.6
            pen.line_clip([(turn_x(x0, z(x0)), 57.6), (turn_x(x0 - 0.8, z(x0 - 0.8)), 58.8)], BLUSH_LINE, {BLUSH})


# ------------------------------------------------------------------ guides
INSET = (70, 25, 5)   # small top-view sketch of the head: centre x, y, radius (grid)


def guides(pen, faint):
    red = GUIDE_FAINT if faint else GUIDE_RED
    blue = GUIDE_FAINT if faint else GUIDE_BLUE
    pen.ellipse((19, 15, 61, 57), outline=blue)                 # skull: a ball looks the same from any side
    lo, hi = silhouette(44)
    chin = chin_point()
    pen.line([(CX + lo, 44), chin, (CX + hi, 44)], fill=blue)   # two lines meeting at the chin
    pen.line([(CX, 8)] + center_line() + [(chin[0], 92)], fill=red)   # centre line bends with the turn
    for y in (EYE_Y, NOSE_Y, MOUTH_Y):
        pen.line([(12, y), (68, y)], fill=blue)
    pen.line([(12, TOP), (68, TOP)], fill=red)
    pen.line([(12, CHIN), (68, CHIN)], fill=red)


def turn_inset(pen):
    """Head seen from above: where 'straight ahead' is, and where the face points now."""
    cx, cy, r = INSET
    pen.ellipse((cx - r, cy - r, cx + r, cy + r), outline=GUIDE_BLUE)
    pen.line([(cx, cy), (cx, cy + r + 3)], fill=GUIDE_FAINT)
    t = math.radians(TURN)
    dx, dy = -math.sin(t), math.cos(t)
    pen.line([(cx, cy), (cx + dx * (r + 3), cy + dy * (r + 3))], fill=GUIDE_RED)
    nx, ny = cx + dx * r, cy + dy * r
    pen.poly([(nx - dy * 1.2, ny + dx * 1.2), (nx + dx * 1.6, ny + dy * 1.6), (nx + dy * 1.2, ny - dx * 1.2)],
             fill=GUIDE_BLUE)


# ------------------------------------------------------------------ hair
def hair_front(pen, step):
    color = step == 4
    hf = HAIR if color else PAPER_HAIR
    for lock in (SIDE_L, SIDE_R):
        pts = warp(lock, lock_z)
        pen.poly(pts, fill=hf)
        pen.aa_line(pts, LINE, closed=True)
    pen.poly(crown_poly(), fill=hf)
    pen.aa_line(CROWN_ARC, LINE)
    pen.poly([(20.5, 25), (59.5, 25), (63, 32), (17, 32)], fill=hf)
    # back layer of bangs
    back = bang_clumps(BANGS_BACK)
    for c in back:
        pen.poly(clump(*c), fill=HAIR_SHADE if color else PAPER_HAIR)
    for c in back:
        pts = clump(*c)
        pen.aa_line(pts[len(pts) // 3:], LINE, width=0.7 * E)
    # front layer of bangs
    front = bang_clumps(BANGS)
    for c in front:
        pen.poly(clump(*c), fill=hf)
    if color:
        # each clump: darker core toward the tip, lighter rim on one side = rounded volume
        for rl, rr, tx, ty in front:
            cx = (rl + rr) / 2
            y0 = (24 + ty) / 2
            top_pt, tip_pt = (cx + (tx - cx) * 0.4, y0 - 1), (tx, ty - 1.2)
            left = smooth([top_pt, (cx - 0.5 + (tx - cx) * 0.6, (y0 + ty) / 2), tip_pt], n=6, closed=False)
            right = smooth([tip_pt, (cx + 0.9 + (tx - cx) * 0.65, (y0 + ty) / 2), top_pt], n=6, closed=False)
            pen.poly(left + right[1:], fill=HAIR_SHADE)
            pen.line(smooth([(rl + 1.4, 23.5), (rl + 1.8 + (tx - cx) * 0.3, (23 + ty) / 2), (tx - 0.4, ty - 3)],
                            n=5, closed=False), fill=HAIR_MID)
        # side locks: shaded inner half, highlight curve following the S
        for pts, c in ((LOCK_SHADE, HAIR_SHADE), (LOCK_HI, HAIR_MID)):
            ln = smooth(pts, n=6, closed=False)
            pen.line(warp(ln, lock_z), fill=c)
            pen.line(warp(mirror(ln), lock_z), fill=c)
    for c in front:
        pts = clump(*c)
        pen.aa_line(pts[len(pts) // 3:], LINE, width=0.7 * E)
    # one thin strand falling between the eyes
    strand = smooth(warp([(43.5, 23), (42.6, 34), (41.7, 44), (41.3, 51)], bang_z), n=6, closed=False)
    pen.aa_line(strand, HAIR if color else LINE, width=0.55 * E)
    # strand flow from the part, curving with the dome
    flow = HAIR_SHADE if color else GUIDE_FAINT
    for x, y_end in FLOW:
        side = -1 if x < 44 else 1
        mid = (44 + side * 6 + x) / 2
        ctrl = [(44 + side * 0.5, 14), (mid, 16 + abs(x - 44) * 0.25), (x, y_end)]
        pen.line(smooth(warp(ctrl, shell_z), n=6, closed=False), fill=flow)
    pen.line(warp([(44, 12.5 + i * 0.5) for i in range(14)], shell_z), fill=LINE)      # the part
    dx = turn_x(43, shell_z(43, 13)) - 43                                                # ahoge rides along
    pen.aa_line(smooth([(43 + dx, 13), (42 + dx, 10), (43.5 + dx, 8), (46.5 + dx, 8.8)], n=5, closed=False), LINE,
                width=0.65 * E)


HAIR_RAMP = [HAIR_DEEP, HAIR_SHADE, HAIR, HAIR_MID, HAIR_HI, HAIR_SHINE]
STRAND_LINES = (HAIR_DEEP, HAIR_SHADE, LINE)


def hair_light(pen, img):
    """Hair shading in the style of the 'add glow' technique:
    1. light at the top, darker toward the ends (banded gradient)
    2. a glow band across the hair...
    3. ...cut by the strand lines ("eraser")...
    4. ...and smudged into streaks along the strands ("blur + finger tip").
    Plus a thin rim light on the lit (left) edge of the silhouette."""
    ox, oy = pen.ox, pen.oy
    W, H = PW, PH
    px = img.load()
    hair = {c: i for i, c in enumerate(HAIR_RAMP)}

    # 1. gradient: shift each hair pixel along the ramp by height. The boundaries wave and
    #    jitter per column so they read as strand ends instead of a straight cut.
    wob = random.Random(3)
    top_cut, low_cut = [], []
    for x in range(W):
        gx = x / K
        top_cut.append(16.2 + 1.2 * math.sin(gx * 0.87) + wob.uniform(-0.8, 0.8))
        low_cut.append(63 + 2.2 * math.sin(gx * 0.65) + 1.3 * math.sin(gx * 1.43 + 1) + wob.uniform(-1.2, 1.2))
    for y in range(H):
        gy = y / K
        for x in range(W):
            if gy < top_cut[x]:
                shift = 1
            elif gy < low_cut[x]:
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
        half *= E
        run = 0
        for x in range(*x_range):
            gx = x / K
            cy = center_y(gx) * K
            if run <= 0:                     # start a new strand of hair
                run = rng.choice((1, 2, 2, 3, 3, 4))
                up = rng.choice((2, 3, 3, 4, 5, 6)) * E
                down = rng.choice((3, 4, 6, 7, 9, 11)) * E
                if rng.random() < 0.18:      # some strands barely shine
                    up, down = E, 2 * E
            run -= 1
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

    # rim light: the first hair pixels just inside the outline on the left edge
    for y in range(int(16 * K), H):
        edge = None
        for x in range(W - 1):
            c = px[ox + x, oy + y]
            if c == BG:
                continue
            if edge is None:
                edge = x
            if x - edge > 4 * E:
                break
            if c in hair:
                for k in range(int(E)):
                    n = px[ox + x + k, oy + y]
                    if n in hair and hair[n] < hair[HAIR_MID]:
                        px[ox + x + k, oy + y] = HAIR_MID
                break


# ------------------------------------------------------------------ panels
def panel(img, ox, oy, step):
    pen = Pen(ImageDraw.Draw(img), ox, oy, img)
    face = face_outline()
    neck = neck_warp(NECK)
    if step == 1:
        guides(pen, faint=False)
        pen.poly(face, outline=GUIDE_BLUE)
        pen.poly(neck, outline=GUIDE_BLUE)
        pen.line(SHOULDERS[1:-1], fill=GUIDE_BLUE)
        if TURN:
            turn_inset(pen)
        return
    color = step == 4
    if not color:
        guides(pen, faint=True)
    # back hair
    if step >= 3:
        pen.poly(BACK_HAIR, fill=HAIR_SHADE if color else PAPER_HAIR)
        if color:
            pen.poly(smooth([(22, 56), (58, 56), (55, 86), (25, 86)], n=4), fill=HAIR_DEEP)
            for k, pts in enumerate(BACK_STRANDS):
                c = HAIR_MID if k == 1 else HAIR
                ln = smooth(pts, n=6, closed=False)
                for side_ln in (ln, mirror(ln)):
                    pen.line(side_ln, fill=c)                                       # lit ridge
                    pen.line([(x + 1 / K, y) for x, y in side_ln], fill=HAIR_DEEP)  # its shadow side
        else:
            for pts in BACK_STRANDS[:2]:
                ln = smooth(pts, n=6, closed=False)
                pen.line(ln, fill=GUIDE_FAINT)
                pen.line(mirror(ln), fill=GUIDE_FAINT)
        pen.aa_line(BACK_HAIR, LINE, closed=True)
    # clothes
    if color:
        pen.poly(SHOULDERS, fill=CLOTH)
        pen.aa_line(SHOULDERS[1:-1], LINE)
        pen.line([(8, 90), (22, 85.5), (34.5, 81.5)], fill=CLOTH_HI)
        pen.line(mirror([(8, 90), (22, 85.5), (34.5, 81.5)]), fill=CLOTH_HI)
    else:
        pen.aa_line(SHOULDERS[1:-1], LINE)
    # neck
    pen.poly(neck, fill=SKIN if color else BG)
    pen.aa_line(neck[1:] + neck[:1], LINE)
    if color:
        pen.poly_clip([(x, y + 1.4) for x, y in face if y >= 62], SKIN_SHADE, {SKIN})   # under the chin
        pen.poly_clip([(34.5, 79), (45.5, 79), (45.5, 81), (34.5, 81)], SKIN_SHADE, {SKIN})
        if TURN:   # neck muscle shows on the side that turns away
            pen.line_clip([(CX + 4.2 + neck_shift(69), 69), (CX + 0.8, 80.5)], SKIN_SHADE, {SKIN}, width=round(E))
    # ears and face
    skin = SKIN if color else BG
    for side in ((-1, 1) if step < 3 else ()):            # long hair covers the ears later
        pts = EAR_L if side < 0 else mirror(EAR_L)
        ear = ear_map(pts, side)
        pen.poly(ear, fill=skin)
        pen.aa_line(ear, LINE, closed=True)
        if color:
            sh = EAR_SHADE_L if side < 0 else mirror(EAR_SHADE_L)
            pen.line(ear_map(sh, side), fill=SKIN_SHADE2)
    pen.poly(face, fill=skin)
    pen.aa_line(face, LINE, closed=True)
    if color:
        face_shading(pen)
    if step >= 3:
        hair_front(pen, step)
        if color:
            hair_light(pen, img)
    features(pen, color)       # eyes and brows sit on top of the bangs


def render(turn, name, title, labels):
    global TURN
    TURN = turn
    cells = ((0, 0), (PW, 0), (0, PH), (PW, PH))
    sheet = Image.new("RGB", (PW * 2, PH * 2), BG)
    for k, (ox, oy) in enumerate(cells, start=1):
        panel(sheet, ox, oy, k)
    sheet.save(f"{name}_raw.png")
    big = sheet.resize((PW * 2 * SCALE, PH * 2 * SCALE), Image.NEAREST)

    f = K * SCALE / 6                     # text was sized for 6 screen pixels per grid square
    bar = round(80 * f)
    out = Image.new("RGB", (big.width, big.height + bar), BG)
    out.paste(big, (0, bar))
    d = ImageDraw.Draw(out)
    d.text((24 * f, 22 * f), title, fill=LINE, font=ImageFont.load_default(size=round(36 * f)))
    small = ImageFont.load_default(size=round(26 * f))
    for k, (ox, oy) in enumerate(cells):
        x, y = ox * SCALE + 14 * f, bar + oy * SCALE + 10 * f
        l, t, r, bt = d.textbbox((x, y), labels[k], font=small)
        d.rectangle((l - 6 * f, t - 4 * f, r + 6 * f, bt + 4 * f), fill=BG)
        d.text((x, y), labels[k], fill=(170, 90, 100), font=small)
    if turn:
        tiny = ImageFont.load_default(size=round(19 * f))
        cx, cy, r = INSET
        for i, txt in enumerate(("top view", f"{turn:g}° turn")):
            l, t, rr, bt = d.textbbox((0, 0), txt, font=tiny)
            x = cx * K * SCALE - (rr - l) / 2
            y = bar + (cy + r + 3.8) * K * SCALE + i * 22 * f
            d.text((x, y), txt, fill=(170, 90, 100), font=tiny)
    d.line((PW * SCALE, bar, PW * SCALE, out.height), fill=(210, 205, 210), width=round(2 * f))
    d.line((0, bar + PH * SCALE, out.width, bar + PH * SCALE), fill=(210, 205, 210), width=round(2 * f))
    out.save(f"{name}.png")
    print("saved", name, out.size)


def main():
    render(0, "face_study", "Anime face construction - pixel study",
           ["1  Guides: circle, chin lines, eye line",
            "2  Features: eyes, nose, mouth, ears",
            "3  Hair: crown, bangs, side locks",
            "4  Color + shading"])
    render(15, "face_study_turn15", "Head turned 15° - pixel study",
           ["1  Guides: curved center line",
            "2  Features: far side squeezes",
            "3  Hair: bangs + part turn too",
            "4  Color + shading"])
    # side-by-side of the two finished portraits
    z = 3
    f = K * z / 8
    gap, bar = round(24 * f), round(70 * f)
    pair = Image.new("RGB", (PW * z * 2 + gap, PH * z + bar), BG)
    d = ImageDraw.Draw(pair)
    font = ImageFont.load_default(size=round(30 * f))
    for i, (name, label) in enumerate((("face_study", "front"), ("face_study_turn15", "turned 15°"))):
        raw = Image.open(f"{name}_raw.png").crop((PW, PH, PW * 2, PH * 2))
        x = i * (PW * z + gap)
        pair.paste(raw.resize((PW * z, PH * z), Image.NEAREST), (x, bar))
        d.text((x + 16 * f, 20 * f), label, fill=(170, 90, 100), font=font)
    pair.save("face_turn_compare.png")
    print("saved face_turn_compare", pair.size)


main()
