"""Original anime girl, drawn stage by stage with the raster engine.
Proportions measured from the references: 6.35 heads tall, shoulders ~1.05 head wide,
eyes on the 0.57 line, nose 0.70, mouth 0.81 (book p.20 / the Aoi sheet)."""
import math, sys, os
import numpy as np
from engine import *

W, H = 1600, 3400
CX, Y0, HU = 800.0, 190.0, 480.0           # canvas centre x, skull top y, head unit (px)
TILT = -5.0                                  # head tilt in degrees (top leans to viewer's left)
PIV = (0.0, 0.95)                            # head pivot (head units)
HEAD_DX = -0.015                             # head shifts slightly toward the weight leg
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')

def B(x, y):                                 # body head-units -> canvas px
    return (CX + x * HU, Y0 + y * HU)

def Hd(x, y):                                # head-local head-units -> canvas px (tilted)
    a = math.radians(TILT); c, s = math.cos(a), math.sin(a)
    dx, dy = x - PIV[0], y - PIV[1]
    return B(PIV[0] + dx * c - dy * s + HEAD_DX, PIV[1] + dx * s + dy * c)

def Hs(pts, mirror=1):
    return [(Hd(mirror * p[0], p[1]) + tuple(p[2:])) for p in pts]

def lerp(a, b, t): return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)

# ------------------------------------------------------------------ palette
def mul(base, target):
    return np.clip(rgb(target) / np.maximum(rgb(base), 1e-3), 0, 1)

SKIN, HAIR = '#fde9de', '#b97f52'
C = dict(
    skin=rgb(SKIN), skin_sh=mul(SKIN, '#f3c3b1'), skin_sh2=mul(SKIN, '#e3a092'), blush=rgb('#f59a9a'),
    skin_ln=rgb('#a55c4f'),
    hair=rgb(HAIR), hair_sh=mul(HAIR, '#8c5636'), hair_sh2=mul(HAIR, '#5e3324'), hair_hl=rgb('#f7deb8'), hair_ln=rgb('#4a2619'),
    iris=rgb('#4f80c6'), iris_dk=rgb('#1f3170'), iris_lt=rgb('#a6dcf5'), pupil=rgb('#16183d'), lash=rgb('#2b1a1d'),
    white=rgb('#ffffff'), eyesh=rgb('#d4dbef'), mouth=rgb('#b5525a'),
)

def SHADE(mode): return mode == 'final'
def FILL(mode, col): return C['white'] if mode == 'line' else col

# ------------------------------------------------------------------ head geometry (head units, untilted)
FACE_R = [(0.384, 0.40), (0.379, 0.53), (0.360, 0.635), (0.320, 0.73), (0.255, 0.82), (0.170, 0.905),
          (0.085, 0.968), (0.030, 0.994)]
def face_pts():
    left = [(-x, y) for x, y in FACE_R]
    right = list(reversed(FACE_R))
    cranium = [(0.36, 0.22), (0.27, 0.08), (0.14, 0.015), (0.0, 0.0), (-0.14, 0.015), (-0.27, 0.08), (-0.36, 0.22)]
    return left + [(0.0, 1.0)] + right + cranium   # left temple -> chin -> right temple -> over the top

EYE_TOP = [(0.118, 0.600, 'c'), (0.135, 0.553), (0.170, 0.522), (0.218, 0.510), (0.264, 0.520), (0.294, 0.545), (0.308, 0.578, 'c')]
EYE_BOT = [(0.300, 0.615), (0.272, 0.645), (0.225, 0.660), (0.180, 0.656), (0.145, 0.640), (0.124, 0.618)]
LASH = [(0.108, 0.606, 'c'), (0.121, 0.550), (0.158, 0.507), (0.215, 0.489), (0.268, 0.499), (0.305, 0.525),
        (0.340, 0.538, 'c'), (0.320, 0.556), (0.308, 0.580, 'c'), (0.294, 0.545), (0.264, 0.520), (0.218, 0.510),
        (0.170, 0.522), (0.135, 0.553)]
IRIS = (0.212, 0.592, 0.058, 0.076)          # cx, cy, rx, ry
CREASE = [(0.150, 0.488), (0.21, 0.470), (0.268, 0.476), (0.305, 0.498)]
BROW = [(0.112, 0.430), (0.16, 0.408), (0.225, 0.400), (0.29, 0.410)]

# hair ------------------------------------------------------------------------
PART = (-0.10, -0.045)
MAIN_R = [(0.03, -0.068), (0.17, -0.060), (0.30, -0.016), (0.41, 0.074), (0.49, 0.20), (0.545, 0.36), (0.575, 0.52),
          (0.578, 0.66), (0.555, 0.79), (0.51, 0.90)]
MAIN_RB = [(0.45, 0.965), (0.37, 0.985), (0.30, 0.95), (0.298, 0.85), (0.333, 0.70), (0.350, 0.56), (0.345, 0.44), (0.33, 0.36)]
MAIN_L = [(-0.22, -0.036), (-0.32, 0.008), (-0.42, 0.084), (-0.50, 0.21), (-0.552, 0.37), (-0.578, 0.53), (-0.578, 0.67),
          (-0.552, 0.80), (-0.505, 0.91)]
MAIN_LB = [(-0.44, 0.972), (-0.365, 0.99), (-0.30, 0.95), (-0.298, 0.85), (-0.333, 0.70), (-0.348, 0.56), (-0.343, 0.44), (-0.33, 0.36)]
HAIRLINE = [(0.22, 0.25), (0.0, 0.20), (-0.22, 0.25)]

# clumps: centreline (root -> tip), max width, root width, bias, line_from (fraction of length where edge lines start)
SIDE = [
    ([(0.49, 0.40), (0.528, 0.60), (0.526, 0.80), (0.498, 0.95), (0.462, 1.045)], 0.095, 0.08, 0.0, 0.40),
    ([(0.42, 0.28), (0.462, 0.50), (0.460, 0.72), (0.420, 0.90), (0.374, 1.03)], 0.13, 0.10, 0.0, 0.35),
    ([(0.33, 0.30), (0.362, 0.50), (0.352, 0.70), (0.314, 0.86), (0.286, 0.965)], 0.10, 0.08, -0.1, 0.30),
    ([(-0.49, 0.42), (-0.528, 0.62), (-0.522, 0.82), (-0.488, 0.96), (-0.447, 1.032)], 0.095, 0.08, 0.0, 0.40),
    ([(-0.42, 0.29), (-0.462, 0.50), (-0.458, 0.73), (-0.416, 0.91), (-0.368, 1.042)], 0.13, 0.10, 0.0, 0.35),
    ([(-0.33, 0.30), (-0.356, 0.50), (-0.346, 0.70), (-0.306, 0.86), (-0.281, 0.955)], 0.10, 0.08, 0.1, 0.30),
]
FLICKS = [
    ([(0.500, 0.78), (0.540, 0.905), (0.572, 0.978), (0.606, 1.012)], 0.058, 0.05, 0.0, 0.0),
    ([(-0.505, 0.80), (-0.544, 0.92), (-0.578, 0.982), (-0.612, 1.004)], 0.052, 0.045, 0.0, 0.0),
]
BANGS = [   # part at x=-0.10: small section swept left, big section swept right; back to front
    ([(-0.17, 0.00), (-0.255, 0.18), (-0.315, 0.34), (-0.338, 0.432), (-0.352, 0.470)], 0.135, 0.11, -0.10, 0.40),
    ([(0.18, 0.02), (0.300, 0.14), (0.365, 0.28), (0.388, 0.385), (0.404, 0.438)], 0.125, 0.10, 0.10, 0.45),
    ([(0.08, 0.00), (0.190, 0.14), (0.262, 0.30), (0.296, 0.430), (0.318, 0.500)], 0.150, 0.11, 0.10, 0.40),
    ([(0.00, -0.01), (0.088, 0.15), (0.142, 0.31), (0.160, 0.420), (0.180, 0.468)], 0.150, 0.11, 0.08, 0.50),
    ([(-0.10, 0.00), (-0.138, 0.18), (-0.168, 0.34), (-0.172, 0.452), (-0.160, 0.515)], 0.140, 0.10, -0.05, 0.40),
    ([(-0.06, -0.01), (-0.004, 0.17), (0.026, 0.34), (0.030, 0.470), (0.044, 0.556)], 0.150, 0.11, 0.05, 0.45),
]
WISPS = [
    ([(-0.03, 0.10), (0.05, 0.27), (0.085, 0.400), (0.104, 0.452)], 0.045, 0.04, 0.0, 0.3),
    ([(-0.16, 0.12), (-0.222, 0.30), (-0.240, 0.420), (-0.252, 0.478)], 0.04, 0.035, 0.0, 0.3),
]
AHOGE = ([(-0.075, -0.045), (-0.060, -0.110), (-0.010, -0.150), (0.035, -0.135)], 0.032, 0.03, 0.0, 0.0)

def clump_poly(center, wmax, w0, bias=0.0, at=0.30, p=1.55):
    """leaf-shaped hair clump around a centreline (canvas px). returns polygon, left edge, right edge."""
    c, u = resample(spline(center), 1.0)
    t = np.gradient(c, axis=0); t /= (np.linalg.norm(t, axis=1, keepdims=True) + 1e-9)
    n = np.stack([-t[:, 1], t[:, 0]], 1)
    w = np.where(u < at, w0 + (wmax - w0) * np.sin(np.clip(u / at, 0, 1) * math.pi / 2),
                 wmax * (1 - np.clip((u - at) / (1 - at), 0, 1) ** p))
    w = w * HU / 2.0
    left = c + n * (w * (1 + bias))[:, None]
    right = c - n * (w * (1 - bias))[:, None]
    return np.vstack([left, right[::-1]]), left, right

# ------------------------------------------------------------------ drawing
def head_masks():
    face = Path(Hs(face_pts()), closed=True)
    return face

def draw_back_hair(L, mode):
    pts = [(-0.34, 0.55), (-0.37, 0.80), (-0.352, 0.95), (-0.322, 1.040, 'c'), (-0.300, 0.995), (-0.262, 1.018, 'c'),
           (-0.232, 0.965), (-0.15, 0.925), (0.0, 0.915), (0.15, 0.925), (0.232, 0.965),
           (0.262, 1.018, 'c'), (0.300, 0.995), (0.322, 1.040, 'c'), (0.352, 0.95), (0.37, 0.80), (0.34, 0.55)]
    p = Path(Hs(pts), closed=True)
    m = p.mask()
    L.paint(m, FILL(mode, C['hair']))
    if SHADE(mode):
        L.paint(m, C['hair_sh2'], 1.0, mode='multiply')
    L.paint(stroke(p.seg(1, 7), 2.4, taper=(0.3, 0.4)), C['hair_ln'])
    L.paint(stroke(p.seg(10, 16), 2.4, taper=(0.4, 0.3)), C['hair_ln'])
    return m

def draw_neck_stub(L, mode):
    pts = [(-0.135, 0.80), (-0.128, 1.00), (-0.15, 1.20), (-0.30, 1.30), (0.30, 1.30), (0.15, 1.20), (0.128, 1.00), (0.135, 0.80)]
    m = shape([B(*p) for p in pts])
    L.paint(m, FILL(mode, C['skin']))
    for side in (-1, 1):
        L.paint(line([B(side * 0.130, 0.86), B(side * 0.129, 1.02), B(side * 0.16, 1.19)], 3.2, taper=(0.2, 0.3)), C['skin_ln'])
    return m

def draw_face(L, mode, hair_front_mask=None, bang_masks=()):
    face = head_masks()
    fm = face.mask()
    L.paint(fm, FILL(mode, C['skin']))
    if SHADE(mode):
        u = HU
        # cast shadow of the bangs (light from the upper left -> shadow falls down-right)
        if bang_masks:
            cast = union(*[b.shift(0.022 * u, 0.042 * u) for b in bang_masks]).blur(1.2)
            L.paint(cast & fm, C['skin_sh'], 1.0, mode='multiply')
        # form shadow on the far (right) side of the face
        side = fm - fm.shift(-0.06 * u, -0.01 * u)
        L.paint(side.blur(1.5) & fm, C['skin_sh'], 1.0, mode='multiply')
        # blush
        for sx in (-1, 1):
            cx, cy = Hd(sx * 0.245, 0.705)
            b = ellipse(cx, cy, 0.075 * u, 0.036 * u, TILT).blur(6)
            L.paint(b, C['blush'], 0.55, mode='multiply')
            for k in range(3):
                x0 = sx * 0.245 + (k - 1) * 0.028
                L.paint(line(Hs([(x0 + 0.010, 0.688), (x0, 0.705), (x0 - 0.010, 0.722)]), 2.0, taper=(0.4, 0.4)), C['skin_ln'] * 0.5 + C['blush'] * 0.5, 0.55)
    # jaw line (silhouette), drawn as two strokes from the cheeks to the chin
    n = len(FACE_R)
    L.paint(stroke(face.seg(1, n), 4.0, taper=(0.25, 0.08), floor=0.15), FILL('x', C['skin_ln']))
    L.paint(stroke(face.seg(n, 2 * n - 1), 4.0, taper=(0.08, 0.25), floor=0.15), FILL('x', C['skin_ln']))
    return face, fm

EYE_DX = -0.009
def draw_eye(L, mode, m):
    E = lambda pts: Hs([(p[0] + EYE_DX,) + tuple(p[1:]) for p in pts], m)
    opening = Path(E(EYE_TOP + EYE_BOT), closed=True)
    om = opening.mask()
    L.paint(om, C['white'])
    cx, cy, rx, ry = IRIS
    icx, icy = Hd(m * (cx + EYE_DX), cy)
    iris = ellipse(icx, icy, rx * HU, ry * HU, TILT) & om
    L.paint(iris, C['iris'] if mode != 'line' else C['white'])
    if SHADE(mode):
        # upper half darker, lower part lighter and saturated (book p.110)
        L.paint(lin_grad(iris, (icx, icy + 0.01 * HU), (icx, icy - 0.06 * HU)), C['iris_dk'], 0.95)
        low = iris - ellipse(icx, icy - 0.03 * HU, rx * HU * 1.05, ry * HU * 0.92, TILT)
        L.paint(low.blur(3) & iris, C['iris_lt'], 0.85)
        # eye white: shadow under the upper lid
        lid = (Path(E(LASH), closed=True).mask().shift(0, 0.02 * HU).blur(2) & om) - iris
        L.paint(lid, C['eyesh'], 1.0, mode='multiply')
    # pupil
    L.paint(ellipse(icx, icy + 0.006 * HU, 0.021 * HU, 0.036 * HU, TILT) & om, C['pupil'] if mode != 'line' else C['lash'], 0.95)
    # iris rim
    ring = ellipse(icx, icy, rx * HU, ry * HU, TILT) - ellipse(icx, icy, rx * HU - 2.2, ry * HU - 2.2, TILT)
    L.paint(ring & om, C['iris_dk'] if mode != 'line' else C['lash'], 0.9)
    # lash band (thick, heaviest at the outer corner) + two lash flicks
    L.paint(Path(E(LASH), closed=True).mask(), C['lash'])
    for a, b in (((0.282, 0.522), (0.318, 0.494)), ((0.302, 0.537), (0.340, 0.522))):
        L.paint(line(E([a, lerp(a, b, 0.5), b]), 4.0, taper=(0.0, 0.9), floor=0.05), C['lash'])
    # lower lid (outer two thirds only), crease
    low = E([(0.302, 0.594), (0.27, 0.624), (0.225, 0.638), (0.18, 0.636)])
    L.paint(line(low, 2.6, taper=(0.3, 0.6)), C['lash'] * 0.6 + C['skin_ln'] * 0.4)
    L.paint(line(E(CREASE), 2.0, taper=(0.4, 0.4)), C['skin_ln'])
    # highlights
    L.paint(ellipse(icx - 0.022 * HU, icy - 0.03 * HU, 0.017 * HU, 0.015 * HU, TILT), C['white'])
    L.paint(ellipse(icx + 0.024 * HU, icy + 0.032 * HU, 0.007 * HU, 0.006 * HU, TILT), C['white'], 0.95)
    return om

def draw_features(L, mode):
    nose = Hs([(0.020, 0.692), (0.016, 0.708), (0.004, 0.720)])
    L.paint(line(nose, 2.6, taper=(0.5, 0.3)), C['skin_ln'])
    mouth = Hs([(-0.045, 0.806), (-0.015, 0.818), (0.018, 0.817), (0.047, 0.803)])
    L.paint(line(mouth, 3.0, taper=(0.35, 0.35), floor=0.1), C['skin_ln'] * 0.8 + C['mouth'] * 0.2)
    if SHADE(mode):
        L.paint(line(Hs([(-0.012, 0.842), (0.016, 0.842)]), 2.0, taper=(0.5, 0.5)), C['skin_sh2'], 0.6)

def draw_brows(L, mode):
    for m in (-1, 1):
        L.paint(line(Hs(BROW, m), 3.2, taper=(0.15, 0.6), floor=0.1), C['hair_ln'], 0.75)

def hair_pieces():
    """z-ordered hair pieces: (kind, mask, left_edge, right_edge, line_from)"""
    pieces = []
    pts = [PART + ('c',)] + MAIN_R + MAIN_RB + HAIRLINE + list(reversed(MAIN_LB)) + list(reversed(MAIN_L))
    main = Path(Hs(pts), closed=True)
    pieces.append(('main', main.mask(), main, len(MAIN_R), None))
    for group, kind in ((FLICKS, 'flick'), (SIDE, 'side'), ([AHOGE], 'ahoge'), (BANGS, 'bang'), (WISPS, 'wisp')):
        for center, wmax, w0, bias, lf in group:
            pp, le, ri = clump_poly(Hs(center), wmax, w0, bias)
            pieces.append((kind, poly(pp), le, ri, lf))
    return pieces

def ring_highlight():
    """'angel ring' on the crown: crisp zigzag band + comet streaks along the hair flow (lit side stronger)"""
    rng = np.random.default_rng(7)
    arc = lambda x: 0.150 + 0.15 * (x / 0.5) ** 2
    top, bot = [], []
    for i, x in enumerate(np.linspace(-0.47, 0.46, 34)):
        j = rng.uniform(0.008, 0.022) * (1 if i % 2 else -1)
        top.append((x, arc(x) - 0.020 + j)); bot.append((x, arc(x) + 0.020 + j * 1.4))
    band = poly(np.asarray(Hs(top + bot[::-1]))).blur(0.6)
    streaks = []
    for x in np.linspace(-0.45, 0.42, 17):
        if rng.random() < 0.15:
            continue
        y = arc(x) + rng.uniform(-0.012, 0.012)
        fx = (x - PART[0]) * 0.6; fy = 1.0
        n = math.hypot(fx, fy); fx, fy = fx / n, fy / n
        ln = rng.uniform(0.07, 0.16) * (1.2 if x < 0.05 else 0.75)
        p0 = (x - fx * ln * 0.45, y - fy * ln * 0.45); p1 = (x + fx * ln * 0.55, y + fy * ln * 0.55)
        w = rng.uniform(7, 13) * (1.0 if x < 0.1 else 0.6)
        streaks.append(line(Hs([p0, lerp(p0, p1, 0.5), p1]), w, taper=(0.25, 0.75), floor=0.0))
    return band, union(*streaks)

def draw_front_hair(L, mode):
    u = HU
    pieces = hair_pieces()
    shade = SHADE(mode)
    bang_ms = [p[1] for p in pieces if p[0] in ('bang', 'wisp')]
    band, streaks = ring_highlight() if shade else (None, None)
    main_path = None
    for kind, m, a, b, lf in pieces:
        L.paint(m, FILL(mode, C['hair']))
        if shade:
            if kind == 'main':
                L.paint(lin_grad(m, B(0, 0.48), B(0, 0.98)), C['hair_sh'], 0.95, mode='multiply')
                cast = union(*[x.shift(0.010 * u, 0.022 * u) for x in bang_ms]).blur(5)
                L.paint(cast & m, C['hair_sh'], 0.6, mode='multiply')
                far = shape(Hs([(0.30, -0.08), (0.64, 0.18), (0.64, 1.1), (0.46, 1.1), (0.50, 0.45)])).blur(22)
                L.paint(far & m, C['hair_sh'], 0.55, mode='multiply')
            else:
                # wedge on the side away from the light, growing from nothing toward the tip
                n = len(a); k = int(n * 0.38)
                s = np.linspace(0, 1, n - k)[:, None]
                frac = 0.62 * np.clip(s / 0.45, 0, 1) ** 0.8
                inner = b[k:] + (a[k:] - b[k:]) * frac
                wedge = poly(np.vstack([inner, b[k:][::-1]])).blur(1.0)
                L.paint(wedge & m, C['hair_sh'], 0.9, mode='multiply')
                if kind in ('side', 'flick'):
                    L.paint(lin_grad(m, B(0, 0.60), B(0, 1.04)), C['hair_sh'], 0.65, mode='multiply')
            L.paint(band & m, C['hair_hl'], 0.45)
            L.paint(streaks & m, C['hair_hl'], 0.95)
        if kind == 'main':
            main_path = a
            continue
        n = len(a)
        k_left = int(n * lf); k_right = int(n * min(0.92, lf + (0.15 if kind == 'bang' else 0.0)))
        w = 2.9 if kind in ('bang', 'side') else 2.3
        L.paint(stroke(a[k_left:], w, taper=(0.45, 0.10), floor=0.08), C['hair_ln'])
        L.paint(stroke(b[k_right:], w * 0.9, taper=(0.45, 0.10), floor=0.08), C['hair_ln'])
    # silhouette of the bell, drawn last (side clumps stay inside it)
    npts = len(main_path.ctrl)
    L.paint(stroke(main_path.seg(0, len(MAIN_R)), 3.8, taper=(0.06, 0.12), floor=0.25), C['hair_ln'])
    L.paint(stroke(main_path.seg(npts - len(MAIN_L), npts), 3.8, taper=(0.12, 0.06), floor=0.25), C['hair_ln'])
    # part line and a few crown flow lines (thin, partial)
    for pts_, w in (([(-0.098, -0.04), (-0.085, 0.03), (-0.07, 0.09)], 2.0),
                    ([(-0.05, -0.05), (0.08, -0.03), (0.20, 0.03)], 1.4),
                    ([(-0.15, -0.035), (-0.26, 0.01), (-0.35, 0.09)], 1.4),
                    ([(0.30, 0.05), (0.40, 0.16), (0.46, 0.30)], 1.4)):
        L.paint(line(Hs(pts_), w, taper=(0.3, 0.5)), C['hair_ln'], 0.7)
    return pieces, bang_ms

def m_path_seg(path, i0, i1):
    return path.seg(i0, i1)

def render_head(mode, path, crop=True):
    cv = Canvas(W, H, '#fbf8f4')
    L = cv.layer(); draw_back_hair(L, mode); cv.comp(L)
    L = cv.layer(); draw_neck_stub(L, mode); cv.comp(L)
    scratch = cv.layer(); _, bang_masks = draw_front_hair(scratch, 'flat')
    L = cv.layer(); draw_face(L, mode, bang_masks=bang_masks)
    for m in (-1, 1): draw_eye(L, mode, m)
    draw_features(L, mode); cv.comp(L)
    L = cv.layer(); draw_front_hair(L, mode); cv.comp(L)
    L = cv.layer(); draw_brows(L, mode); cv.comp(L, 0.55)
    im = cv.save(path)
    if crop:
        x0, y0 = B(-0.75, -0.25); x1, y1 = B(0.75, 1.3)
        im.crop((int(x0), int(y0), int(x1), int(y1))).save(path.replace('.png', '_crop.png'))

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'final'
    render_head(mode, os.path.join(OUT, f'head_{mode}.png'))
    print('ok', mode)
