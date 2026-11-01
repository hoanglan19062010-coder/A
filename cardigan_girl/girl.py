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
    iris=rgb('#4a7cc6'), iris_dk=rgb('#1a2760'), iris_lt=rgb('#8fd3f2'), iris_glow=rgb('#d2f6ff'), pupil=rgb('#16183d'), lash=rgb('#2b1a1d'),
    white=rgb('#ffffff'), eyesh=rgb('#d4dbef'), mouth=rgb('#b5525a'),
)

def SHADE(mode): return mode == 'final'
def FILL(mode, col): return C['white'] if mode == 'line' else col

# ------------------------------------------------------------------ head geometry (head units, untilted)
FACE_R = [(0.384, 0.40), (0.385, 0.52), (0.380, 0.62), (0.366, 0.70), (0.340, 0.765), (0.300, 0.828),
          (0.243, 0.886), (0.170, 0.938), (0.092, 0.978), (0.032, 0.996)]      # cheeks stay full, then tuck into a small chin
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
MAIN_RB = [(0.478, 0.985, 'c'), (0.442, 0.935), (0.405, 1.010, 'c'), (0.368, 0.945), (0.322, 0.990, 'c'), (0.300, 0.90),
           (0.333, 0.70), (0.350, 0.56), (0.345, 0.44), (0.33, 0.36)]
MAIN_L = [(-0.22, -0.036), (-0.32, 0.008), (-0.42, 0.084), (-0.50, 0.21), (-0.552, 0.37), (-0.578, 0.53), (-0.578, 0.67),
          (-0.552, 0.80), (-0.505, 0.91)]
MAIN_LB = [(-0.470, 0.992, 'c'), (-0.435, 0.940), (-0.392, 1.005, 'c'), (-0.355, 0.950), (-0.318, 0.982, 'c'), (-0.300, 0.90),
           (-0.333, 0.70), (-0.348, 0.56), (-0.343, 0.44), (-0.33, 0.36)]
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
LOW_WISPS = [   # a few thin strands that slip out below the side masses
    ([(0.400, 0.76), (0.410, 0.94), (0.396, 1.080)], 0.026, 0.003, 0.0, 0.30),
    ([(0.318, 0.78), (0.306, 0.94), (0.286, 1.022)], 0.022, 0.003, 0.0, 0.30),
    ([(-0.405, 0.78), (-0.418, 0.96), (-0.404, 1.088)], 0.026, 0.003, 0.0, 0.30),
    ([(-0.312, 0.79), (-0.300, 0.95), (-0.276, 1.015)], 0.022, 0.003, 0.0, 0.30),
]
BACK_STRANDS = [   # hair behind the neck: thin clumps ending in fine points at different lengths
    ([(-0.30, 0.60), (-0.318, 0.80), (-0.322, 0.93), (-0.316, 1.035)], 0.085, 0.075),
    ([(0.30, 0.60), (0.320, 0.80), (0.326, 0.92), (0.322, 0.998)], 0.085, 0.075),
    ([(-0.24, 0.62), (-0.256, 0.82), (-0.262, 0.92), (-0.270, 1.002)], 0.080, 0.070),
    ([(0.24, 0.62), (0.262, 0.82), (0.270, 0.93), (0.278, 1.042)], 0.080, 0.070),
    ([(-0.18, 0.64), (-0.198, 0.83), (-0.206, 0.93), (-0.214, 1.022)], 0.075, 0.065),
    ([(0.18, 0.64), (0.200, 0.83), (0.208, 0.94), (0.212, 1.012)], 0.075, 0.065),
    ([(-0.12, 0.66), (-0.140, 0.85), (-0.150, 0.955)], 0.070, 0.060),
    ([(0.12, 0.66), (0.140, 0.85), (0.152, 0.960)], 0.070, 0.060),
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
    """the hair behind the neck: an upper base plus thin strands that end in fine points"""
    base = shape(Hs([(-0.345, 0.52), (-0.36, 0.70), (-0.30, 0.80), (0.0, 0.79), (0.30, 0.80), (0.36, 0.70), (0.345, 0.52)]))
    L.paint(base, FILL(mode, C['hair']))
    if SHADE(mode):
        L.paint(base, C['hair_sh2'], 1.0, mode='multiply')
    ms = [base]
    for center, wmax, w0 in BACK_STRANDS:
        pp, le, ri = clump_poly(Hs(center), wmax, w0, 0.0, at=0.25, p=0.8)
        m = poly(pp); ms.append(m)
        L.paint(m, FILL(mode, C['hair']))
        if SHADE(mode):
            L.paint(m, C['hair_sh2'], 0.85, mode='multiply')
            n = len(le); k = int(n * 0.35)
            rim = poly(np.vstack([le[k:], (le[k:] * 0.72 + ri[k:] * 0.28)[::-1]])).blur(1.0)
            L.paint(rim & m, C['hair'], 0.35)                 # a little light on the strand's lit edge
        n = len(le)
        L.paint(stroke(le[int(n * 0.50):], 1.8, taper=(0.5, 0.05), floor=0.08), C['hair_ln'], 0.9)
        L.paint(stroke(ri[int(n * 0.50):], 1.8, taper=(0.5, 0.05), floor=0.08), C['hair_ln'], 0.9)
    return union(*ms)

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
    """book p.13 / p.110: thick upper lash band with flicks, iris dark on top and bright below,
    a bright crescent in the lower iris, hatching at the iris sides, highlights on top."""
    E = lambda pts: Hs([(p[0] + EYE_DX,) + tuple(p[1:]) for p in pts], m)
    opening = Path(E(EYE_TOP + EYE_BOT), closed=True)
    om = opening.mask()
    lash_m = Path(E(LASH), closed=True).mask()
    L.paint(om, C['white'])
    cx, cy, rx, ry = IRIS
    icx, icy = Hd(m * (cx + EYE_DX), cy)
    R, RY = rx * HU, ry * HU
    iris_full = ellipse(icx, icy, R, RY, TILT)
    iris = iris_full & om
    L.paint(iris, C['iris'] if mode != 'line' else C['white'])
    if SHADE(mode):
        lid = (lash_m.shift(0, 0.026 * HU).blur(3) & om) - iris          # eye white: shade under the upper lid
        L.paint(lid, C['eyesh'], 1.0, mode='multiply')
        L.paint(lin_grad(iris, (icx, icy + 0.006 * HU), (icx, icy - 0.068 * HU), gamma=0.8), C['iris_dk'], 1.0)
        L.paint(lin_grad(iris, (icx, icy - 0.004 * HU), (icx, icy + 0.074 * HU), gamma=1.1), C['iris_lt'], 0.9)
        cres = (ellipse(icx, icy + 0.044 * HU, R * 0.80, RY * 0.38, TILT) - ellipse(icx, icy + 0.026 * HU, R * 0.82, RY * 0.38, TILT))
        L.paint(cres.blur(1.2) & iris, C['iris_glow'], 0.9)
        for ang in np.linspace(28, 152, 7):                               # light streaks in the lower iris
            t = math.radians(ang + TILT)
            p0 = (icx + math.cos(t) * R * 0.45, icy + math.sin(t) * RY * 0.45)
            p1 = (icx + math.cos(t) * R * 0.90, icy + math.sin(t) * RY * 0.90)
            L.paint(line([p0, p1], 2.2, taper=(0.3, 0.3)) & iris, C['iris_glow'], 0.45)
        for sx in (-1, 1):                                                # dark hatching under the lid
            for k in range(3):
                x = icx + sx * R * (0.58 + 0.12 * k)
                L.paint(line([(x, icy - RY * 0.50), (x, icy - RY * (0.02 + 0.10 * k))], 2.2, taper=(0.2, 0.6)) & iris, C['iris_dk'], 0.7)
    pupil = ellipse(icx, icy + 0.004 * HU, 0.021 * HU, 0.034 * HU, TILT) & om
    L.paint(pupil, C['pupil'] if mode != 'line' else C['lash'], 0.95)
    if SHADE(mode):                                                       # soft ring around the pupil
        pr = (ellipse(icx, icy + 0.004 * HU, 0.031 * HU, 0.046 * HU, TILT) - ellipse(icx, icy + 0.004 * HU, 0.021 * HU, 0.034 * HU, TILT))
        L.paint(lin_grad(pr & iris, (icx, icy - 0.02 * HU), (icx, icy + 0.05 * HU)), C['iris_lt'], 0.45)
    ring = iris_full - ellipse(icx, icy, R - 2.4, RY - 2.4, TILT)
    L.paint(ring & om, C['iris_dk'] if mode != 'line' else C['lash'], 0.9)
    # upper lash band + three curved flicks at the outer corner
    L.paint(lash_m, C['lash'])
    for a_, mid, b_, w in (((0.276, 0.508), (0.298, 0.490), (0.318, 0.478), 4.4),
                           ((0.298, 0.524), (0.324, 0.510), (0.350, 0.504), 4.0),
                           ((0.312, 0.546), (0.334, 0.546), (0.354, 0.554), 3.4)):
        L.paint(line(E([a_, mid, b_]), w, taper=(0.0, 0.95), floor=0.05), C['lash'])
    # lower lash line on the bottom edge of the white (outer half) + a tiny flick; crease above
    low = E([(0.304, 0.610), (0.282, 0.642), (0.246, 0.660), (0.204, 0.664)])
    L.paint(line(low, 2.8, taper=(0.15, 0.85), floor=0.1), C['lash'] * 0.65 + C['skin_ln'] * 0.35)
    L.paint(line(E([(0.300, 0.618), (0.316, 0.630), (0.328, 0.642)]), 2.2, taper=(0.1, 0.9)), C['lash'], 0.9)
    L.paint(line(E(CREASE), 2.2, taper=(0.4, 0.4)), C['skin_ln'])
    # highlights (on top): big upper-left, small lower-right, a sparkle
    L.paint(ellipse(icx - 0.022 * HU, icy - 0.028 * HU, 0.020 * HU, 0.016 * HU, TILT - 25) & om, C['white'])
    L.paint(ellipse(icx + 0.026 * HU, icy + 0.032 * HU, 0.009 * HU, 0.007 * HU, TILT) & om, C['white'], 0.95)
    L.paint(ellipse(icx - 0.031 * HU, icy + 0.030 * HU, 0.0045 * HU, 0.0045 * HU, TILT) & om, C['white'], 0.85)
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

TIP_P = dict(flick=0.85, side=0.85, lowwisp=0.9, ahoge=1.55, bang=1.25, wisp=1.0)

def hair_pieces():
    """z-ordered hair pieces: (kind, mask, left_edge, right_edge, line_from)"""
    pieces = []
    pts = [PART + ('c',)] + MAIN_R + MAIN_RB + HAIRLINE + list(reversed(MAIN_LB)) + list(reversed(MAIN_L))
    main = Path(Hs(pts), closed=True)
    pieces.append(('main', main.mask(), main, len(MAIN_R), None))
    for group, kind in ((FLICKS, 'flick'), (SIDE, 'side'), (LOW_WISPS, 'lowwisp'), ([AHOGE], 'ahoge'), (BANGS, 'bang'), (WISPS, 'wisp')):
        for center, wmax, w0, bias, lf in group:
            pp, le, ri = clump_poly(Hs(center), wmax, w0, bias, p=TIP_P[kind])
            pieces.append((kind, poly(pp), le, ri, lf))
    return pieces

def leaf(p0, p1, w, w0=0.003, at=0.3, p=1.0, bend=0.0):
    """leaf/comet shape from p0 to p1 (head-local units), widest at `at`"""
    mid = lerp(p0, p1, 0.5)
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]; n = math.hypot(dx, dy) + 1e-9
    mid = (mid[0] - dy / n * bend, mid[1] + dx / n * bend)
    pp, _, _ = clump_poly(Hs([p0, mid, p1]), w, w0, 0.0, at=at, p=p)
    return poly(pp)

def hair_shadow():
    """cel shadow for the bell: the far side and the underside are in shadow. The boundary breaks into
    strand-shaped fingers (shadow reaching up, light reaching down) that follow the hair flow."""
    base = shape(Hs([(0.30, -0.14), (0.78, -0.14), (0.78, 1.25), (-0.78, 1.25), (-0.78, 0.90), (-0.56, 0.86), (-0.42, 0.84),
                     (-0.20, 0.80), (-0.02, 0.66), (0.14, 0.46), (0.23, 0.28), (0.27, 0.10)]))
    up = [leaf((x0, y0), (x1, y1), w, w0=w, at=0.05, p=1.0) for (x0, y0), (x1, y1), w in (
        ((0.25, 0.30), (0.160, 0.000), 0.050), ((0.33, 0.26), (0.262, -0.040), 0.055), ((0.43, 0.32), (0.400, 0.040), 0.050),
        ((0.52, 0.48), (0.508, 0.200), 0.050), ((-0.36, 0.86), (-0.380, 0.600), 0.050), ((-0.47, 0.88), (-0.500, 0.640), 0.050),
        ((-0.25, 0.82), (-0.270, 0.640), 0.045))]
    # light catching single strands inside the shadow: pointed at both ends, rooted in the lit area
    down = [leaf((x0, y0), (x1, y1), w, w0=0.003, at=0.30, p=1.0) for (x0, y0), (x1, y1), w in (
        ((0.25, -0.02), (0.350, 0.380), 0.040), ((0.33, 0.03), (0.455, 0.470), 0.036), ((0.44, 0.14), (0.545, 0.600), 0.032),
        ((-0.43, 0.62), (-0.448, 0.960), 0.040), ((-0.53, 0.66), (-0.552, 0.930), 0.034))]
    return union(base, *up) - union(*down)

def ring_highlight():
    """'angel ring' on the crown (book p.112: a crown that sits on the head): long comet-shaped strands
    that follow the hair flow, bold on the lit side and shorter/dimmer on the far side."""
    rng = np.random.default_rng(5)
    arc = lambda x: 0.135 + 0.17 * (x / 0.5) ** 2
    lit, far = [], []
    for x in np.linspace(-0.46, 0.40, 12):
        x += rng.uniform(-0.012, 0.012)
        f = float(np.clip(x / 0.40, 0, 1))                       # 0 on the lit side -> 1 on the far side
        ln_ = rng.uniform(0.17, 0.27) * (1 - 0.40 * f)
        w = rng.uniform(0.030, 0.044) * (1 - 0.35 * f)
        y = arc(x) + rng.uniform(-0.012, 0.012)
        fx, fy = (x - PART[0]) * 0.55, 1.0; n = math.hypot(fx, fy); fx, fy = fx / n, fy / n
        top = (x - fx * ln_ * 0.38, y - fy * ln_ * 0.38); tip = (x + fx * ln_ * 0.62, y + fy * ln_ * 0.62)
        (lit if f < 0.5 else far).append(leaf(top, tip, w, at=0.28, p=1.0, bend=0.006))
        if rng.random() < 0.5:                                    # thin companion strand
            off = 0.028 * (1 if rng.random() < 0.5 else -1)
            t2 = (top[0] + off, top[1] + 0.02); p2 = (tip[0] + off * 1.2, tip[1] - ln_ * 0.30)
            (lit if f < 0.5 else far).append(leaf(t2, p2, w * 0.45, at=0.3, p=1.0))
    band = shape(Hs([(x, arc(x) - 0.050) for x in np.linspace(-0.50, 0.44, 14)] +
                    [(x, arc(x) + 0.004) for x in np.linspace(0.44, -0.50, 14)])).blur(7)
    return band, union(*lit), union(*far)

def draw_front_hair(L, mode):
    u = HU
    pieces = hair_pieces()
    shade = SHADE(mode)
    bang_ms = [p[1] for p in pieces if p[0] in ('bang', 'wisp')]
    band, hl_lit, hl_far = ring_highlight() if shade else (None, None, None)
    sh_mask = hair_shadow() if shade else None
    main_path = None
    for kind, m, a, b, lf in pieces:
        L.paint(m, FILL(mode, C['hair']))
        if shade:
            if kind == 'main':
                L.paint(sh_mask & m, C['hair_sh'], 0.95, mode='multiply')
                cast = union(*[x.shift(0.012 * u, 0.026 * u) for x in bang_ms]).blur(1.0)   # bangs cast a hard shadow
                L.paint(cast & m, C['hair_sh'], 0.85, mode='multiply')
            else:
                # ribbon shading: the edge away from the light is in shadow, growing from a point near the root
                n = len(a); k = int(n * 0.15)
                s = np.linspace(0, 1, n - k)[:, None]
                far_side = float(np.mean(a[:, 0])) > Hd(0.18, 0.5)[0]
                cover = 0.80 if (kind in ('side', 'flick', 'lowwisp') and far_side) else 0.50
                frac = cover * np.clip(s / 0.55, 0, 1) ** 0.9
                inner = b[k:] + (a[k:] - b[k:]) * frac
                wedge = poly(np.vstack([inner, b[k:][::-1]])).blur(0.8)
                L.paint(wedge & m, C['hair_sh'], 0.9, mode='multiply')
                def tip_shadow(k1, k2):
                    """lower part of a clump in shadow, with a diagonal boundary that runs along the strand"""
                    return poly(np.vstack([b[int(n * k1):], a[int(n * k2):][::-1]])).blur(0.8)
                if kind in ('side', 'flick', 'lowwisp'):
                    if far_side:
                        L.paint((sh_mask & m), C['hair_sh'], 0.6, mode='multiply')
                    else:
                        L.paint(tip_shadow(0.58, 0.86) & m, C['hair_sh'], 0.7, mode='multiply')
                if kind == 'bang':                       # bang tips turn under, away from the light
                    L.paint(tip_shadow(0.70, 0.92) & m, C['hair_sh'], 0.6, mode='multiply')
            L.paint(band & m, C['hair_hl'], 0.30)
            L.paint(hl_lit & m, C['hair_hl'], 0.95)
            L.paint(hl_far & m, C['hair_hl'], 0.55)
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
    r0 = len(MAIN_R); l0 = 1 + len(MAIN_R) + len(MAIN_RB) + len(HAIRLINE) + 4
    L.paint(stroke(main_path.seg(r0, r0 + 6), 2.6, taper=(0.1, 0.3), floor=0.1), C['hair_ln'])
    L.paint(stroke(main_path.seg(l0, l0 + 6), 2.6, taper=(0.3, 0.1), floor=0.1), C['hair_ln'])
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
