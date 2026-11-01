"""Blonde anime girl, light-brown eyes, seen from below: chin lifted, head turned 16 degrees,
eyes looking down at the camera.

Built on a 3D head (cranium sphere + tapered jaw slices) that is rotated by yaw / pitch / roll
and rendered through a depth buffer. The hair is a thicker shell on the same sphere (scalp +
bangs clumps) plus a curtain of long hair hanging from the hairline in world space. Per-pixel
surface normals give cel shading from one light, and the face features are painted as decals in
the head's own coordinates, so they wrap around the head with the turn.
"""
import math
import numpy as np
from PIL import Image, ImageDraw

YAW, PITCH, ROLL = 12.0, 32.0, -7.0   # deg: slight turn, chin lifted high, a little tilt (like the reference)
R_HEAD, R_HAIR = 21.0, 23.6
CAM_D = 120.0                         # camera distance in head units (perspective strength)
SS = 2                                # render at 2x, then box-downsample
W1, H1 = 320, 420
W, H = W1 * SS, H1 * SS
U = 4.0 * SS                          # pixels per head unit
CX, CY = 160 * SS, 142 * SS
DISPLAY = 3

FACE, HAIR, BANGS, CURTAIN = 1, 5, 6, 7

# ------------------------------------------------------------------ palette
BG_TOP, BG_BOT = (206, 224, 242), (248, 238, 230)
LINE = (96, 60, 52)
LINE_SOFT = (150, 104, 80)
SKIN = (254, 232, 220)
SKIN_SH = (240, 198, 192)
SKIN_SH2 = (214, 160, 162)
BLUSH = (250, 190, 190)
LIP = (214, 128, 128)
LIP_HI = (255, 224, 218)
NOSE_SH = (226, 176, 172)
HAIR_BASE = (240, 206, 128)
HAIR_SH = (206, 160, 92)
HAIR_DEEP = (160, 114, 66)
HAIR_HI = (252, 236, 186)
HAIR_SHINE = (255, 251, 232)
HAIR_LINE = (138, 92, 54)
BROW = (196, 146, 84)
SCLERA = (252, 250, 252)
SCLERA_SH = (218, 212, 226)
IRIS = [(74, 44, 30), (96, 58, 34), (128, 80, 42), (160, 108, 54), (190, 136, 66), (214, 162, 82), (232, 186, 100)]
PUPIL = (40, 24, 20)
LASH = (62, 36, 34)
LASH_LOW = (160, 104, 96)
TOP = (70, 76, 104)
TOP_SH = (48, 52, 76)
TOP_HI = (96, 104, 134)
TOP_LINE = (34, 36, 54)


def unit(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def rot_matrix():
    a, p, r = map(math.radians, (YAW, PITCH, ROLL))
    Ry = np.array([[math.cos(a), 0, -math.sin(a)], [0, 1, 0], [math.sin(a), 0, math.cos(a)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    Rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return Rz @ Rx @ Ry


R = rot_matrix()


def to_world(P):
    return P @ R.T


def project(Pw):
    """World point(s) -> screen pixels (perspective: nearer things look bigger)."""
    k = CAM_D / (CAM_D - Pw[..., 2])
    return np.stack([CX + U * Pw[..., 0] * k, CY + U * Pw[..., 1] * k], -1)


# ------------------------------------------------------------------ head surface (head space)
# slices below the sphere centre: half-width A, front depth F, back depth G, by height Y
_Y = [0, 8, 14, 20, 25, 29.5, 32, 33.5]
_A = [21, 20.8, 19.8, 17.8, 14.2, 9.5, 4.8, 0.0]
_F = [21, 21, 20.5, 20, 19.5, 18.5, 17.6, 17.0]
_G = [21, 19.4, 12, 6, 0, -8, -11.5, -13]


def slice_params(y):
    y = np.asarray(y, float)
    sph = y < 0
    r = np.sqrt(np.maximum(0.0, R_HEAD ** 2 - y ** 2))
    a = np.where(sph, r, np.interp(y, _Y, _A))
    f = np.where(sph, r, np.interp(y, _Y, _F))
    g = np.where(sph, r, np.interp(y, _Y, _G))
    return a, f, g


def head_point(y, t):
    a, f, g = slice_params(y)
    x = a * np.cos(t)
    z = (f - g) / 2 + (f + g) / 2 * np.sin(t)
    return np.stack([x, np.broadcast_to(y, x.shape), z], -1)


def grid_normals(P, centre):
    """Outward normals of a parametric surface grid P[i, j, 3]."""
    d0 = np.gradient(P, axis=0)
    d1 = np.gradient(P, axis=1)
    n = unit(np.cross(d1, d0))
    flip = ((P - centre) * n).sum(-1) < 0
    n[flip] *= -1
    return n


def head_surface(ny=460, nt=900):
    y = np.linspace(-R_HEAD + 0.02, 33.5, ny)
    t = np.linspace(0, 2 * math.pi, nt, endpoint=False)
    Y, T = np.meshgrid(y, t, indexing="ij")
    P = head_point(Y, T)
    a, f, g = slice_params(Y)
    centre = np.stack([np.zeros_like(Y), Y, (f - g) / 2], -1)
    return P, grid_normals(P, centre)


# ------------------------------------------------------------------ depth buffer
depth = np.full((H, W), -1e9)
mat = np.zeros((H, W), np.int8)
nrm = np.zeros((H, W, 3))
hs = np.zeros((H, W, 3))          # head-space position (or hair-curtain coordinates)


def splat(Pw, Nw, m, Ph=None):
    Pw = Pw.reshape(-1, 3)
    Nw = Nw.reshape(-1, 3)
    Ph = Pw if Ph is None else Ph.reshape(-1, 3)
    S = project(Pw)
    z = Pw[:, 2]
    cells = []
    for ox in (0, 1):
        for oy in (0, 1):
            px = np.floor(S[:, 0]).astype(int) + ox
            py = np.floor(S[:, 1]).astype(int) + oy
            ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
            np.maximum.at(depth, (py[ok], px[ok]), z[ok])
            cells.append((px, py, ok))
    for px, py, ok in cells:
        win = ok.copy()
        win[ok] = z[ok] >= depth[py[ok], px[ok]] - 1e-7
        mat[py[win], px[win]] = m
        nrm[py[win], px[win]] = Nw[win]
        hs[py[win], px[win]] = Ph[win]


def splat_head_space(P, N, m):
    splat(to_world(P), N @ R.T, m, P)


def fill_holes(rounds=4):
    """Pixels left empty between splats take the attributes of their nearest covered neighbour."""
    for _ in range(rounds):
        nb = [(shift(mat, dx, dy), shift(depth, dx, dy), dx, dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))]
        count = sum((m > 0) for m, _, _, _ in nb)
        hole = (mat == 0) & (count >= 2)
        best = np.full((H, W), -1e9)
        pick = np.full((H, W), -1)
        for i, (m, d, dx, dy) in enumerate(nb):
            better = hole & (m > 0) & (d > best)
            best[better] = d[better]
            pick[better] = i
        for i, (m, d, dx, dy) in enumerate(nb):
            sel = pick == i
            mat[sel] = m[sel]
            depth[sel] = d[sel]
            nrm[sel] = shift(nrm, dx, dy)[sel]
            hs[sel] = shift(hs, dx, dy)[sel]


def shift(M, dx, dy):
    out = np.zeros_like(M)
    h, w = M.shape[:2]
    ys, yd = slice(max(0, dy), min(h, h + dy)), slice(max(0, -dy), min(h, h - dy))
    xs, xd = slice(max(0, dx), min(w, w + dx)), slice(max(0, -dx), min(w, w - dx))
    out[ys, xs] = M[yd, xd]
    return out


def erode(M, n=1):
    for _ in range(n):
        M = M & shift(M, 1, 0) & shift(M, -1, 0) & shift(M, 0, 1) & shift(M, 0, -1)
    return M


def dilate(M, n=1):
    for _ in range(n):
        M = M | shift(M, 1, 0) | shift(M, -1, 0) | shift(M, 0, 1) | shift(M, 0, -1)
    return M




# ------------------------------------------------------------------ hair
PART = math.radians(-14)          # the hair parts a little left of centre (head space)


def hairline_theta(phi):
    """Polar angle (from the top of the head) of the hairline, by longitude phi (0 = front)."""
    return np.radians(70 + 48 * (1 - np.cos(phi)) / 2)


def shell_point(theta, phi, r=R_HAIR):
    return np.stack([r * np.sin(theta) * np.sin(phi), -r * np.cos(theta), r * np.sin(theta) * np.cos(phi)], -1)


def scalp_surface():
    th = np.linspace(0.01, math.radians(128), 560)
    ph = np.linspace(-math.pi, math.pi, 1200, endpoint=False)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    P = shell_point(TH, PH)
    keep = TH <= hairline_theta(PH) + math.radians(3)
    return P[keep], unit(P[keep])


# bangs: wide soft clumps of different lengths (centre longitude, half width, tip polar angle), in degrees
CLUMPS = [(-56, 12, 93), (-36, 11, 99), (-17, 10, 96), (2, 12, 100), (22, 10, 95), (42, 12, 99), (60, 11, 93),
          (-49, 2.0, 124), (-31, 1.6, 113), (47, 1.9, 120)]     # thin strands crossing the cheeks


def bangs_surface():
    pts = []
    for cdeg, hw, tip in CLUMPS:
        c, hw, tip = math.radians(cdeg), math.radians(hw), math.radians(tip)
        lean = math.radians(4.0) * (1 if c > PART else -1)
        s = np.linspace(0, 1, 360)[:, None]
        q = np.linspace(-1, 1, 130)[None, :]
        th0 = hairline_theta(c) - math.radians(8)
        # full width down to 55 percent of the length, then a gently curved taper to the tip
        wprof = np.where(s < 0.5, 1.0 - 0.1 * s, 0.95 * np.sqrt(np.clip(1 - ((s - 0.5) / 0.5) ** 2, 0, 1)))
        TH = th0 + (tip - th0) * s + 0 * q
        PH = c + lean * s + hw * wprof * q
        pts.append(shell_point(TH, PH, R_HAIR + (1.3 if hw < math.radians(3) else 0.35)).reshape(-1, 3))
    P = np.concatenate(pts)
    return P, unit(P)


def curtain_surface():
    """Long hair hanging from the hairline: a smooth, slightly puffed curtain around the back and
    sides, with side locks that reach down past the jaw and frame the face on both sides."""
    ph = np.linspace(math.radians(50), math.radians(310), 900)
    th = hairline_theta(ph) - math.radians(7)
    A = to_world(shell_point(th, ph))                        # attachment ring on the scalp
    a = np.degrees(np.abs(np.arctan2(np.sin(ph), np.cos(ph))))   # angle from the front, 0..180
    front = np.clip((a - 50) / 28, 0, 1)                     # 0 at the front edge, 1 by 78 degrees
    ends = 44 + 19 * front ** 0.8 + 3.5 * np.sin(5 * ph + 1.0) * front
    rows = []
    for k in range(640):
        s = k / 639
        y = A[:, 1] + (ends - A[:, 1]) * s
        drop = y - A[:, 1]
        base_r = np.hypot(A[:, 0], A[:, 2]) + 1.4 * (1 - front)     # side locks sit a touch in front of the cheeks
        puff = 2.2 * np.sin(math.pi * np.clip(drop / 30, 0, 1))      # the hair puffs out, then falls back in
        wave = 1.4 * np.sin(y / 5.5 + 1.6 * ph) * np.clip(drop / 12, 0, 1)
        r = base_r + 0.07 * drop + puff + wave
        d = unit(np.stack([A[:, 0], np.zeros_like(y), A[:, 2]], -1))
        P = np.stack([d[:, 0] * r, y, d[:, 2] * r], -1)
        rows.append((P, d, np.stack([ph, y, drop], -1)))
    P = np.concatenate([r_[0] for r_ in rows])
    N = unit(np.concatenate([r_[1] for r_ in rows]) + np.array([0, -0.2, 0]))
    C = np.concatenate([r_[2] for r_ in rows])
    return P, N, C


# ------------------------------------------------------------------ face features (decals)
EYE_X, EYE_Y = 10.5, 8.0           # eye centres: one eye width apart, on the eye line


def eye_frame(sign):
    a = slice_params(EYE_Y)[0]
    t = math.acos(np.clip(sign * EYE_X / float(a), -1, 1))
    y = np.array([EYE_Y])
    C = head_point(y, np.array([t]))[0]
    dy = head_point(y + 0.01, np.array([t]))[0] - head_point(y - 0.01, np.array([t]))[0]
    dt = head_point(y, np.array([t + 1e-3]))[0] - head_point(y, np.array([t - 1e-3]))[0]
    u = unit(dt if dt[0] > 0 else -dt)
    return C, u, unit(dy)


def tri(u, v, A, B, C):
    def side(p, q):
        return (q[0] - p[0]) * (v - p[1]) - (q[1] - p[1]) * (u - p[0])
    a, b, c = side(A, B), side(B, C), side(C, A)
    return ((a >= 0) & (b >= 0) & (c >= 0)) | ((a <= 0) & (b <= 0) & (c <= 0))


def ell(u, v, cu, cv, ru, rv):
    return ((u - cu) / ru) ** 2 + ((v - cv) / rv) ** 2 <= 1


def paint_eyes(img):
    """The eye design from the face study, in head units (1 unit = 2 of its 'eye units'):
    11 wide, 8.5 tall, tall iris, thick lash line heaviest at the outer corner, outer flick,
    crease, soft lower lash. u runs outward from the eye centre, v down. She looks down at the
    camera, so the irises sit low and slide a little toward the near side."""
    face = mat == FACE
    hairy = mat == BANGS
    sel = face | hairy
    Q = hs[sel]
    col = img[sel].copy()
    is_face = face[sel]
    for sign in (1, -1):
        C, uvec, vvec = eye_frame(sign)
        d = Q - C
        du, dv = d @ uvec, d @ vvec
        u = -du * sign                                    # negative = outer corner, like the study
        w, h = 5.6, 7.2
        uo, vo = -w + 0.15, 2.2
        k = np.clip((-w + 1.5 - u) / (-w + 1.5 - uo), 0, 1)
        top = 0.5 + (vo - 0.5) * k * k
        bot = np.where(u <= 0, vo + (h - vo) * np.sqrt(np.clip(1 - (u / uo) ** 2, 0, 1)),
                       h * (0.45 + 0.55 * np.sqrt(np.clip(1 - (u / w) ** 2, 0, 1))))
        inside = is_face & (u >= uo) & (u <= w) & (dv >= top) & (dv <= bot)
        lid = dv - top
        # iris (tall ellipse), low in the opening, nudged toward the near side (+x)
        icu, icv, irx, iry = -0.5, 2.3, 3.4, 4.4
        e = ((du - icu) / irx) ** 2 + ((dv - icv) / iry) ** 2
        c = np.where((lid < 1.6)[:, None], SCLERA_SH, SCLERA)
        kk = np.clip(((dv - (icv - iry)) / (2 * iry) * len(IRIS)).astype(int), 0, len(IRIS) - 1)
        c = np.where((e <= 1)[:, None], np.asarray(IRIS)[kk], c)
        c = np.where(((e > 0.8) & (e <= 1))[:, None], IRIS[0], c)
        c = np.where(((lid < 2.0) & (e <= 1))[:, None], IRIS[0], c)
        pupil = ell(du, dv, icu, icv - 0.7, 1.55, 2.3)
        c = np.where(pupil[:, None], PUPIL, c)
        hl1 = ell(du, dv, icu - 1.65, icv - 2.0, 0.72, 0.82)       # light from the upper left
        hl2 = ell(du, dv, icu + 1.3, icv + 2.1, 0.48, 0.32)
        c = np.where(hl1[:, None], (255, 255, 255), c)
        c = np.where((hl2 & (e <= 1))[:, None], (250, 232, 206), c)
        col[inside] = c[inside]
        # thick upper lash, heaviest at the outer corner, tapering at the inner end
        th = 1.05 + 0.75 * np.clip((1.0 - u) / 4.0, 0, 1)
        th = np.where(u > w - 1.5, th * np.clip((w + 0.6 - u) / 2.1, 0.3, 1), th)
        lash = (u >= uo - 0.6) & (u <= w + 0.6) & (dv >= top - th) & (dv <= top + 0.15)
        lash |= tri(u, dv, (uo - 0.7, -0.5), (uo - 2.4, 1.45), (uo - 2.05, 1.8)) \
            | tri(u, dv, (uo - 0.85, 1.5), (uo - 0.2, 2.35), (uo + 0.3, 0.75))
        col[is_face & lash] = LASH
        # lower lid: a soft line, plus a dark tapered stroke closing the outer corner
        low = (u >= uo + 1.25) & (u <= 0.8 * w) & (dv >= bot + 0.2) & (dv <= bot + 0.65)
        col[is_face & low] = LASH_LOW
        tt = np.clip((u - uo) / 1.8, 0, 1)
        thc = 0.1 + 0.4 * (1 - tt)
        corner = (u >= uo) & (u <= uo + 1.8) & (np.abs(dv - bot - 0.15) <= thc)
        col[is_face & corner] = LASH
        # crease above the lid
        # brow: thin, a touch lower at the outer end; shows faintly through the bangs
        vb = -6.3 + 0.5 * np.clip((-w + 2.5 - u) / 1.5, 0, 1)
        qb = (u - 0.5) / 6.0
        tb = 0.28 + 0.16 * (1 - np.clip(qb * qb, 0, 1))
        brow = (np.abs(qb) <= 1) & (np.abs(dv - vb) <= tb) & ((d * d).sum(-1) < 90)
        col[is_face & brow] = BROW
        bb = ~is_face & brow
        col[bb] = col[bb] * 0.55 + np.asarray(BROW) * 0.45
        # blush: low on the cheek, under the outer half of the eye, with three little strokes
        bu = u + 2.5
        bl = ell(bu, dv, 0, 14.0, 3.5, 1.3)
        col[is_face & bl] = BLUSH
        for kx in (-1.6, 0, 1.6):
            strk = (np.abs((bu - kx) + 0.6 * (dv - 14.0)) < 0.22) & (np.abs(dv - 14.0) < 0.7)
            col[is_face & bl & strk] = BLUSH_LINE
    img[sel] = col


BLUSH_LINE = (238, 160, 166)


def surface_xy(x, y):
    a = float(slice_params(y)[0])
    t = math.acos(np.clip(x / a, -1, 1))
    return head_point(np.array([float(y)]), np.array([t]))[0]


def line_cov(pts2, width, closed=False):
    im = Image.new("L", (W, H), 0)
    q = [tuple(p) for p in pts2]
    if closed:
        q.append(q[0])
    ImageDraw.Draw(im).line(q, fill=255, width=max(1, round(width)), joint="curve")
    return np.asarray(im, float) / 255


def poly_cov(pts2):
    im = Image.new("L", (W, H), 0)
    ImageDraw.Draw(im).polygon([tuple(p) for p in pts2], fill=255)
    return np.asarray(im, float) / 255


def blend(img, cov, color, where=None):
    if where is not None:
        cov = cov * where
    img[:] = img * (1 - cov[..., None]) + np.asarray(color, float) * cov[..., None]


def paint_nose_mouth(img):
    face = (mat == FACE).astype(float)
    nose = np.array([[0.5, 19.6, 23.0], [-1.1, 22.6, 24.6], [-0.7, 24.2, 24.9], [0.6, 24.8, 24.0]])
    blend(img, line_cov(project(to_world(nose)), 1.1 * SS), LINE_SOFT, face)
    sh = np.array([[-0.8, 24.4, 24.3], [1.9, 25.1, 23.5], [-0.1, 25.9, 22.8]])
    blend(img, poly_cov(project(to_world(sh))), NOSE_SH, face)
    xs = np.linspace(-3.0, 3.0, 25)
    ys = 29.0 + 0.45 * (1 - (xs / 3.0) ** 2)
    m = np.array([surface_xy(x, y) + [0, 0, 0.4] for x, y in zip(xs, ys)])
    blend(img, line_cov(project(to_world(m)), 1.2 * SS), LIP, face)
    for x in (-3.0, 3.0):
        p0 = surface_xy(x, 29.2) + [0, 0, 0.4]
        blend(img, line_cov(project(to_world(np.array([p0, p0 + [0, -0.6, 0]]))), 1.2 * SS), LIP, face)
    gap = np.array([surface_xy(x, y) + [0, 0, 0.4] for x, y in ((-1.7, 29.5), (1.7, 29.5), (1.1, 30.7), (-1.1, 30.7))])
    blend(img, poly_cov(project(to_world(gap))), (150, 78, 88), face)
    lo = np.array([surface_xy(x, 31.4 + 0.25 * (1 - (x / 2.0) ** 2)) + [0, 0, 0.5] for x in np.linspace(-2.0, 2.0, 9)])
    blend(img, line_cov(project(to_world(lo)), 0.9 * SS) * 0.7, LIP_HI, face)


# ------------------------------------------------------------------ body (flat, screen space)
def body_shapes():
    """Neck and a simple dark top, placed under the projected chin. Returns 2x-pixel polygons."""
    chin = project(to_world(np.array([0.0, 33.5, 17.0])))
    cx = 0.55 * CX + 0.45 * chin[0]
    yb = CY + 37.5 * U                         # neck base / collar line, fixed under the head
    neck = [(cx - 6.0 * U, chin[1] - 12 * U), (cx + 6.2 * U, chin[1] - 12 * U),
            (cx + 5.8 * U, yb + 2 * U), (cx - 5.8 * U, yb + 2 * U)]
    cxb = CX * 0.75 + cx * 0.25
    top = [(cxb - 42 * U, H + 10), (cxb - 40 * U, yb + 15 * U), (cxb - 28 * U, yb + 6.5 * U), (cxb - 14 * U, yb + 2.2 * U),
           (cxb - 7.5 * U, yb + 0.8 * U), (cxb, yb + 2.4 * U), (cxb + 7.5 * U, yb + 0.8 * U), (cxb + 14 * U, yb + 2.2 * U),
           (cxb + 28 * U, yb + 6.5 * U), (cxb + 40 * U, yb + 15 * U), (cxb + 42 * U, H + 10)]
    neckline = top[3:8]
    collar = [(cx - 7.6 * U, yb - 6.2 * U), (cx + 7.8 * U, yb - 7.0 * U), (cx + 7.3 * U, yb + 0.6 * U), (cx - 7.1 * U, yb + 1.4 * U)]
    return neck, top, neckline, collar, cxb, yb


def smooth2d(pts, n=10):
    pts = list(pts)
    out = []
    for i in range(len(pts) - 1):
        p0 = pts[max(i - 1, 0)]
        p1, p2 = pts[i], pts[i + 1]
        p3 = pts[min(i + 2, len(pts) - 1)]
        for k in range(n):
            t = k / n
            t2, t3 = t * t, t * t * t
            out.append(tuple(0.5 * (2 * p1[j] + (-p0[j] + p2[j]) * t + (2 * p0[j] - 5 * p1[j] + 4 * p2[j] - p3[j]) * t2
                                    + (-p0[j] + 3 * p1[j] - 3 * p2[j] + p3[j]) * t3) for j in range(2)))
    out.append(pts[-1])
    return out


# ------------------------------------------------------------------ render
LIGHT = unit([-0.42, -0.78, 0.46])      # from the viewer's upper left


def background():
    yy = np.linspace(0, 1, H)[:, None]
    bg = np.asarray(BG_TOP, float) * (1 - yy) + np.asarray(BG_BOT, float) * yy
    img = np.broadcast_to(bg[:, None, :], (H, W, 3)).copy()
    xx = np.arange(W)[None, :]
    g = np.exp(-(((xx - 0.1 * W) / (0.55 * W)) ** 2 + ((np.arange(H)[:, None] - 0.1 * H) / (0.55 * H)) ** 2))
    return img * (1 - 0.35 * g[..., None]) + 255 * 0.35 * g[..., None]


def hair_colours(img, ndl, M, bangs):
    base = np.where((ndl < 0.2)[..., None], HAIR_SH, HAIR_BASE)
    base = np.where((ndl < -0.2)[..., None], HAIR_DEEP, base)
    P = hs
    theta = np.arccos(np.clip(-P[..., 1] / R_HAIR, -1, 1))
    phi = np.arctan2(P[..., 0], P[..., 2])
    rel = np.degrees(phi - PART)
    # a few soft strand lines flowing from the part, fading out near the crown and the tips
    gw = np.sin(np.radians(rel) * 360 / 17.0) + 0.35 * np.sin(np.radians(rel) * 360 / 9.5 + 1.0)
    groove = (gw > 1.26) & (theta > math.radians(36)) & (theta < math.radians(96))
    base = np.where((groove & (ndl >= 0.2))[..., None], HAIR_SH, base)
    # angel ring on the top of the head and a second one across the bangs, both cut by the strands
    ring_c = np.where(bangs, math.radians(80) + math.radians(2.5) * np.sin(3 * phi), math.radians(50))
    ring = (np.abs(theta - ring_c) < math.radians(4.4)) & (np.abs(phi) < math.radians(82)) & (ndl > 0.15) & ~groove
    base = np.where(ring[..., None], HAIR_HI, base)
    core = (np.abs(theta - ring_c) < math.radians(1.3)) & ring
    base = np.where(core[..., None], HAIR_SHINE, base)
    tipdark = bangs & (theta > math.radians(94)) & (ndl >= 0.2)
    base = np.where(tipdark[..., None], HAIR_SH, base)
    img[M] = base[M]


def curtain_colours(img, ndl, M):
    ph, y, drop = hs[..., 0], hs[..., 1], hs[..., 2]
    base = np.where((ndl < 0.16)[..., None], HAIR_SH, HAIR_BASE)
    base = np.where((ndl < -0.22)[..., None], HAIR_DEEP, base)
    streak = (np.sin(4 * ph + 1.0) > 0.88) & (ndl > 0.3) & (drop < 24) & (drop > 3)
    base = np.where(streak[..., None], HAIR_HI, base)
    groove = np.sin(6 * ph + 2.0) > 0.965
    base = np.where((groove & (ndl >= 0.16))[..., None], HAIR_SH, base)
    low = drop > 34 + 3 * np.sin(5 * ph)
    base = np.where((low & (ndl >= 0.16) & ~streak)[..., None], HAIR_SH, base)
    img[M] = base[M]


def render():
    P, N = head_surface()
    splat_head_space(P, N, FACE)
    P, N = scalp_surface()
    splat_head_space(P, N, HAIR)
    P, N = bangs_surface()
    splat_head_space(P, N, BANGS)
    P, N, C = curtain_surface()
    splat(P, N, CURTAIN, C)
    fill_holes()

    ndl = (nrm * LIGHT).sum(-1)
    face = mat == FACE
    hair = (mat == HAIR) | (mat == BANGS)
    curtain = mat == CURTAIN
    fig = mat > 0
    down = nrm[..., 1]

    # ---- the figure layer (head + hair)
    img = background()
    sk = np.where((ndl < 0.3)[..., None], SKIN_SH, SKIN)
    sk = np.where((down > 0.55)[..., None], SKIN_SH, sk)           # the underside of the jaw, lightly shaded
    img[face] = sk[face]
    hairline_sh = face & dilate(hair, 2 * SS) & ~shift(face, 0, -2 * SS)
    img[hairline_sh] = SKIN_SH
    hair_colours(img, ndl, hair, mat == BANGS)
    curtain_colours(img, ndl, curtain)
    bsh = shift(mat == BANGS, 2 * SS, 3 * SS) & face              # bangs shadow on the forehead
    img[bsh] = SKIN_SH
    paint_eyes(img)
    paint_nose_mouth(img)

    # ---- body: drawn flat, behind the hair that hangs in front of it
    neck, top, neckline, collar, cxb, yb = body_shapes()
    neck_m = poly_cov(neck) >= 0.5
    top_m = poly_cov(smooth2d(top)) >= 0.5
    col_m = poly_cov(collar) >= 0.5
    body_z = np.where(neck_m | col_m, 2.0, -3.0)
    body = neck_m | top_m | col_m
    in_front = fig & (depth > body_z)
    show_body = body & ~in_front
    base = background()
    nk = np.full((H, W, 3), SKIN, float)
    chin_sh = dilate(face, 8 * SS) & ~face
    nk[chin_sh] = SKIN_SH2
    xcol = np.broadcast_to(np.arange(W)[None, :], (H, W))
    nk[~chin_sh & (xcol > cxb + 2.5 * U)] = SKIN_SH
    tp = np.full((H, W, 3), TOP, float)
    tp[xcol > cxb + 6 * U] = TOP_SH
    lit = poly_cov([(cxb - 40 * U, yb + 15 * U), (cxb - 28 * U, yb + 6.5 * U), (cxb - 14 * U, yb + 2.2 * U),
                    (cxb - 13 * U, yb + 7 * U), (cxb - 26 * U, yb + 12 * U), (cxb - 36 * U, yb + 19 * U)]) >= 0.5
    tp[lit] = TOP_HI
    base[neck_m & show_body] = nk[neck_m & show_body]
    base[top_m & show_body] = tp[top_m & show_body]
    cl = np.full((H, W, 3), (238, 232, 226), float)
    cl[xcol > cxb + 5 * U] = (206, 196, 200)
    base[col_m & show_body] = cl[col_m & show_body]
    band = poly_cov([collar[0], collar[1], (collar[1][0] - 0.3 * U, collar[1][1] + 1.6 * U), (collar[0][0] + 0.3 * U, collar[0][1] + 1.6 * U)]) >= 0.5
    base[band & show_body] = (54, 46, 60)
    blend(base, line_cov(collar, 1.1 * SS, closed=True), TOP_LINE, (show_body & dilate(col_m, 2)).astype(float))
    # neckline band and outlines of the body
    nl = smooth2d(neckline)
    blend(base, line_cov(nl, 2.2 * SS), TOP_HI, show_body & top_m)
    blend(base, line_cov(nl, 0.9 * SS), TOP_LINE, show_body & top_m)
    outline = line_cov(smooth2d(top), 1.2 * SS)
    blend(base, outline, TOP_LINE, show_body.astype(float))
    blend(base, line_cov([neck[0], neck[3]], 1.1 * SS) + line_cov([neck[1], neck[2]], 1.1 * SS), LINE_SOFT,
          (show_body & neck_m).astype(float))
    # composite the figure over the body where it is in front
    vis = fig & (in_front | ~body)
    base[vis] = img[vis]
    img = base

    # ---- outlines: silhouette of the figure, skin / hair edges, a soft line between hair layers
    def edge(A, B):
        return (A & ~erode(A) & dilate(B)) | (B & ~erode(B) & dilate(A))
    sil = vis & ~erode(vis, SS)
    lines = sil | (edge(face, hair | curtain) & vis)
    hl = edge(hair, curtain) & vis & ~lines
    img[hl] = HAIR_LINE
    hair_edge = lines & (hair | curtain)
    img[lines] = LINE
    img[hair_edge] = HAIR_LINE

    out = np.clip(img, 0, 255).reshape(H1, SS, W1, SS, 3).mean((1, 3))
    return Image.fromarray(out.astype(np.uint8))


def guide_image(im):
    """The book's construction lines over the finished portrait."""
    from PIL import ImageFont
    big = im.resize((W1 * DISPLAY, H1 * DISPLAY), Image.NEAREST)
    out = Image.blend(big, Image.new("RGB", big.size, (255, 255, 255)), 0.5)
    d = ImageDraw.Draw(out)
    k = DISPLAY / SS
    RED, BLUE = (225, 60, 60), (50, 110, 235)

    def line3(pts, col, w=4):
        q = project(to_world(np.asarray(pts, float))) * k
        d.line([tuple(v) for v in q], fill=col, width=w)
    th = np.linspace(-math.pi / 2, math.pi / 2, 80)
    c = project(to_world(np.zeros(3))) * k
    rr = R_HEAD * U * k
    d.ellipse((c[0] - rr, c[1] - rr, c[0] + rr, c[1] + rr), outline=BLUE, width=4)
    line3([(0, R_HEAD * math.sin(u), R_HEAD * math.cos(u)) for u in th], RED)
    r = math.sqrt(R_HEAD ** 2 - EYE_Y ** 2)
    line3([(r * math.sin(u), EYE_Y, r * math.cos(u)) for u in th], BLUE)
    line3([(0, EYE_Y, 22), (0, 24, 25), (0, 29.2, 22.5), (0, 33.5, 17)], RED)
    line3([(-19, EYE_Y, 9), (0, 33.5, 17), (19, EYE_Y, 9)], BLUE)
    f = ImageFont.load_default(size=30)
    d.text((24, 20), f"turn {YAW:g}°  |  chin up {PITCH:g}° (camera below)", fill=(90, 50, 50), font=f)
    return out


if __name__ == "__main__":
    im = render()
    im.save("blonde_girl_raw.png")
    im.resize((W1 * DISPLAY, H1 * DISPLAY), Image.NEAREST).save("blonde_girl.png")
    guide_image(im).save("blonde_girl_guide.png")
    print("saved", im.size)
