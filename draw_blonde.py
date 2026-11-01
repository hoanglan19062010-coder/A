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

YAW, PITCH, ROLL = 16.0, 18.0, -3.0   # deg: turn toward the viewer's left, chin up, slight tilt
R_HEAD, R_HAIR = 21.0, 23.6
CAM_D = 120.0                         # camera distance in head units (perspective strength)
SS = 2                                # render at 2x, then box-downsample
W1, H1 = 320, 420
W, H = W1 * SS, H1 * SS
U = 4.0 * SS                          # pixels per head unit
CX, CY = 160 * SS, 142 * SS
DISPLAY = 3

FACE, NECK, TORSO, HAIR, BANGS, CURTAIN = 1, 2, 3, 5, 6, 7

# ------------------------------------------------------------------ palette
BG_TOP, BG_BOT = (206, 224, 242), (248, 238, 230)
LINE = (82, 52, 46)
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
TOP = (132, 162, 192)
TOP_SH = (96, 122, 156)
TOP_LINE = (62, 80, 108)


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


def neck_surface():
    y = np.linspace(19, 64, 260)
    t = np.linspace(0, 2 * math.pi, 420, endpoint=False)
    Y, T = np.meshgrid(y, t, indexing="ij")
    rx = 5.9 + 1.0 * np.clip((Y - 40) / 18, 0, 1)
    P = np.stack([rx * np.cos(T), Y, -4 + 6.0 * np.sin(T)], -1)
    centre = np.stack([np.zeros_like(Y), Y, np.full_like(Y, -4.0)], -1)
    return P, grid_normals(P, centre)


def torso_surface():
    y = np.linspace(47, 118, 2200)
    t = np.linspace(0, 2 * math.pi, 900, endpoint=False)
    Y, T = np.meshgrid(y, t, indexing="ij")
    rx = 8.5 + 27.5 * np.clip((Y - 47) / 13, 0, 1) ** 0.6
    rz = 14.0 + 2.0 * np.clip((Y - 47) / 13, 0, 1)
    P = np.stack([rx * np.cos(T), Y, -11 + rz * np.sin(T)], -1)
    centre = np.stack([np.zeros_like(Y), Y, np.full_like(Y, -11.0)], -1)
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
PART = math.radians(-12)          # the hair parts slightly left of centre (head space)


def hairline_theta(phi):
    """Polar angle (from the top of the head) of the hairline, by longitude phi (0 = front)."""
    return np.radians(70 + 48 * (1 - np.cos(phi)) / 2)


def shell_point(theta, phi, r=R_HAIR):
    return np.stack([r * np.sin(theta) * np.sin(phi), -r * np.cos(theta), r * np.sin(theta) * np.cos(phi)], -1)


def scalp_surface():
    th = np.linspace(0.01, math.radians(125), 520)
    ph = np.linspace(-math.pi, math.pi, 1100, endpoint=False)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    P = shell_point(TH, PH)
    keep = TH <= hairline_theta(PH)
    return P[keep], unit(P[keep])


# bangs clumps: (centre longitude deg, half width deg, tip polar angle deg)
CLUMPS = [(-60, 7, 95), (-46, 9, 104), (-31, 6, 99), (-18, 9, 107), (-5, 6, 101),
          (9, 10, 109), (24, 6, 100), (38, 9, 106), (53, 7, 98), (64, 5, 94)]


def bangs_surface():
    pts = []
    for cdeg, hw, tip in CLUMPS:
        c, hw, tip = math.radians(cdeg), math.radians(hw), math.radians(tip)
        lean = math.radians(5.0) * (1 if c > PART else -1)
        s = np.linspace(0, 1, 340)[:, None]
        q = np.linspace(-1, 1, 120)[None, :]
        th0 = hairline_theta(c) - math.radians(6)
        TH = th0 + (tip - th0) * s + 0 * q
        PH = c + lean * s + hw * 1.7 * np.minimum(1.0, (1 - s) / 0.62) ** 0.7 * q
        pts.append(shell_point(TH, PH, R_HAIR + 0.35).reshape(-1, 3))
    P = np.concatenate(pts)
    return P, unit(P)


def curtain_surface():
    """Long hair hanging from the hairline at the sides and back, in world space."""
    ph = np.linspace(math.radians(62), math.radians(298), 760)
    th = hairline_theta(ph) - math.radians(4)
    A = to_world(shell_point(th, ph))                       # attachment ring
    ends = 60 - 9 * np.exp(-((ph - math.radians(62)) / 0.42) ** 2) \
        - 9 * np.exp(-((ph - math.radians(298)) / 0.42) ** 2) + 3.0 * np.sin(4 * ph + 1.0)
    rows = []
    for k in range(560):
        s = k / 559
        y = A[:, 1] + (ends - A[:, 1]) * s
        drop = y - A[:, 1]
        base_r = np.hypot(A[:, 0], A[:, 2])
        r = base_r + 0.11 * drop + 1.3 * np.sin(y / 3.5 + 2 * ph) * np.clip(drop / 8, 0, 1)
        d = unit(np.stack([A[:, 0], np.zeros_like(y), A[:, 2]], -1))
        P = np.stack([d[:, 0] * r, y, d[:, 2] * r], -1)
        rows.append((P, d, np.stack([ph, y, r], -1)))
    P = np.concatenate([r_[0] for r_ in rows])
    N = unit(np.concatenate([r_[1] for r_ in rows]) + np.array([0, -0.25, 0]))
    C = np.concatenate([r_[2] for r_ in rows])
    return P, N, C


# ------------------------------------------------------------------ face features (decals)
def eye_frame(sign):
    y = 8.5
    a = slice_params(y)[0]
    t = math.acos(np.clip(sign * 9.6 / float(a), -1, 1))
    C = head_point(np.array([y]), np.array([t]))[0]
    dy = head_point(np.array([y + 0.01]), np.array([t]))[0] - head_point(np.array([y - 0.01]), np.array([t]))[0]
    dt = head_point(np.array([y]), np.array([t + 1e-3]))[0] - head_point(np.array([y]), np.array([t - 1e-3]))[0]
    u = unit(dt if dt[0] > 0 else -dt)
    return C, u, unit(dy)


def tri(u, v, A, B, C):
    def side(p, q):
        return (q[0] - p[0]) * (v - p[1]) - (q[1] - p[1]) * (u - p[0])
    a, b, c = side(A, B), side(B, C), side(C, A)
    return ((a >= 0) & (b >= 0) & (c >= 0)) | ((a <= 0) & (b <= 0) & (c <= 0))


def paint_eyes(img, ndl):
    face = mat == FACE
    skinlike = face | (mat == BANGS)
    Q = hs[skinlike]
    col = img[skinlike].copy()
    is_face = face[skinlike]
    for sign in (1, -1):
        C, u, v = eye_frame(sign)
        d = Q - C
        du, dv = d @ u, d @ v
        up = du * sign                                   # + toward the outer corner
        q = np.clip(up / 7.1, -1.2, 1.2)
        top = -4.0 + 3.3 * q * q - 0.5 * q               # lids lowered a little: she looks down at us
        bot = 4.4 - 5.0 * q * q - 0.5 * q
        inside = is_face & (np.abs(up) <= 7.1) & (dv >= top) & (dv <= bot)
        # iris: low in the opening (looking down) and toward the near side (looking at the camera)
        iu, iv, ir = 1.0, 1.4, 4.6
        e = ((du - iu) / ir) ** 2 + ((dv - iv) / ir) ** 2
        lid = dv - top
        c = np.where((lid < 1.6)[:, None], SCLERA_SH, SCLERA)
        k = np.clip(((dv - (iv - ir)) / (2 * ir) * len(IRIS)).astype(int), 0, len(IRIS) - 1)
        iris_c = np.asarray(IRIS)[k]
        c = np.where((e <= 1)[:, None], iris_c, c)
        c = np.where(((e > 0.8) & (e <= 1))[:, None], IRIS[0], c)
        c = np.where(((lid < 2.2) & (e <= 1))[:, None], IRIS[0], c)
        pupil = ((du - iu) / 1.5) ** 2 + ((dv - iv + 0.3) / 1.9) ** 2 <= 1
        c = np.where(pupil[:, None], PUPIL, c)
        hl1 = ((du - iu + 1.4) / 1.15) ** 2 + ((dv - iv + 1.5) / 1.25) ** 2 <= 1
        hl2 = ((du - iu - 1.5) / 0.7) ** 2 + ((dv - iv - 1.9) / 0.5) ** 2 <= 1
        c = np.where(hl1[:, None], (255, 255, 255), c)
        c = np.where((hl2 & (e <= 1))[:, None], (250, 226, 196), c)
        col[inside] = c[inside]
        # thick upper lash line, heavier toward the outer corner, with a flick
        th = 1.15 + 1.6 * np.clip((up + 1.5) / 5.5, 0, 1)
        lash = (np.abs(up) <= 7.35) & (dv >= top - th) & (dv <= top + 0.4)
        lash |= tri(up, dv, (6.2, -1.7), (9.4, -4.0), (5.3, -3.0))
        col[is_face & lash] = LASH
        # lower lash: a soft line on the outer half, a small dark wedge at the corner
        low = (up >= 0.3) & (up <= 6.5) & (dv >= bot + 0.15) & (dv <= bot + 0.75)
        col[is_face & low] = LASH_LOW
        wedge = (up >= 4.8) & (up <= 7.2) & (dv >= bot - 0.1) & (dv <= bot + 0.25 + (up - 4.8) * 0.55)
        col[is_face & wedge] = LASH
        # eyelid crease
        crease = (np.abs(up) <= 5.1) & (dv >= top - th - 1.35) & (dv <= top - th - 0.75)
        col[is_face & crease & ~lash] = SKIN_SH2
        # brow: thin arch, shows faintly through the bangs
        qb = (up - 1.4) / 7.6
        vb = -9.3 - 1.4 * (1 - qb * qb)
        tb = 0.42 + 0.32 * np.clip(1 - (up + 2.5) / 6.0, 0, 1)
        brow = (np.abs(qb) <= 1) & (np.abs(dv - vb) <= tb) & ((d * d).sum(-1) < 130)
        col[is_face & brow] = BROW
        bb = ~is_face & brow
        col[bb] = col[bb] * 0.5 + np.asarray(BROW) * 0.5
        # blush under the outer half of the eye
        bl = ((up - 4.0) / 3.6) ** 2 + ((dv - 8.2) / 1.7) ** 2 <= 1
        col[is_face & bl & ~low & ~wedge] = BLUSH
    img[skinlike] = col


def surface_xy(x, y):
    """Point on the face surface at head-space (x, y)."""
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
    # nose: drawn on the far side of the bridge, with a small shadow under the tip
    nose = np.array([[0.6, 12.5, 22.4], [-1.2, 16.4, 24.4], [-1.0, 19.2, 25.2], [0.5, 20.3, 24.3]])
    blend(img, line_cov(project(to_world(nose)), 1.2 * SS), LINE_SOFT, face)
    sh = np.array([[-1.0, 19.6, 24.6], [1.8, 20.8, 23.8], [-0.2, 21.6, 22.8]])
    blend(img, poly_cov(project(to_world(sh))), NOSE_SH, face)
    # mouth: a small closed smile
    xs = np.linspace(-4.3, 4.3, 25)
    ys = 26.4 + 1.1 * (1 - (xs / 4.3) ** 2)
    m = np.array([surface_xy(x, y) + [0, 0, 0.4] for x, y in zip(xs, ys)])
    blend(img, line_cov(project(to_world(m)), 1.25 * SS), LIP, face)
    for x in (-4.3, 4.3):
        p = surface_xy(x, 26.4) + [0, 0, 0.4]
        blend(img, line_cov(project(to_world(np.array([p, p + [0, -0.7, 0]]))), 1.3 * SS), LIP, face)
    lo = np.array([surface_xy(x, 28.4 + 0.3 * (1 - (x / 2.3) ** 2)) + [0, 0, 0.5] for x in np.linspace(-2.3, 2.3, 9)])
    blend(img, line_cov(project(to_world(lo)), 1.0 * SS) * 0.8, LIP_HI, face)


# ------------------------------------------------------------------ render
LIGHT = unit([-0.42, -0.78, 0.46])      # from the viewer's upper left


def background():
    yy = np.linspace(0, 1, H)[:, None]
    bg = np.asarray(BG_TOP, float) * (1 - yy) + np.asarray(BG_BOT, float) * yy
    img = np.broadcast_to(bg[:, None, :], (H, W, 3)).copy()
    # soft light from the upper left corner
    xx = np.arange(W)[None, :]
    g = np.exp(-(((xx - 0.1 * W) / (0.55 * W)) ** 2 + ((np.arange(H)[:, None] - 0.1 * H) / (0.55 * H)) ** 2))
    img = img * (1 - 0.35 * g[..., None]) + 255 * 0.35 * g[..., None]
    return img


def hair_colours(img, ndl, M, bangs):
    base = np.where((ndl < 0.22)[..., None], HAIR_SH, HAIR_BASE)
    base = np.where((ndl < -0.15)[..., None], HAIR_DEEP, base)
    P = hs
    theta = np.arccos(np.clip(-P[..., 1] / R_HAIR, -1, 1))
    phi = np.arctan2(P[..., 0], P[..., 2])
    rel = np.degrees(phi - PART)
    # strand grooves running from the part down toward the hairline
    gw = np.sin(np.radians(rel) * 360 / 11.5) + 0.4 * np.sin(np.radians(rel) * 360 / 7.3 + 1.0)
    groove = (gw > 1.22) & (theta > math.radians(32))
    base = np.where((groove & (ndl >= 0.22))[..., None], HAIR_SH, base)
    base = np.where((groove & (ndl < 0.22))[..., None], HAIR_DEEP, base)
    # angel ring near the top, cut by the grooves
    ring_c = np.where(bangs, math.radians(73) + math.radians(3) * np.sin(3 * phi), math.radians(52))
    ring = (np.abs(theta - ring_c) < math.radians(4.6)) & (np.abs(phi) < math.radians(80)) & (ndl > 0.2) & ~groove
    base = np.where(ring[..., None], HAIR_HI, base)
    core = (np.abs(theta - ring_c) < math.radians(1.4)) & ring
    base = np.where(core[..., None], HAIR_SHINE, base)
    # bangs: darker toward the tips
    tipdark = bangs & (theta > math.radians(93)) & (ndl >= 0.22)
    base = np.where(tipdark[..., None], HAIR_SH, base)
    img[M] = base[M]


def curtain_colours(img, ndl, M):
    ph, y, r = hs[..., 0], hs[..., 1], hs[..., 2]
    base = np.where((ndl < 0.18)[..., None], HAIR_SH, HAIR_BASE)
    base = np.where((ndl < -0.2)[..., None], HAIR_DEEP, base)
    streak = (np.sin(5 * ph + 1.0) > 0.86) & (ndl > 0.3) & (y < 42)
    base = np.where(streak[..., None], HAIR_HI, base)
    groove = np.sin(7 * ph + 2.0) > 0.95
    base = np.where((groove & (ndl >= 0.18))[..., None], HAIR_SH, base)
    base = np.where((groove & (ndl < 0.18))[..., None], HAIR_DEEP, base)
    low = y > 46 + 3 * np.sin(5 * ph)
    base = np.where((low & (ndl >= 0.18) & ~streak)[..., None], HAIR_SH, base)
    img[M] = base[M]


def render():
    P, N = head_surface()
    splat_head_space(P, N, FACE)
    P, N = scalp_surface()
    splat_head_space(P, N, HAIR)
    P, N = bangs_surface()
    splat_head_space(P, N, BANGS)
    P, N = neck_surface()
    splat(P, N, NECK)
    P, N = torso_surface()
    splat(P, N, TORSO)
    P, N, C = curtain_surface()
    splat(P, N, CURTAIN, C)
    fill_holes()

    ndl = (nrm * LIGHT).sum(-1)
    img = background()
    skin = (mat == FACE) | (mat == NECK)
    hair = (mat == HAIR) | (mat == BANGS)
    curtain = mat == CURTAIN
    torso = mat == TORSO
    fig = mat > 0

    # skin: one light, cel shadows; the underside of the jaw turns away from the light
    sk = np.where((ndl < 0.32)[..., None], SKIN_SH, SKIN)
    sk = np.where((ndl < -0.12)[..., None], SKIN_SH2, sk)
    down = nrm[..., 1]
    sk = np.where((down > 0.42)[..., None], SKIN_SH2, sk)
    img[skin] = sk[skin]
    under_chin = (mat == NECK) & dilate(mat == FACE, 9 * SS // 2)
    img[under_chin] = SKIN_SH2
    hairline_sh = (mat == FACE) & dilate(hair, 2 * SS) & ~shift(mat == FACE, 0, -2 * SS)
    img[hairline_sh] = SKIN_SH
    # torso: a plain top with a round neckline; the chest above it is skin
    tp = np.where((ndl < 0.3)[..., None], TOP_SH, TOP)
    img[torso] = tp[torso]
    wx, wy = hs[..., 0], hs[..., 1]
    opening = np.hypot(wx, (wy - 47) * 0.9)            # crew neck: a small opening around the neck
    chest = torso & (opening < 9.5)
    img[chest] = np.where((ndl < 0.3)[..., None], SKIN_SH, SKIN)[chest]
    rib = torso & (opening >= 9.5) & (opening < 11.2)
    img[rib] = TOP_SH
    collar = torso & (np.abs(opening - 9.5) < 0.45)
    img[collar] = TOP_LINE

    hair_colours(img, ndl, hair, mat == BANGS)
    curtain_colours(img, ndl, curtain)
    # the bangs throw a shadow onto the forehead (light from the upper left)
    bsh = shift(mat == BANGS, 2 * SS, 3 * SS) & (mat == FACE)
    img[bsh] = SKIN_SH
    paint_eyes(img, ndl)
    paint_nose_mouth(img)

    # outlines: silhouette, skin/hair, skin/clothes; a softer line between hair layers
    def edge(A, B):
        return (A & ~erode(A) & dilate(B)) | (B & ~erode(B) & dilate(A))
    sil = fig & ~erode(fig, SS)
    skin_all = skin | chest
    lines = sil | edge(skin_all, hair | curtain) | edge(skin_all, torso & ~chest) | edge(torso, curtain)
    jaw = (mat == FACE) & (hs[..., 1] > 15)
    ridge = edge(jaw & (down > 0.42), jaw & (down <= 0.42)) & (hs[..., 1] > 24)
    img[ridge] = (206, 148, 146)
    neckline_l = edge(mat == NECK, chest)
    img[neckline_l] = LINE_SOFT
    hl = edge(hair, curtain) & ~lines
    img[hl] = HAIR_LINE
    hair_edge = lines & (hair | curtain) & ~dilate(skin_all, 2) & ~dilate(torso, 2)
    img[lines] = LINE
    img[hair_edge] = HAIR_LINE

    out = np.clip(img, 0, 255).reshape(H1, SS, W1, SS, 3).mean((1, 3))
    return Image.fromarray(out.astype(np.uint8))


def guide_image(im):
    """The book's construction lines over the finished portrait: sphere, centre line, eye line,
    the two lines meeting at the chin."""
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
    rr = R_HEAD * U * k * CAM_D / (CAM_D - 0)
    d.ellipse((c[0] - rr, c[1] - rr, c[0] + rr, c[1] + rr), outline=BLUE, width=4)
    line3([(0, R_HEAD * math.sin(u), R_HEAD * math.cos(u)) for u in th], RED)
    r = math.sqrt(R_HEAD ** 2 - 8.5 ** 2)
    line3([(r * math.sin(u), 8.5, r * math.cos(u)) for u in th], BLUE)
    line3([(0, 8.5, 22), (0, 19.5, 25), (0, 26.4, 22.5), (0, 33.5, 17)], RED)
    line3([(-19, 8.5, 9), (0, 33.5, 17), (19, 8.5, 9)], BLUE)
    from PIL import ImageFont
    f = ImageFont.load_default(size=30)
    d.text((24, 20), f"turn {YAW:g}\u00b0  |  chin up {PITCH:g}\u00b0 (camera below)  |  tilt {abs(ROLL):g}\u00b0", fill=(90, 50, 50), font=f)
    return out


if __name__ == "__main__":
    im = render()
    im.save("blonde_girl_raw.png")
    im.resize((W1 * DISPLAY, H1 * DISPLAY), Image.NEAREST).save("blonde_girl.png")
    guide_image(im).save("blonde_girl_guide.png")
    print("saved", im.size)
