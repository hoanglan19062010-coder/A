"""Pixel-art anime portrait drawn from a low-angle selfie.

How it was built (the same steps the manga book teaches):
- Angle: a 3D head (sphere + centre line + chin) was fitted to landmarks on the photo
  (brows, glasses bridge, nose, mouth corners, chin, ear, hairline). Result: head turned
  ~24 deg to his right, camera ~21 deg below the face (selfie angle), head tilted ~6 deg,
  and strong close-camera perspective (the chin and jaw look big, the top of the head small).
- Light: the sun is behind him on the right. So the right edges (hair, ear, cheek, side of
  the neck, shoulder) get warm rim light, the face sits in cool sky light, the underside of
  the jaw falls into shadow, and the backlit ear glows - "warm light, cool shadows".
- Style: male anime features (narrower, sharper eyes, simple nose and mouth, clumped hair),
  one light source, flat cel shadows, a soft low-contrast background so he stands out.

Shapes are measured on the photo in photo pixels and converted to canvas pixels with P().
MODE switches the stage: "guides", "line", "flat" or "final".
"""
import math
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 384, 512                     # canvas pixels
SCALE = 4                           # display upscale
SS = 4                              # supersampling used to measure shape coverage
X0, Y0, S = 230, 410, 1636 / 384    # photo pixel -> canvas pixel
MODE = "final"

# ----------------------------------------------------------------------------- palette
PAPER = (250, 248, 244)
INK = (40, 30, 40)
SKY = [(116, 172, 220), (132, 184, 226), (152, 198, 232), (176, 212, 236), (198, 224, 240),
       (216, 233, 241), (232, 240, 242), (244, 246, 242), (252, 251, 246)]
SKIN_HI = (255, 233, 214)      # sunlit (warm)
SKIN = (236, 200, 182)         # sky-lit base
SKIN_SH = (206, 162, 162)      # shadow (cool)
SKIN_SH2 = (174, 128, 140)     # deep shadow (cooler)
SKIN_LINE = (186, 128, 124)    # soft lines drawn inside the face
EAR_GLOW = (250, 172, 150)     # light shining through the ear
NOSTRIL = (124, 74, 84)
LIP = (226, 170, 162)
LIP_SH = (204, 144, 146)
LIP_LINE = (128, 70, 78)
LINE_DARK = (64, 38, 48)       # outline on the shadow side
LINE_LIT = (182, 112, 94)      # outline where the sun hits
HAIR_DEEP = (13, 13, 22)
HAIR_SH = (22, 23, 37)
HAIR = (35, 37, 56)
HAIR_MID = (54, 62, 92)
HAIR_HI = (88, 112, 156)       # sky reflected in black hair (cool)
HAIR_SHINE = (146, 174, 212)
HAIR_RIM = (244, 230, 212)     # sun on the edge (warm)
HAIR_LINE = (10, 10, 18)
BROW = (38, 28, 36)
SCLERA = (240, 238, 242)
SCLERA_SH = (200, 196, 212)
IRIS = [(22, 16, 20), (30, 21, 24), (40, 28, 28), (54, 37, 33), (72, 49, 40), (94, 64, 49), (118, 82, 60)]
PUPIL = (12, 9, 12)
LASH = (22, 16, 22)
LASH_LOW = (92, 60, 66)
FRAME = (20, 20, 28)
FRAME_MID = (88, 92, 106)
FRAME_CLEAR = (176, 184, 196)
FRAME_SHINE = (236, 238, 244)
LENS = (214, 230, 242)
SHIRT = (36, 36, 46)
SHIRT_SH = (22, 22, 30)
SHIRT_HI = (56, 58, 74)
SHIRT_RIM = (196, 190, 190)
SHIRT_LINE = (12, 12, 18)
COLLAR = (50, 50, 62)
COLLAR_DARK = (34, 34, 44)
PRINT = (206, 206, 158)
PRINT_SH = (176, 178, 134)
PRINT_INK = (34, 34, 30)


def P(x, y):
    return ((x - X0) / S, (y - Y0) / S)


def PP(pts):
    return [P(x, y) for x, y in pts]


def mix(a, b, t):
    return tuple(a[i] * (1 - t) + b[i] * t for i in range(3))


def smooth(ctrl, n=8, closed=True):
    """Catmull-Rom spline through the control points."""
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


def path(spec, n=6):
    """Closed outline from [(x, y, sharp), ...]: runs between sharp corners become smooth curves."""
    pts = [(x, y) for x, y, _ in spec]
    sharp = [i for i, (_, _, s) in enumerate(spec) if s]
    if not sharp:
        return smooth(pts, n)
    m, out = len(pts), []
    for j, i0 in enumerate(sharp):
        i1 = sharp[(j + 1) % len(sharp)]
        run, i = [pts[i0]], i0
        while True:
            i = (i + 1) % m
            run.append(pts[i])
            if i == i1:
                break
        out.extend(smooth(run, n, closed=False)[:-1] if len(run) > 2 else run[:-1])
    return out


def band(pts, widths, side=1):
    """Strip along a polyline, on one side; widths: one number or one per point."""
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


def resample(pts, n):
    """n evenly spaced points along a polyline."""
    d = [0.0]
    for (xa, ya), (xb, yb) in zip(pts, pts[1:]):
        d.append(d[-1] + math.hypot(xb - xa, yb - ya))
    out, j = [], 0
    for k in range(n):
        s = d[-1] * k / (n - 1)
        while j < len(d) - 2 and d[j + 1] < s:
            j += 1
        t = (s - d[j]) / ((d[j + 1] - d[j]) or 1)
        out.append((pts[j][0] + (pts[j + 1][0] - pts[j][0]) * t, pts[j][1] + (pts[j + 1][1] - pts[j][1]) * t))
    return out


# ----------------------------------------------------------------------------- canvas
img = np.zeros((H, W, 3), np.float32)
SKYMAP = np.zeros((H, W, 3), np.float32)
YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)


def raster(pts, width=None, closed=True):
    """Coverage (0..1) of a polygon, or of a stroke when width is given."""
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


def fill(pts, c, clip=None, thr=0.5):
    m = raster(pts) >= thr
    if clip is not None:
        m &= clip
    img[m] = colour(c)[m]
    return m


def paint(a, c):
    """Pixel-art anti-aliasing: full colour where a shape covers a pixel well, two in-between
    tones on the stair steps, nothing elsewhere."""
    col = colour(c)
    full = a >= 0.55
    mid = (a >= 0.36) & ~full
    low = (a >= 0.18) & (a < 0.36)
    img[full] = col[full]
    img[mid] = img[mid] * 0.4 + col[mid] * 0.6
    img[low] = img[low] * 0.68 + col[low] * 0.32


def stroke(pts, c, width=1.0, closed=False, clip=None):
    a = raster(pts, width, closed)
    if clip is not None:
        a = a * clip
    paint(a, c)


def xgrad(c1, c2, xa, xb):
    """Colour that changes from c1 to c2 between canvas x = xa and x = xb."""
    t = np.clip((XX - xa) / (xb - xa), 0, 1)[..., None]
    return np.asarray(c1, np.float32) * (1 - t) + np.asarray(c2, np.float32) * t


def hazy(c, k):
    """Background colour pushed toward the sky behind it (distance haze)."""
    return np.asarray(c, np.float32) * (1 - k) + SKYMAP * k


def C(c):
    return PAPER if MODE in ("line", "guides") else c


def shading():
    return MODE == "final"


# ----------------------------------------------------------------------------- shapes (photo pixels)
FACE = [(640, 790), (598, 845), (578, 920), (571, 1000), (574, 1062), (586, 1122), (604, 1182), (623, 1236),
        (640, 1286), (653, 1332), (668, 1382), (689, 1438), (717, 1494), (753, 1544), (796, 1594), (846, 1630),
        (896, 1651), (934, 1657), (978, 1651), (1032, 1631), (1092, 1601), (1150, 1563), (1204, 1518), (1243, 1470),
        (1268, 1418), (1286, 1356), (1297, 1296), (1298, 1240), (1290, 1180), (1286, 1100), (1288, 1030),
        (1278, 968), (1255, 905), (1215, 842), (1150, 792), (1050, 762), (930, 750), (800, 756), (700, 770)]
JAW_FAR = FACE[3:17]                     # far cheek down to the chin
JAW_NEAR = FACE[16:28]                   # chin up to below the ear
EAR = [(1288, 1022), (1301, 1013), (1326, 1021), (1344, 1041), (1353, 1076), (1354, 1121), (1350, 1176),
       (1341, 1226), (1329, 1259), (1311, 1271), (1296, 1266), (1288, 1240), (1286, 1180), (1285, 1100), (1286, 1050)]
NECK = [(955, 1590), (1270, 1230), (1300, 1275), (1300, 1367), (1307, 1433), (1320, 1487), (1340, 1533),
        (1373, 1573), (1413, 1600), (1453, 1620), (1540, 1655), (1540, 1790), (990, 1940), (968, 1800), (962, 1700)]
NECK_NEAR = [(1300, 1270), (1300, 1367), (1307, 1433), (1320, 1487), (1340, 1533), (1373, 1573), (1413, 1600),
             (1453, 1620), (1500, 1636)]
NECK_FAR = [(950, 1655), (958, 1705), (966, 1760), (974, 1800)]
SHIRT_SHAPE = [(530, 2600), (588, 2567), (620, 2439), (652, 2280), (676, 2152), (707, 2056), (771, 1961), (835, 1881),
         (899, 1809), (935, 1786), (979, 1793), (1027, 1825), (1106, 1825), (1186, 1793), (1298, 1729),
         (1394, 1673), (1473, 1625), (1505, 1610), (1617, 1633), (1777, 1681), (1890, 1718), (1890, 2600)]
SHIRT_FAR_EDGE = SHIRT_SHAPE[1:9]
SHOULDER_NEAR = [(1505, 1610), (1617, 1633), (1777, 1681), (1890, 1718)]
COLLAR_IN = [(935, 1786), (979, 1793), (1027, 1825), (1106, 1825), (1186, 1793), (1298, 1729), (1394, 1673),
             (1473, 1625), (1505, 1610)]
COLLAR_OUT = [(1545, 1640), (1492, 1682), (1400, 1744), (1298, 1806), (1170, 1886), (1058, 1918), (962, 1908),
              (916, 1882), (897, 1842), (903, 1803)]
PRINT_BOX = [(694, 2203), (979, 2218), (1505, 2282), (1505, 2600), (628, 2600), (640, 2450), (660, 2320)]


NEAR_LENS = [(843, 1062), (876, 1028), (926, 994), (981, 967), (1036, 953), (1090, 945), (1113, 944),
             (1116, 985), (1115, 1042), (1110, 1095), (1100, 1128), (1078, 1146), (1028, 1156), (968, 1163),
             (910, 1163), (874, 1155), (851, 1134), (843, 1104)]
FAR_LENS = [(773, 1069), (720, 1074), (667, 1081), (613, 1094), (567, 1114), (540, 1134), (528, 1156),
            (533, 1191), (548, 1231), (570, 1263), (600, 1279), (645, 1282), (690, 1273), (730, 1253),
            (755, 1226), (768, 1186), (773, 1130)]


def hair_spec():
    """Hair outline: a rounded crop with a few pointed clumps of different sizes on top (the book:
    vary the clump sizes), sideburns, and a fringe of locks hanging onto the forehead."""
    S_, C_ = True, False
    outer = [
        (586, 1050, S_), (563, 1000, C_), (556, 938, C_), (557, 878, C_), (548, 846, S_), (561, 818, C_),
        (566, 772, C_), (582, 726, C_), (610, 688, C_), (612, 650, S_), (646, 640, C_), (676, 612, C_),
        (712, 590, C_), (748, 574, C_), (770, 546, S_), (800, 556, C_), (846, 545, C_), (884, 538, C_),
        (918, 508, S_), (940, 536, C_), (980, 538, C_), (1012, 516, S_), (1028, 549, C_), (1072, 560, C_),
        (1112, 582, C_), (1150, 584, S_), (1150, 618, C_), (1180, 652, C_), (1206, 690, C_), (1244, 704, S_),
        (1236, 742, C_), (1258, 796, C_), (1277, 846, C_), (1306, 868, S_), (1288, 900, C_), (1295, 958, C_),
        (1300, 1000, C_), (1303, 1016, S_),
        # near side: above the ear, sideburn, temple
        (1282, 1004, C_), (1266, 1012, S_), (1248, 970, C_), (1224, 928, C_), (1207, 884, C_), (1192, 846, S_),
    ]
    # fringe locks: (notch on the right, tip, how far the lock bows out)
    locks = [((1192, 846), (1168, 852)), ((1150, 800), (1118, 840)), ((1094, 786), (1056, 826)),
             ((1032, 772), (984, 836)), ((960, 768), (926, 814)), ((906, 770), (862, 840)),
             ((836, 766), (800, 818)), ((776, 772), (736, 828)), ((714, 784), (682, 832)),
             ((662, 800), (640, 860))]
    fr = []
    for i, ((nx, ny), (tx, ty)) in enumerate(locks):
        if i > 0:
            fr.append((nx, ny, S_))
            fr.append((nx - (nx - tx) * 0.35 + 4, ny + (ty - ny) * 0.55, C_))
        fr.append((tx, ty, S_))
        nxt = locks[i + 1][0] if i + 1 < len(locks) else (632, 900)
        fr.append((tx - (tx - nxt[0]) * 0.35 - 3, ty - (ty - nxt[1]) * 0.6, C_))
    far = [(632, 902, S_), (626, 962, C_), (610, 1012, C_)]
    return outer + fr + far


FRINGE_TIPS = []    # filled by draw_hair (for the forehead shadow)


# ----------------------------------------------------------------------------- background
def draw_background():
    if MODE in ("line", "guides"):
        img[:] = PAPER
        SKYMAP[:] = PAPER
        return
    sun_x, sun_y = 300, 150
    d = np.hypot(XX - sun_x, (YY - sun_y) * 1.15)
    t = np.clip(1 - d / 370, 0, 1) ** 1.3
    t = np.clip(t + 0.2 * np.clip((YY - 40) / 440, 0, 1), 0, 1)
    v = t * (len(SKY) - 1)
    base = np.floor(v)
    frac = v - base
    bayer = (np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], np.float32) + 0.5) / 16
    thr = bayer[YY.astype(int) % 4, XX.astype(int) % 4]
    step = (np.clip((frac - 0.5) * 2.4 + 0.5, 0, 1) > thr)
    idx = np.clip(base + step, 0, len(SKY) - 1).astype(int)
    if MODE == "flat":
        idx[:] = 4
    img[:] = np.asarray(SKY, np.float32)[idx]
    SKYMAP[:] = img
    rng = random.Random(11)

    # glass tower on the right
    m = raster([(350, 26), (384, 21), (384, 330), (350, 330)]) >= 0.5
    glass = np.zeros_like(img)
    glass[:] = (104, 146, 160)
    rows = (YY.astype(int) - 26) % 6 == 0
    cols = (XX.astype(int) - 350) % 9 == 0
    glass[rows | cols] = (78, 116, 132)
    refl = np.clip(1 - (YY - 30) / 220, 0, 1)[..., None] * np.clip((XX - 350) / 34, 0, 1)[..., None]
    glass = glass * (1 - 0.35 * refl) + np.asarray((214, 232, 236), np.float32) * 0.35 * refl
    top = raster([(350, 26), (384, 21), (384, 48), (350, 50)]) >= 0.5
    glass[top] = (66, 92, 110)
    stripe = raster([(350, 40), (384, 37), (384, 39), (350, 42)]) >= 0.5
    glass[stripe] = (170, 200, 212)
    img[m] = (glass * 0.62 + SKYMAP * 0.38)[m]

    # distant pink apartment towers and trees on the left
    for x0, x1, y0 in ((24, 50, 284), (54, 74, 300), (2, 20, 312), (78, 92, 318)):
        mt = raster([(x0, y0), (x1, y0), (x1, 380), (x0, 380)]) >= 0.5
        t_ = np.zeros_like(img)
        t_[:] = (214, 178, 168)
        t_[((YY.astype(int) - y0) % 5 == 2) & ((XX.astype(int) - x0) % 3 != 0)] = (192, 152, 148)
        t_[XX > x0 + (x1 - x0) * 0.7] *= 0.94
        img[mt] = (t_ * 0.55 + SKYMAP * 0.45)[mt]
    for cx, cy, r in ((10, 352, 16), (34, 346, 14), (58, 356, 15), (84, 350, 13), (108, 360, 14), (130, 366, 12)):
        for k in range(9):
            a = rng.uniform(0, 2 * math.pi)
            rr = r * rng.uniform(0.3, 0.75)
            bx, by = cx + math.cos(a) * rr, cy + math.sin(a) * rr * 0.7
            br = r * rng.uniform(0.35, 0.55)
            mb = raster(smooth([(bx + br * math.cos(q), by + br * 0.8 * math.sin(q)) for q in np.linspace(0, 6.28, 9)[:-1]])) >= 0.5
            col = (98, 140, 100) if by > cy else (128, 166, 118)
            img[mb] = hazy(col, 0.35)[mb]
    # fence and road
    road = raster([(0, 384), (140, 384), (140, 512), (0, 512)]) >= 0.5
    img[road] = hazy((176, 178, 182), 0.12)[road]
    for x in range(2, 140, 5):
        mf = raster([(x, 366), (x + 1.4, 366), (x + 1.4, 384), (x, 384)]) >= 0.5
        img[mf] = hazy((150, 152, 156), 0.3)[mf]
    mf = raster([(0, 371), (140, 371), (140, 373), (0, 373)]) >= 0.5
    img[mf] = hazy((150, 152, 156), 0.3)[mf]
    for y in (430, 470, 506):
        md = raster([(30, y), (40, y - 2), (41, y + 1), (31, y + 3)]) >= 0.5
        img[md] = hazy((236, 236, 232), 0.1)[md]
    # a white car parked on the left
    car = smooth([(8, 404), (10, 394), (20, 388), (40, 387), (50, 392), (56, 398), (57, 406), (8, 407)])
    mc = raster(car) >= 0.5
    img[mc] = hazy((236, 238, 238), 0.15)[mc]
    mw = raster([(20, 390), (40, 389), (47, 395), (16, 396)]) >= 0.5
    img[mw] = hazy((96, 110, 122), 0.2)[mw]
    for wx in (17, 47):
        mwh = raster(smooth([(wx + 3.2 * math.cos(q), 406 + 3.2 * math.sin(q)) for q in np.linspace(0, 6.28, 9)[:-1]])) >= 0.5
        img[mwh] = hazy((44, 44, 50), 0.15)[mwh]

    # palm trees behind his shoulder (backlit: dark but softened by haze)
    for (cx, cy, sz, trunk_x) in ((276, 236, 30, 282), (330, 222, 34, 334), (382, 250, 30, 384)):
        mt = raster([(trunk_x - 2, cy), (trunk_x + 1, cy), (trunk_x + 3, 330), (trunk_x - 1, 330)]) >= 0.5
        img[mt] = hazy((112, 110, 100), 0.45)[mt]
        for k in range(10):
            a = math.radians(-170 + k * 36 + rng.uniform(-10, 10))
            L = sz * rng.uniform(0.7, 1.0)
            droop = 0.55 * L
            ctrl = [(cx, cy), (cx + math.cos(a) * L * 0.5, cy + math.sin(a) * L * 0.5 - 4),
                    (cx + math.cos(a) * L, cy + math.sin(a) * L * 0.6 + droop)]
            spine = smooth(ctrl, n=8, closed=False)
            frond = band(spine, [2.6, 2.2, 1.6, 1.0][:1] * 1, 1)
            leaf = []
            for i, (x, y) in enumerate(spine[1:-1], 1):
                xa, ya = spine[i - 1]
                xb, yb = spine[i + 1]
                tx, ty = xb - xa, yb - ya
                L2 = math.hypot(tx, ty) or 1
                w = 4.2 * (1 - i / len(spine)) + 1.0
                leaf.append((x - ty / L2 * w + tx / L2 * 1.5, y + tx / L2 * w + ty / L2 * 1.5))
            poly = [spine[0]] + leaf + [spine[-1]] + [(x + ty_ * 0, y) for (x, y), ty_ in zip(spine[::-1], [0] * len(spine))]
            mp = raster(poly) >= 0.45
            col = (70, 104, 94) if k % 3 else (58, 90, 84)
            img[mp] = hazy(col, 0.42)[mp]
            stroke(spine, hazy((52, 80, 74), 0.45), 1.0)

    # leaves in the top-right corner
    for cx, cy in ((352, 6), (372, 22), (330, 0), (384, 40), (316, 12)):
        for k in range(14):
            a = rng.uniform(0, 2 * math.pi)
            r = rng.uniform(2, 13)
            lx, ly = cx + math.cos(a) * r, cy + math.sin(a) * r * 0.8
            la = rng.uniform(0, math.pi)
            ll, lw = rng.uniform(4.5, 7.0), rng.uniform(1.8, 2.6)
            leaf = [(lx + math.cos(la) * ll * c - math.sin(la) * lw * s_, ly + math.sin(la) * ll * c + math.cos(la) * lw * s_)
                    for c, s_ in ((-1, 0), (-0.3, 1), (0.4, 0.8), (1, 0), (0.4, -0.8), (-0.3, -1))]
            ml = raster(leaf) >= 0.5
            col = (56, 90, 62) if rng.random() < 0.6 else (86, 124, 80)
            img[ml] = hazy(col, 0.12)[ml]


# ----------------------------------------------------------------------------- body
def draw_neck():
    neck = PP(NECK)
    M = raster(smooth(neck, 6)) >= 0.5
    img[M] = C(SKIN) if MODE != "flat" else SKIN
    if shading():
        lit = PP([(1300, 1262), (1300, 1367), (1307, 1433), (1320, 1487), (1340, 1533), (1373, 1573), (1413, 1600),
                  (1453, 1620), (1520, 1648), (1440, 1700), (1350, 1736), (1262, 1772), (1222, 1700), (1238, 1600),
                  (1256, 1500), (1272, 1404), (1286, 1320)])
        scm = PP([(1286, 1318), (1262, 1420), (1236, 1530), (1206, 1640), (1176, 1760), (1160, 1830)])
        front = scm + PP([(1000, 1900), (940, 1600), (1100, 1450)])
        fill(smooth(front, 6), SKIN_SH, M)
        fill(smooth(lit, 6), SKIN_HI, M)
        # the jaw throws a shadow onto the front of the neck
        jaw = PP(JAW_NEAR[:9])
        drop = [(x, y + 7.5 + 3.5 * i / len(jaw)) for i, (x, y) in enumerate(jaw)]
        fill(jaw + drop[::-1], SKIN_SH2, M & ~(raster(smooth(lit, 6)) >= 0.5))
        fill(band(PP(COLLAR_IN), 3.0, -1), SKIN_SH2, M)
    stroke(smooth(PP(NECK_NEAR), 6, closed=False), LINE_LIT if MODE != "line" else INK, 1.1)
    stroke(smooth(PP(NECK_FAR), 6, closed=False), LINE_DARK if MODE != "line" else INK, 1.2)
    if MODE == "line":
        stroke(smooth(PP([(1280, 1330), (1250, 1470), (1210, 1620), (1180, 1740)]), 6, closed=False), INK, 0.8)
    return M


def draw_shirt():
    shirt = smooth(PP(SHIRT_SHAPE), 6)
    M = raster(shirt) >= 0.5
    img[M] = C(SHIRT)
    collar = smooth(PP(COLLAR_IN), 6, closed=False) + smooth(PP(COLLAR_OUT), 6, closed=False)
    MC = raster(collar) >= 0.5
    if MODE != "line":
        if shading():
            fill(smooth(PP([(903, 1803), (835, 1881), (771, 1961), (707, 2056), (684, 2140), (760, 2092),
                            (842, 2024), (920, 1966), (990, 1934), (950, 1912), (912, 1880)]), 6), SHIRT_HI, M)
            fill(band(smooth(PP(COLLAR_OUT[1:-2]), 6, closed=False), 4.0, 1), SHIRT_SH, M & ~MC)
            for fold in ([(1010, 1935), (930, 2010), (860, 2100), (800, 2170)],
                         [(1130, 1925), (1100, 2030), (1080, 2140)],
                         [(1600, 1700), (1520, 1800), (1470, 1900)]):
                f = smooth(PP(fold), 6, closed=False)
                stroke(f, SHIRT_SH, 1.6, clip=M & ~MC)
                stroke([(x - 1.2, y - 0.6) for x, y in f], SHIRT_HI, 0.8, clip=M & ~MC)
        # print on the front (pale, with distressed ink)
        pb = smooth(PP(PRINT_BOX), 4)
        MP = raster(pb) >= 0.5
        img[MP] = PRINT if MODE != "line" else PAPER
        if shading():
            fill(band(smooth(PP(PRINT_BOX[:3]), 6, closed=False), 3.0, 1), PRINT_SH, MP)
            rng = random.Random(5)
            for _ in range(140):
                x = rng.uniform(640, 1500)
                y = rng.uniform(2205, 2600)
                if rng.random() < 0.55:
                    x = rng.uniform(640, 1050)
                r = rng.uniform(4, 16)
                ang = rng.uniform(0, math.pi)
                blob = [(x + math.cos(ang + q) * r * rng.uniform(0.5, 1.2), y + math.sin(ang + q) * r * rng.uniform(0.4, 1.0))
                        for q in np.linspace(0, 2 * math.pi, 7)[:-1]]
                fill(PP(blob), PRINT_INK, MP)
            cx, cy = P(835, 2560)
            fill(smooth([(cx + 46 * math.cos(q), cy + 42 * math.sin(q)) for q in np.linspace(0, 6.28, 17)[:-1]]), PRINT_INK, MP)
        else:
            cx, cy = P(835, 2560)
            fill(smooth([(cx + 46 * math.cos(q), cy + 42 * math.sin(q)) for q in np.linspace(0, 6.28, 17)[:-1]]),
                 PRINT_INK if MODE == "flat" else PAPER, MP)
        img[MC] = COLLAR
        if shading():
            ins, outs = resample(smooth(PP(COLLAR_IN), 6, closed=False), 40), resample(smooth(PP(COLLAR_OUT[::-1]), 6, closed=False), 40)
            for k in range(1, 39, 2):
                stroke([ins[k], outs[k]], COLLAR_DARK, 0.7, clip=MC)
            fill(band(smooth(PP(SHOULDER_NEAR), 6, closed=False), 2.2, 1), SHIRT_RIM, M)
            fill(band(smooth(PP(COLLAR_OUT[:3]), 6, closed=False), 1.6, -1), SHIRT_RIM, M | MC)
    if MODE == "line":
        cx, cy = P(835, 2560)
        stroke(smooth([(cx + 46 * math.cos(q), cy + 42 * math.sin(q)) for q in np.linspace(0, 6.28, 17)[:-1]]), INK, 0.8, True, clip=raster(smooth(PP(PRINT_BOX), 4)))
        stroke(smooth(PP(PRINT_BOX[:3]), 6, closed=False), INK, 0.8)
        for fold in ([(1010, 1935), (930, 2010), (860, 2100)], [(1130, 1925), (1100, 2030)]):
            stroke(smooth(PP(fold), 6, closed=False), INK, 0.7)
    line = SHIRT_LINE if MODE != "line" else INK
    stroke(smooth(PP(SHIRT_FAR_EDGE), 6, closed=False), line, 1.3)
    stroke(smooth(PP(SHOULDER_NEAR), 6, closed=False), line, 1.1)
    stroke(smooth(PP(COLLAR_OUT), 6, closed=False), line, 1.0)
    stroke(smooth(PP(COLLAR_IN), 6, closed=False), line, 1.1)
    return M


# ----------------------------------------------------------------------------- head
def draw_ear():
    ear = smooth(PP(EAR), 6)
    M = raster(ear) >= 0.5
    img[M] = C(SKIN_HI if shading() else SKIN)
    if shading():
        inner = smooth(PP([(1296, 1046), (1320, 1040), (1336, 1070), (1340, 1130), (1334, 1190), (1318, 1236),
                           (1300, 1248), (1294, 1180), (1292, 1100)]), 6)
        fill(inner, EAR_GLOW, M)
        fill(smooth(PP([(1292, 1104), (1312, 1098), (1322, 1140), (1318, 1192), (1300, 1206), (1293, 1160)]), 6), SKIN_SH, M)
    lin = INK if MODE == "line" else (164, 96, 84)
    stroke(smooth(PP([(1302, 1040), (1324, 1056), (1332, 1100), (1330, 1160), (1320, 1206), (1306, 1226)]), 6, closed=False),
           lin if MODE == "line" else (190, 112, 100), 0.8)
    stroke(ear, LINE_LIT if MODE != "line" else INK, 1.0, closed=True)
    return M


def draw_face():
    face = smooth(PP(FACE), 6)
    M = raster(face) >= 0.5
    img[M] = C(SKIN)
    if shading():
        # sunlit side plane: from the edge of the glasses back to the ear and down the jaw
        lit = PP([(1288, 960), (1214, 862), (1150, 900), (1124, 960), (1124, 1040), (1120, 1120), (1104, 1184),
                  (1090, 1252), (1086, 1322), (1100, 1392), (1134, 1452), (1178, 1500), (1214, 1520), (1258, 1474),
                  (1288, 1392), (1300, 1300), (1296, 1200), (1290, 1100)])
        ML = fill(smooth(lit, 6), SKIN_HI, M)
        edge = ML & ~shifted(ML, 2, 0) & M
        img[edge] = mix(SKIN, SKIN_HI, 0.5)
        # the far cheek turns away from the sky: a shadow band along that edge
        far = smooth(PP(JAW_FAR), 6, closed=False)
        ws = [2.5 + 9.0 * math.sin(math.pi * min(1.0, i / (len(far) * 0.8))) for i in range(len(far))]
        fill(band(far, ws, 1), SKIN_SH, M)
        # nose: the side facing away, the underside, and the nostrils
        fill(smooth(PP([(778, 1092), (796, 1104), (797, 1180), (792, 1236), (772, 1256), (764, 1206), (768, 1140)]), 6),
             SKIN_SH, M)
        fill(smooth(PP([(764, 1250), (786, 1262), (820, 1270), (860, 1258), (896, 1240), (904, 1260), (874, 1286),
                        (830, 1296), (792, 1290), (770, 1274)]), 6), SKIN_SH, M)
        # eye sockets either side of the nose bridge
        fill(smooth(PP([(752, 1090), (770, 1096), (774, 1150), (758, 1150)]), 6), SKIN_SH, M)
        fill(smooth(PP([(846, 1072), (868, 1080), (860, 1130), (846, 1118)]), 6), SKIN_SH, M)
        # under the lower lip, and the soft dip above the chin
        fill(smooth(PP([(842, 1488), (880, 1497), (930, 1495), (968, 1482), (950, 1504), (900, 1514), (856, 1506)]), 6),
             SKIN_SH, M)
    # outline: dark on the shadow side, warm where the sun reaches
    col = xgrad(LINE_DARK, LINE_LIT, P(1080, 0)[0], P(1240, 0)[0]) if MODE != "line" else INK
    stroke(face, col, 1.25, closed=True)
    return M


def shifted(M, dx, dy):
    """M moved by (dx, dy) pixels; empty where it slides in from outside."""
    out = np.zeros_like(M)
    h, w = M.shape
    ys = slice(max(0, dy), min(h, h + dy))
    yd = slice(max(0, -dy), min(h, h - dy))
    xs = slice(max(0, dx), min(w, w + dx))
    xd = slice(max(0, -dx), min(w, w - dx))
    out[ys, xs] = M[yd, xd]
    return out


def rim(M, dx, dy):
    """Pixels of M whose neighbour (dx, dy) toward the light is outside M: the lit edge."""
    return M & ~shifted(M, -dx, -dy)


def draw_hair(skin):
    global FRINGE_TIPS
    spec = hair_spec()
    outline = path([(P(x, y)[0], P(x, y)[1], s) for x, y, s in spec], n=6)
    M = raster(outline) >= 0.5
    FRINGE_TIPS = [P(x, y) for x, y, s in spec if s and 780 < y < 900 and 630 < x < 1180]
    if shading():
        # the fringe throws its shape as a shadow on the forehead
        sh = shifted(M, -2, 5) & skin & ~M
        img[sh] = SKIN_SH
    img[M] = C(HAIR)
    if MODE in ("line", "guides"):
        for ln in STRANDS:
            stroke(smooth(PP(ln), 6, closed=False), (150, 140, 150), 0.7, clip=M)
        stroke(outline, INK, 1.2, closed=True)
        return M
    if shading():
        # darker underneath: the lower part of the locks, the far side, around the sideburns
        low = shifted(M, 0, -9) & M
        img[M & ~low] = HAIR_SH
        for (x, y) in FRINGE_TIPS:
            fill([(x - 4.5, y - 11), (x + 3.5, y - 11), (x, y)], HAIR_DEEP, M)
        far = M & (XX < P(640, 0)[0] + (YY - P(0, 700)[1]) * 0.12)
        img[far & ~low] = HAIR_DEEP
        img[far & low] = HAIR_SH
        for ln in STRANDS:
            stroke(smooth(PP(ln), 6, closed=False), HAIR_DEEP, 0.9, clip=M)
        hair_shine(M)
        # the sun is behind him on the right: rim light on the edges that face it
        r1 = rim(M, 3, -1) & (XX > P(980, 0)[0])
        r2 = rim(M, 5, -2) & ~r1 & (XX > P(1040, 0)[0])
        img[r2] = HAIR_HI
        img[r1] = HAIR_RIM
    if MODE == "flat":
        for ln in STRANDS:
            stroke(smooth(PP(ln), 6, closed=False), HAIR_SH, 0.8, clip=M)
    col = xgrad(HAIR_LINE, (70, 64, 76), P(1120, 0)[0], P(1300, 0)[0])
    stroke(outline, col, 1.2, closed=True)
    return M


STRANDS = [   # lock edges, flowing from the crown down to the fringe
    [(1092, 790), (1088, 740), (1070, 690)],
    [(1030, 776), (1024, 714), (1004, 660)],
    [(958, 772), (950, 724), (936, 676)],
    [(904, 774), (894, 720), (876, 668)],
    [(834, 770), (820, 716), (800, 670)],
    [(774, 776), (758, 726), (736, 684)],
    [(712, 788), (692, 744), (672, 708)],
    [(1150, 804), (1160, 752), (1156, 704)],
    [(1226, 930), (1236, 870), (1230, 812)],
    [(600, 940), (594, 880), (598, 820)],
]


def hair_shine(M):
    """Sky reflected on the top of the black hair: a band that the strand lines cut, smudged
    into streaks along the strands (the 'glow, eraser, blur' method)."""
    ramp = [HAIR_DEEP, HAIR_SH, HAIR, HAIR_MID, HAIR_HI, HAIR_SHINE]
    idx = {c: i for i, c in enumerate(ramp)}
    pix = img
    rng = random.Random(9)
    cx_pts = PP([(560, 790), (640, 690), (760, 628), (900, 606), (1030, 616), (1140, 652), (1220, 716), (1270, 800)])
    xs = [p[0] for p in cx_pts]
    ys = [p[1] for p in cx_pts]
    run = 0
    for x in range(int(xs[0]), int(xs[-1]) + 1):
        cy = float(np.interp(x, xs, ys))
        if run <= 0:
            run = rng.choice((1, 2, 2, 3, 3, 4))
            up = rng.choice((2, 3, 4, 5, 6))
            down = rng.choice((3, 5, 7, 9, 12))
            if rng.random() < 0.2:
                up, down = 1, 2
        run -= 1
        edge = min(1.0, (x - xs[0]) / 30) * min(1.0, (xs[-1] - x) / 20)
        half = 2.5 * edge
        for y in range(int(cy - half - up), int(cy + half + down) + 1):
            if not (0 <= y < H) or not M[y, x]:
                continue
            cur = tuple(int(v) for v in pix[y, x])
            if cur not in idx or cur in (HAIR_DEEP,):
                continue
            dy = y - cy
            if abs(dy) <= half * 0.45:
                nc = HAIR_SHINE
            elif abs(dy) <= half or (dy < 0 and -dy - half <= up * 0.5) or (dy > 0 and dy - half <= down * 0.45):
                nc = HAIR_HI
            else:
                nc = HAIR_MID
            if x < P(700, 0)[0]:          # the far side catches less of the sky
                nc = ramp[max(2, idx[nc] - 1)]
            if idx[nc] > idx[cur]:
                pix[y, x] = nc


# ----------------------------------------------------------------------------- face features
def tapered(centre, w0, w1):
    """Closed band around a centre line, w0 thick at the start and w1 at the end."""
    n = len(centre)
    ws = [w0 + (w1 - w0) * i / (n - 1) for i in range(n)]
    up = band(centre, [w / 2 for w in ws], 1)[n:][::-1]
    dn = band(centre, [w / 2 for w in ws], -1)[n:][::-1]
    return up + dn[::-1]


def draw_brows(skin):
    near = smooth(PP([(866, 922), (930, 900), (1000, 884), (1060, 878), (1108, 884)]), 8, closed=False)
    far = smooth(PP([(726, 1032), (690, 1018), (648, 1008), (602, 1008)]), 8, closed=False)
    col = BROW if MODE != "line" else INK
    for c, w0, w1 in ((near, 7.0, 2.0), (far, 5.6, 1.6)):
        m = raster(tapered(c, w0, w1)) >= 0.5
        m &= skin
        img[m] = col


def draw_eye(centre, ang, sx, sy, mirror, iris_u, w=19.0):
    """Sharp, narrow male anime eye. Local u runs toward the outer corner, v down. The local
    shape is mapped to the screen with a rotation and a squash, so the far eye narrows."""
    cx, cy = centre
    ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))

    def L2S(u, v):
        x, y = mirror * sx * u, sy * v
        return (cx + ca * x - sa * y, cy + sa * x + ca * y)

    def S2L(X, Y):
        dx, dy = X - cx, Y - cy
        return (ca * dx + sa * dy) / (mirror * sx), (-sa * dx + ca * dy) / sy

    top = smooth([(-w, 1.6), (-0.55 * w, -2.6), (0.05 * w, -4.2), (0.6 * w, -4.1), (w, -2.9)], 10, closed=False)
    bot = smooth([(-w, 1.6), (-0.62 * w, 6.0), (0.0, 7.4), (0.55 * w, 5.6), (w, -2.9)], 10, closed=False)
    tu, tv = [p[0] for p in top], [p[1] for p in top]
    bu, bv = [p[0] for p in bot], [p[1] for p in bot]
    top_v = lambda u: float(np.interp(u, tu, tv))
    bot_v = lambda u: float(np.interp(u, bu, bv))
    opening = [L2S(u, v) for u, v in top] + [L2S(u, v) for u, v in bot[::-1]]
    M = raster(opening) >= 0.5
    iu, iv, r = iris_u, 1.9, 7.0
    ys, xs = np.nonzero(M)
    line = MODE in ("line", "guides")
    for y, x in zip(ys, xs):
        u, v = S2L(x + 0.5, y + 0.5)
        e = ((u - iu) / r) ** 2 + ((v - iv) / r) ** 2
        pupil = ((u - iu) / 3.1) ** 2 + ((v - iv + 0.4) / 3.4) ** 2 <= 1
        lid = v - top_v(u)
        if line:
            c = PAPER
            if 0.78 < e <= 1.0 or pupil:
                c = INK
        else:
            c = SCLERA_SH if lid < 1.8 and MODE == "final" else SCLERA
            if e <= 1:
                k = int(np.clip((v - (iv - r)) / (2 * r) * len(IRIS), 0, len(IRIS) - 1))
                c = IRIS[k] if MODE == "final" else IRIS[3]
                if e > 0.8:
                    c = IRIS[0]
                if pupil:
                    c = PUPIL
                if lid < 2.4 and MODE == "final":
                    c = IRIS[0]
        img[y, x] = c
    if not line:
        icx, icy = L2S(iu, iv)
        k = max(0.7, sx)
        for (dx, dy, rx, ry, c) in ((-2.1 * k, -2.3, 1.5 * k, 1.4, (250, 250, 252)),
                                    (2.2 * k, 3.0, 0.9 * k, 0.6, (150, 112, 86))):
            hx, hy = icx + dx, icy + dy
            hm = raster(smooth([(hx + rx * math.cos(q), hy + ry * math.sin(q)) for q in np.linspace(0, 6.28, 9)[:-1]])) >= 0.45
            img[hm & M] = c
    # thick upper lash line, heaviest toward the outer corner, with an upswept flick
    thick = lambda u: 1.7 + 2.0 * float(np.clip((u + 0.3 * w) / (0.9 * w), 0, 1))
    us = np.linspace(-w - 0.5, w + 0.6, 40)
    lash = [L2S(u, top_v(u) - thick(u)) for u in us] + [L2S(u, top_v(u) + 0.7) for u in us[::-1]]
    fill(lash, LASH if not line else INK, thr=0.45)
    flick = [L2S(w - 4.0, top_v(w - 4.0) - thick(w - 4.0)), L2S(w + 5.8, -6.6), L2S(w + 1.0, -2.2)]
    fill(flick, LASH if not line else INK, thr=0.45)
    # lower lash: a soft line along the outer part, darker at the corner
    low = [L2S(u, bot_v(u) + 0.75) for u in np.linspace(0.05 * w, w - 0.5, 14)]
    stroke(low, LASH_LOW if not line else INK, 1.0)
    wedge = [L2S(w - 6.5, bot_v(w - 6.5) + 0.2), L2S(w + 0.7, -2.6), L2S(w - 4.0, bot_v(w - 4.0) + 1.5)]
    fill(wedge, LASH if not line else INK, thr=0.45)
    if MODE == "final":
        ic = [L2S(-w + 0.4, 1.3), L2S(-w + 2.6, 0.2), L2S(-w + 2.4, 3.0)]
        fill(ic, (214, 150, 152), thr=0.4)


def draw_nose_mouth():
    line = MODE in ("line", "guides")
    nose_line = INK if line else SKIN_LINE
    stroke(smooth(PP([(778, 1098), (775, 1160), (772, 1215), (768, 1244)]), 6, closed=False), nose_line, 0.8)
    stroke(smooth(PP([(762, 1250), (778, 1274), (808, 1290), (842, 1292), (872, 1280), (896, 1258)]), 6, closed=False),
           nose_line, 0.9)
    nos = INK if line else NOSTRIL
    stroke(smooth(PP([(786, 1270), (800, 1266), (812, 1272)]), 6, closed=False), nos, 1.3)
    stroke(PP([(864, 1246), (880, 1242)]), nos, 1.0)
    if MODE == "final":
        fill(smooth(PP([(830, 1426), (872, 1420), (922, 1414), (972, 1409), (1004, 1410), (988, 1440), (950, 1464),
                        (902, 1476), (860, 1470), (836, 1450)]), 6), mix(SKIN, LIP, 0.45))
        fill(smooth(PP([(890, 1446), (916, 1441), (940, 1442), (926, 1451), (902, 1454)]), 6), mix(SKIN_HI, LIP, 0.25))
    mouth = smooth(PP([(796, 1430), (806, 1421), (840, 1417), (880, 1412), (918, 1406), (965, 1401), (1000, 1400),
                       (1020, 1404)]), 6, closed=False)
    stroke(mouth, INK if line else LIP_LINE, 1.0)
    if line:
        stroke(smooth(PP([(846, 1490), (890, 1498), (940, 1494), (966, 1484)]), 6, closed=False), INK, 0.7)


def draw_glasses(face):
    """Rectangular frames: thick black on top fading to clear at the bottom (like his), a lens
    tint with a couple of glare streaks, and the temple arm running back to the ear."""
    line = MODE in ("line", "guides")
    for lens in (NEAR_LENS, FAR_LENS):
        poly = smooth(PP(lens), 6)
        ML = raster(poly) >= 0.5
        if MODE == "final":
            img[ML] = img[ML] * 0.9 + np.asarray(LENS, np.float32) * 0.1
            outside = ML & ~face
            img[outside] = img[outside] * 0.8 + np.asarray(LENS, np.float32) * 0.2
    if MODE == "final":
        for a, b in (((548, 1180), (590, 1130)), ((556, 1214), (604, 1150)), ((1062, 1138), (1096, 1100))):
            g = PP([a, b])
            stroke(g, (250, 252, 255), 0.9, clip=raster(smooth(PP(FAR_LENS if a[0] < 800 else NEAR_LENS), 6)))
    # far temple arm, seen through the far lens where it sticks out past the cheek
    arm_far = PP([(532, 1160), (575, 1170), (618, 1184)])
    stroke(arm_far, FRAME if not line else INK, 2.4, clip=(~face).astype(np.float32))
    # near temple arm: black outer edge over a silver inner side
    top_edge = smooth(PP([(1112, 936), (1180, 950), (1240, 966), (1286, 982)]), 6, closed=False)
    bot_edge = smooth(PP([(1286, 994), (1240, 986), (1186, 980), (1114, 975)]), 6, closed=False)
    arm = top_edge + bot_edge
    if not line:
        fill(arm, FRAME_SHINE if MODE == "final" else FRAME_CLEAR)
    stroke(top_edge, FRAME if not line else INK, 1.6)
    stroke(bot_edge, FRAME_MID if not line else INK, 0.8)
    # rims: thick and dark along the top, fading to a thin clear line at the bottom
    for lens, top_n in ((NEAR_LENS, 9), (FAR_LENS, 9)):
        pts = smooth(PP(lens), 6)
        n = len(pts)
        ys_ = [p[1] for p in pts]
        ytop, ybot = min(ys_), max(ys_)
        for i in range(n):
            a, b = pts[i], pts[(i + 1) % n]
            t = ((a[1] + b[1]) / 2 - ytop) / (ybot - ytop)
            if line:
                stroke([a, b], INK, 1.0)
                continue
            if t < 0.3:
                c, wdt = FRAME, 3.2 - 2.0 * t
            elif t < 0.7:
                k = (t - 0.3) / 0.4
                c, wdt = mix(FRAME, FRAME_CLEAR, k), 2.6 - 1.6 * k
            else:
                c, wdt = FRAME_CLEAR, 0.9
            stroke([a, b], c, wdt)
    # bridge over the nose
    stroke(smooth(PP([(770, 1068), (790, 1062), (812, 1058), (846, 1062)]), 6, closed=False), FRAME if not line else INK, 2.2)
    if MODE == "final":
        stroke(PP([(1060, 945), (1112, 941)]), FRAME_MID, 0.8)
        stroke(PP([(1118, 950), (1118, 1000)]), (120, 124, 136), 0.7)


# ----------------------------------------------------------------------------- finishing
def glow(figure):
    """Backlight bloom: background pixels right next to his lit edges get brighter."""
    near = np.zeros_like(figure)
    for dx, dy in ((1, 0), (2, 0), (1, 1), (1, -1), (0, 1), (0, -1)):
        near |= shifted(figure, dx, dy)
    near &= ~figure
    right = XX > W * 0.55
    k = (near & right)
    img[k] = img[k] * 0.55 + np.asarray((255, 252, 240), np.float32) * 0.45


def stray_hairs():
    col = HAIR_RIM if MODE == "final" else (INK if MODE == "line" else HAIR)
    for pts, wdt in (([(904, 500), (912, 470), (930, 448)], 0.6), ([(968, 526), (998, 492), (1036, 476)], 0.6),
                     ([(874, 540), (858, 506), (842, 488)], 0.5), ([(1180, 612), (1214, 586), (1246, 578)], 0.6),
                     ([(1306, 842), (1330, 830), (1352, 832)], 0.5), ([(548, 824), (522, 812), (500, 812)], 0.5),
                     ([(1318, 962), (1342, 954)], 0.5)):
        stroke(smooth(PP(pts), 6, closed=False), col, wdt)


def render(mode="final"):
    global MODE
    MODE = mode
    img[:] = 0
    draw_background()
    neck = draw_neck()
    shirt = draw_shirt()
    ear = draw_ear()
    face = draw_face()
    hair = draw_hair(face)
    skin = face & ~hair
    draw_brows(skin)
    draw_nose_mouth()
    draw_eye(P(997, 1038), -11.4, 1.03, 1.0, 1, 0.6)
    draw_eye(P(702, 1122), -16.7, 0.6, 0.95, -1, -3.2)
    draw_glasses(face)
    stray_hairs()
    if MODE == "final":
        glow(neck | ear | hair | shirt | face)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


if __name__ == "__main__":
    raw = render("final")
    raw.save("me_pixel_raw.png")
    raw.resize((W * SCALE, H * SCALE), Image.NEAREST).save("me_pixel.png")
    print("saved me_pixel.png", raw.size)
