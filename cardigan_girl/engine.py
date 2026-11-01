"""Tiny raster paint engine, drawing-app style.

Everything is pixels in numpy float32:
  * Mask      - bbox-local coverage (0..1), produced by brushes / fills
  * Layer     - full-canvas premultiplied RGBA with paint modes
                (normal, lock-alpha, multiply, screen, erase) = clipping-mask workflow
  * strokes   - pressure-tapered brush strokes (union of capsules, analytic AA)
  * fills     - polygon fill (4x supersampled), ellipses, gradients, blur
"""
import math
import numpy as np
from PIL import Image, ImageDraw

F32 = np.float32


def rgb(h):
    h = h.lstrip('#')
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], F32)


# ----------------------------------------------------------------------------- masks
class Mask:
    __slots__ = ('x0', 'y0', 'a')

    def __init__(self, x0, y0, a):
        self.x0, self.y0, self.a = int(x0), int(y0), np.asarray(a, F32)

    @property
    def x1(self): return self.x0 + self.a.shape[1]

    @property
    def y1(self): return self.y0 + self.a.shape[0]

    def empty(self): return self.a.size == 0 or float(self.a.max()) <= 0

    def pad_to(self, x0, y0, x1, y1):
        out = np.zeros((y1 - y0, x1 - x0), F32)
        sx0, sy0 = max(self.x0, x0), max(self.y0, y0)
        sx1, sy1 = min(self.x1, x1), min(self.y1, y1)
        if sx1 > sx0 and sy1 > sy0:
            out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = self.a[sy0 - self.y0:sy1 - self.y0, sx0 - self.x0:sx1 - self.x0]
        return out

    def __or__(self, o):          # union
        x0, y0, x1, y1 = min(self.x0, o.x0), min(self.y0, o.y0), max(self.x1, o.x1), max(self.y1, o.y1)
        return Mask(x0, y0, np.maximum(self.pad_to(x0, y0, x1, y1), o.pad_to(x0, y0, x1, y1)))

    def __and__(self, o):         # intersection (product)
        x0, y0, x1, y1 = max(self.x0, o.x0), max(self.y0, o.y0), min(self.x1, o.x1), min(self.y1, o.y1)
        if x1 <= x0 or y1 <= y0:
            return Mask(x0, y0, np.zeros((0, 0), F32))
        return Mask(x0, y0, self.pad_to(x0, y0, x1, y1) * o.pad_to(x0, y0, x1, y1))

    def __sub__(self, o):         # self minus o
        return Mask(self.x0, self.y0, self.a * (1.0 - o.pad_to(self.x0, self.y0, self.x1, self.y1)))

    def __mul__(self, k):
        return Mask(self.x0, self.y0, self.a * F32(k))

    def blur(self, r):
        if r <= 0:
            return self
        p = int(math.ceil(r * 2)) + 1
        a = np.pad(self.a, p)
        for _ in range(3):
            a = _box(_box(a, r, 0), r, 1)
        return Mask(self.x0 - p, self.y0 - p, a)

    def grow(self, r):
        """morphological-ish dilation via blur + threshold"""
        b = self.blur(r)
        return Mask(b.x0, b.y0, np.clip(b.a * 2.2, 0, 1))

    def shift(self, dx, dy):
        return Mask(self.x0 + int(round(dx)), self.y0 + int(round(dy)), self.a)

    def full(self, W, H):
        return self.pad_to(0, 0, W, H)


def _box(a, r, axis):
    r = int(round(r))
    if r < 1:
        return a
    c = np.cumsum(np.pad(a, [(r + 1, r) if i == axis else (0, 0) for i in range(2)]), axis=axis, dtype=np.float64)
    n = a.shape[axis]
    hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
    lo = np.take(c, np.arange(0, n), axis=axis)
    return ((hi - lo) / (2 * r + 1)).astype(F32)


def union(*ms):
    ms = [m for m in ms if m is not None]
    out = ms[0]
    for m in ms[1:]:
        out = out | m
    return out


# ----------------------------------------------------------------------------- geometry
def _cr(p0, p1, p2, p3, n, alpha=0.5):
    def tj(ti, a, b):
        return ti + max(float(np.hypot(*(b - a))), 1e-4) ** alpha
    t0 = 0.0; t1 = tj(t0, p0, p1); t2 = tj(t1, p1, p2); t3 = tj(t2, p2, p3)
    t = np.linspace(t1, t2, n, endpoint=False)[:, None]
    A1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
    A2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
    A3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
    B1 = (t2 - t) / (t2 - t0) * A1 + (t - t0) / (t2 - t0) * A2
    B2 = (t3 - t) / (t3 - t1) * A2 + (t - t1) / (t3 - t1) * A3
    return (t2 - t) / (t2 - t1) * B1 + (t - t1) / (t2 - t1) * B2


def spline_idx(pts, closed=False):
    """Centripetal Catmull-Rom through pts; (x, y, 'c') marks a sharp corner.
    Returns (dense_points, index_of_each_control_point_in_dense)."""
    P = [np.array(p[:2], float) for p in pts]
    corner = [len(p) > 2 and p[2] == 'c' for p in pts]
    n = len(P)
    if n == 1:
        return np.array([P[0]]), [0]
    nseg = n if closed else n - 1
    out, idx, count = [], [], 0
    for i in range(nseg):
        j = (i + 1) % n
        p1, p2 = P[i], P[j]
        if (not closed and i == 0) or corner[i]:
            p0 = 2 * p1 - p2
        else:
            p0 = P[i - 1]
        if (not closed and j == n - 1) or corner[j]:
            p3 = 2 * p2 - p1
        else:
            p3 = P[(j + 1) % n]
        k = max(3, int(np.hypot(*(p2 - p1)) / 1.2))
        seg = _cr(p0, p1, p2, p3, k)
        idx.append(count)
        out.append(seg); count += len(seg)
    if not closed:
        out.append(P[-1][None, :]); idx.append(count); count += 1
    return np.vstack(out), idx


def spline(pts, closed=False):
    return spline_idx(pts, closed)[0]


class Path:
    """dense smooth path through control points; sub-paths share the exact same curve"""

    def __init__(self, pts, closed=True):
        self.ctrl = pts
        self.closed = closed
        self.d, self.idx = spline_idx(pts, closed)

    def mask(self):
        return poly(self.d)

    def seg(self, i0, i1):
        """dense sub-path from control point i0 to i1 (forward; wraps if closed)"""
        a, b = self.idx[i0 % len(self.idx)], self.idx[i1 % len(self.idx)]
        if b >= a and not (self.closed and i1 > i0 and b == a):
            return self.d[a:b + 1]
        return np.vstack([self.d[a:], self.d[:b + 1]])


def resample(path, step=1.0):
    path = np.asarray(path, float)
    d = np.hypot(*np.diff(path, axis=0).T)
    s = np.concatenate([[0], np.cumsum(d)])
    if s[-1] < 1e-6:
        return path[:1], np.array([0.0])
    n = max(2, int(s[-1] / step) + 1)
    u = np.linspace(0, s[-1], n)
    return np.stack([np.interp(u, s, path[:, 0]), np.interp(u, s, path[:, 1])], 1), u / s[-1]


# ----------------------------------------------------------------------------- brushes
def capsules(pts, radii, W=None, H=None):
    pts = np.asarray(pts, float); radii = np.asarray(radii, float)
    R = float(radii.max()) + 2
    x0 = int(math.floor(pts[:, 0].min() - R)); y0 = int(math.floor(pts[:, 1].min() - R))
    x1 = int(math.ceil(pts[:, 0].max() + R)) + 1; y1 = int(math.ceil(pts[:, 1].max() + R)) + 1
    a = np.zeros((y1 - y0, x1 - x0), F32)
    if len(pts) == 1:
        pts = np.vstack([pts, pts + 1e-3]); radii = np.array([radii[0], radii[0]])
    for i in range(len(pts) - 1):
        (ax, ay), (bx, by) = pts[i], pts[i + 1]
        ra, rb = radii[i], radii[i + 1]
        if max(ra, rb) <= 0.02:
            continue
        rr = max(ra, rb) + 1.5
        sx0 = int(math.floor(min(ax, bx) - rr)) - x0; sx1 = int(math.ceil(max(ax, bx) + rr)) - x0 + 1
        sy0 = int(math.floor(min(ay, by) - rr)) - y0; sy1 = int(math.ceil(max(ay, by) + rr)) - y0 + 1
        X = (np.arange(sx0, sx1) + x0 + 0.5)[None, :]
        Y = (np.arange(sy0, sy1) + y0 + 0.5)[:, None]
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy + 1e-9
        t = np.clip(((X - ax) * dx + (Y - ay) * dy) / L2, 0, 1)
        d = np.hypot(X - (ax + t * dx), Y - (ay + t * dy))
        r = ra + (rb - ra) * t
        cov = np.clip(r - d + 0.5, 0, 1) * np.clip(r * 2.0, 0, 1)
        sub = a[sy0:sy1, sx0:sx1]
        np.maximum(sub, cov.astype(F32), out=sub)
    return Mask(x0, y0, a)


def pressure(u, taper=(0.18, 0.3), floor=0.12, shape=1.0):
    """pen pressure along normalized arc length u: ramps up, holds, ramps down."""
    ti, to = taper
    rin = np.clip(u / ti, 0, 1) if ti > 0 else np.ones_like(u)
    rout = np.clip((1 - u) / to, 0, 1) if to > 0 else np.ones_like(u)
    rin = rin * rin * (3 - 2 * rin); rout = rout * rout * (3 - 2 * rout)
    return floor + (1 - floor) * (rin * rout) ** shape


def line(pts, w, taper=(0.18, 0.3), floor=0.12, closed=False, step=0.8, profile=None):
    """Pressure stroke through control points. w = max width in px."""
    path = spline(pts, closed=closed) if len(pts) > 2 else np.asarray([p[:2] for p in pts], float)
    p, u = resample(path, step)
    prof = pressure(u, taper, floor) if profile is None else profile(u)
    return capsules(p, prof * w / 2.0)


def stroke(dense, w, taper=(0.18, 0.3), floor=0.12, step=0.8, profile=None):
    """pressure stroke along an already-dense path (e.g. Path.seg)"""
    p, u = resample(dense, step)
    prof = pressure(u, taper, floor) if profile is None else profile(u)
    return capsules(p, prof * w / 2.0)


def dot(x, y, r):
    return ellipse(x, y, r, r)


# ----------------------------------------------------------------------------- fills
def poly(pts, ss=4):
    pts = np.asarray(pts, float)
    x0 = int(math.floor(pts[:, 0].min())) - 2; y0 = int(math.floor(pts[:, 1].min())) - 2
    x1 = int(math.ceil(pts[:, 0].max())) + 3; y1 = int(math.ceil(pts[:, 1].max())) + 3
    w, h = x1 - x0, y1 - y0
    img = Image.new('L', (w * ss, h * ss), 0)
    ImageDraw.Draw(img).polygon([((x - x0) * ss, (y - y0) * ss) for x, y in pts], fill=255)
    a = np.asarray(img, F32).reshape(h, ss, w, ss).mean(axis=(1, 3)) / 255.0
    return Mask(x0, y0, a)


def shape(pts, closed=True):
    """smooth closed shape through control points (corners allowed) -> filled mask"""
    return poly(spline(pts, closed=closed))


def ellipse(cx, cy, rx, ry, rot=0.0):
    R = max(rx, ry) + 2
    x0, y0 = int(math.floor(cx - R)), int(math.floor(cy - R))
    x1, y1 = int(math.ceil(cx + R)) + 1, int(math.ceil(cy + R)) + 1
    X = (np.arange(x0, x1) + 0.5)[None, :] - cx
    Y = (np.arange(y0, y1) + 0.5)[:, None] - cy
    c, s = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    u, v = X * c + Y * s, -X * s + Y * c
    d = np.sqrt((u / rx) ** 2 + (v / ry) ** 2) + 1e-9
    g = np.sqrt((u / rx ** 2) ** 2 + (v / ry ** 2) ** 2) / d + 1e-9
    a = np.clip((1 - d) / g + 0.5, 0, 1)
    return Mask(x0, y0, a)


def lin_grad(m, p0, p1, gamma=1.0):
    """mask * t where t goes 0 at p0 -> 1 at p1"""
    X = (np.arange(m.x0, m.x1) + 0.5)[None, :]
    Y = (np.arange(m.y0, m.y1) + 0.5)[:, None]
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    t = np.clip(((X - p0[0]) * dx + (Y - p0[1]) * dy) / (dx * dx + dy * dy), 0, 1) ** gamma
    return Mask(m.x0, m.y0, m.a * t)


def rad_grad(m, c, r_in, r_out, inv=False):
    X = (np.arange(m.x0, m.x1) + 0.5)[None, :]
    Y = (np.arange(m.y0, m.y1) + 0.5)[:, None]
    d = np.hypot(X - c[0], Y - c[1])
    t = np.clip((d - r_in) / max(r_out - r_in, 1e-6), 0, 1)
    t = t * t * (3 - 2 * t)
    return Mask(m.x0, m.y0, m.a * (t if inv else 1 - t))


def rect(x0, y0, x1, y1):
    return Mask(int(x0), int(y0), np.ones((int(y1) - int(y0), int(x1) - int(x0)), F32))


def smooth_noise(W, H, cell, seed=0, octaves=3):
    rng = np.random.default_rng(seed)
    out = np.zeros((H, W), F32); amp = 1.0; tot = 0
    for o in range(octaves):
        c = max(2, int(cell / (2 ** o)))
        g = rng.random((H // c + 3, W // c + 3)).astype(F32)
        im = Image.fromarray(g, mode='F').resize(((W // c + 3) * c, (H // c + 3) * c), Image.BICUBIC)
        out += amp * np.asarray(im, F32)[:H, :W]; tot += amp; amp *= 0.5
    out /= tot
    return (out - out.min()) / (out.max() - out.min() + 1e-6)


# ----------------------------------------------------------------------------- layers
class Layer:
    """premultiplied RGBA, full canvas"""

    def __init__(self, W, H):
        self.W, self.H = W, H
        self.p = np.zeros((H, W, 3), F32)
        self.a = np.zeros((H, W), F32)

    def _win(self, m):
        x0, y0 = max(m.x0, 0), max(m.y0, 0)
        x1, y1 = min(m.x1, self.W), min(m.y1, self.H)
        if x1 <= x0 or y1 <= y0:
            return None
        return (slice(y0, y1), slice(x0, x1)), m.a[y0 - m.y0:y1 - m.y0, x0 - m.x0:x1 - m.x0]

    def paint(self, m, color, opacity=1.0, mode='normal', tex=None):
        """mode: normal | lock (paint inside existing alpha) | multiply | screen | add | erase"""
        if m is None:
            return
        w = self._win(m)
        if w is None:
            return
        sl, a = w
        a = a * F32(opacity)
        if tex is not None:
            a = a * tex[sl]
        c = np.asarray(color, F32)
        P, A = self.p[sl], self.a[sl]
        a3 = a[..., None]
        if mode == 'normal':
            P[...] = c * a3 + P * (1 - a3)
            A[...] = a + A * (1 - a)
        elif mode == 'lock':
            P[...] = P * (1 - a3) + c * a3 * A[..., None]
        elif mode == 'multiply':
            P[...] = P * (1 - a3 + a3 * c)
        elif mode == 'screen':
            A3 = A[..., None]
            P[...] = A3 - (A3 - P) * (1 - a3 * c)
        elif mode == 'add':
            A3 = A[..., None]
            P[...] = np.minimum(P + c * a3 * A3, A3)
        elif mode == 'erase':
            P[...] = P * (1 - a3)
            A[...] = A * (1 - a)
        else:
            raise ValueError(mode)

    def alpha_mask(self):
        ys, xs = np.where(self.a > 0.003)
        if len(ys) == 0:
            return Mask(0, 0, np.zeros((0, 0), F32))
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        return Mask(x0, y0, self.a[y0:y1, x0:x1].copy())


class Canvas:
    def __init__(self, W, H, paper='#ffffff'):
        self.W, self.H = W, H
        self.rgb = np.ones((H, W, 3), F32) * rgb(paper)

    def layer(self):
        return Layer(self.W, self.H)

    def comp(self, L, opacity=1.0):
        a = L.a[..., None] * opacity
        self.rgb[...] = L.p * opacity + self.rgb * (1 - a)

    def paint(self, m, color, opacity=1.0):
        L = Layer(self.W, self.H)
        L.paint(m, color, opacity)
        self.comp(L)

    def save(self, path, scale=1.0):
        im = Image.fromarray((np.clip(self.rgb, 0, 1) * 255 + 0.5).astype(np.uint8))
        if scale != 1.0:
            im = im.resize((int(self.W * scale), int(self.H * scale)), Image.LANCZOS)
        im.save(path)
        return im
