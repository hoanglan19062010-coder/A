"""Body parts: fills, clipped shading, and lineart for every piece of the outfit."""
from body import *

def bp(pts, closed=True):
    return Path([B(*p[:2]) + tuple(p[2:]) for p in pts], closed=closed)

def Bs(pts):
    return [B(*p[:2]) + tuple(p[2:]) for p in pts]

def edge(L, path, i0, i1, w, col, taper=(0.15, 0.15), floor=0.2, op=1.0):
    L.paint(stroke(path.seg(i0, i1), w, taper=taper, floor=floor), col, op)

def ln(L, pts, w, col, taper=(0.3, 0.3), floor=0.1, op=1.0):
    L.paint(line(Bs(pts), w, taper=taper, floor=floor), col, op)

def form_shadow(m, k, dy=0.0):
    """strip on the side away from the light (light from upper left)"""
    return (m - m.shift(-k * HU, -dy * HU)).blur(1.2) & m

SAGE, NAVY, CAMEL, TIGHT, SHOE = '#9fb995', '#2f3b60', '#d6b68b', '#2c2830', '#6f3f28'
P_ = dict(
    sage=rgb(SAGE), sage_sh=mul(SAGE, '#788f73'), sage_sh2=mul(SAGE, '#5d725b'), sage_hl=rgb('#c8d9bd'), sage_ln=rgb('#3f4f3d'),
    navy=rgb(NAVY), navy_sh=mul(NAVY, '#1d2541'), navy_hl=rgb('#55669a'), navy_ln=rgb('#11152a'),
    camel=rgb(CAMEL), camel_sh=mul(CAMEL, '#b28e64'), camel_sh2=mul(CAMEL, '#94704a'), camel_ln=rgb('#6a4a2e'),
    tight=rgb(TIGHT), tight_sh=mul(TIGHT, '#1a171d'), tight_sheen=rgb('#6a5560'), tight_ln=rgb('#120f14'),
    shoe=rgb(SHOE), shoe_sh=mul(SHOE, '#45251a'), shoe_hl=rgb('#b07a5c'), sole=rgb('#2b1a12'), shoe_ln=rgb('#21130d'),
    button=rgb('#efe6d6'), button_ln=rgb('#6b5a48'),
)

def F(mode, col):
    return C['white'] if mode == 'line' else col

# ------------------------------------------------------------------ legs + shoes
LEG_L_OUT = [(-0.47, 3.26), (-0.458, 3.55), (-0.425, 3.95), (-0.368, 4.40), (-0.296, 4.80), (-0.262, 4.97), (-0.252, 5.07),
             (-0.274, 5.28), (-0.252, 5.60), (-0.192, 5.95), (-0.150, 6.14)]
LEG_L_IN = [(-0.02, 3.64), (-0.026, 3.95), (-0.040, 4.40), (-0.046, 4.80), (-0.030, 4.96), (-0.030, 5.10), (-0.008, 5.30),
            (-0.014, 5.62), (0.0, 5.95), (0.004, 6.14)]
LEG_R_OUT = [(0.47, 3.32), (0.458, 3.60), (0.425, 4.00), (0.365, 4.45), (0.290, 4.84), (0.256, 5.00), (0.258, 5.11),
             (0.300, 5.32), (0.320, 5.62), (0.332, 5.92), (0.338, 6.10)]
LEG_R_IN = [(0.02, 3.64), (0.026, 3.95), (0.036, 4.40), (0.031, 4.80), (0.013, 4.99), (0.023, 5.12), (0.062, 5.35),
            (0.128, 5.68), (0.170, 5.95), (0.184, 6.10)]

def leg_path(out, inn):
    return bp(out + list(reversed(inn)))

def draw_legs(L, mode):
    ms = []
    for out, inn, name in ((LEG_R_OUT, LEG_R_IN, 'R'), (LEG_L_OUT, LEG_L_IN, 'L')):   # free leg behind the weight leg
        p = leg_path(out, inn); m = p.mask(); ms.append(m)
        L.paint(m, F(mode, P_['tight']))
        if SHADE(mode):
            # denier sheen: skin shows through where the knit stretches over the front of the leg
            k = 0 if name == 'L' else 1
            ctr = [lerp(out[i], inn[min(i, len(inn) - 1)], 0.42) for i in range(len(out))]
            sheen = line(Bs(ctr[2:-1]), 0.11 * HU, taper=(0.25, 0.25), floor=0.3).blur(10) & m
            L.paint(sheen, P_['tight_sheen'], 0.75)
            kn = ctr[5]
            L.paint(ellipse(*B(kn[0], kn[1] + 0.01), 0.05 * HU, 0.07 * HU).blur(8) & m, P_['tight_sheen'], 0.55)
            L.paint(form_shadow(m, 0.075), P_['tight_sh'], 0.85, mode='multiply')
        n = len(out)
        edge(L, p, 0, n - 1, 4.2, P_['tight_ln'], taper=(0.0, 0.05))
        edge(L, p, n, 2 * n - 1, 3.6, P_['tight_ln'], taper=(0.05, 0.0))
        # knee hint
        kx, ky = out[5]; ix, iy = inn[4]
        ln(L, [lerp((kx, ky), (ix, iy), 0.35), lerp((kx, ky), (ix, iy), 0.5), lerp((kx, ky), (ix, iy), 0.65)], 1.6,
           P_['tight_ln'] * 0.4 + P_['tight_sheen'] * 0.6, taper=(0.4, 0.4), op=0.8)
    return ms

def draw_shoes(L, mode):
    for ank, ang, name in ((ANK_R, -24, 'R'), (ANK_L, 3, 'L')):
        ax, ay = ank
        def T(p):
            x, y = p; a = math.radians(ang)
            return (ax + x * math.cos(a) - y * math.sin(a), ay + x * math.sin(a) + y * math.cos(a))
        # outline: narrow at the ankle, widest at the toe box, rounded toe
        body = [(-0.074, -0.005), (-0.090, 0.07), (-0.112, 0.15), (-0.118, 0.200), (-0.090, 0.236), (-0.040, 0.250), (0.0, 0.252),
                (0.040, 0.250), (0.090, 0.236), (0.118, 0.200), (0.112, 0.15), (0.090, 0.07), (0.074, -0.005)]
        p = bp([T(q) for q in body]); m = p.mask()
        L.paint(m, F(mode, P_['shoe']))
        # inside of the shoe behind the U-shaped opening
        inside = bp([T(q) for q in [(-0.072, 0.0), (-0.05, 0.035), (0.0, 0.052), (0.05, 0.035), (0.072, 0.0), (0.0, 0.012)]])
        L.paint(inside.mask(), F(mode, P_['sole']))
        sole = bp([T(q) for q in [(-0.117, 0.205), (-0.088, 0.240), (0.0, 0.256), (0.088, 0.240), (0.117, 0.205),
                                  (0.119, 0.226), (0.090, 0.262), (0.0, 0.276), (-0.090, 0.262), (-0.119, 0.226)]])
        L.paint(sole.mask(), F(mode, P_['sole']))
        if SHADE(mode):
            L.paint(form_shadow(m, 0.05), P_['shoe_sh'], 0.9, mode='multiply')
            L.paint(lin_grad(m, B(*T((0, 0.09))), B(*T((0, 0.03)))), P_['shoe_sh'], 0.7, mode='multiply')
            shine = bp([T(q) for q in [(-0.085, 0.165), (-0.06, 0.150), (-0.03, 0.152), (-0.04, 0.172), (-0.075, 0.190)]]).mask().blur(0.8)
            L.paint(shine & m, P_['shoe_hl'], 0.9)
        edge(L, p, 0, len(body) - 1, 3.8, P_['shoe_ln'], taper=(0.1, 0.1))
        edge(L, sole, 5, 9, 3.4, P_['shoe_ln'], taper=(0.05, 0.05))
        edge(L, inside, 0, 4, 2.6, P_['shoe_ln'], taper=(0.1, 0.1))
        ln(L, [T(q) for q in [(-0.100, 0.108), (-0.05, 0.128), (0.0, 0.133), (0.05, 0.128), (0.100, 0.108)]], 2.4, P_['shoe_ln'], taper=(0.15, 0.15))
        ln(L, [T(q) for q in [(-0.095, 0.132), (-0.05, 0.152), (0.0, 0.157), (0.05, 0.152), (0.095, 0.132)]], 1.8, P_['shoe_ln'], taper=(0.2, 0.2), op=0.7)
        ln(L, [T(q) for q in [(-0.026, 0.138), (0.0, 0.142), (0.026, 0.138)]], 2.2, P_['shoe_ln'], taper=(0.3, 0.3))

# ------------------------------------------------------------------ skin torso (neck + chest)
NECK = [(-0.138, 0.84), (-0.137, 1.00), (-0.148, 1.13), (-0.18, 1.22), (-0.30, 1.29), (-0.48, 1.38), (-0.46, 1.72),
        (0.46, 1.72), (0.48, 1.36), (0.30, 1.27), (0.18, 1.21), (0.148, 1.13), (0.137, 1.00), (0.138, 0.84)]

def draw_skin_torso(L, mode, face_mask=None):
    p = bp(NECK); m = p.mask()
    L.paint(m, F(mode, C['skin']))
    if SHADE(mode):
        if face_mask is not None:
            cast = face_mask.shift(0.012 * HU, 0.115 * HU).blur(1.5)
            L.paint(cast & m, C['skin_sh2'], 1.0, mode='multiply')
        L.paint(form_shadow(m, 0.05), C['skin_sh'], 0.9, mode='multiply')
    edge(L, p, 0, 3, 3.2, C['skin_ln'], taper=(0.0, 0.5))
    edge(L, p, 10, 13, 3.2, C['skin_ln'], taper=(0.5, 0.0))
    # collarbones + pit of the neck
    ln(L, [(-0.035, 1.338), (-0.11, 1.312), (-0.205, 1.31)], 2.0, C['skin_ln'], taper=(0.5, 0.6), op=0.75)
    ln(L, [(0.035, 1.338), (0.11, 1.31), (0.205, 1.305)], 2.0, C['skin_ln'], taper=(0.5, 0.6), op=0.75)
    ln(L, [(-0.012, 1.295), (0.0, 1.312), (0.012, 1.295)], 1.6, C['skin_ln'], taper=(0.4, 0.4), op=0.6)
    return m

# ------------------------------------------------------------------ top (navy)
TOP = [(-0.235, 1.355), (-0.17, 1.50), (-0.08, 1.585), (0.0, 1.60), (0.08, 1.585), (0.17, 1.50), (0.235, 1.355),
       (0.42, 1.40), (0.44, 1.75), (0.40, 2.10), (0.31, 2.52), (-0.31, 2.52), (-0.40, 2.10), (-0.44, 1.75), (-0.42, 1.40)]

def draw_top(L, mode):
    p = bp(TOP); m = p.mask()
    L.paint(m, F(mode, P_['navy']))
    if SHADE(mode):
        L.paint(form_shadow(m, 0.10), P_['navy_sh'], 0.9, mode='multiply')
        for sx in (-1, 1):
            bx, by = B(sx * 0.17, 1.93)
            cres = (ellipse(bx, by + 0.05 * HU, 0.17 * HU, 0.13 * HU) - ellipse(bx, by, 0.17 * HU, 0.13 * HU)).blur(4)
            L.paint(cres & m, P_['navy_sh'], 0.85, mode='multiply')
        hl = shape(Bs([(-0.125, 1.80), (-0.085, 1.785), (-0.07, 1.83), (-0.095, 1.86), (-0.12, 1.85)])).blur(2)
        L.paint(hl & m, P_['navy_hl'], 0.35)
    edge(L, p, 0, 6, 2.8, P_['navy_ln'], taper=(0.2, 0.2))
    ln(L, [(-0.005, 2.05), (0.002, 2.13), (0.0, 2.20)], 1.6, P_['navy_ln'], taper=(0.5, 0.5), op=0.6)
    ln(L, [(-0.07, 2.24), (-0.045, 2.38), (-0.055, 2.50)], 1.5, P_['navy_ln'], taper=(0.6, 0.4), op=0.55)
    ln(L, [(0.085, 2.25), (0.06, 2.40), (0.07, 2.50)], 1.5, P_['navy_ln'], taper=(0.6, 0.4), op=0.55)
    # contrast binding inside the scoop neckline
    bind = [(x * 0.90, y + 0.022) for x, y in TOP[1:6]]
    ln(L, bind, 2.2, P_['navy_hl'] * 0.4 + C['white'] * 0.6, taper=(0.2, 0.2), op=0.85)
    return m

# ------------------------------------------------------------------ skirt (camel, button front)
SK_HEM = [(-0.575, 3.72), (-0.49, 3.745), (-0.40, 3.722), (-0.29, 3.762), (-0.17, 3.738), (-0.05, 3.776), (0.07, 3.748),
          (0.19, 3.778), (0.31, 3.745), (0.43, 3.768), (0.53, 3.735), (0.585, 3.745)]
SKIRT = [(-0.315, 2.595), (-0.385, 2.90), (-0.455, 3.25), (-0.528, 3.60)] + SK_HEM + [(0.538, 3.60), (0.468, 3.25), (0.398, 2.90), (0.325, 2.605)]

def draw_skirt(L, mode, cardigan_mask=None):
    wb = bp([(-0.305, 2.50, 'c'), (0.315, 2.51, 'c'), (0.325, 2.605, 'c'), (-0.315, 2.595, 'c')]); wbm = wb.mask()
    p = bp(SKIRT); m = p.mask()
    L.paint(m, F(mode, P_['camel'])); L.paint(wbm, F(mode, P_['camel']))
    if SHADE(mode):
        L.paint(form_shadow(m, 0.16), P_['camel_sh'], 0.8, mode='multiply')
        # folds: shadow wedges from the hip toward the hem valleys
        for x0, x1 in ((-0.32, -0.29), (-0.12, -0.05), (0.09, 0.19), (0.30, 0.43)):
            w = shape(Bs([(x0 + 0.005, 3.05), (x1 + 0.02, 3.77), (x1 - 0.045, 3.77)])).blur(4)
            L.paint(w & m, P_['camel_sh'], 0.75, mode='multiply')
        if cardigan_mask is not None:
            cast = cardigan_mask.shift(0.015 * HU, 0.05 * HU).blur(3)
            L.paint(cast & m, P_['camel_sh2'], 0.85, mode='multiply')
        L.paint(form_shadow(wbm, 0.05), P_['camel_sh'], 0.8, mode='multiply')
    n = len(SKIRT)
    edge(L, p, 0, n - 1, 3.2, P_['camel_ln'], taper=(0.05, 0.05))
    edge(L, wb, 0, 3, 2.6, P_['camel_ln'], taper=(0.05, 0.05))
    edge(L, wb, 3, 4, 2.6, P_['camel_ln'], taper=(0.05, 0.05))
    # fold lines + centre placket + buttons
    for a, b_ in (((-0.32, 3.10), (-0.29, 3.762)), ((-0.11, 3.15), (-0.05, 3.776)), ((0.09, 3.12), (0.19, 3.778)), ((0.30, 3.08), (0.43, 3.768))):
        ln(L, [a, lerp(a, b_, 0.5), b_], 2.0, P_['camel_ln'], taper=(0.7, 0.05), op=0.85)
    ln(L, [(0.012, 2.605), (0.012, 3.2), (0.010, 3.77)], 2.0, P_['camel_ln'], taper=(0.05, 0.05), op=0.9)
    for y in (2.72, 2.97, 3.22, 3.47):
        x, yy = B(-0.012, y)
        L.paint(ellipse(x, yy, 0.016 * HU, 0.016 * HU), F(mode, P_['button']))
        L.paint(ellipse(x, yy, 0.016 * HU, 0.016 * HU) - ellipse(x, yy, 0.016 * HU - 2.2, 0.016 * HU - 2.2), P_['button_ln'])
    return m

# ------------------------------------------------------------------ cardigan
CARD_L = [(-0.175, 1.20), (-0.168, 1.40), (-0.150, 1.70), (-0.133, 2.00), (-0.130, 2.40), (-0.138, 2.90), (-0.148, 3.30),
          (-0.155, 3.55, 'c'), (-0.30, 3.565), (-0.45, 3.555), (-0.565, 3.545, 'c'), (-0.555, 3.10), (-0.53, 2.60), (-0.51, 2.10),
          (-0.50, 1.70), (-0.585, 1.50), (-0.50, 1.39), (-0.38, 1.32), (-0.27, 1.26), (-0.20, 1.19, 'c')]
CARD_R = [(0.170, 1.20), (0.163, 1.40), (0.146, 1.70), (0.130, 2.00), (0.128, 2.40), (0.136, 2.90), (0.146, 3.30),
          (0.152, 3.55, 'c'), (0.30, 3.562), (0.45, 3.552), (0.57, 3.540, 'c'), (0.56, 3.10), (0.535, 2.60), (0.515, 2.10),
          (0.505, 1.70), (0.59, 1.48), (0.50, 1.37), (0.38, 1.30), (0.27, 1.245), (0.195, 1.185, 'c')]
COLLAR_BACK = [(-0.235, 1.255), (-0.205, 1.175), (-0.10, 1.128), (0.0, 1.118), (0.10, 1.126), (0.20, 1.168), (0.232, 1.245),
               (0.17, 1.215), (0.0, 1.20), (-0.17, 1.222)]

def draw_collar_back(L, mode):
    p = bp(COLLAR_BACK); m = p.mask()
    L.paint(m, F(mode, P_['sage']))
    if SHADE(mode):
        L.paint(m, P_['sage_sh2'], 1.0, mode='multiply')
    edge(L, p, 0, 7, 2.6, P_['sage_ln'], taper=(0.1, 0.1))
    return m

def placket_strip(side, pts, width=0.058):
    """band along the cardigan front edge (inside the panel)"""
    front = pts[:8]
    inner = [(x + side * width, y) for x, y, *_ in front]
    return Bs([q[:2] for q in front]) + list(reversed(Bs(inner)))

def draw_cardigan(L, mode):
    ms = []
    for pts, side in ((CARD_R, 1), (CARD_L, -1)):
        p = bp(pts); m = p.mask(); ms.append(m)
        L.paint(m, F(mode, P_['sage']))
        pl = poly(spline(placket_strip(side, pts), closed=True))
        hem = shape(Bs([pts[7][:2], pts[8], pts[9], pts[10][:2]]) + Bs([(pts[10][0], pts[10][1] - 0.085), (pts[9][0], 3.468), (pts[8][0], 3.478), (pts[7][0], 3.465)]))
        if SHADE(mode):
            if side > 0:   # panel turned away from the light: overall shadow, lit strip near the front edge
                L.paint(m, P_['sage_sh'], 0.65, mode='multiply')
            L.paint(form_shadow(m, 0.09), P_['sage_sh'], 0.85, mode='multiply')
            # drape folds: long soft shadow wedges
            for (x0, y0), (x1, y1) in ((((side * 0.37), 1.95), ((side * 0.41), 3.52)), (((side * 0.26), 2.45), ((side * 0.27), 3.52))):
                w = shape(Bs([(x0, y0), (x1 + 0.03, y1), (x1 - 0.035, y1)])).blur(5)
                L.paint(w & m, P_['sage_sh'], 0.8, mode='multiply')
            # inside of the front edge rolls slightly: thin dark line of the inner face
            L.paint(form_shadow(pl, 0.03) & m, P_['sage_sh'], 0.6, mode='multiply')
            L.paint(hem & m, P_['sage_sh'], 0.25, mode='multiply')
        # outline: front edge, hem, side (side mostly covered by the sleeve)
        edge(L, p, 0, 7, 3.0, P_['sage_ln'], taper=(0.1, 0.02))
        edge(L, p, 7, 10, 3.0, P_['sage_ln'], taper=(0.02, 0.02))
        edge(L, p, 10, 15, 3.0, P_['sage_ln'], taper=(0.02, 0.1))
        edge(L, p, 15, 19, 3.0, P_['sage_ln'], taper=(0.1, 0.1))
        # placket inner line + hem rib band line + rib ticks
        inner = [(x + side * 0.058, y) for x, y, *_ in pts[:8]]
        ln(L, inner[1:], 1.8, P_['sage_ln'], taper=(0.3, 0.05), op=0.8)
        ln(L, [(pts[7][0], 3.465), (pts[8][0], 3.478), (pts[9][0], 3.468), (pts[10][0], 3.46)], 1.8, P_['sage_ln'], taper=(0.05, 0.05), op=0.8)
        for k in range(1, 12):
            x = pts[7][0] + (pts[10][0] - pts[7][0]) * k / 12
            ln(L, [(x, 3.49), (x, 3.535)], 1.2, P_['sage_ln'], taper=(0.3, 0.3), op=0.45)
        # drape fold lines
        ln(L, [(side * 0.37, 2.05), (side * 0.39, 2.8), (side * 0.41, 3.46)], 1.8, P_['sage_ln'], taper=(0.7, 0.1), op=0.75)
        ln(L, [(side * 0.26, 2.55), (side * 0.265, 3.0), (side * 0.27, 3.46)], 1.6, P_['sage_ln'], taper=(0.7, 0.1), op=0.7)
        if side < 0:   # buttons on the viewer's left placket
            for y in (1.95, 2.30, 2.65, 3.00, 3.35):
                x0 = np.interp(y, [q[1] for q in pts[:8]], [q[0] for q in pts[:8]]) + side * 0.029
                x, yy = B(x0, y)
                L.paint(ellipse(x, yy, 0.018 * HU, 0.018 * HU), F(mode, P_['button']))
                L.paint(ellipse(x, yy, 0.018 * HU, 0.018 * HU) - ellipse(x, yy, 0.018 * HU - 2.2, 0.018 * HU - 2.2), P_['button_ln'])
        else:          # buttonholes on the right placket
            for y in (1.95, 2.30, 2.65, 3.00, 3.35):
                x0 = np.interp(y, [q[1] for q in pts[:8]], [q[0] for q in pts[:8]]) + side * 0.029
                ln(L, [(x0 - 0.012, y), (x0 + 0.012, y)], 1.6, P_['sage_ln'], taper=(0.2, 0.2), op=0.7)
    return union(*ms)

# ------------------------------------------------------------------ sleeves + sweater-paw hands
SLV = {   # forearms angle in so the hands rest beside the thighs (references: Aoi, Miko)
    -1: ([(-0.36, 1.31), (-0.50, 1.385), (-0.575, 1.47), (-0.640, 1.63), (-0.672, 2.00), (-0.690, 2.42), (-0.668, 2.85), (-0.640, 3.18)],
         [(-0.470, 1.76), (-0.462, 2.00), (-0.458, 2.42), (-0.430, 2.85), (-0.408, 3.18)]),
    +1: ([(0.36, 1.30), (0.50, 1.37), (0.575, 1.45), (0.645, 1.61), (0.678, 1.98), (0.698, 2.38), (0.664, 2.80), (0.622, 3.13)],
         [(0.472, 1.74), (0.466, 1.98), (0.462, 2.38), (0.424, 2.80), (0.390, 3.13)]),
}

def draw_arm(L, mode, side):
    out, inn = SLV[side]
    IN = -side                                     # unit direction toward the body centre
    ytop = out[-1][1]                              # cuff top
    co, ci = out[-1][0], inn[-1][0]                # outer / inner x at the cuff top
    slant = ((out[-1][0] - out[-2][0]) / (out[-1][1] - out[-2][1]) + (inn[-1][0] - inn[-2][0]) / (inn[-1][1] - inn[-2][1])) / 2
    yb = ytop + 0.14                               # cuff bottom
    sb = slant * 0.14                              # horizontal drift of the cuff bottom along the forearm
    cx = (co + ci) / 2 + sb
    # ---- hand: hidden in the cuff, only curled fingertips and the thumb tip peek out
    H_ = lambda q: (cx + IN * q[0], yb + q[1])
    tips = [(-0.050, 0.046, 0.026), (-0.020, 0.058, 0.027), (0.012, 0.060, 0.027), (0.042, 0.050, 0.025)]   # x, depth, half-width
    for x0, d, hw in tips:
        f = bp([H_(q) for q in [(x0 - hw, -0.03), (x0 - hw * 1.02, d * 0.55), (x0 - hw * 0.62, d * 0.95), (x0, d * 1.06),
                                (x0 + hw * 0.62, d * 0.95), (x0 + hw * 1.02, d * 0.55), (x0 + hw, -0.03)]])
        fm = f.mask(); L.paint(fm, F(mode, C['skin']))
        if SHADE(mode):
            L.paint(lin_grad(fm, B(cx, yb + d * 0.6), B(cx, yb - 0.01)), C['skin_sh'], 0.9, mode='multiply')
        edge(L, f, 1, 6, 2.5, C['skin_ln'], taper=(0.3, 0.3))
    th = bp([H_(q) for q in [(0.050, -0.03), (0.084, 0.0), (0.094, 0.045), (0.078, 0.072), (0.056, 0.060), (0.048, 0.02)]])
    thm = th.mask(); L.paint(thm, F(mode, C['skin']))
    if SHADE(mode):
        L.paint(form_shadow(thm, 0.015), C['skin_sh'], 0.9, mode='multiply')
    edge(L, th, 0, 5, 2.4, C['skin_ln'], taper=(0.2, 0.3))
    # ---- sleeve tube
    pts = out + list(reversed(inn))
    p = bp(pts); m = p.mask()
    L.paint(m, F(mode, P_['sage']))
    # ---- ribbed cuff: a little narrower; the sleeve blouses over it
    cuff = bp([(co + IN * 0.008, ytop - 0.012), (co + IN * 0.014 + sb * 0.5, ytop + 0.07), (co + IN * 0.020 + sb, yb),
               (ci - IN * 0.016 + sb, yb), (ci - IN * 0.010 + sb * 0.5, ytop + 0.07), (ci - IN * 0.004, ytop - 0.012)])
    cm = cuff.mask(); L.paint(cm, F(mode, P_['sage']))
    if SHADE(mode):
        L.paint(form_shadow(m, 0.085), P_['sage_sh'], 0.9, mode='multiply')
        L.paint(form_shadow(cm, 0.06), P_['sage_sh'], 0.9, mode='multiply')
        L.paint(lin_grad(cm, B(0, ytop + 0.045), B(0, ytop - 0.005)), P_['sage_sh'], 0.6, mode='multiply')
        if side > 0:
            L.paint(m, P_['sage_sh'], 0.35, mode='multiply')
        for y0, ln_ in ((2.37, 0.085), (2.49, 0.11)):        # elbow folds on the inner side
            x0 = np.interp(y0, [q[1] for q in inn], [q[0] for q in inn])
            w = line(Bs([(x0, y0), (x0 - IN * ln_ * 0.5, y0 + 0.02), (x0 - IN * ln_, y0 + 0.035)]), 0.026 * HU, taper=(0.1, 0.9)).blur(2)
            L.paint(w & m, P_['sage_sh'], 0.8, mode='multiply')
        L.paint(ellipse(*B(out[4][0] + IN * 0.06, 2.10), 0.035 * HU, 0.24 * HU).blur(12) & m, P_['sage_hl'], 0.35)
    n = len(out)
    edge(L, p, 1, n - 1, 3.8, P_['sage_ln'], taper=(0.1, 0.02))
    edge(L, p, n, len(pts) - 1, 3.2, P_['sage_ln'], taper=(0.02, 0.2))
    edge(L, cuff, 0, 3, 3.2, P_['sage_ln'], taper=(0.02, 0.02))
    edge(L, cuff, 3, 6, 3.2, P_['sage_ln'], taper=(0.02, 0.02))
    ln(L, [(co + IN * 0.004, ytop - 0.008), (cx, ytop + 0.016), (ci - IN * 0.002, ytop - 0.008)], 2.6, P_['sage_ln'], taper=(0.05, 0.05))
    for k in range(1, 8):
        x = co + IN * 0.02 + (ci - co - IN * 0.04) * k / 8
        ln(L, [(x + sb * 0.25, ytop + 0.035), (x + sb * 0.9, yb - 0.012)], 1.2, P_['sage_ln'], taper=(0.3, 0.3), op=0.45)
    # folds: elbow (inner side, pointing out), gathers above the cuff, a drape line from the shoulder
    for y0, ln_ in ((2.37, 0.085), (2.49, 0.11)):
        x0 = np.interp(y0, [q[1] for q in inn], [q[0] for q in inn])
        ln(L, [(x0, y0), (x0 - IN * ln_ * 0.5, y0 + 0.02), (x0 - IN * ln_, y0 + 0.035)], 2.0, P_['sage_ln'], taper=(0.05, 0.85))
    ln(L, [(co + IN * 0.006, ytop - 0.09), (co + IN * 0.05, ytop - 0.055), (co + IN * 0.09, ytop - 0.065)], 1.8, P_['sage_ln'], taper=(0.05, 0.85), op=0.85)
    ln(L, [(ci - IN * 0.006, ytop - 0.13), (ci - IN * 0.045, ytop - 0.09), (ci - IN * 0.085, ytop - 0.08)], 1.8, P_['sage_ln'], taper=(0.05, 0.85), op=0.85)
    ln(L, [(out[3][0] + IN * 0.06, 1.70), (out[4][0] + IN * 0.065, 2.05), (out[4][0] + IN * 0.06, 2.24)], 1.6, P_['sage_ln'], taper=(0.6, 0.6), op=0.6)
    # drop-shoulder seam across the top of the arm
    ln(L, [(out[2][0] + IN * 0.01, out[2][1] + 0.01), (out[3][0] + IN * 0.09, 1.66), (inn[0][0] - IN * 0.005, inn[0][1] - 0.03)], 1.8,
       P_['sage_ln'], taper=(0.3, 0.3), op=0.7)
    return m
