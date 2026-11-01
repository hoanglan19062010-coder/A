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
# each leg is an axis (hip joint -> knee -> ankle) with a width profile (outer, inner half-widths)
LEG_AXIS = {'L': (HIP_L, KNE_L, ANK_L, -1), 'R': (HIP_R, KNE_R, ANK_R, +1)}
THIGH = [(0.00, 0.230, 0.215), (0.30, 0.215, 0.186), (0.50, 0.200, 0.172), (0.70, 0.178, 0.156), (0.87, 0.150, 0.136),
         (1.00, 0.124, 0.118)]
SHIN = [(0.00, 0.124, 0.118), (0.07, 0.120, 0.110), (0.20, 0.140, 0.121), (0.33, 0.137, 0.130), (0.50, 0.118, 0.113),
        (0.70, 0.096, 0.090), (0.88, 0.079, 0.074), (0.96, 0.075, 0.074), (1.00, 0.073, 0.071), (1.06, 0.078, 0.076),
        (1.12, 0.086, 0.082)]                       # past 1.0 = the top of the foot, hidden inside the shoe

def leg_geometry(name):
    hip, knee, ank, so = LEG_AXIS[name]
    outer, inner, axis = [], [], []
    for (p0, p1, prof, smax) in ((hip, knee, THIGH, 1.0), (knee, ank, SHIN, 1.12)):
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]; L_ = math.hypot(dx, dy); ux, uy = dx / L_, dy / L_
        nx, ny = uy, -ux                            # normal pointing to the viewer's right of the axis direction
        if nx * so < 0: nx, ny = -nx, -ny           # make it point to the leg's outer side
        ss = np.linspace(0, smax, 40)
        wo = np.interp(ss, [q[0] for q in prof], [q[1] for q in prof])
        wi = np.interp(ss, [q[0] for q in prof], [q[2] for q in prof])
        for s, o, i in zip(ss, wo, wi):
            cx_, cy_ = p0[0] + dx * s, p0[1] + dy * s
            outer.append((cx_ + nx * o, cy_ + ny * o)); inner.append((cx_ - nx * i, cy_ - ny * i)); axis.append((cx_, cy_))
    return outer, inner, axis

def draw_legs(L, mode):
    ms = []
    for name in ('R', 'L'):                         # free leg behind the weight leg
        outer, inner, axis = leg_geometry(name)
        poly_pts = np.asarray(Bs(outer) + Bs(inner[::-1]))
        m = poly(poly_pts); ms.append(m)
        L.paint(m, F(mode, P_['tight']))
        so = LEG_AXIS[name][3]
        if SHADE(mode):
            # denier sheen down the front of the leg, a little toward the light; brighter on the knee
            sheen_pts = [(x - 0.035, y) for x, y in axis[20:74]]
            sheen = line(Bs(sheen_pts), 0.10 * HU, taper=(0.25, 0.25), floor=0.3).blur(10) & m
            L.paint(sheen, P_['tight_sheen'], 0.75)
            kx, ky = LEG_AXIS[name][1]
            L.paint(ellipse(*B(kx - 0.02, ky + 0.01), 0.05 * HU, 0.065 * HU).blur(8) & m, P_['tight_sheen'], 0.55)
            L.paint(form_shadow(m, 0.075), P_['tight_sh'], 0.85, mode='multiply')
        n = len(outer)
        L.paint(stroke(np.asarray(Bs(outer[18:n - 6])), 4.2, taper=(0.0, 0.05), floor=0.3), P_['tight_ln'])
        L.paint(stroke(np.asarray(Bs(inner[18:n - 6])), 3.6, taper=(0.0, 0.05), floor=0.3), P_['tight_ln'])
        # knee: a small curve under the kneecap
        kx, ky = LEG_AXIS[name][1]
        ln(L, [(kx - 0.045, ky + 0.035), (kx - 0.01, ky + 0.052), (kx + 0.03, ky + 0.040)], 1.7,
           P_['tight_ln'] * 0.4 + P_['tight_sheen'] * 0.6, taper=(0.4, 0.4), op=0.85)
    return ms

SHOE_ANG = {'L': 4, 'R': -26}
SHOE_SIL = [(-0.100, 0.030), (-0.090, 0.002), (-0.050, -0.012), (0.0, -0.016), (0.050, -0.012), (0.090, 0.002), (0.100, 0.030),
            (0.108, 0.10), (0.118, 0.17), (0.120, 0.215), (0.100, 0.250), (0.055, 0.268), (0.0, 0.272), (-0.055, 0.268),
            (-0.100, 0.250), (-0.120, 0.215), (-0.118, 0.17), (-0.108, 0.10)]
SHOE_OPEN = [(-0.088, 0.034), (-0.060, 0.012), (0.0, 0.004), (0.060, 0.012), (0.088, 0.034), (0.060, 0.064), (0.0, 0.076), (-0.060, 0.064)]
SHOE_VAMP = [(-0.1025, 0.050, 'c'), (-0.064, 0.076), (0.0, 0.088), (0.064, 0.076), (0.1025, 0.050, 'c'), (0.108, 0.10), (0.118, 0.17),
             (0.120, 0.215), (0.100, 0.250), (0.055, 0.268), (0.0, 0.272), (-0.055, 0.268), (-0.100, 0.250), (-0.120, 0.215),
             (-0.118, 0.17), (-0.108, 0.10)]
SHOE_SOLE = [(-0.119, 0.222), (-0.100, 0.252), (-0.055, 0.270), (0.0, 0.274), (0.055, 0.270), (0.100, 0.252), (0.119, 0.222),
             (0.121, 0.240), (0.102, 0.270), (0.056, 0.286), (0.0, 0.290), (-0.056, 0.286), (-0.102, 0.270), (-0.121, 0.240)]

def shoe_T(name):
    ax, ay = (ANK_L if name == 'L' else ANK_R)
    a = math.radians(SHOE_ANG[name]); c, s = math.cos(a), math.sin(a)
    return lambda p: (ax + p[0] * c - p[1] * s, ay + p[0] * s + p[1] * c) + tuple(p[2:])

def draw_shoes_back(L, mode):
    """heel counter + dark lining, drawn BEFORE the leg so the leg goes into the shoe"""
    for name in ('R', 'L'):
        T = shoe_T(name)
        sil = bp([T(q) for q in SHOE_SIL]); sm = sil.mask()
        L.paint(sm, F(mode, P_['shoe']))
        if SHADE(mode):
            L.paint(form_shadow(sm, 0.05), P_['shoe_sh'], 0.9, mode='multiply')
        op = bp([T(q) for q in SHOE_OPEN]); L.paint(op.mask(), F(mode, P_['sole']))
        edge(L, sil, 0, 6, 3.4, P_['shoe_ln'], taper=(0.1, 0.1))
        edge(L, op, 0, 4, 2.4, P_['shoe_ln'], taper=(0.1, 0.1))

def draw_shoes_front(L, mode):
    """vamp, toe box and sole, drawn AFTER the leg: its top edge (the throat) crosses in front of the foot"""
    for name in ('R', 'L'):
        T = shoe_T(name)
        vamp = bp([T(q) for q in SHOE_VAMP]); vm = vamp.mask()
        L.paint(vm, F(mode, P_['shoe']))
        sole = bp([T(q) for q in SHOE_SOLE]); L.paint(sole.mask(), F(mode, P_['sole']))
        if SHADE(mode):
            L.paint(form_shadow(vm, 0.05), P_['shoe_sh'], 0.9, mode='multiply')
            L.paint(lin_grad(vm, B(*T((0, 0.13))), B(*T((0, 0.06)))), P_['shoe_sh'], 0.55, mode='multiply')   # under the strap
            shine = bp([T(q) for q in [(-0.090, 0.180), (-0.068, 0.165), (-0.036, 0.168), (-0.048, 0.188), (-0.080, 0.205)]]).mask().blur(0.8)
            L.paint(shine & vm, P_['shoe_hl'], 0.95)
        n = len(SHOE_VAMP)
        edge(L, vamp, 0, 4, 2.8, P_['shoe_ln'], taper=(0.1, 0.1))                 # throat across the foot
        edge(L, vamp, 4, n, 3.8, P_['shoe_ln'], taper=(0.05, 0.05))              # sides and toe
        edge(L, sole, 7, 13, 3.4, P_['shoe_ln'], taper=(0.05, 0.05))
        edge(L, sole, 0, 6, 1.8, P_['shoe_ln'], taper=(0.1, 0.1), op=0.8)
        # penny strap + slit, moc-toe apron seam
        ln(L, [T(q) for q in [(-0.104, 0.108), (-0.05, 0.124), (0.0, 0.128), (0.05, 0.124), (0.104, 0.108)]], 2.2, P_['shoe_ln'], taper=(0.15, 0.15))
        ln(L, [T(q) for q in [(-0.108, 0.140), (-0.05, 0.156), (0.0, 0.160), (0.05, 0.156), (0.108, 0.140)]], 2.0, P_['shoe_ln'], taper=(0.15, 0.15), op=0.85)
        ln(L, [T(q) for q in [(-0.022, 0.142), (0.0, 0.135), (0.022, 0.142), (0.0, 0.149), (-0.022, 0.142)]], 1.8, P_['shoe_ln'], taper=(0.0, 0.0))
        ln(L, [T(q) for q in [(-0.086, 0.165), (-0.070, 0.215), (0.0, 0.236), (0.070, 0.215), (0.086, 0.165)]], 1.6, P_['shoe_ln'], taper=(0.3, 0.3), op=0.7)

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

# ------------------------------------------------------------------ hands (book p.52-55)
# a wedge-shaped palm block + fingers whose segments shrink by 2/3, curling a little more from index to pinky.
# Built in 3D (u: down the hand, v: toward the palm, w: toward the thumb), projected orthographically.
HAND_SCALE = 1.1
FINGERS = [   # name, knuckle (u, v, w), segment lengths, base/tip radius, splay (deg), flex at the three joints (deg)
    ('pinky',  (0.215, 0.000, -0.055), (0.080, 0.054, 0.038), 0.0170, 0.0132, -9, (34, 54, 28)),
    ('ring',   (0.233, -0.004, -0.019), (0.100, 0.067, 0.045), 0.0200, 0.0155, -4, (28, 46, 24)),
    ('middle', (0.240, -0.006, 0.021), (0.106, 0.071, 0.047), 0.0210, 0.0162, 0, (22, 38, 20)),
    ('index',  (0.230, -0.004, 0.060), (0.100, 0.067, 0.045), 0.0205, 0.0160, 5, (15, 30, 18)),
]
PALM_PTS = [(0.215, 0.030, 0.040, 0.018), (0.222, 0.030, 0.000, 0.018), (0.205, 0.028, -0.040, 0.018),
            (0.090, 0.035, 0.060, 0.030), (0.120, 0.022, -0.050, 0.026), (0.060, 0.030, 0.062, 0.026),
            (0.135, 0.040, 0.078, 0.025)]

def finger_chain(mcp, segs, splay, flex):
    a = math.radians(splay)
    d = np.array([math.cos(a), 0.0, math.sin(a)])
    p = np.array(mcp, float); pts = [p.copy()]
    for seg, fl in zip(segs, flex):
        vp = np.array([0.0, 1.0, 0.0]) - d * d[1]; vp /= np.linalg.norm(vp)
        t = math.radians(fl); d = math.cos(t) * d + math.sin(t) * vp; d /= np.linalg.norm(d)
        p = p + d * seg; pts.append(p.copy())
    return np.array(pts)

def hand_parts():
    parts = []
    for name, mcp, segs, r0, r1, splay, flex in FINGERS:
        pts = finger_chain(mcp, segs, splay, flex)
        parts.append((name, pts, np.linspace(r0, r1, len(pts))))
    tp = [np.array([0.135, 0.040, 0.078])]                      # thumb: from its knuckle, curling toward the index
    for seg, dvec in ((0.062, (0.86, 0.36, -0.12)), (0.050, (0.74, 0.58, -0.22))):
        dv = np.array(dvec, float); dv /= np.linalg.norm(dv); tp.append(tp[-1] + dv * seg)
    parts.append(('thumb', np.array(tp), np.array([0.025, 0.021, 0.018])))
    return parts

def convex_hull(P):
    P = sorted(set(map(tuple, np.round(P, 3))))
    if len(P) < 3:
        return np.array(P)
    def cross(o, a, b): return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in P:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0: lo.pop()
        lo.append(p)
    for p in reversed(P):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0: up.pop()
        up.append(p)
    return np.array(lo[:-1] + up[:-1])

def draw_hand(L, mode, side, wrist, alpha, theta=38.0):
    """relaxed hanging hand, palm toward the thigh, back of the hand turned a little toward the viewer"""
    IN = -side
    th = math.radians(theta); c, s = math.cos(th), math.sin(th)
    ca, sa = math.cos(alpha), math.sin(alpha)
    def proj(P):
        P = np.asarray(P, float) * HAND_SCALE
        u, v, w = P[..., 0], P[..., 1], P[..., 2]
        x = IN * (c * v + s * w); y = u; z = -s * v + c * w
        X = wrist[0] + x * ca - y * sa; Y = wrist[1] + x * sa + y * ca
        return np.stack([CX + X * HU, Y0 + Y * HU], -1), z
    lw = 2.7
    def hull_mask(centers, radii, grow=0.0):
        ring = []
        for (px, py), r in zip(centers, radii):
            for t in np.linspace(0, 2 * math.pi, 20, endpoint=False):
                ring.append((px + math.cos(t) * (r + grow), py + math.sin(t) * (r + grow)))
        return poly(convex_hull(np.array(ring)))
    # palm block: wrist cross-section + knuckles + palm pads + thumb base
    pp3 = [(-0.03, 0.029 * math.sin(t) - 0.003, 0.048 * math.cos(t), 0.006) for t in np.linspace(0, 2 * math.pi, 10, endpoint=False)]
    pp3 += [(m[0], m[1], m[2], r0 + 0.003) for _, m, _, r0, _, _, _ in FINGERS] + list(PALM_PTS)
    pc, _ = proj([q[:3] for q in pp3]); pr = [q[3] * HAND_SCALE * HU for q in pp3]
    palm = hull_mask(pc, pr); palm_ink = hull_mask(pc, pr, lw)
    palm_core = Mask(palm.x0, palm.y0, np.clip((palm.blur(3).a - 0.9) * 10, 0, 1))
    ink_col = C['skin_ln']
    L.paint(palm_ink, ink_col); L.paint(palm, F(mode, C['skin']))
    if SHADE(mode):
        L.paint(form_shadow(palm, 0.03), C['skin_sh'], 0.9, mode='multiply')
    hand_all = [palm]
    parts = []
    for name, pts, radii in hand_parts():
        p2, z = proj(pts)
        parts.append((float(np.mean(z)), name, p2, radii * HAND_SCALE * HU))
    for _, name, p2, rpx in sorted(parts, key=lambda q: q[0]):          # far fingers first
        m = capsules(p2, rpx); ink = capsules(p2, rpx + lw) - palm_core
        L.paint(ink, ink_col); L.paint(m, F(mode, C['skin']))
        hand_all.append(m)
        if SHADE(mode):
            L.paint(form_shadow(m, 0.012), C['skin_sh'], 0.9, mode='multiply')
            L.paint(lin_grad(m, tuple(p2[1]), tuple(p2[-1])), C['skin_sh'], 0.55, mode='multiply')   # curl turns away
        if name in ('index', 'middle', 'thumb'):                                                     # nails
            a_, b_ = p2[-2], p2[-1]
            d = (b_ - a_) / (np.linalg.norm(b_ - a_) + 1e-6)
            ctr = a_ + (b_ - a_) * 0.62 - np.array([d[1], -d[0]]) * rpx[-1] * 0.15 * IN
            ang = math.degrees(math.atan2(d[1], d[0]))
            nl, nw = np.linalg.norm(b_ - a_) * 0.36, rpx[-1] * 0.62
            nail = ellipse(ctr[0], ctr[1], nl, nw, ang) & m
            rim = (ellipse(ctr[0], ctr[1], nl, nw, ang) - ellipse(ctr[0], ctr[1], nl - 1.4, nw - 1.4, ang)) & m
            L.paint(nail, rgb('#fde4de') if mode != 'line' else C['white'], 0.95)
            L.paint(rim, C['skin_ln'], 0.55)
    return union(*hand_all)

def sleeve_fold_lists(out, inn, IN, folds):
    """insert small notches into the sleeve contours where a fold starts (fabric dents in, bulges below)"""
    def notch(pts, y0, inward):
        xs = [q[1] for q in pts]
        xo = lambda y: float(np.interp(y, xs, [q[0] for q in pts]))
        keep = [q for q in pts if abs(q[1] - y0) > 0.045 or q is pts[0] or q is pts[-1]]
        new = [(xo(y0 - 0.03), y0 - 0.03), (xo(y0) + inward * 0.011, y0), (xo(y0 + 0.035) - inward * 0.005, y0 + 0.035)]
        return sorted(keep + new, key=lambda q: q[1])
    for y0, s0, y1, s1, kind in folds:
        if kind != 'z' or y0 > out[-1][1] - 0.06 or y0 < 1.75:
            continue
        if s0 == 0.0:
            out = notch(out, y0, IN)
        elif s0 == 1.0:
            inn = notch(inn, y0, -IN)
    return out, inn

SLEEVE_FOLDS = [   # (y start, s start, y end, s end, kind); s: 0 = outer contour, 1 = inner contour
    (1.70, 0.20, 2.12, 0.24, 'drape'), (1.74, 0.62, 2.02, 0.70, 'drape'),
    (1.84, 1.0, 1.99, 0.50, 'z'), (2.06, 0.0, 2.17, 0.46, 'z'),
    (2.36, 1.0, 2.30, 0.56, 'lock'), (2.42, 1.0, 2.41, 0.42, 'lock'), (2.47, 1.0, 2.55, 0.52, 'lock'),
    (2.40, 0.0, 2.46, 0.30, 'z'),
    (2.62, 0.0, 2.71, 0.62, 'z'), (2.72, 1.0, 2.81, 0.42, 'z'), (2.83, 0.0, 2.92, 0.60, 'z'), (2.93, 1.0, 3.01, 0.45, 'z'),
    (3.03, 0.0, 3.09, 0.55, 'z'),
]

def draw_arm(L, mode, side):
    out0, inn0 = SLV[side]
    IN = -side                                     # unit direction toward the body centre
    ytop = out0[-1][1]                             # cuff top
    co, ci = out0[-1][0], inn0[-1][0]              # outer / inner x at the cuff top
    slant = ((out0[-1][0] - out0[-2][0]) / (out0[-1][1] - out0[-2][1]) + (inn0[-1][0] - inn0[-2][0]) / (inn0[-1][1] - inn0[-2][1])) / 2
    yb = ytop + 0.14                               # cuff bottom
    sb = slant * 0.14                              # horizontal drift of the cuff bottom along the forearm
    cx = (co + ci) / 2 + sb
    # ---- hand first: the cuff covers the wrist
    alpha = -math.atan(slant) - IN * math.radians(6)
    hand = draw_hand(L, mode, side, (cx - 0.035 * math.sin(-alpha), yb - 0.035), alpha)
    # ---- sleeve tube with fold notches in the contour
    out, inn = sleeve_fold_lists(list(out0), list(inn0), IN, SLEEVE_FOLDS)
    xo = lambda y: float(np.interp(y, [q[1] for q in out], [q[0] for q in out]))
    xi = lambda y: float(np.interp(y, [q[1] for q in inn], [q[0] for q in inn]))
    def sp(y, s_):
        if y < inn[0][1]:                          # above the armpit: measure across to the seam side
            return (xo(y) + s_ * IN * 0.20, y)
        return (xo(y) + s_ * (xi(y) - xo(y)), y)
    pts = out + list(reversed(inn))
    p = bp(pts); m = p.mask()
    L.paint(m, F(mode, P_['sage']))
    # ---- ribbed cuff: a little narrower; the sleeve blouses over it
    cuff = bp([(co + IN * 0.008, ytop - 0.012), (co + IN * 0.014 + sb * 0.5, ytop + 0.07), (co + IN * 0.020 + sb, yb),
               (ci - IN * 0.016 + sb, yb), (ci - IN * 0.010 + sb * 0.5, ytop + 0.07), (ci - IN * 0.004, ytop - 0.012)])
    cm = cuff.mask()
    if SHADE(mode):                                # the cuff casts a soft shadow on the back of the hand
        L.paint((cm.shift(0.006 * HU, 0.028 * HU).blur(3) & hand) - cm, C['skin_sh2'], 0.8, mode='multiply')
    L.paint(cm, F(mode, P_['sage']))
    if SHADE(mode):
        L.paint(form_shadow(m, 0.085), P_['sage_sh'], 0.9, mode='multiply')
        L.paint(form_shadow(cm, 0.06), P_['sage_sh'], 0.9, mode='multiply')
        L.paint(lin_grad(cm, B(0, ytop + 0.045), B(0, ytop - 0.005)), P_['sage_sh'], 0.6, mode='multiply')
        if side > 0:
            L.paint(m, P_['sage_sh'], 0.35, mode='multiply')
        L.paint(ellipse(*B(out0[4][0] + IN * 0.06, 2.10), 0.035 * HU, 0.24 * HU).blur(12) & m, P_['sage_hl'], 0.35)
        # fold shading: the fabric below each fold line turns away from the light; a light ridge above it
        for y0, s0, y1, s1, kind in SLEEVE_FOLDS:
            P0, P1 = sp(y0, s0), sp(y1, s1)
            if kind == 'drape':
                w = line(Bs([P0, lerp(P0, P1, 0.5), P1]), 0.035 * HU, taper=(0.6, 0.6)).blur(4)
                L.paint(w & m, P_['sage_sh'], 0.6, mode='multiply')
                continue
            depth = 0.030 if kind == 'z' else 0.022
            sh = poly(np.asarray(Bs([P0, P1, (P0[0], P0[1] + depth)]))).blur(2.5)
            L.paint(sh & m, P_['sage_sh'], 0.85, mode='multiply')
            hl = poly(np.asarray(Bs([(P0[0], P0[1] - 0.012), P1, P0]))).blur(1.5)
            L.paint(hl & m, P_['sage_hl'], 0.35)
    n = len(out)
    edge(L, p, 1, n - 1, 3.8, P_['sage_ln'], taper=(0.1, 0.02))
    edge(L, p, n, len(pts) - 1, 3.2, P_['sage_ln'], taper=(0.02, 0.2))
    edge(L, cuff, 0, 3, 3.2, P_['sage_ln'], taper=(0.02, 0.02))
    edge(L, cuff, 3, 6, 3.2, P_['sage_ln'], taper=(0.02, 0.02))
    ln(L, [(co + IN * 0.004, ytop - 0.008), (cx, ytop + 0.016), (ci - IN * 0.002, ytop - 0.008)], 2.6, P_['sage_ln'], taper=(0.05, 0.05))
    for k in range(1, 8):
        x = co + IN * 0.02 + (ci - co - IN * 0.04) * k / 8
        ln(L, [(x + sb * 0.25, ytop + 0.035), (x + sb * 0.9, yb - 0.012)], 1.2, P_['sage_ln'], taper=(0.3, 0.3), op=0.45)
    # fold lines: thick where they leave the contour, tapering inside the sleeve
    for y0, s0, y1, s1, kind in SLEEVE_FOLDS:
        P0, P1 = sp(y0, s0), sp(y1, s1)
        mid = lerp(P0, P1, 0.5); mid = (mid[0], mid[1] - 0.008)
        if kind == 'drape':
            ln(L, [P0, mid, P1], 1.6, P_['sage_ln'], taper=(0.6, 0.6), op=0.6)
        else:
            ln(L, [P0, mid, P1], 2.2 if kind == 'z' else 2.0, P_['sage_ln'], taper=(0.05, 0.85), op=0.9)
    # drop-shoulder seam across the top of the arm
    ln(L, [(out0[2][0] + IN * 0.01, out0[2][1] + 0.01), (out0[3][0] + IN * 0.09, 1.66), (inn0[0][0] - IN * 0.005, inn0[0][1] - 0.03)], 1.8,
       P_['sage_ln'], taper=(0.3, 0.3), op=0.7)
    return m
