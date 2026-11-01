"""Body landmarks (head units) + stage 1 construction drawing."""
from girl import *

# ---------------------------------------------------------------- skeleton landmarks (body head units)
PIT = (0.0, 1.30)
SH_L, SH_R = (-0.50, 1.40), (0.50, 1.37)          # viewer's left shoulder slightly lower (contrapposto)
ELB_L, ELB_R = (-0.575, 2.44), (0.582, 2.40)
WRI_L, WRI_R = (-0.528, 3.30), (0.508, 3.25)
HIP_L, HIP_R = (-0.24, 3.30), (0.245, 3.36)       # weight on viewer's left leg -> that hip rides higher
KNE_L, KNE_R = (-0.14, 4.94), (0.13, 4.99)
ANK_L, ANK_R = (-0.07, 6.12), (0.255, 6.08)        # supporting ankle under the pit of the neck
SOLE = 6.36
NAVEL = (-0.01, 2.75)
CROTCH = (-0.01, 3.62)
LOA = [(-0.03, -0.06), (0.0, 0.55), (0.0, 1.30), (0.04, 2.10), (0.0, 2.75), (-0.09, 3.40), (-0.13, 4.94), (-0.07, 6.30)]

BLUE = rgb('#4a7fd0'); RED = rgb('#d8484a'); GREY = rgb('#9aa3b5')

def pencil(L, pts, w=2.4, col=BLUE, op=0.85, taper=(0.15, 0.2), closed=False):
    L.paint(line(pts, w, taper=taper, floor=0.25, closed=closed), col, op)

def draw_bones(L):
    P = lambda pts: [B(*p) for p in pts]
    # head-unit ruler
    for k in range(7):
        y = Y0 + k * HU
        L.paint(line([(70, y), (150, y)], 2.0, taper=(0, 0)), GREY, 0.9)
    L.paint(line([(110, Y0), (110, Y0 + SOLE * HU)], 2.0, taper=(0, 0)), GREY, 0.9)
    y = Y0 + SOLE * HU
    L.paint(line([(70, y), (150, y)], 2.0, taper=(0, 0)), GREY, 0.9)
    # line of action (S curve)
    pencil(L, P(LOA), 4.0, RED, 0.75, taper=(0.1, 0.1))
    # ---- head: sphere + jaw + centre line + feature lines (in the tilted head frame)
    circ = [(0.39 * math.cos(t), 0.39 + 0.39 * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 40, endpoint=False)]
    pencil(L, Hs(circ), 3.0, closed=True, taper=(0, 0))
    jaw = [(-0.384, 0.40), (-0.36, 0.635), (-0.255, 0.82), (-0.085, 0.968), (0, 1.0), (0.085, 0.968), (0.255, 0.82), (0.36, 0.635), (0.384, 0.40)]
    pencil(L, Hs(jaw), 2.8)
    pencil(L, Hs([(0.0, 0.0), (0.0, 0.5), (0.0, 1.0)]), 2.0)
    for yy, hw in ((0.592, 0.40), (0.42, 0.34), (0.705, 0.10), (0.81, 0.12)):
        pencil(L, Hs([(-hw, yy), (0, yy), (hw, yy)]), 1.8, op=0.7)
    for m in (-1, 1):  # eye boxes, one eye apart
        x0 = 0.11 * m; x1 = 0.30 * m
        pencil(L, Hs([(x0, 0.52), (x1, 0.52), (x1, 0.655), (x0, 0.655)]), 1.8, op=0.7, closed=True, taper=(0, 0))
    # hair volume (floats above the skull)
    hv = [(0.52 * math.cos(t), 0.43 + 0.50 * math.sin(t)) for t in np.linspace(math.pi, 2 * math.pi, 24)]
    pencil(L, Hs(hv), 1.6, op=0.45)
    # ---- neck + shoulders + clavicles
    for m in (-1, 1):
        pencil(L, P([(m * 0.13, 0.86), (m * 0.13, 1.05), (m * 0.16, 1.22)]), 2.6)
    pencil(L, P([SH_L, (-0.25, 1.30), PIT, (0.25, 1.28), SH_R]), 2.6)
    # ---- 3-box torso: ribcage egg, abdomen, hip box
    rib = [(0.0, 1.31), (0.33, 1.42), (0.39, 1.75), (0.34, 2.10), (0.0, 2.25), (-0.34, 2.12), (-0.39, 1.77), (-0.33, 1.43)]
    pencil(L, P(rib), 2.8, closed=True, taper=(0, 0))
    for side in (-1, 1):
        pencil(L, P([(side * 0.33, 2.12), (side * 0.28, 2.55), (side * 0.44, 3.10)]), 2.4)
    hipbox = [(-0.40, 2.98), (0.40, 3.02), (0.47, 3.30), (0.05, 3.64), (-0.05, 3.62), (-0.47, 3.26)]
    pencil(L, P(hipbox), 2.8, closed=True, taper=(0, 0))
    pencil(L, P([(-0.40, 2.98), (0.40, 3.02)]), 2.0)
    pencil(L, P([PIT, (0.02, 1.8), NAVEL, CROTCH]), 1.8, op=0.6)
    # bust
    for side in (-1, 1):
        cx = side * 0.19
        bust = [(cx + 0.16 * math.cos(t), 1.92 + 0.14 * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 24, endpoint=False)]
        pencil(L, P(bust), 1.8, op=0.6, closed=True, taper=(0, 0))
    # ---- limbs: bones + cylinders + joints
    def limb(a, b, ra, rb):
        pencil(L, P([a, b]), 4.0, taper=(0.05, 0.05))
        dx, dy = b[0] - a[0], b[1] - a[1]; n = math.hypot(dx, dy); nx, ny = -dy / n, dx / n
        for s in (-1, 1):
            pencil(L, P([(a[0] + nx * ra * s, a[1] + ny * ra * s), (b[0] + nx * rb * s, b[1] + ny * rb * s)]), 1.6, op=0.5)
    limb(SH_L, ELB_L, 0.085, 0.065); limb(ELB_L, WRI_L, 0.07, 0.045)
    limb(SH_R, ELB_R, 0.085, 0.065); limb(ELB_R, WRI_R, 0.07, 0.045)
    limb(HIP_L, KNE_L, 0.16, 0.10); limb(KNE_L, ANK_L, 0.11, 0.05)
    limb(HIP_R, KNE_R, 0.16, 0.10); limb(KNE_R, ANK_R, 0.11, 0.05)
    for p in (SH_L, SH_R, ELB_L, ELB_R, WRI_L, WRI_R, HIP_L, HIP_R, KNE_L, KNE_R, ANK_L, ANK_R):
        x, y = B(*p); L.paint(ellipse(x, y, 13, 13) - ellipse(x, y, 10.5, 10.5), BLUE, 0.9)
    # hands (wedge) and feet (1:3 wedge, foreshortened from the front)
    for w, d in ((WRI_L, -1), (WRI_R, 1)):
        pencil(L, P([(w[0] - 0.06, w[1] + 0.02), (w[0] + 0.07, w[1] + 0.02), (w[0] + 0.05, w[1] + 0.26), (w[0] - 0.05, w[1] + 0.27)]),
               2.2, closed=True, taper=(0, 0))
    pencil(L, P([(-0.135, 6.10), (0.0, 6.10), (0.03, 6.30), (-0.065, 6.37), (-0.165, 6.30)]), 2.2, closed=True, taper=(0, 0))
    pencil(L, P([(0.195, 6.06), (0.32, 6.06), (0.385, 6.27), (0.33, 6.35), (0.20, 6.27)]), 2.2, closed=True, taper=(0, 0))
    # balance: plumb line from the pit of the neck
    L.paint(line(P([PIT, (0.0, SOLE)]), 1.6, taper=(0, 0)), RED, 0.35)

def render_bones(path):
    cv = Canvas(W, H, '#fbf8f4')
    L = cv.layer(); draw_bones(L); cv.comp(L)
    return cv.save(path)

if __name__ == '__main__':
    render_bones(os.path.join(OUT, 'stage1_bones.png'))
    print('ok')
