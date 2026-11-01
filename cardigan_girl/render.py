"""Full figure pipeline: stage 1 bones, stage 2 lineart, stage 3 flats, stage 4 final."""
from parts import *
from girl import draw_back_hair, draw_face, draw_eye, draw_features, draw_front_hair, draw_brows, head_masks

def render(mode, path, bones_under=False, bg=True):
    paper = '#fbf8f4'
    cv = Canvas(W, H, paper)
    if mode == 'final' and bg:
        # soft colour block behind the figure + floor shadow
        blk = ellipse(*B(0.0, 2.2), 1.05 * HU, 1.95 * HU).blur(30)
        cv.paint(blk, rgb('#eef0e6'), 0.9)
        sh = ellipse(*B(0.08, 6.36), 0.55 * HU, 0.07 * HU).blur(14)
        cv.paint(sh, rgb('#b9b3c4'), 0.6)
    if bones_under:
        L = cv.layer(); draw_bones(L); cv.comp(L, 0.22)
    face_mask = head_masks().mask()
    fig = np.zeros((H, W), F32)
    def put(L, op=1.0):
        cv.comp(L, op)
        np.maximum(fig, L.a * op, out=fig)
    L = cv.layer(); draw_back_hair(L, mode); put(L)
    L = cv.layer(); draw_collar_back(L, mode); put(L)
    L = cv.layer(); draw_shoes_back(L, mode); put(L)
    L = cv.layer(); draw_legs(L, mode); put(L)
    L = cv.layer(); draw_shoes_front(L, mode); put(L)
    L = cv.layer(); draw_skin_torso(L, mode, face_mask); put(L)
    L = cv.layer(); draw_top(L, mode); put(L)
    scratch = cv.layer(); card_mask = draw_cardigan(scratch, 'flat')
    L = cv.layer(); draw_skirt(L, mode, card_mask); put(L)
    L = cv.layer(); draw_cardigan(L, mode); put(L)
    for side in (-1, 1):
        L = cv.layer(); draw_arm(L, mode, side); put(L)
    scratch = cv.layer(); _, bang_masks = draw_front_hair(scratch, 'flat')
    L = cv.layer(); draw_face(L, mode, bang_masks=bang_masks)
    for m in (-1, 1): draw_eye(L, mode, m)
    draw_features(L, mode); put(L)
    L = cv.layer(); draw_front_hair(L, mode); put(L)
    L = cv.layer(); draw_brows(L, mode); cv.comp(L, 0.55)
    # heavier outer silhouette line, like the book's coloured figures: a ring that straddles the outer edge
    # (1px inside, ~1.8px outside) so it merges with the existing outline instead of leaving a seam
    solid = np.clip((fig - 0.5) * 4.0 + 0.5, 0, 1).astype(F32)
    blurf = lambda a, r: Mask(0, 0, a).blur(r).full(W, H)
    outer = np.clip((blurf(solid, 1.6) - 0.10) * 7.0, 0, 1)
    inner = 1.0 - np.clip((blurf(1.0 - solid, 1.0) - 0.10) * 7.0, 0, 1)
    ring = Mask(0, 0, outer * (1.0 - inner))
    O = cv.layer(); O.paint(ring, rgb('#2a1c1e'), 0.9); cv.comp(O)
    return cv.save(path)

if __name__ == '__main__':
    import sys, time
    t = time.time()
    if len(sys.argv) > 1 and sys.argv[1] == 'all':
        from body import render_bones
        render_bones(os.path.join(OUT, 'stage1_bones.png'))
        render('line', os.path.join(OUT, 'stage2_lineart.png'), bones_under=True)
        render('flat', os.path.join(OUT, 'stage3_flats.png'))
        render('final', os.path.join(OUT, 'stage4_final.png'))
        print('all stages %.1fs' % (time.time() - t))
    else:
        mode = sys.argv[1] if len(sys.argv) > 1 else 'final'
        im = render(mode, os.path.join(OUT, f'full_{mode}.png'), bones_under=(mode == 'line'))
        im.resize((800, 1700), Image.LANCZOS).save(os.path.join(OUT, f'full_{mode}_small.png'))
        print('ok', mode, '%.1fs' % (time.time() - t))
