import json, sys
from pathlib import Path
import numpy as np, cv2
from .metrics import _geometry, _mask_and_detection

def report_scene(scene_dir):
    O = Path(scene_dir)
    a = np.load(O / 'labels.npz')
    N = len(a['K'])
    srcs = ['gt', 'stage2_mask', 'sam3_tracking_1hz']
    versions = ['stage2']
    pr = {v: np.load(O / (v + '.npz')) for v in versions}
    report = {}
    matched = []
    for v in versions:
        for s in srcs:
            key = v + '/' + s
            (report[key], valid, _) = _geometry(pr[v][s + '_joints'], a['gt_joints'], a['gt_valid'], pr[v][s + '_valid'], a['point_valid'])
            matched.append(valid)
    common = np.logical_and.reduce(matched)
    common_report = {v + '/' + s: _geometry(pr[v][s + '_joints'], a['gt_joints'], common, pr[v][s + '_valid'], a['point_valid'])[0] for v in versions for s in srcs}
    maskstats = {s: {k: [] for k in ['mask_gt_count', 'mask_pred_count', 'mask_intersection', 'mask_union', 'mask_supervision_valid', 'lr_overlap_iou', 'correct_iou', 'swapped_iou']} for s in srcs[1:]}
    risk = []
    previews = []
    chosen = set(np.linspace(0, N - 1, min(10, N)).round().astype(int))
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
    cols = [(60, 80, 235), (230, 130, 40)]

    def masks(path, swap=False):
        m = cv2.imread(str(path))
        assert m is not None, path
        return (m[:, :, [1, 0] if swap else [0, 1]] > 0).transpose(2, 0, 1)

    def overlay(im, m):
        im = im.copy()
        for h in range(2):
            im[m[h]] = (0.6 * im[m[h]] + 0.4 * np.array(cols[h])).astype(np.uint8)
        return im

    def label(im, t):
        im = im.copy()
        cv2.rectangle(im, (0, 0), (im.shape[1], 25), (245, 245, 245), -1)
        cv2.putText(im, t, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1)
        return im

    def skeleton(im, j, valid, K):
        im = im.copy()
        for h in range(2):
            if not valid[h]:
                continue
            q = j[h] @ K.T
            uv = q[:, :2] / np.maximum(q[:, 2:], 1e-06)
            good = np.isfinite(uv).all(-1) & (q[:, 2] > 1e-06) & (abs(uv).max(-1) < 10000.0)
            for (u, v) in edges:
                if good[u] and good[v]:
                    cv2.line(im, tuple(np.rint(uv[u]).astype(int)), tuple(np.rint(uv[v]).astype(int)), cols[h], 2, cv2.LINE_AA)
        return im

    def mesh(im, verts, valid, K):
        fpath = O / 'mano_faces.npy'
        if not fpath.exists():
            return im
        faces = np.load(fpath)
        triangles = []
        for side in range(2):
            if not valid[side]:
                continue
            q = verts[side] @ K.T
            uv = q[:, :2] / np.maximum(q[:, 2:], 1e-06)
            for face in faces:
                if np.isfinite(uv[face]).all() and (q[face, 2] > 0).all() and (abs(uv[face]).max() < 10000):
                    triangles.append((float(q[face, 2].mean()), np.rint(uv[face]).astype(np.int32), side))
        layer = im.copy()
        coverage = np.zeros(im.shape[:2], np.uint8)
        for (_, tri, side) in sorted(triangles, key=lambda t: -t[0]):
            cv2.fillConvexPoly(layer, tri, cols[side])
            cv2.fillConvexPoly(coverage, tri, 1)
        out = im.copy()
        use = coverage > 0
        out[use] = (0.55 * layer[use] + 0.45 * out[use]).astype(np.uint8)
        return out
    for i in range(N):
        g = masks(O / 'gt_masks' / f'{i:06d}.png')
        ps = {'stage2_mask': masks(O / 'stage2_masks' / f'{i:06d}.png'), 'sam3_tracking_1hz': masks(O / 'sam/masks' / f'{i:06d}.png', True)}
        for (s, p) in ps.items():
            d = maskstats[s]
            d['mask_gt_count'].append(g.sum((-1, -2)))
            d['mask_pred_count'].append(p.sum((-1, -2)))
            d['mask_intersection'].append((g & p).sum((-1, -2)))
            d['mask_union'].append((g | p).sum((-1, -2)))
            d['mask_supervision_valid'].append(a['mask_valid'][i])
            ov = (p[0] & p[1]).sum() / max(1, (p[0] | p[1]).sum())
            d['lr_overlap_iou'].append(ov)
            correct = ((g & p).sum((-1, -2)) / np.maximum(1, (g | p).sum((-1, -2)))).mean()
            swap = ((g & p[::-1]).sum((-1, -2)) / np.maximum(1, (g | p[::-1]).sum((-1, -2)))).mean()
            d['correct_iou'].append(correct)
            d['swapped_iou'].append(swap)
            if ov > 0.5 or (a['mask_valid'][i].all() and swap > correct + 0.2):
                risk.append(dict(index=i, source_id=int(a['source_ids'][i]), source=s, overlap=float(ov), correct=float(correct), swapped=float(swap)))
        if i in chosen or (risk and risk[-1]['index'] == i and (len(previews) < 20)):
            im = cv2.imread(str(O / 'frames' / f'{i:06d}.png'))
            rows = [np.hstack([label(overlay(im, g), 'GT masks'), label(overlay(im, ps['stage2_mask']), 'Stage2 mask'), label(overlay(im, ps['sam3_tracking_1hz']), 'SAM3 + tracking1Hz')])]
            for v in versions:
                cells = []
                for s in srcs:
                    base = im
                    if i in chosen and s + '_preview_vertices' in pr[v]:
                        base = mesh(im, pr[v][s + '_preview_vertices'][sorted(chosen).index(i)], pr[v][s + '_valid'][i], a['K'][i])
                    cells.append(label(skeleton(base, pr[v][s + '_joints'][i], pr[v][s + '_valid'][i], a['K'][i]), v + ' / ' + s))
                rows.append(np.hstack(cells))
            file = f'preview_{i:06d}.jpg'
            cv2.imwrite(str(O / file), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
            previews.append(file)
    ms = {}
    for (s, d) in maskstats.items():
        d = {k: np.asarray(v) for (k, v) in d.items()}
        np.savez_compressed(O / (s + '_mask_stats.npz'), **d)
        ms[s] = _mask_and_detection(d)
    (O / 'PREVIEW_FRAMES.json').write_text(json.dumps([{'file': f, 'index': int(f[8:14]), 'source_frame_id': int(a['source_ids'][int(f[8:14])]), 'source_timestamp': float(a['timestamps'][int(f[8:14])]), 'source_timestamp_valid': bool(a['timestamps_valid'][int(f[8:14])])} for f in previews], indent=2))
    (O / 'METRICS.json').write_text(json.dumps({'geometry': report, 'common_intersection': common_report, 'mask': ms, 'identity_risk_count': len(risk), 'identity_risks': risk, 'protocol': 'same-side diagnostic, not paper bbox-matched benchmark; swapped IoU is audit only, never corrects predictions'}, indent=2))
    (O / 'index.html').write_text('<meta charset="utf-8"><h1>' + O.name + '</h1><p>Rows: masks and Stage2 MANO. Columns: GT / Stage2 Mask / SAM3 tracking1Hz. Right red, left blue. Same-side diagnostic. Identity flags are candidates, not confirmed track swaps.</p>' + ''.join(('<p>source frame ' + str(int(a['source_ids'][int(f[8:14])])) + '</p><img style="width:100%" src="' + f + '">' for f in previews)))
    (O / 'SUCCESS.json').write_text(json.dumps({'frames': N, 'combinations': 3, 'preview_count': len(previews), 'identity_risk_count': len(risk)}))

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Score and visualize a prepared camera-space scene.')
    parser.add_argument('scene_dir', type=Path)
    args = parser.parse_args()
    report_scene(args.scene_dir)

if __name__ == '__main__':
    main()
