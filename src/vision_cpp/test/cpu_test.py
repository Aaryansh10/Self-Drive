import subprocess, numpy as np
rng = np.random.default_rng(1)
H, W, r0 = 544, 960, 160
cx, cy, fx, fy = 480.0, 272.5, 343.25, 343.25
cam_pitch, cam_h, max_range, step, pitch = 0.4, 1.606, 8.0, 1, 0.02
# random skeleton-like components (polylines), unique pixels
pix, lab = set(), {}
for c in range(7):
    u = rng.integers(50, 900); v = rng.integers(400, 540)
    for _ in range(rng.integers(20, 300)):
        v -= rng.integers(0, 2); u += rng.integers(-1, 2)
        if r0 <= v < H and 0 <= u < W and (v, u) not in lab:
            lab[(v, u)] = 1000 + c * 7919
items = [(l, v, u) for (v, u), l in lab.items()]
rng.shuffle(items)
with open("test_in.txt", "w") as f:
    f.write(f"{cx} {cy} {fx} {fy} {cam_pitch} {cam_h} {max_range} {step} {H} {r0} {H-r0} {pitch}\n{len(items)}\n")
    for l, v, u in items: f.write(f"{l} {v} {u}\n")
subprocess.check_call(["./cpu_test"])
# ---- reference (python originals) ----
a = np.array(items); lb, v, u = a[:, 0], a[:, 1], a[:, 2]
n = lb.max() + 1
counts = np.bincount(lb, minlength=n); mean_u = np.bincount(lb, weights=u, minlength=n) / np.maximum(counts, 1)
order = np.lexsort((u, -v, lb, mean_u[lb])); rv, ru = v[order], u[order]
def ipm(v_sub, u_sub):
    alpha = np.arctan((v_sub * step - cy) / fy); ang = alpha + cam_pitch + pitch
    m = ang >= 0.01; alpha, ang, uu = alpha[m], ang[m], u_sub[m] * step
    fwd = cam_h / np.tan(ang); m2 = (fwd > 0) & (fwd <= max_range)
    alpha, ang, uu, fwd = alpha[m2], ang[m2], uu[m2], fwd[m2]
    return np.stack([fwd + 0.02, (cx - uu) / fx * fwd * np.cos(alpha) / np.cos(ang), 0 * fwd], 1)
ref = ipm(rv, ru)
lines = open("test_out.txt").read().split("\n")
no, k = map(int, lines[0].split())
got_o = np.array([list(map(int, l.split())) for l in lines[1:1 + no]])
got_p = np.array([list(map(float, l.split())) for l in lines[1 + no:1 + no + k]])
print("pixels", len(items), "ordering identical:", np.array_equal(got_o[:, 0], rv) and np.array_equal(got_o[:, 1], ru))
print("ipm points", k, "ref", len(ref), "max abs diff:", np.abs(got_p - ref).max() if k == len(ref) else "COUNT MISMATCH")
