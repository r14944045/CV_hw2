import os, torch, numpy as np, torchvision, sys
HERE = os.path.dirname(os.path.abspath(__file__)); CACHE = os.environ.get('HW2_DEV_CACHE', os.path.join(HERE, 'cache'))
split = sys.argv[1]; K = int(sys.argv[2])
assert split == "val", "val only: the test stream is read only by the official harness"
d = torch.load(os.path.join(CACHE, f"{split}.pt")); x = d["x"]
ck = torch.load(os.path.join(HERE, "../../source_resnet50_160.pth"), map_location="cpu", weights_only=False)
m = torchvision.models.resnet50(num_classes=20); m.load_state_dict(ck["state_dict"]); m = m.cuda().eval()
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda(); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
feats = []
with torch.no_grad():
    for i in range(0, len(x), 256):
        xb = (x[i:i+256].cuda().float()/255 - MEAN)/STD
        h = m.maxpool(m.relu(m.bn1(m.conv1(xb)))); h1 = m.layer1(h)
        f = torch.cat([h.mean((2,3)), h.std((2,3)), h1.mean((2,3)), h1.std((2,3))], 1)
        feats.append(f.cpu())
F = torch.cat(feats).numpy().astype(np.float64)
F = (F - F.mean(0)) / (F.std(0) + 1e-6)
# PCA to 32 dims
U, S, Vt = np.linalg.svd(F - F.mean(0), full_matrices=False); Z = U[:, :32] * S[:32]
N = len(Z)
# optimal segmentation into K segments, cost = within-seg SSE; use block resolution of 16 images for speed then refine
blk = 16; nb = N // blk
cs = np.concatenate([np.zeros((1, Z.shape[1])), np.cumsum(Z, 0)]); cs2 = np.concatenate([[0], np.cumsum((Z**2).sum(1))])
def cost(a, b):  # images [a,b)
    n = b - a; s = cs[b] - cs[a]; return cs2[b] - cs2[a] - (s**2).sum() / n
# DP over image-level boundaries restricted to multiples of 4 for speed
step = 4; P = list(range(0, N + 1, step)); M = len(P)
INF = 1e30
dp = np.full((K + 1, M), INF); arg = np.zeros((K + 1, M), int); dp[0, 0] = 0
Pa = np.array(P)
for k in range(1, K + 1):
    for j in range(1, M):
        i = np.arange(max(0, j - 2400 // step), j)
        a, bb = Pa[i], Pa[j]; n = bb - a; sm = cs[bb] - cs[a]
        c = cs2[bb] - cs2[a] - (sm**2).sum(1) / n
        v = dp[k - 1, i] + c
        t = np.argmin(v); dp[k, j], arg[k, j] = v[t], i[t]
b = [M - 1]
for k in range(K, 0, -1): b.append(arg[k, b[-1]])
bounds = [P[i] for i in reversed(b)]
print("bounds", bounds, "lengths", np.diff(bounds))
np.save(os.path.join(HERE, f"{split}_bounds.npy"), np.array(bounds))
