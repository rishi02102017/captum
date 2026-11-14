import torch, captum
from captum.robust import PGD
print("captum", captum.__version__, "torch", torch.__version__)
torch.manual_seed(0)

class Net(torch.nn.Module):
    def __init__(self, d, k=3):
        super().__init__(); self.lin = torch.nn.Linear(d, k)
    def forward(self, x):
        return torch.softmax(self.lin(x.flatten(1)), dim=1)

def run(name, shape, step_num=0, mask=None):
    x = torch.randn(*shape)
    d = x[0].numel()
    adv = PGD(Net(d))
    try:
        out = adv.perturb(x, radius=0.5, step_size=0.1, step_num=step_num, target=0,
                          norm="L2", random_start=True, mask=mask)
        n = (out - x).flatten(1).norm(dim=1)
        print(f"[{name}] shape={tuple(shape)} OK per-example L2 dist={n.tolist()}")
        return out - x
    except Exception as e:
        print(f"[{name}] shape={tuple(shape)} ERROR {type(e).__name__}: {e}")

run("batch1 control", (1, 5))
run("batch4 x 3 features", (4, 3))
run("batch2 image", (2, 3, 8, 8))
run("batch2 x 2 features", (2, 2))
run("batch4 x 3, step_num=2", (4, 3), step_num=2)
# Distribution check where B == last dim: radius r_i should be per example (row),
# so each row's perturbation is r_i * unit vector -> row norms = r_i <= radius and
# direction == unit_u. If r is broadcast along the last dim instead, row norm is
# sqrt(sum_j r_j^2 u_ij^2). Show this by fixing the RNG and recomputing manually.
torch.manual_seed(123)
x = torch.zeros(3, 3)
adv = PGD(Net(3))
torch.manual_seed(7)
out = adv.perturb(x, 0.5, 0.1, 0, 0, norm="L2", random_start=True)
torch.manual_seed(7)
u = torch.randn(3, 3); unit_u = torch.nn.functional.normalize(u)
r = (torch.rand(3) ** (1.0 / 3)) * 0.5
expected = r[:, None] * unit_u
print("[B==D] actual delta:\n", out)
print("[B==D] expected (radius per example, r[:,None]*unit_u):\n", expected)
print("[B==D] matches per-example scaling:", torch.allclose(out, expected))
print("[B==D] matches per-column scaling r[None,:]*unit_u:", torch.allclose(out, r[None, :] * unit_u))
