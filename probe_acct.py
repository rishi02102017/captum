import warnings; warnings.filterwarnings("ignore")
import torch, opacus
from opacus.accountants import RDPAccountant, PRVAccountant, GaussianAccountant
from opacus import PrivacyEngine
print("opacus", opacus.__version__)
for cls in (RDPAccountant, PRVAccountant, GaussianAccountant):
    a = cls()
    try:
        print(f"[{cls.__name__} empty history] get_epsilon(1e-5) ->", a.get_epsilon(delta=1e-5))
    except Exception as e:
        print(f"[{cls.__name__} empty history] ERROR {type(e).__name__}: {e}")
for acc in ("rdp", "prv", "gdp"):
    pe = PrivacyEngine(accountant=acc)
    try:
        print(f"[PrivacyEngine({acc}) before training] ->", pe.get_epsilon(1e-5))
    except Exception as e:
        print(f"[PrivacyEngine({acc}) before training] ERROR {type(e).__name__}: {e}")
