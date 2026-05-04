import sympy as sp
import bisect
import csv
from collections import defaultdict
import numpy as np
import re
from scipy.stats import entropy
import signal
import time
import joblib


# ------------------ Core Operations ------------------
def spoly(f, g, lmf=None, lmg=None):
    """S(f,g) = (lcm/LM(f)) f - (lcm/LM(g)) g."""
    lmf = f.LM if lmf is None else lmf
    lmg = g.LM if lmg is None else lmg
    R = f.ring
    lcm = R.monomial_lcm(lmf, lmg)
    return f.mul_monom(R.monomial_div(lcm, lmf)) - g.mul_monom(R.monomial_div(lcm, lmg))


def reduce_poly(g, G, lmG=None):
    """Full multivariate division: reduce until LT(g) irreducible."""
    if not G or g == 0:
        return g
    ring = g.ring
    lmG = [f.LM for f in G] if lmG is None else lmG
    h = g.copy()
    while h != 0:
        lmh, lch = h.LT
        found = False
        for f, lmf in zip(G, lmG):
            m = ring.monomial_div(lmh, lmf)
            if m is not None:
                h -= f.mul_term((m, lch / f.LC))
                found = True
                break
        if not found:
            break
    return h


# ------------------ Basis Post-Processing ------------------
def clear_denominators(f):
    if f == 0:
        return f
    ring = f.ring
    if hasattr(ring.domain, 'is_QQ'):
        if not ring.domain.is_QQ:
            return f
    denoms = []
    for _, coeff in f.terms():
        if hasattr(coeff, 'q'):
            denoms.append(abs(coeff.q))
        elif hasattr(coeff, 'denominator'):
            denoms.append(abs(coeff.denominator))
        else:
            denoms.append(1)
    if all(d == 1 for d in denoms):
        return f
    from math import gcd
    def lcm(a, b):
        return abs(a * b) // gcd(a, b)
    multiplier = denoms[0]
    for d in denoms[1:]:
        multiplier = lcm(multiplier, d)
    return f * multiplier


def minimalize(G):
    if not G:
        return G
    R = G[0].ring
    Gmin = []
    for f in sorted(G, key=lambda h: R.order(h.LM)):
        if all(not R.monomial_div(f.LM, g.LM) for g in Gmin):
            Gmin.append(f)
    return Gmin


def interreduce(G):
    if not G:
        return []
    ring = G[0].ring
    Gred = []
    for i, g in enumerate(G):
        others = [G[j] for j in range(len(G)) if j != i]
        if not others:
            g_red = g.monic()
        else:
            lm_others = [h.LM for h in others]
            g_red = reduce_poly(g, others, lm_others)
            if g_red == 0:
                continue
            g_red = g_red.monic()
        Gred.append(g_red)
    return Gred


# ------------------ Gebauer-Möller Filter ------------------
def _is_coprime(a, b):
    """Test gcd of monomials (exponent tuples) = 1, i.e. componentwise min is zero."""
    return all(min(x, y) == 0 for x, y in zip(a, b))


def gebauer_moller_update(P, lmG, ring, new_index):
    """
    Gebauer-Möller UPDATE (Becker-Weispfenning Alg. 5.5.7,
    Gebauer-Möller 1988). Filters P (old pairs) and appends
    minimal new pairs (i, k) for i < k = new_index.

    Three criteria:
      F  (coprimality): drop (i,k) if gcd(LM(g_i), LM(g_k)) = 1.
      M  (minimality among new pairs): among candidate pairs
          {(i,k) : i<k}, drop (i,k) if some other candidate
          (l,k) has lcm(LM(g_l), LM(g_k)) properly dividing
          lcm(LM(g_i), LM(g_k)).
      B  (chain / backward on old pairs): remove existing
          (i,j) in P iff LM(g_k) | lcm(LM(g_i), LM(g_j))  AND
          lcm(LM(g_i), LM(g_k)) != lcm(LM(g_i), LM(g_j))   AND
          lcm(LM(g_j), LM(g_k)) != lcm(LM(g_i), LM(g_j)).
    """
    k = new_index
    tk = lmG[k]

    # -------- Criterion B: prune old pairs --------
    P_new = []
    for (i, j) in P:
        ti, tj = lmG[i], lmG[j]
        lcm_ij = ring.monomial_lcm(ti, tj)
        # tk divides lcm_ij ?
        if ring.monomial_div(lcm_ij, tk) is not None:
            lcm_ik = ring.monomial_lcm(ti, tk)
            lcm_jk = ring.monomial_lcm(tj, tk)
            if lcm_ik != lcm_ij and lcm_jk != lcm_ij:
                # (i,j) is covered by (i,k) and (j,k) -> discard
                continue
        P_new.append((i, j))

    # -------- Build candidate new pairs {(i, k) : i < k} --------
    candidates = []  # list of (i, lcm_ik)
    for i in range(k):
        ti = lmG[i]
        # Criterion F: coprime LMs -> S-poly reduces to zero -> drop
        if _is_coprime(ti, tk):
            continue
        lcm_ik = ring.monomial_lcm(ti, tk)
        candidates.append((i, lcm_ik))

    # -------- Criterion M: minimality among candidates --------
    # Drop (i,k) if some other (l,k) has lcm_lk | lcm_ik and lcm_lk != lcm_ik.
    # (Proper divisibility; ties broken by keeping the first occurrence.)
    kept = []
    for idx, (i, lcm_ik) in enumerate(candidates):
        drop = False
        for jdx, (l, lcm_lk) in enumerate(candidates):
            if idx == jdx:
                continue
            if ring.monomial_div(lcm_ik, lcm_lk) is not None:
                # lcm_lk divides lcm_ik
                if lcm_lk != lcm_ik:
                    drop = True
                    break
                # Equal lcm: keep only the earliest index to avoid
                # mutually dropping each other.
                if jdx < idx:
                    drop = True
                    break
        if not drop:
            kept.append((i, lcm_ik))

    # -------- Criterion F (again, redundancy cleanup) --------
    # Already handled above; kept explicit for clarity.
    for (i, _) in kept:
        P_new.append((i, k))

    P[:] = P_new
    return P


# ------------------ Caches ------------------
spoly_cache = defaultdict(lambda: None)
lcm_deg_cache = defaultdict(lambda: None)
entropy_cache = defaultdict(lambda: None)


def clear_caches():
    spoly_cache.clear()
    lcm_deg_cache.clear()
    entropy_cache.clear()
    entropy_state.clear()

def canonical_pair(i, j):
    return (min(i, j), max(i, j))


def get_spoly(i, j, G, lmG):
    pair = canonical_pair(i, j)
    if spoly_cache[pair] is None:
        spoly_cache[pair] = spoly(G[i], G[j], lmG[i], lmG[j])
    return spoly_cache[pair]


def get_spoly_entropy(i, j, G, lmG):
    pair = canonical_pair(i, j)
    if spoly_cache[pair] is None:
        spoly_cache[pair] = spoly(G[i], G[j], lmG[i], lmG[j])
    if entropy_cache[pair] is None:
        S = spoly_cache[pair]
        if S == 0:
            entropy_cache[pair] = 0.0
        else:
            term_degs = np.array([sum(m) for m, _ in S.terms()], dtype=float)
            if term_degs.sum() > 0:
                probs = term_degs / term_degs.sum()
                entropy_cache[pair] = float(entropy(probs))
            else:
                entropy_cache[pair] = 0.0        
    return entropy_cache[pair]

entropy_state = {}


def get_lcm_degree(i, j, lmG, ring):
    pair = canonical_pair(i, j)
    if lcm_deg_cache[pair] is None:
        lcm_deg_cache[pair] = sum(ring.monomial_lcm(lmG[i], lmG[j]))
    return lcm_deg_cache[pair]


# ------------------ Selection Strategies ------------------
def select(G, P, strategy='degree', lmG=None, sugarG=None):
    ring = G[0].ring
    if lmG is None or len(lmG) == 0:
        lmG = [g.LM for g in G]

    if strategy == 'degree':
        return min(P, key=lambda p: get_lcm_degree(*p, lmG, ring))
    elif strategy == 'sugar':
        if sugarG is None:
            raise ValueError("sugarG must be provided for 'sugar' strategy")
        def key(p):
            i, j = p
            lcm = ring.monomial_lcm(lmG[i], lmG[j])
            deg_lcm = sum(lcm)
            s1 = sugarG[i] + deg_lcm - sum(lmG[i])
            s2 = sugarG[j] + deg_lcm - sum(lmG[j])
            return (max(s1, s2), ring.order(lcm))
        return min(P, key=key)
    elif strategy == 'normal':
        def key(p):
            i, j = p
            lcm = ring.monomial_lcm(lmG[i], lmG[j])
            return ring.order(lcm)
        return min(P, key=key)
    elif strategy == 's_poly_entropy':
        return min(P, key=lambda p: (get_spoly_entropy(*p, G, lmG)))

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# ------------------ Main Algorithm ------------------
def buchberger(F, selection="degree"):
    """
    Buchberger with Gebauer-Möller pair management.
    Applies GM UPDATE incrementally as each input polynomial is inserted
    (Becker-Weispfenning Alg. 5.5.7), rather than seeding P with all pairs.
    """
    clear_caches()
    if not F:
        return [], 0
    F = [f.monic() for f in F if f != 0]
    if not F:
        return [], 0

    ring = F[0].ring
    n_steps = 0

    # Incremental initialization via GM UPDATE
    G = []
    lmG = []
    sugarG = []
    P = []
    for f in F:
        G.append(f)
        lmG.append(f.LM)
        sugarG.append(f.degree())
        k = len(G) - 1
        if k > 0:
            gebauer_moller_update(P, lmG, ring, k)

    # Main Buchberger loop
    while P:
        pair = select(G, P, selection, lmG, sugarG)
        i, j = pair
        P.remove(pair)


        S = get_spoly(i, j, G, lmG)

        n_steps += 1

        r = reduce_poly(S, G, lmG)
        if r == 0:
            continue

        r = r.monic()
        k = len(G)

        lcm_ij = ring.monomial_lcm(lmG[i], lmG[j])
        deg_lcm = sum(lcm_ij)
        shift_i = deg_lcm - sum(lmG[i])
        shift_j = deg_lcm - sum(lmG[j])
        newsugar = max(sugarG[i] + shift_i, sugarG[j] + shift_j, r.degree())
        sugarG.append(newsugar)

        G.append(r)
        lmG.append(r.LM)

        gebauer_moller_update(P, lmG, ring, k)

    # Post-process
    Gmin = minimalize(G)
    try:
        Gred = interreduce(Gmin)
    except Exception as e:
        print(f"Warning: Interreduce failed: {e}")
        Gred = Gmin

    return Gred, n_steps



# ------------------ CSV loading / driver (unchanged behaviour) ------------------
def parse_modular_polynomial(poly_str, gen_dict):
    if not poly_str or poly_str.strip().lower() in ("none", ""):
        return None
    poly_str = poly_str.replace("^", "**")
    poly_str = re.sub(r'(\d+)\s*mod\s*(\d+)', r'(\1 % \2)', poly_str)
    try:
        return eval(poly_str, {"__builtins__": None}, gen_dict)
    except Exception as e:
        print("Error parsing polynomial:", poly_str)
        print("Exception:", e)
        return None


def load_systems_grouped(csv_file, ring, has_header=False):
    systems = defaultdict(list)
    gen_dict = {str(g): g for g in ring.gens}
    with open(csv_file, newline='') as f:
        reader = csv.DictReader(f) if has_header else csv.reader(f)
        for row in reader:
            if has_header:
                sid = int(row["system_id"])
                poly_str = row["polynomial"]
            else:
                if not row or len(row) < 3:
                    continue
                sid = int(row[0])
                poly_str = row[2]
            poly = parse_modular_polynomial(poly_str, gen_dict)
            if poly is not None:
                systems[sid].append(poly)
    return dict(systems)


class TimeoutException(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutException()


signal.signal(signal.SIGALRM, _timeout_handler)