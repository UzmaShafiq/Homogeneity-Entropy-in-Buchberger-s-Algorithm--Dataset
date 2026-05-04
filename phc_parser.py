import csv
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import sympy as sp
from sympy.polys import ring, QQ, RR, CC, GF
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

# Parsing transforms:
# - implicit_multiplication_application lets "2x" parse as 2*x if needed
# - convert_xor turns ^ into ** if it ever appears
TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)


@dataclass
class SystemRing:
    system_id: int
    variables: List[str]
    ring: object
    generators: List[object]
    n_variables: int
    field: str


class PHCRingLoader:
    # Keep only true mathematical constants/reserved names out.
    # Do NOT exclude 'e' because system 6 uses it as a variable.
    EXCLUDE_VARS = {"i", "I", "pi", "oo", "zoo", "nan", "E"}

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.systems: Dict[int, SystemRing] = {}

    def _clean_poly(self, poly_str: str) -> str:
        return poly_str.strip().rstrip(";")

    def _extract_vars(self, poly_str: str) -> set:
        """
        Extract variable names from already explicit PHC strings like:
        x1**2 + y*z - 1
        """
        vars_set = set()

        # Match variable-like tokens starting with a letter/underscore.
        for match in re.finditer(r"[A-Za-z_][A-Za-z0-9_]*", poly_str):
            tok = match.group()

            # Skip reserved constants/special names
            if tok in self.EXCLUDE_VARS:
                continue

            # Skip SymPy scientific notation artifact names if any
            if tok in {"Integer", "Rational", "Float"}:
                continue

            vars_set.add(tok)

        return vars_set

    def load_all_rings(self, field: str = "Q", modulus: Optional[int] = None, order: str = "grevlex") -> Dict[int, SystemRing]:
        if field == "Q":
            domain = QQ
        elif field == "RR":
            domain = RR
        elif field == "CC":
            domain = CC
        elif field == "GF":
            if modulus is None:
                raise ValueError("modulus needed for GF")
            domain = GF(modulus)
        else:
            raise ValueError(f"field must be Q/RR/CC/GF, got {field}")

        grouped: Dict[int, List[str]] = {}

        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = int(row["system_id"])
                poly = self._clean_poly(row["polynomial"])
                grouped.setdefault(sid, []).append(poly)

        rings: Dict[int, SystemRing] = {}

        for sid, polys in grouped.items():
            vars_set = self._extract_vars(" ".join(polys))
            variables = sorted(vars_set)
            
            order="grevlex"
            R, *gens = ring(variables, domain, order=order)

            rings[sid] = SystemRing(
                system_id=sid,
                variables=variables,
                ring=R,
                generators=gens,
                n_variables=len(variables),
                field=field,
            )

        self.systems = rings
        return rings

    def get_ring(self, system_id: int):
        return self.systems.get(system_id)

    def summary(self):
        return {
            "total_systems": len(self.systems),
            "rings": {
                sid: {
                    "nvars": r.n_variables,
                    "vars": r.variables,
                    "field": r.field,
                }
                for sid, r in self.systems.items()
            },
        }


class FixedPHCPolyLoader:
    def __init__(self, csv_path: str, rings: Dict[int, SystemRing]):
        self.csv_path = csv_path
        self.rings = rings
        self.system_polys: Dict[int, List[object]] = {}
        self.failed: List[dict] = []

    def _clean_poly(self, poly_str: str) -> str:
        poly_str = poly_str.strip().rstrip(";")
        poly_str = re.sub(r"\s+", "", poly_str)
        return poly_str

    def _build_local_dict(self, R) -> Dict[str, sp.Symbol]:
        """
        parse_expr wants normal SymPy Symbols, not PolyElement generators.
        R.symbols gives the symbol names/objects for the ring.
        """
        local_dict = {}

        for s in R.symbols:
            name = str(s)
            local_dict[name] = sp.Symbol(name)

        # complex unit support for systems with complex coefficients
        local_dict["I"] = sp.I
        local_dict["i"] = sp.I

        return local_dict

    def _parse_to_ring(self, poly_str: str, R):
        """
        Parse a polynomial string with SymPy first, then coerce into the ring.
        """
        poly_str = self._clean_poly(poly_str)

        # Normalize imaginary unit if the CSV uses lowercase i
        poly_str = re.sub(r'(?<![A-Za-z0-9_])i(?![A-Za-z0-9_])', 'I', poly_str)

        local_dict = self._build_local_dict(R)

        expr = parse_expr(
            poly_str,
            local_dict=local_dict,
            transformations=TRANSFORMATIONS,
            evaluate=True,
        )

        # Convert into the target polynomial ring
        return R.from_expr(expr)

    def load_all_polys(self, verbose: bool = False) -> Dict[int, List[object]]:
        polys_by_sys: Dict[int, List[object]] = {}
        self.failed = []

        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                sid = int(row["system_id"])
                pidx = int(row["poly_index"])

                if sid not in self.rings:
                    continue

                R = self.rings[sid].ring
                poly_str = row["polynomial"]

                try:
                    p = self._parse_to_ring(poly_str, R)
                    polys_by_sys.setdefault(sid, []).append(p)

                    if verbose and len(polys_by_sys[sid]) == 1:
                        print(f"Sys{sid} first poly OK: {self._clean_poly(poly_str)[:80]}")

                except Exception as e:
                    self.failed.append(
                        {
                            "system_id": sid,
                            "poly_index": pidx,
                            "polynomial": poly_str,
                            "error": repr(e),
                        }
                    )
                    if verbose:
                        print(
                            f"Sys{sid} poly{pidx}: "
                            f"{self._clean_poly(poly_str)[:80]} ... ERROR: {e}"
                        )

        self.system_polys = polys_by_sys

        total = sum(len(v) for v in polys_by_sys.values())
        print(f"Successfully parsed {total} polynomials")

        if self.failed:
            print(f"Failed to parse {len(self.failed)} polynomials")

        return polys_by_sys

    def get_polys(self, system_id: int) -> List[object]:
        return self.system_polys.get(system_id, [])

    def failure_report(self) -> List[dict]:
        return self.failed


# ----------------------------
# Usage
# ----------------------------

#csv_path = "/home/uzma/Documents/ISSAC_up/PHC_pack_Final.csv"

def load_smart_rings(csv_path: str) -> Dict[int, SystemRing]:
    """Auto-detects QQ vs CC per system"""
    loader = PHCRingLoader(csv_path)
    
    # First pass: check for complex coefficients
    complex_systems = set()
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "i" in row["polynomial"] or "I" in row["polynomial"]:
                complex_systems.add(int(row["system_id"]))
    
    # Load rings with correct domains
    rational_rings = loader.load_all_rings(field="Q")
    rings = {}
    
    for sid in rational_rings:
        if sid in complex_systems:
            # Reload complex systems with CC domain
            cc_loader = PHCRingLoader(csv_path)
            rings[sid] = cc_loader.load_all_rings(field="CC", order="grevlex")[sid]
        else:
            rings[sid] = rational_rings[sid]
    
    return rings

# Usage:

# rings_smart = load_smart_rings("PHC_pack_Final_noimag.csv")
# polyloader = FixedPHCPolyLoader("PHC_pack_Final_noimag.csv", rings_smart)
# polys_all = polyloader.load_all_polys(verbose=True)
# print(f"Failures now: {len(polyloader.failure_report())}")

# Example access
# system 1 ring
# print(f"Number of systems loaded: {len(rings_smart)}")
# print(f"System IDs available: {sorted(rings_smart.keys())[:10]}...")
# R1 = rings_smart[1].ring
# print("\nSystem 1 ring:", R1)
# print("System 1 polys:")
# for p in polyloader.get_polys(1):
#     print("  ", p)

# Failed rows, if any
# if polyloader.failure_report():
#     print("\nFirst 10 failures:")
#     for item in polyloader.failure_report()[:10]:
#         print(item)

def load_gf_rings(csv_path: str, modulus: int = 32003) -> Dict[int, SystemRing]:
    """
    Load every system over GF(modulus). Use this for the integer-converted
    PHCpack CSV produced by convert_phc_to_gf.py — every coefficient is
    already a non-negative integer in [0, modulus), and there are no
    complex systems (those were skipped during conversion).
    """
    loader = PHCRingLoader(csv_path)
    return loader.load_all_rings(field="GF", modulus=modulus, order="grevlex")


# Usage for the GF(32003) converted file:
# GF_CSV = "PHC_pack_GF32003.csv"

# rings_gf = load_gf_rings(GF_CSV, modulus=32003)
# polyloader_gf = FixedPHCPolyLoader(GF_CSV, rings_gf)
# polys_gf = polyloader_gf.load_all_polys(verbose=True)

# print(f"\nLoaded {len(polys_gf)} systems over GF(32003)")
# print(f"System IDs: {sorted(polys_gf.keys())[:10]}...")

# # Quick spot check on one system
# sid = sorted(polys_gf.keys())[0]
# print(f"\nSystem {sid} ring: {rings_gf[sid].ring}")
# print(f"System {sid} first poly: {polys_gf[sid][0]}")
# print(f"System {sid} first poly coefficients: "
#       f"{[c for _, c in polys_gf[sid][0].terms()]}")

# if polyloader_gf.failure_report():
#     print("\nFailures:")
#     for item in polyloader_gf.failure_report()[:10]:
#         print(item)