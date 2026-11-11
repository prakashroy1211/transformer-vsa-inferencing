"""
Transformer-VSA: single-image inference and traceable TPS report generation.

Stage 1 (neural perception): frozen MedSigLIP encoder + learnable morphology queries
          -> projected VSA vectors -> rectified atomic scores (Eqs. 4-6).
Stage 2 (symbolic reasoning, parameter-free): composite findings from the frozen
          rule base (COMPOSITE_RULES) using geometric mean (AND / bind) and
          max (OR / bundle) (Eqs. 7-8).
Stage 3 (diagnosis): per-diagnosis hypothesis scoring against the frozen Paris System
          rule base (DIAGNOSIS_DEFS) and softmax competition.
Stage 4 (report): deterministic three-tier population of the report fields.

Usage
-----
    export HF_TOKEN=<your own Hugging Face token>   # google/medsiglip-448 is a gated model
    python transformer_vsa_inference.py \
        --image path/to/image.jpg \
        --dictionary frozen_dictionary_atomic_terms_v12.pth \
        --weights best_model.pth \
        --json_out trace.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchhd as hd
from PIL import Image
from torchvision import transforms as T
from transformers import SiglipVisionModel

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VSA_DIM = 1000
HIDDEN_DIM = 512
TEMPERATURE_SCALE = 4.0     # tau in Eq. (5)
DIAGNOSIS_SCALE = 15.0      # softmax scale in the diagnosis competition (Stage 3)
CELL_THRESHOLD = 0.70       # Tier-3 threshold for "Cell Types Observed"

# Ordered atomic-term list; the order must match the trained query embeddings.
ALL_ATOMS = [
    "BACKGROUND_CLEAN", "BLOOD_PRESENT", "CELLS_ABUNDANT", "CELLS_ADEQUATE", "CELLS_SPARSE",
    "CELL_COHESIVE", "CELL_ISOLATED", "CELL_PLEOMORPHIC", "CELL_UNIFORM", "CHROMATIN_COARSE",
    "CHROMATIN_DELICATE", "CHROMATIN_DENSE", "CHROMATIN_HYPERCHROMATIC", "CHROMATIN_VESICULAR",
    "CYTOPLASM_ABUNDANT", "CYTOPLASM_MODERATE", "CYTOPLASM_SCANT", "DEGENERATIVE_CHANGES",
    "DISRUPTED_ARCHITECTURE", "GLANDULAR_ORIGIN", "INFLAMMATION_MILD", "INFLAMMATION_MODERATE",
    "KOILOCYTIC_CHANGES", "NECROSIS_PRESENT", "NUCLEAR_MEMBRANE_GROOVED", "NUCLEAR_MEMBRANE_IRREGULAR",
    "NUCLEAR_MEMBRANE_SMOOTH", "NUCLEUS_LARGE", "NUCLEUS_MODERATE", "NUCLEUS_SMALL",
    "NUCLEUS_VERY_LARGE", "SQUAMOUS_ORIGIN", "UMBRELLA_CELL_MORPHOLOGY", "UROTHELIAL_ORIGIN",
    "VACUOLIZATION",
]
NUM_QUERIES = len(ALL_ATOMS)

# ----------------------------------------------------------------------------
# Frozen rule base 1: composite findings
#   op = "bind"   : AND  -> geometric mean of atomic scores, Hadamard binding of vectors
#   op = "bundle" : OR   -> max of atomic scores, bundling of vectors
#   op = "direct" : single atom
#   op = "negate" : 1 - score of the atom
# ----------------------------------------------------------------------------
COMPOSITE_RULES = {
    "specimen_type": {
        "Voided urine": (["UROTHELIAL_ORIGIN"], "direct"),
        "Catheterized": (["BLOOD_PRESENT"], "direct"),
        "Bladder wash": (["CELLS_ABUNDANT"], "direct"),
        "Instrumented": (["DISRUPTED_ARCHITECTURE"], "direct"),
    },
    "gross_examination": {
        "Clear": (["BACKGROUND_CLEAN"], "direct"),
        "Haemorrhagic": (["BLOOD_PRESENT"], "direct"),
        "Turbid": (["INFLAMMATION_MODERATE"], "direct"),
        "Others": (["DEGENERATIVE_CHANGES"], "direct"),
    },
    "background": {
        "Clean": (["BACKGROUND_CLEAN"], "direct"),
        "Inflammatory": (["INFLAMMATION_MODERATE", "CELLS_ADEQUATE"], "bundle"),
        "Hemorrhagic": (["BLOOD_PRESENT", "CELLS_ADEQUATE"], "bundle"),
        "Necrotic": (["NECROSIS_PRESENT", "DEGENERATIVE_CHANGES"], "bundle"),
        "Degenerative": (["DEGENERATIVE_CHANGES"], "direct"),
    },
    "cell_types_observed": {
        "Benign urothelial cells": (["UROTHELIAL_ORIGIN", "NUCLEUS_SMALL", "CELL_UNIFORM"], "bind"),
        "Atypical urothelial cells": (["UROTHELIAL_ORIGIN", "NUCLEUS_MODERATE", "CELL_PLEOMORPHIC"], "bind"),
        "Squamous cells": (["SQUAMOUS_ORIGIN", "CYTOPLASM_ABUNDANT"], "bind"),
        "Glandular cells": (["GLANDULAR_ORIGIN", "CELL_COHESIVE"], "bind"),
        "Inflammatory cells": (["INFLAMMATION_MODERATE", "CELL_ISOLATED"], "bind"),
    },
    "cellularity": {
        "Adequate": (["CELLS_ADEQUATE", "CELLS_ABUNDANT"], "bundle"),
        "Inadequate": (["CELLS_SPARSE"], "direct"),
    },
    "nc_ratio": {
        "Normal": (["NUCLEUS_SMALL", "CYTOPLASM_ABUNDANT"], "bind"),
        "0.5-0.7": (["NUCLEUS_MODERATE", "CYTOPLASM_MODERATE"], "bind"),
        ">0.7": (["NUCLEUS_LARGE", "CYTOPLASM_SCANT"], "bind"),
    },
    "nuclear_contour": {
        "Smooth and regular": (["NUCLEAR_MEMBRANE_SMOOTH", "CELL_UNIFORM"], "bind"),
        "Irregular": (["NUCLEAR_MEMBRANE_IRREGULAR", "CELL_PLEOMORPHIC"], "bind"),
    },
    "chromatin_pattern": {
        "Fine and evenly distributed": (["CHROMATIN_DELICATE", "CELL_UNIFORM"], "bind"),
        "Vesicular": (["CHROMATIN_VESICULAR"], "direct"),
        "Coarsely granular": (["CHROMATIN_COARSE", "CHROMATIN_DENSE"], "bind"),
        "Irregular chromatinic rim": (["CHROMATIN_HYPERCHROMATIC", "NUCLEAR_MEMBRANE_IRREGULAR"], "bind"),
    },
    "hyperchromasia": {
        "Present": (["CHROMATIN_HYPERCHROMATIC"], "direct"),
        "Absent": (["CHROMATIN_HYPERCHROMATIC"], "negate"),
    },
    "umbrella_cells": {
        "Present": (["UMBRELLA_CELL_MORPHOLOGY", "UROTHELIAL_ORIGIN"], "bind"),
        "Absent": (["UMBRELLA_CELL_MORPHOLOGY"], "negate"),
    },
}

REPORT_LABEL = {
    "specimen_type": "Specimen Type",
    "gross_examination": "Gross Examination",
    "cellularity": "Cellularity",
    "background": "Background",
    "nc_ratio": "N:C Ratio",
    "nuclear_contour": "Nuclear Contour",
    "chromatin_pattern": "Chromatin Pattern",
    "hyperchromasia": "Hyperchromasia",
    "umbrella_cells": "Umbrella Cells",
}

# ----------------------------------------------------------------------------
# Frozen rule base 2: Paris System diagnostic definitions
# ----------------------------------------------------------------------------
DIAGNOSIS_DEFS = {
    "Negative for High-Grade Urothelial Carcinoma (NHGUC)": {
        "composites": [
            ("cellularity", "Adequate"),
            ("nc_ratio", "Normal"),
            ("nuclear_contour", "Smooth and regular"),
        ],
        "raw_atoms": [],
    },
    "Atypical Urothelial Cells (AUC)": {
        "composites": [
            ("cellularity", "Adequate"),
            ("nc_ratio", "0.5-0.7"),
            ("nuclear_contour", "Irregular"),
            ("hyperchromasia", "Present"),
        ],
        "raw_atoms": [],
    },
    "High-Grade Urothelial Carcinoma (HGUC)": {
        "composites": [
            ("cellularity", "Adequate"),
            ("nc_ratio", ">0.7"),
            ("nuclear_contour", "Irregular"),
            ("chromatin_pattern", "Coarsely granular"),
            ("hyperchromasia", "Present"),
        ],
        "raw_atoms": [],
    },
    "Suspicious for High-Grade Urothelial Carcinoma (SHGUC)": {
        "composites": [
            ("nc_ratio", ">0.7"),
            ("nuclear_contour", "Irregular"),
            ("cellularity", "Inadequate"),
        ],
        "raw_atoms": [],
    },
    "Other Malignancy": {
        "composites": [("cellularity", "Adequate")],
        "raw_atoms": ["CELL_PLEOMORPHIC"],
        "bundled_atoms": ["GLANDULAR_ORIGIN", "SQUAMOUS_ORIGIN"],
        "negated_atoms": ["UROTHELIAL_ORIGIN"],
    },
    "Nondiagnostic / Unsatisfactory": {
        "composites": [("cellularity", "Inadequate")],
        "raw_atoms": [],
    },
}


# ----------------------------------------------------------------------------
# VSA vector construction (frozen, parameter-free)
# ----------------------------------------------------------------------------
def _score_to_vsa(atoms_list, op, fbd):
    vecs = [fbd[a] for a in atoms_list if a in fbd]
    if not vecs:
        return None
    if op in ("bind", "direct"):
        result = vecs[0]
        for v in vecs[1:]:
            result = hd.bind(result, v)
    elif op == "bundle":
        result = vecs[0]
        for v in vecs[1:]:
            result = hd.bundle(result, v)
    else:  # "negate" and any other op: geometry of the single atom
        result = vecs[0]
    return result


def build_vsa_dicts(fbd):
    """Build composite-term vectors and per-diagnosis prototype vectors."""
    comp_vsa = {}
    for cat, terms in COMPOSITE_RULES.items():
        comp_vsa[cat] = {}
        for term, (atoms, op) in terms.items():
            vec = _score_to_vsa(atoms, op, fbd)
            if vec is not None:
                comp_vsa[cat][term] = vec

    diag_vsa = {}
    for diag_name, defn in DIAGNOSIS_DEFS.items():
        parts = []
        for (cat, term) in defn["composites"]:
            if cat in comp_vsa and term in comp_vsa[cat]:
                parts.append(comp_vsa[cat][term])
        for atom in defn.get("raw_atoms", []):
            if atom in fbd:
                parts.append(fbd[atom])
        for atom in defn.get("negated_atoms", []):
            if atom in fbd:
                parts.append(fbd[atom])

        bundled = defn.get("bundled_atoms", [])
        if bundled:
            vecs = [fbd[a] for a in bundled if a in fbd]
            v = vecs[0]
            for x in vecs[1:]:
                v = hd.bundle(v, x)
            v = F.normalize(v, dim=-1)
            parts.append(v)

        if not parts:
            continue

        proto = parts[0]
        for p in parts[1:]:
            proto = hd.bind(proto, p)
        diag_vsa[diag_name] = proto

    return comp_vsa, diag_vsa


# ----------------------------------------------------------------------------
# Stage 1 model: frozen MedSigLIP + morphology queries + shared VSA projection
# ----------------------------------------------------------------------------
class AtomicVSASupervised(nn.Module):
    def __init__(self, hidden_dim=512, num_queries=35, vsa_dim=1000):
        super().__init__()
        self.medsiglip = SiglipVisionModel.from_pretrained("google/medsiglip-448")
        for p in self.medsiglip.parameters():
            p.requires_grad = False
        self.medsiglip.eval()

        D = self.medsiglip.config.hidden_size
        self.patch_proj = nn.Linear(D, hidden_dim)
        self.vsa_to_hidden = nn.Linear(vsa_dim, hidden_dim)  # used only for query initialisation at training time
        self.query_embeddings = nn.Parameter(torch.zeros(num_queries, hidden_dim))

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=8, batch_first=True, dropout=0.2)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=8, batch_first=True, dropout=0.2)
        self.norm1, self.norm2, self.norm3 = (nn.LayerNorm(hidden_dim) for _ in range(3))

        def ffn():
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Dropout(0.3),
                nn.Linear(hidden_dim * 4, hidden_dim), nn.Dropout(0.3))
        self.ffn1, self.ffn2 = ffn(), ffn()
        self.vsa_projection = nn.Sequential(nn.Linear(hidden_dim, vsa_dim), nn.Tanh())

    def forward(self, pixel_values):
        with torch.no_grad():
            patch_tokens = self.medsiglip(pixel_values=pixel_values).last_hidden_state
        mem = self.patch_proj(patch_tokens)
        q = self.query_embeddings.unsqueeze(0).expand(pixel_values.size(0), -1, -1)
        q = self.norm1(q + self.cross_attention(q, mem, mem, need_weights=False)[0])
        q = self.norm2(q + self.ffn1(q))
        q = self.norm3(q + self.self_attention(q, q, q, need_weights=False)[0])
        q = q + self.ffn2(q)
        return self.vsa_projection(q)


# ----------------------------------------------------------------------------
# Inference stages
# ----------------------------------------------------------------------------
def stage1_atomic_probs(vsa_vectors, ideal_atomic_vecs):
    """Cosine similarity to frozen atoms -> temperature sigmoid -> rectification (Eqs. 5-6)."""
    norm_out = F.normalize(vsa_vectors, dim=-1)
    raw_cosine = (norm_out * ideal_atomic_vecs).sum(dim=-1)
    rectified = torch.clamp(2.0 * torch.sigmoid(raw_cosine * TEMPERATURE_SCALE) - 1.0, min=0.0)
    return {name: float(p) for name, p in zip(ALL_ATOMS, rectified.cpu())}


def stage2_composite_scores(atomic_probs):
    """Composite scores: AND = geometric mean (Eq. 7), OR = max (Eq. 8)."""
    def p(atom):
        return atomic_probs.get(atom, 0.0)

    scores = {}
    for cat, terms in COMPOSITE_RULES.items():
        scores[cat] = {}
        for term, (atoms, op) in terms.items():
            probs = [p(a) for a in atoms]
            if not probs:
                s = 0.0
            elif op == "bind":
                s = float(np.prod(probs) ** (1.0 / len(probs)))
            elif op == "bundle":
                s = float(np.max(probs))
            elif op == "direct":
                s = float(probs[0])
            elif op == "negate":
                s = 1.0 - float(probs[0])
            else:
                s = 0.0
            scores[cat][term] = s
    return scores


def stage3_paris_diagnosis(comp_scores, atomic_probs, comp_vsa, diag_vsa, fbd):
    """
    For each diagnosis: joint score = geometric mean of its constituent scores;
    hypothesis vector = joint score * (bound unscaled constituent vectors);
    similarity = scalar projection of the hypothesis onto the frozen prototype.
    Because the hypothesis and the prototype are built from the same bound vectors,
    the projection reduces to the joint score; the vectors are retained so that the
    composition can be inspected term by term.
    Diagnoses compete through softmax(DIAGNOSIS_SCALE * similarity).
    """
    similarities = {}

    for diag_name, defn in DIAGNOSIS_DEFS.items():
        parts_scores = []
        parts_vecs = []

        for (cat, term) in defn.get("composites", []):
            parts_scores.append(comp_scores[cat][term])
            parts_vecs.append(comp_vsa[cat][term].to(DEVICE))

        for atom in defn.get("raw_atoms", []):
            parts_scores.append(atomic_probs.get(atom, 0.0))
            parts_vecs.append(fbd[atom].to(DEVICE))

        for atom in defn.get("negated_atoms", []):  # required absence
            parts_scores.append(1.0 - atomic_probs.get(atom, 0.0))
            parts_vecs.append(fbd[atom].to(DEVICE))

        bundled = defn.get("bundled_atoms", [])  # OR gate
        if bundled:
            score = max(atomic_probs.get(a, 0.0) for a in bundled)
            vecs = [fbd[a].to(DEVICE) for a in bundled if a in fbd]
            v_bundle = vecs[0]
            for x in vecs[1:]:
                v_bundle = hd.bundle(v_bundle, x)
            v_bundle = F.normalize(v_bundle, dim=-1)
            parts_scores.append(score)
            parts_vecs.append(v_bundle)

        if not parts_vecs:
            sim = 0.0
        else:
            final_p = float(np.prod(parts_scores) ** (1.0 / len(parts_scores)))

            hypothesis_v = parts_vecs[0]
            for v in parts_vecs[1:]:
                hypothesis_v = hd.bind(hypothesis_v, v)
            hypothesis_vec = final_p * hypothesis_v

            prototype_vec = diag_vsa[diag_name].to(DEVICE)
            raw_dot = torch.dot(hypothesis_vec, prototype_vec).item()
            proto_mag_sq = torch.dot(prototype_vec, prototype_vec).item()
            projection = raw_dot / proto_mag_sq if proto_mag_sq > 0 else 0.0
            sim = projection if projection > 0 else 0.0

        similarities[diag_name] = sim

    names = list(similarities.keys())
    sim_tensor = torch.tensor([similarities[n] for n in names], dtype=torch.float32)
    confidences = F.softmax(sim_tensor * DIAGNOSIS_SCALE, dim=0).numpy()

    conf_dict = {n: float(c) for n, c in zip(names, confidences)}
    winner = names[int(np.argmax(confidences))]
    if max(similarities.values()) == 0.0:   # zero-hypothesis rule: no Paris category supported
        winner = "Nondiagnostic / Unsatisfactory"
    return winner, conf_dict, similarities


def stage4_populate_report(winner, comp_scores):
    """Tier 1: slots locked by the winning rule. Tier 2: argmax. Tier 3: threshold."""
    report = {}
    locked_slots = set()

    for (cat, term) in DIAGNOSIS_DEFS[winner]["composites"]:
        report[cat] = {"value": term, "score": comp_scores[cat][term], "tier": 1}
        locked_slots.add(cat)

    for cat in COMPOSITE_RULES:
        if cat in locked_slots or cat == "cell_types_observed":
            continue
        slot_scores = comp_scores[cat]
        best_term = max(slot_scores, key=slot_scores.get)
        report[cat] = {"value": best_term, "score": slot_scores[best_term], "tier": 2}

    cell_scores = comp_scores.get("cell_types_observed", {})
    selected = sorted([(t, s) for t, s in cell_scores.items() if s >= CELL_THRESHOLD],
                      key=lambda x: x[1], reverse=True)
    report["cell_types_observed"] = {
        "value": [t for t, _ in selected] or ["None above threshold"],
        "scores": [s for _, s in selected],
        "tier": 3,
    }
    return report


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------
def print_report(winner, conf_dict, comp_scores, report, atomic_probs):
    print("\n[Stage 1] Atomic scores")
    for name, prob in sorted(atomic_probs.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name:<30s} {prob:.4f}")

    print("\n[Stage 2] Composite scores")
    for cat, terms in comp_scores.items():
        print(f"  {cat}")
        for term, sc in sorted(terms.items(), key=lambda x: x[1], reverse=True):
            print(f"    {term:<30s} {sc:.4f}")

    print("\n[Stage 3] Paris System diagnosis")
    ranked = sorted(conf_dict.items(), key=lambda x: x[1], reverse=True)
    for name, conf in ranked:
        print(f"  {conf * 100:6.2f}%  {name}")
    margin = (ranked[0][1] - ranked[1][1]) * 100 if len(ranked) > 1 else 100.0
    print(f"  Winner: {winner}   (margin over runner-up: {margin:.2f} percentage points)")

    print("\n[Stage 4] Report")
    for key in ("specimen_type", "gross_examination", "cellularity", "background",
                "nc_ratio", "nuclear_contour", "chromatin_pattern",
                "hyperchromasia", "umbrella_cells"):
        e = report[key]
        print(f"  {REPORT_LABEL[key]:<18s}: {e['value']} (score={e['score']:.4f}, tier {e['tier']})")
    ct = report["cell_types_observed"]
    if ct["scores"]:
        cells = ", ".join(f"{v} ({s:.4f})" for v, s in zip(ct["value"], ct["scores"]))
    else:
        cells = "None above threshold"
    print(f"  {'Cell Types':<18s}: {cells}")


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------
def run_inference(image_path, model, ideal_atomic_vecs, comp_vsa, diag_vsa, fbd):
    img = Image.open(image_path).convert("RGB")
    xform = T.Compose([T.Resize((448, 448)), T.ToTensor(),
                       T.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])
    pv = xform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=(DEVICE == "cuda")):
            vsa_out = model(pv).squeeze(0).float()

    atomic_probs = stage1_atomic_probs(vsa_out, ideal_atomic_vecs)
    comp_scores = stage2_composite_scores(atomic_probs)
    winner, conf, _ = stage3_paris_diagnosis(comp_scores, atomic_probs, comp_vsa, diag_vsa, fbd)
    report = stage4_populate_report(winner, comp_scores)
    return atomic_probs, comp_scores, conf, winner, report


def load_all(dictionary_path, weights_path):
    # The dictionary file is a pickled object; load only files you trust.
    fbd = torch.load(dictionary_path, map_location=DEVICE, weights_only=False)
    for k, v in fbd.items():
        fbd[k] = v.tensor().to(DEVICE) if hasattr(v, "tensor") else v.to(DEVICE)

    comp_vsa, diag_vsa = build_vsa_dicts(fbd)
    ideal_atomic_vecs = F.normalize(
        torch.stack([fbd[a] for a in ALL_ATOMS]).to(DEVICE), dim=1)

    model = AtomicVSASupervised(hidden_dim=HIDDEN_DIM, num_queries=NUM_QUERIES, vsa_dim=VSA_DIM)
    sd = torch.load(weights_path, map_location=DEVICE)
    if list(sd.keys())[0].startswith("module."):
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
    # transformers >= 5 names the SigLIP vision weights without the "vision_model." level;
    # map the checkpoint names to the installed version
    model_keys = set(model.state_dict())

    def _rename(k):
        if k in model_keys or not k.startswith("medsiglip."):
            return k
        rest = k[len("medsiglip."):]
        alt = ("medsiglip." + rest[len("vision_model."):] if rest.startswith("vision_model.")
               else "medsiglip.vision_model." + rest)
        return alt if alt in model_keys else k

    sd = {_rename(k): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.to(DEVICE).eval()
    return model, ideal_atomic_vecs, comp_vsa, diag_vsa, fbd


def main(argv=None):
    global DEVICE
    ap = argparse.ArgumentParser(description="Transformer-VSA single-image inference")
    ap.add_argument("--image", required=True, help="path to an RGB cytology patch")
    ap.add_argument("--dictionary", required=True, help="frozen atomic dictionary (.pth)")
    ap.add_argument("--weights", required=True, help="trained Stage-1 weights (.pth)")
    ap.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    ap.add_argument("--json_out", default=None, help="optional path for the full trace as JSON")
    args = ap.parse_args(argv)

    if args.device:
        DEVICE = args.device
    if not os.path.exists(args.image):
        sys.exit(f"Image not found: {args.image}")

    model, ideal, comp_vsa, diag_vsa, fbd = load_all(args.dictionary, args.weights)
    atomic_probs, comp_scores, conf, winner, report = run_inference(
        args.image, model, ideal, comp_vsa, diag_vsa, fbd)

    print_report(winner, conf, comp_scores, report, atomic_probs)

    if args.json_out:
        ranked = sorted(conf.values(), reverse=True)
        trace = {
            "image": os.path.basename(args.image),
            "atomic_scores": atomic_probs,
            "composite_scores": comp_scores,
            "diagnosis": winner,
            "diagnosis_confidence": conf,
            "diagnosis_margin": ranked[0] - ranked[1],
            "report": report,
        }
        with open(args.json_out, "w") as f:
            json.dump(trace, f, indent=2)


if __name__ == "__main__":
    main()
