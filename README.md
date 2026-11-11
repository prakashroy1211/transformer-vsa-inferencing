# Transformer-VSA – Supplementary Code

Inference code for the manuscript *"Transformer-VSA: A Neuro-Symbolic Framework for Explainable
Structured Report Generation in Urine Cytology"*. It reproduces the worked example in
Table 8 / Figure 5(a) (image `P9142585.JPG`) and writes the complete evidence trace
(atomic scores → composite findings → diagnosis → report) for any input patch.

## Contents

| File | Purpose |
|---|---|
| `transformer_vsa_inference.py` | Stage 1 model, frozen rule bases (`COMPOSITE_RULES`, `DIAGNOSIS_DEFS`), Stages 2–4, CLI |
| `requirements.txt` | Python dependencies |
| `frozen_dictionary_atomic_terms_v12.pth` | Frozen bipolar atomic dictionary |
| `best_model.pth` | Trained Stage-1 weights |
| `P9142585.JPG` | Input image of the worked example (Fig. 5(a), Table 8) |
| `expected_output/P9142585_trace.json` | Reference output of the command below |
| `expected_output/P9142585_console.txt` | Reference console printout of the same run |

## Setup

```bash
pip install -r requirements.txt
export HF_TOKEN=<your own Hugging Face access token>
```

The vision backbone `google/medsiglip-448` is a gated model. Accept its terms on the Hugging Face
model page and provide your own token via `HF_TOKEN` (or `huggingface-cli login`). No credentials
are included in this package. A GPU is recommended; the script also runs on the CPU (`--device cpu`).

## Run

```bash
python transformer_vsa_inference.py \
    --image P9142585.JPG \
    --dictionary frozen_dictionary_atomic_terms_v12.pth \
    --weights best_model.pth \
    --json_out trace.json
```

`trace.json` contains the atomic scores, all composite scores, the diagnosis confidences, the margin
between the top two diagnoses, and the report fields with their tier (1 = locked by the winning rule,
2 = argmax, 3 = threshold). Compare it with `expected_output/P9142585_trace.json`.

## Where the manuscript equations live

| Manuscript | Code |
|---|---|
| Eqs. (4)–(6): projection, temperature sigmoid, rectification | `AtomicVSASupervised.forward`, `stage1_atomic_probs` |
| Eq. (7): AND gate, geometric mean | `stage2_composite_scores` (`op == "bind"`) |
| Eq. (8): OR gate, maximum | `stage2_composite_scores` (`op == "bundle"`) |
| Table 2, Eq. (9): diagnostic prototypes and hypothesis vectors | `DIAGNOSIS_DEFS`, `build_vsa_dicts`, `stage3_paris_diagnosis` |
| Eq. (10): scalar projection, softmax with λ = 15, zero-hypothesis rule | `stage3_paris_diagnosis` (`DIAGNOSIS_SCALE = 15`) |
| Section 3.4: three-tier report | `stage4_populate_report` |

## Expected output for `P9142585` (Figure 5a / Table 8)

Diagnosis confidences: AUC 50.84 %, HGUC 49.15 %, all other categories 0.00 %.
Selected composites: Cellularity Adequate 0.9636; N:C 0.5–0.7 0.9633 (>0.7: 0.9518);
Nuclear Contour Irregular 0.9631; Chromatin Coarsely granular 0.9635; Hyperchromasia Present 0.9632.

Small last-digit differences can occur across GPU models and library versions.

## Notes

* The rule bases are the frozen knowledge base described in the paper. Changing a rule changes the
  report without any retraining of the neural stage.
* The dictionary file is a pickled object (`torch.load(..., weights_only=False)`); load only the file
  supplied with this package.
* The code runs with transformers 4.x and 5.x (the checkpoint's MedSigLIP weight names are mapped
  to the installed version).
* Apart from the example image `P9142585.JPG`, the clinical image data cannot be redistributed
  (institutional ethics approval).
