# KG-Causal: A Knowledge-Guided Causal Framework for Interpretable Radiology Report Generation

**Official research implementation (release preparation)** · [Paper manuscript](#-citation)

## 📖 Abstract

Radiology Report Generation (RRG) aims to produce clinically meaningful descriptions from chest X-ray images. However, conventional data-driven methods can over-rely on statistical co-occurrences between findings, leading to spurious associations and clinically inconsistent reports.

We propose **KG-Causal**, a knowledge-guided causal reasoning framework built around a **Perception–Intervention–Generation (PerInGen)** pipeline. It constructs structured radiological knowledge from reports and uses semantic counterfactual interventions to distinguish intervention-sensitive diagnostic evidence from potentially spurious correlations.

1. **Knowledge Graph Construction (KGC):** Extract radiological entities and relations with RadGraph, organize them into disease-state semantic clusters with CheXbert labels and LLM-assisted semantic aggregation, and retrieve image-specific knowledge subgraphs.
2. **Semantic Counterfactual Reasoning (SCR):** Modify selected diagnostic cluster states (e.g., present → absent) in the knowledge space while keeping the chest X-ray evidence unchanged.
3. **Causal Interaction Fusion (CIF):** Perform bidirectional vision–knowledge attention for knowledge-guided perception and evidence-corrected knowledge reasoning.
4. **Causal Gating Unit (CGU):** Use factual–counterfactual representation discrepancies to modulate the fused features before report generation.

The method is evaluated on **IU-Xray**, **MIMIC-CXR**, and **CheXpert Plus** using both language-generation and clinical-efficacy metrics.

## 🧠 Framework

The method comprises **offline knowledge induction** and an **online PerInGen pipeline**:

```text
Radiology reports ── RadGraph + CheXbert + LLM-SA ──► Global knowledge graph
                                                            │
Chest X-ray ──► Visual encoder ──► Disease-state retrieval ──► Factual KG
      │                                                     │
      │                                           Semantic intervention (SCR)
      │                                                     │
      │                                              Counterfactual KG
      │                                                     │
      └──────────────► Bidirectional fusion (CIF) ◄─────────┘
                                   │
                       Factual–counterfactual difference
                                   │
                          Causal gating (CGU)
                                   │
                         Transformer decoder
                                   │
                           Generated report
```

The paper's **Fig. 2** provides the full architectural diagram. The LLM is used for *offline semantic structuring and intervention design*, not as the report-generation backbone.

## 🛠️ Environment Setup

**1. Clone the repository**

```bash
git clone https://github.com/hustselab511/KG-Causal.git
cd KG-Causal
```

**2. Create an environment**

```bash
conda create -n kg-causal python=3.10 -y
conda activate kg-causal
# Install a PyTorch / torchvision build compatible with your CUDA environment.
```

**3. Install dependencies**

```bash
# Run after a validated requirements.txt has been added to the release.
pip install -r requirements.txt
```

The supplied code imports PyTorch, torchvision, Transformers, NumPy, pandas, SciPy, and Pillow. Some evaluation scripts additionally use Java-based tools. **Exact tested versions and a complete dependency manifest have not yet been provided.**

## 💾 Data Preparation

| Dataset | Role | Source |
| :--- | :--- | :--- |
| **IU-Xray (Indiana University Chest X-ray)** | Small-scale RRG benchmark | [Open-i](https://openi.nlm.nih.gov/) |
| **MIMIC-CXR** | Large-scale RRG benchmark | [PhysioNet MIMIC-CXR](https://physionet.org/content/mimic-cxr/) |
| **CheXpert Plus** | Additional external benchmark in the paper | [CheXpert Plus](https://stanfordaimi.azurewebsites.net/datasets/8c9f4a30-66a0-4e38-b8e7-8a8d1ef10d18) |

Obtain the data through the respective providers and follow their access requirements. **No clinical images or reports are redistributed here.** The provided source snapshot includes IU-Xray and MIMIC-CXR data loaders; a verified CheXpert Plus loader is not present.

### Knowledge graph preparation

1. Extract radiological entities and relations from training reports using **RadGraph**.
2. Assign clinical disease-state categories using **CheXbert/CheXpert-style labels**.
3. Apply **LLM-assisted semantic aggregation (LLM-SA)** to standardize entities and form clinically meaningful concept clusters.
4. Retrieve the relevant subgraphs for each image based on predicted disease states.
5. Generate factual and semantically intervened counterfactual graph representations for model training.

The manuscript describes **28 concept clusters** for IU-Xray and MIMIC-CXR. The exact preprocessing inputs, model resources, and output paths must be configured for the released implementation. The uploaded archive does not contain the completed knowledge-graph assets or a verified end-to-end construction command.


### Training and report generation

The current entry point is:

```bash
python -m trainer.KM_main
```

It loads `config/mimic_cxr.json` and uses its `task` field to select `train` or `sample`. **This is the source entry point, not a verified runnable reproduction command for the provided archive.** The archive lacks that configuration and several imported modules; see [Release checklist](#-release-checklist).

## 📊 Hyperparameters

The following settings are reported in the supplied manuscript; they should be checked against the final released configuration files.

| Component | Parameter | Value |
| :--- | :--- | :--- |
| Visual encoder | Backbone | ImageNet-pretrained ResNet-101 |
| Knowledge encoder | Backbone | Bio ClinicalBERT |
| Input image | Default resolution | 224 × 224 |
| Shared semantic space | Embedding dimension | 512 |
| Training | Batch size | 16 |
| Training | IU-Xray epochs | 30 |
| Training | MIMIC-CXR epochs | 15 |
| Training | ResNet learning rate | 5 × 10⁻⁵ |
| Training | Other-module learning rate | 1 × 10⁻⁴ |
| IU-Xray | Maximum report length | 60 tokens |
| MIMIC-CXR | Maximum report length | 80 tokens |
| Knowledge space | Diagnostic concept clusters | 28 |
| Training hardware | GPU reported in manuscript | NVIDIA RTX 4090 |

The manuscript describes training with cross-entropy loss for the final report; a counterfactual branch contributes discrepancy signals rather than an independently supervised counterfactual-report loss.

## 📈 Results

**Results below are transcribed from the supplied revised manuscript, not independently reproduced from this code archive.**

| Dataset | BLEU-4 | METEOR | ROUGE-L | CIDEr | Clinical Macro-F1 | Clinical Micro-F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| IU-Xray | 0.224 | 0.220 | 0.426 | 0.486 | — | — |
| MIMIC-CXR | 0.150 | 0.184 | 0.317 | 0.237 | 0.402 | 0.544 |
| CheXpert Plus | 0.128 | 0.177 | 0.301 | 0.223 | 0.371 | 0.576 |

Evaluation includes **BLEU-1–4, METEOR, ROUGE-L, CIDEr**, **CheXbert-based clinical efficacy**, and **RadGraph-based entity/relation metrics**. These metrics use different protocols and should not be compared across datasets without accounting for the evaluation setup.

## ⚠️ Release Checklist

Before describing this repository as fully reproducible, the following items need to be resolved in the uploaded code snapshot:

- Add `config/mimic_cxr.json` (and other required experiment configurations).
- Restore missing imported modules, including `models/KM_c.py`, `trainer/KMC_trainer.py`, `modules/radgraph_m.py`, and `utils/radgraph_download.py`, or remove unused imports safely.
- Replace machine-specific data/model paths and configure knowledge resources.
- Fix the malformed f-string in `trainer/KM_main.py` (`print(f"seed:{args['seed']}")`).
- Publish a tested `requirements.txt`, reproducible train/evaluation commands, and checkpoint/data-preparation instructions.
- Review third-party licenses and bundled metric dependencies before distribution.

These limitations apply to the **uploaded snapshot** and may not reflect the full development repository.

## 📝 Citation

If this work is useful, please cite the manuscript. **A final publication venue, DOI, and BibTeX record have not been confirmed**, so the entry below is intentionally marked as a manuscript rather than a published article.

```bibtex
@misc{yao2026kgcausal,
  title  = {A Knowledge-Guided Causal Framework for Interpretable Radiology Report Generation},
  author = {Yao, Keyu and Ju, Jinglin and Shen, Gang},
  year   = {2026},
  note   = {Manuscript; publication details to be updated}
}
```

## 📬 Contact

For questions and issues, please use the repository's GitHub Issues page.

**Disclaimer:** This repository is intended for research only. Generated radiology reports must not be used as a substitute for professional clinical interpretation.
