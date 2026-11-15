# Literature check: what is new in THEIA (AnalogNet model), and what is not

*Checked 2026-11-15 with a literature search (arXiv, PubMed, web). This is a starting point, not a
complete review. Before any paper or award submission, search Google Scholar and NASA ADS again with
the queries at the bottom, and read the papers listed here yourself.*

## Closest prior work: do not claim these ideas as new

| Work | What it does | How AnalogNet differs |
|---|---|---|
| **Meng et al. 2025**, *Deep learning selection of analogues for Mars landing sites in the Qaidam Basin*, arXiv:2501.08584 | Trains a CNN **similarity metric between Earth and Mars surface images** and applies it to Mars analogs in the Qaidam Basin. | The closest work. It shows that "a learned Earth-Mars similarity" is **not new by itself**. AnalogNet adds the Moon, uses **elevation at one physical scale**, searches the **whole Earth**, and adds multi-axis matching, ensemble uncertainty and a rediscovery benchmark. |
| Stern et al. 2025, JGR Planets 10.1029/2024JE008803 | Framework for choosing **ocean-world** analog field sites, traced to science questions (expert-driven). | Expert framework, not a learned global search. Useful for the "mission purpose" idea. |
| Kavitzky et al. 2021, *Topo2vec*, arXiv:2108.08870 | Self-supervised embedding of **Earth** topography (elevation). | Earth only, no planetary analog task. |
| Girona-Mata et al. 2026, *LunarFM*, arXiv:2607.22408 | Multimodal foundation model of the **Moon**, with similarity search. | Moon only, no Earth/Mars, no analog ranking. |
| Fang et al. 2026, *A domain-specific foundation model for Mars* (ESS Open Archive) | Mars imagery foundation model, similarity search on Mars. | Mars only. |
| Naik et al. 2026, *MarsRecon*, arXiv:2609.22379 | Self-supervised HiRISE representations (Olympus Mons). | Mars only, metre-scale imagery. |
| Ward et al. 2022 (arXiv:2210.09234); Goh et al. 2022 (arXiv:2202.00791) | Contrastive learning on Mars images; mixing domains reduces false negatives. | Supports our choice of mixing bodies in every batch. |
| Phillips et al. 2022, Astrobiology 10.1089/ast.2022.0014 | Deep learning maps habitats in Mars-analog terrain, with identification confidence. | Uncertainty matters for analog work; regional, imagery-based. |
| *Earth Embeddings* review, arXiv:2608.03410 (2026) | Reviews Earth-observation embeddings; notes that **retrieval evaluation is weakly benchmarked**. | Motivates our explicit rediscovery benchmark. |

## Claims we can make (carefully worded)

1. A **single self-supervised terrain embedding shared by Earth, Moon and Mars**, built from elevation
   resampled to one physical pixel size (3.7 km) on local tangent planes. *To our knowledge* prior
   learned analog work covers one or two bodies and uses images.
2. **Purpose-weighted multi-axis matching** (terrain, mobility, thermal, bare ground), so a site can be
   a good analog for rover testing but a poor one thermally.
3. **Ensemble uncertainty with a calibration test** (leave-one-network-out reliability).
4. A **rediscovery benchmark**: known analogs held out from all training and design choices, with a
   random-site significance test, a non-neural baseline and ablations.
5. **Per-match explanations** (feature differences plus occlusion saliency).

**Do not claim:** "first AI for analog sites", "first Earth-Mars similarity model", or that new
candidates are confirmed analogs.

## Search queries to repeat before submission

- "terrestrial analog" AND ("machine learning" OR "deep learning" OR "similarity")
- "Mars analog" site selection remote sensing quantitative
- "lunar analog" site selection
- planetary DEM self-supervised OR contrastive embedding
- Earth Mars Moon shared embedding / cross-planet domain adaptation
