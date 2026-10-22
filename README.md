# Fossil2Skeleton

**A research prototype for inferring missing 3D skeletal landmarks from observed fragments.**

This repository contains a **reconstruction of the prototype specification**, not the original source archive. The original ZIP/code mentioned in the prior conversation could not be recovered. This implementation was written from the recoverable record of that work and is labeled accordingly. It is an experimental software artifact, not a validated fossil reconstruction system.

## Concept and attribution

The integrated Fossil2Skeleton research concept is Jared Edward Reser's: use fragmentary skeletal morphology as evidence for machine-generated, uncertainty-aware completion of a plausible whole skeleton. Earlier work on virtual fossil reconstruction, geometric morphometrics, and statistical missing-shape estimation is cited below as methodological context. Those references do not describe or establish authorship of this integrated Fossil2Skeleton proposal.

Fossil2Skeleton is distinct from [Fossil2Genome](https://github.com/jaredreser/Fossil2Genome), which explores the inverse direction from skeletal morphology toward hypothetical genomic information. Fossil2Genome informed repository conventions and serves as a related project, not as the implementation or scientific basis for this code.

## What this version demonstrates

- Synthetic 3D landmark generation from a low-rank toy shape space.
- Landmark masking to represent preserved and missing anatomy.
- A PyTorch transformer that takes coordinates plus an observed/missing mask and predicts landmark coordinates.
- MC-dropout spread across repeated predictions as a rough uncertainty diagnostic.
- Mean-shape and nearest-reference baselines.
- A dependency-light runnable baseline demonstration and a training command.

## What it does not demonstrate

- No real fossil, CT, mesh, or extant-specimen data are included.
- The toy generator is not an anatomical simulator and does not encode phylogeny, allometry, articulation constraints, deformation, taphonomy, or biological priors.
- No trained checkpoint is included. Any training run in this repository learns only from the synthetic generator.
- No predictive accuracy, biological validity, calibration, or cross-taxon transfer has been established.
- MC-dropout standard deviation is an exploratory spread measure, not a calibrated confidence interval or probability that a reconstruction is correct.
- A point cloud/landmark completion output is not a watertight 3D skeleton or an articulated reconstruction.

## Install

Python 3.10 or newer is recommended. The synthetic baseline and its tests require NumPy. The transformer and training workflow require PyTorch.

```bash
python -m pip install -r requirements.txt
```

## Run the baseline demo

```bash
python -m fossil2skeleton.demo
```

It creates toy shapes, holds out one synthetic specimen, hides a subset of landmarks, and reports hidden-landmark RMSE for two simple baselines. The numeric output is a software smoke demonstration only.

## Train the toy transformer

```bash
python -m fossil2skeleton.train --epochs 20 --output fossil2skeleton-toy.pt
```

This command creates a synthetic dataset and trains the landmark transformer on randomly masked shapes. It does not use specimen data.

For a downstream application, load a checkpoint and construct `LandmarkTransformer(n_landmarks=...)`. Provide coordinates shaped `(batch, landmarks, 3)`, with missing coordinates set to zero, and a Boolean observed mask shaped `(batch, landmarks)`. Call `complete_with_uncertainty(model, coordinates, observed)` to get the mean completion and per-landmark coordinate spread. Observed coordinates are copied through unchanged.

## Architecture

Each landmark is represented by its 3D coordinate, a binary observed/missing embedding, and a landmark-position embedding. A Transformer encoder attends across landmark tokens. A coordinate head predicts all landmark positions; the completion routine preserves visible input points and returns repeated stochastic predictions for a rough uncertainty summary. Training loss is computed only at missing landmarks. The baseline demo uses no learned model.

## Tests

```bash
python -m unittest discover -s tests -v
```

The core tests cover deterministic data generation, mask invariants, preservation of observed values, baseline behavior, and missing-landmark-only scoring. Transformer-specific runtime tests require PyTorch; the current automated suite does not imply scientific validation.

## Repository provenance

- **Concept origin:** Jared Edward Reser.
- **Code status:** The earlier prototype archive was unavailable. This source snapshot was reconstructed by ChatGPT on 2026-10-22 from the recoverable conversation specification, under Reser's direction; it is not the original implementation.
- **Version:** 0.1.0-reconstruction.
- **License:** MIT, retained from the existing repository.

## References and related work

These works provide methodological context for virtual reconstruction and the Transformer model family. Fossil2Skeleton's integrated proposal and its code are separately attributed above.

1. Gunz, P., Mitteroecker, P., Neubauer, S., Weber, G. W., & Bookstein, F. L. (2009). Principles for the virtual reconstruction of hominin crania. *Journal of Human Evolution, 57*(1), 48–62. https://doi.org/10.1016/j.jhevol.2009.04.004
2. Couette, S., & White, J. (2010). 3D geometric morphometrics and missing-data: Can extant taxa give clues for the analysis of fossil primates? *Comptes Rendus Palevol, 9*(6–7), 423–433. https://doi.org/10.1016/j.crpv.2010.07.002
3. Bucchi, A., et al. (2022). Digital reconstructions using linear regression: How well can it estimate missing shape data from small damaged areas? *Biology, 11*(12), 1741. https://doi.org/10.3390/biology11121741
4. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems, 30*. https://papers.nips.cc/paper/7181-attention-is-all-you-need
5. Reser, J. E. (2025, July 18). *AI-mediated reconstruction of dinosaur genomes from bird DNA and skeletons*. Observed Impulse. https://www.observedimpulse.com/2025/07/ai-mediated-reconstruction-of-dinosaur.html
6. Reser, J. E. (2025). *Fossil2Genome* [software repository]. https://github.com/jaredreser/Fossil2Genome
