## Feasibility Review
1. User Need, Scope & Acceptance Criteria
User Need
Production of synthetic medical cohorts for prostate MRI training data. Requires generated images that satisfy (1) fidelity (plausible anatomy) (2) diversity (no model collapse) (3) privacy (no memorised scans from training)
Scope
Generating 2D axial pelvic MRI slices (256 × 128) and comparing them with simpler VAE baselines on held-out patients.
Acceptance Criteria
1.	Reconstruction fidelity: Mean SSIM of VQ-VAE reconstructions (> 0.6)
2.	Generated-slice fidelity: Mean SSIM of each sample to nearest real slice (> 0.3)
3.	No Memorisation: (a) Proportion of samples above 99th percentile NN-SSIM of real unseen patients (< real-patient rate) (b) Proportion of near-duplicate samples (NN-SSIM > 0.95)
4.	Coverage & Density: (a) Mean Pairwise SSIM (< Real Test Slices) (b) fidelity and diversity metrics (Naeem et al., 2020) (> 50% of real patient density & coverage)
5.	Resource Budget: (a) Peak training VRAM (< 16 GB) (b) Average generation time per slice on an A100 (< 1 s)
2. Model Choice & Course Concepts
•	VQ-VAE (van den Oord et al., 2017): the encoder downsamples the image into a 64 × 32 grid, with each position assigned one of 512 learned codes. The codebook is gradually updated using exponential moving averages. It uses a straight-through estimator to allow gradients to pass through the discrete code selection during training.
•	Gated PixelCNN Prior (van den Oord et al., 2016): learns the patterns and relationships between the VQ-VAE's discrete codes to generate new code grids one position at a time. Then the VQ-VAE decoder converts these grids into synthetic MRI slices.
•	Baseline: a Gaussian VAE (Kingma & Welling, 2014) uses the same encoder and decoder architecture as the VQ-VAE to compare its discrete code representation & the prior against the VAE’s latent representations in generating synthetic MRI slices.
3. Preliminary Feasibility Evidence
- Data: 11460 / 660 / 540 slices from 31 / 4 / 3 patients (train / validate / test). Confirmed no patient leakage between splits. 60 slices which were 256 x 144 were centre-cropped. Pixel values were percentile-clipped and min-max scaled to [0, 1] as SSIM and PSNR require known range.
•	Smoke Test: 256 slices per split. Peak VRAM of 1.1 GB per autoencoder, 5.5 GB for prior. Evidence of codebook collapse (2/512 codes used), plan to initialise codebook with encoder outputs and reinitialise unused codes every 20 training iterations.
4. Risks, Budget & Fallback
Risks
•	Prior over-fitting: Implement mirror training samples, dropout, weight decay, and decreasing number of channels based on training vs validation performance
•	Unfair baseline: Test VAE with varying KL-weights
Budget
•	A100: 4 GPU hours for main training, prediction and evaluation pipeline and 13 GPU-hours for side experiments (regularisation comparison and hyperparameter optimisation)
Next
•	Optimise regularisation of prior and KL weight of VAE. Then train models and evaluate pairwise / nearest neighbour SSIM, diversity, memorisation, and whether model collapse occurs.
