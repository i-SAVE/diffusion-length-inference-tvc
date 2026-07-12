Local notebook variants (provenance note)
=========================================

mnist_budget_sweep_local.ipynb
    Local source of the MNIST training-budget sweep. Functionally identical to
    ../mnist-budget-sweep-2026-07-08.source.ipynb, which is the version that was
    executed on Kaggle and produced the numbers in Table 1 (see results/logs/).

cifar10_diffusers_scipy_variant.ipynb
    An ALTERNATIVE CIFAR-10 implementation that uses the diffusers UNet2DModel
    (with attention blocks) and scipy.linalg.sqrtm for the FID matrix square root.
    It was NOT the run that produced the numbers in Tables 2-3. The reported
    CIFAR-10 numbers come from the pure-PyTorch variant executed on Kaggle
    (../cifar10-diffusion-strong-2026-07-08.*.ipynb, numpy eigendecomposition
    FID), whose full session log is in results/logs/cifar_kaggle_session.log.
    This variant is included for completeness and as a scipy-based cross-check
    path; the two FID computations are algebraically equivalent.
