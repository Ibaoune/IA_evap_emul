# Author: M. El Aabaribaoune

et_downscaling/
│
├── train.py              → Main training entry point  (RCAN or LSTM)
├── eval.py               → Inference, metrics & diagnostics
│
├── config.yaml           → Global config  (set model_type: rcan | lstm)
├── config.py             → YAML → Python wrapper
│
├── data_loading.py       → Load LMDZ250 predictors + LMDZ35 ET target
├── preprocessing.py      → Split by year, Morocco mask, normalisation
│
├── training.py           → RCAN spatial training loop
│                            + LSTM point-wise training loop
├── evaluation.py         → Inference helpers, metrics, plots, NetCDF save
│
├── rcan_arch.py          → RCAN model + BernoulliGammaLoss (spatial mask)
├── lstm_arch.py          → PointLSTM + BernoulliGammaLossPointwise
│                            + make_sequences helper
│
├── utils.py              → mask_dataset, build_morocco_mask, inspect,
│                            vprint, save_model, load_model
│
├── job_cpu.sh            → SLURM CPU job
├── job_gpu.sh            → SLURM GPU job
│
└── results/
    ├── rcan/
    │   ├── output_data/
    │   ├── output_figs/
    │   └── Metrics/
    └── lstm/
        ├── output_data/
        ├── output_figs/
        └── Metrics/
