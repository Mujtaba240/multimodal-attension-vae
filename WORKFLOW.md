# Development Workflow

## Architecture

```
VS Code (local)               GitHub                 Colab Pro (GPU)
┌─────────────────┐    push   ┌──────┐    pull    ┌──────────────────┐
│  Write code     │ ────────► │      │ ◄──────── │  Train models    │
│  Edit configs   │           │ Repo │            │  Run evals       │
│  Write report   │ ◄──────── │      │ ────────► │  Save → Drive    │
└─────────────────┘    pull   └──────┘            └──────────────────┘
                                                           │
                                                  Google Drive
                                                  ┌──────────────────┐
                                                  │  data/           │
                                                  │  checkpoints/    │
                                                  │  results/        │
                                                  └──────────────────┘
```

## Google Drive Layout

```
DL&AI_Project/
├── data/celeba/
├── checkpoints/{image_vae,attr_vae,mvae,mmvae,attn_fuse}/
└── results/{figures,tables}/
```

