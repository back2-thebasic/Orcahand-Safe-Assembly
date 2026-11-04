# OrcaHand Pen Spin

[English](README.md) | [中文](README.zh-CN.md) | [한국어](README.ko.md)

This directory contains the models, official demonstrations, converted data, and experimental results used for the pen-spin task.

```text
pen_spin/
├── models/
│   ├── pen/              Official MuJoCo pen model, textures, and provenance
│   └── hand/             OrcaHand mesh assets required for playback
├── hardware/             STEP and 3MF print files and visual marker assets
├── data/
│   ├── official/         40 official demonstrations, indexes, and recipes
│   └── converted/        20 initial trajectories retargeted to OrcaHand
├── results/
│   ├── inspection-*     Analysis results and images for clips 003/006
│   └── optimized003-*   Optimization iterations for clip 003 and comparison catalog
├── view.py               Reference and free-dynamics playback tool
├── portable_scene.py     Scene path utilities used by the playback tool
└── requirements.txt      Python dependencies for playback
```

`data/converted` contains the initial conversion results; `results` contains subsequent analyses and optimization outputs. These trajectories are still reference candidates: pen spinning has not yet succeeded under free dynamics, and RL training has not started.

The pen model entry point is `models/pen/assets/pen_tracking.xml`, which depends on `models/pen/textures/` in the provided directory layout. This is the original official XML. The official loading code additionally sets the mass to 30 g and scales the inertia proportionally; that adjustment is not embedded in this XML.
