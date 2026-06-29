---
title: Chemistry Molecule Predictor (FYP)
emoji: 🧪
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
license: mit
---

# Chemistry Molecule Predictor — FYP Demo

Live demo for the MSc Final Year Project:
**Lightweight Domain Adaptation of Small Language Models for Chemistry — A Parameter-Efficient Fine-Tuning Approach Using LoRA and QLoRA.**

## What it does
Type a molecule name (e.g. *Aspirin*) or paste a SMILES, pick a task and an adapter, and the fine-tuned Phi-4-mini model predicts the property.

- **BBBP** — blood-brain barrier penetration (Yes/No)
- **BACE** — BACE-1 enzyme inhibition (Yes/No)
- **ESOL** — aqueous solubility (log mol/L)

Names are resolved to SMILES via the PubChem PUG REST API.

## Models
- **Base:** `microsoft/Phi-4-mini-instruct` (3.8 B parameters)
- **Adapters:** 6 PEFT adapters fine-tuned on MoleculeNet BBBP / BACE / ESOL with scaffold splits

## Hardware
HF Spaces ZeroGPU (Nvidia T4 on demand). First prediction has a cold-start delay; subsequent ones are fast.

## Code
[github.com/gbolahan507/chemistry-peft-fyp](https://github.com/gbolahan507/chemistry-peft-fyp)
