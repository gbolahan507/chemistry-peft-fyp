"""
Chemistry Molecule Predictor — Gradio demo for HF Spaces (ZeroGPU).

Loads microsoft/Phi-4-mini-instruct (fp16) plus 2 LoRA adapters from
Gbolahan507/phi4-lora-{bbbp,bace}.

User types a molecule name (e.g. "Aspirin") or SMILES, picks a task,
and gets a Yes/No prediction with confidence.

FYP: Lightweight Domain Adaptation of Small Language Models for Chemistry.
"""

import gradio as gr
import spaces
import torch
import requests
from functools import lru_cache

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "microsoft/Phi-4-mini-instruct"
HF_USER = "Gbolahan507"

ADAPTER_REPOS = {
    "BBBP": f"{HF_USER}/phi4-lora-bbbp",
    "BACE": f"{HF_USER}/phi4-lora-bace",
}

TASK_DESCRIPTIONS = {
    "BBBP": "Blood-brain barrier penetration (Yes/No)",
    "BACE": "BACE-1 enzyme inhibition (Yes/No) — Alzheimer's drug target",
}

PROMPT_TEMPLATES = {
    "BBBP": "You are a chemistry assistant. Given the SMILES, does this molecule cross the blood-brain barrier? Answer yes or no.\nSMILES: {smiles}\nAnswer:",
    "BACE": "You are a chemistry assistant. Given the SMILES, does this molecule inhibit the BACE-1 enzyme? Answer yes or no.\nSMILES: {smiles}\nAnswer:",
}

EXAMPLES = {
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Ibuprofen": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
}


def looks_like_smiles(text: str) -> bool:
    text = text.strip()
    if not text:
        return False
    smiles_chars = set("()[]=#@+-./\\1234567890")
    has_smiles_char = any(c in smiles_chars for c in text)
    has_space = " " in text
    return has_smiles_char and not has_space


def name_to_smiles(name: str) -> str | None:
    name = name.strip()
    if not name:
        return None
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/CanonicalSMILES/JSON"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json()["PropertyTable"]["Properties"][0]["CanonicalSMILES"]
    except Exception as e:
        print(f"PubChem lookup failed for {name!r}: {e}")
    return None


def resolve_input(user_input: str) -> tuple[str, str]:
    user_input = user_input.strip()
    if looks_like_smiles(user_input):
        return user_input, user_input
    smiles = name_to_smiles(user_input)
    if smiles is None:
        raise gr.Error(f"Could not resolve '{user_input}' to a SMILES. Try pasting a SMILES string directly.")
    return user_input, smiles


_tokenizer = None
_base_model = None


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
    return _tokenizer


def get_base():
    global _base_model
    if _base_model is None:
        _base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.float16, trust_remote_code=True
        ).to("cuda")
    return _base_model


@lru_cache(maxsize=4)
def get_adapter(task: str):
    repo_id = ADAPTER_REPOS[task]
    base = get_base()
    model = PeftModel.from_pretrained(base, repo_id)
    model.eval()
    return model


@spaces.GPU(duration=120)
def predict(user_input: str, task: str):
    if not user_input or not user_input.strip():
        raise gr.Error("Please enter a molecule name or SMILES.")

    name, smiles = resolve_input(user_input)
    tokenizer = get_tokenizer()
    model = get_adapter(task)

    prompt = PROMPT_TEMPLATES[task].format(smiles=smiles)
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to("cuda")

    with torch.no_grad():
        logits = model(**enc).logits[0, -1]

    yes_id = tokenizer.encode(" yes", add_special_tokens=False)[0]
    no_id  = tokenizer.encode(" no",  add_special_tokens=False)[0]
    yes_score = float(logits[yes_id])
    no_score  = float(logits[no_id])
    probs = torch.softmax(torch.tensor([yes_score, no_score]), dim=0)
    yes_prob = float(probs[0])
    verdict = "YES" if yes_prob > 0.5 else "NO"
    confidence = max(yes_prob, 1 - yes_prob)
    result_line = f"**Prediction:** {verdict}  \n**Confidence:** {confidence:.3f}"

    return (
        f"### Result\n\n"
        f"**Name:** {name}  \n"
        f"**SMILES:** `{smiles}`  \n"
        f"**Task:** {task} — {TASK_DESCRIPTIONS[task]}  \n"
        f"**Adapter:** Phi-4-mini-instruct + LoRA  \n\n"
        f"{result_line}"
    )


with gr.Blocks(title="Chemistry Mol Predictor — FYP Demo") as demo:
    gr.Markdown(
        "# Chemistry Molecules Predictor\n"
        "MSc FYP: *Lightweight Domain Adaptation of Small Language Models for Chemistry "
        "— A Parameter-Efficient Fine-Tuning Approach Using LoRA*.\n\n"
        "Type a molecule **name** (e.g. *Aspirin*) or paste a **SMILES** string, pick a task, "
        "and the fine-tuned Phi-4 model predicts the property."
    )

    with gr.Row():
        with gr.Column(scale=2):
            inp = gr.Textbox(
                label="Molecule (name or SMILES)",
                placeholder="e.g. Aspirin   or   CC(=O)Oc1ccccc1C(=O)O",
                value="Aspirin",
            )
            with gr.Row():
                for ex_name in EXAMPLES:
                    gr.Button(ex_name, size="sm").click(
                        lambda n=ex_name: n, outputs=inp
                    )

            task = gr.Dropdown(
                choices=["BBBP", "BACE"],
                value="BBBP",
                label="Task",
                info="BBBP: brain barrier · BACE: enzyme inhibition",
            )
            btn = gr.Button("Predict", variant="primary")

        with gr.Column(scale=3):
            out = gr.Markdown(label="Result")

    btn.click(predict, inputs=[inp, task], outputs=out)

    gr.Markdown(
        "---\n"
        "**Base model:** [microsoft/Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct)  \n"
        "**Adapters:** trained on MoleculeNet BBBP and BACE with scaffold splits  \n"
        "**Code:** [github.com/gbolahan507/chemistry-peft-fyp](https://github.com/gbolahan507/chemistry-peft-fyp)"
    )

if __name__ == "__main__":
    demo.launch()
