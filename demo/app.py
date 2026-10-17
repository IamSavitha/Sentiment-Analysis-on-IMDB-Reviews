"""Gradio demo: IMDB sentiment with the two from-scratch models (BiLSTM vs CNN + BiLSTM)."""
import json
import random
from pathlib import Path

import gradio as gr
import torch

from models import BiLSTMModel, CNNBiLSTMModel, make_preprocessor, build_vocab, tokenise

HERE = Path(__file__).resolve().parent
VOCAB = HERE / "vocab.json"
CHECK = HERE / "verification.json"


def weights(name):
    return next(p for p in [HERE / name, HERE.parent / name] if p.exists())


preprocess = make_preprocessor()

if VOCAB.exists():
    word2idx = json.loads(VOCAB.read_text())
else:  # rebuilt exactly as in training: top 20K words of the preprocessed IMDB train split
    from datasets import load_dataset
    train = load_dataset("stanfordnlp/imdb", split="train")
    word2idx = build_vocab(preprocess(t) for t in train["text"])
    VOCAB.write_text(json.dumps(word2idx))

models = {
    "BiLSTM (baseline)": BiLSTMModel(len(word2idx)),
    "CNN + BiLSTM (hybrid)": CNNBiLSTMModel(len(word2idx)),
}
models["BiLSTM (baseline)"].load_state_dict(torch.load(weights("model1_best.pth"), map_location="cpu"))
models["CNN + BiLSTM (hybrid)"].load_state_dict(torch.load(weights("model2_best.pth"), map_location="cpu"))
for m in models.values():
    m.eval()


@torch.no_grad()
def prob_positive(model, texts):
    seqs = [tokenise(preprocess(t), word2idx) or [1] for t in texts]
    lengths = torch.tensor([len(s) for s in seqs])
    batch = torch.zeros(len(seqs), max(lengths).item(), dtype=torch.long)
    for i, s in enumerate(seqs):
        batch[i, :len(s)] = torch.tensor(s)
    return torch.sigmoid(model(batch, lengths)).tolist()


# Check that the rebuilt vocabulary matches the trained weights: accuracy on a held-out sample.
if CHECK.exists():
    verification = json.loads(CHECK.read_text())
else:
    from datasets import load_dataset
    test = load_dataset("stanfordnlp/imdb", split="test")
    idx = random.Random(0).sample(range(len(test)), 2000)
    texts, labels = [test[i]["text"] for i in idx], [test[i]["label"] for i in idx]
    verification = {"n": len(idx)}
    for name, m in models.items():
        probs = []
        for k in range(0, len(texts), 100):
            probs += prob_positive(m, texts[k:k + 100])
        verification[name] = round(100 * sum((p >= 0.5) == bool(y) for p, y in zip(probs, labels)) / len(labels), 1)
    CHECK.write_text(json.dumps(verification))


def classify(review):
    if not review or not review.strip():
        return {}, {}
    out = []
    for name, m in models.items():
        p = prob_positive(m, [review])[0]
        out.append({"Positive": p, "Negative": 1 - p})
    return out[0], out[1]


with gr.Blocks(title="IMDB Sentiment") as demo:
    gr.Markdown(
        "# IMDB Sentiment: BiLSTM vs CNN + BiLSTM\n"
        "Two models trained from scratch in PyTorch (no pre-trained embeddings) on 25K IMDB reviews. "
        f"Re-checked at startup on {verification['n']:,} held-out test reviews: "
        f"**BiLSTM {verification['BiLSTM (baseline)']}%**, **CNN + BiLSTM {verification['CNN + BiLSTM (hybrid)']}%** accuracy. "
        "[Source code](https://github.com/IamSavitha/Sentiment-Analysis-on-IMDB-Reviews)"
    )
    review = gr.Textbox(label="Movie review", lines=6,
                        value="The plot dragged in the middle, but the performances were wonderful and the ending stayed with me.")
    btn = gr.Button("Classify", variant="primary")
    with gr.Row():
        o1 = gr.Label(label="BiLSTM (baseline)")
        o2 = gr.Label(label="CNN + BiLSTM (hybrid)")
    btn.click(classify, review, [o1, o2])
    gr.Examples([
        ["An absolute masterpiece. I was hooked from the first scene to the last."],
        ["Two hours I will never get back. Wooden acting and a script full of clichés."],
        ["Not as bad as the critics said, but I wouldn't watch it again."],
    ], review)

if __name__ == "__main__":
    demo.launch()
