"""Model definitions and preprocessing, identical to NLP_Sentiment_Analysis.ipynb."""
import re
from collections import Counter

import torch
import torch.nn as nn

MAX_VOCAB, MAX_LEN = 20_000, 256


def make_preprocessor():
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    for pkg in ("stopwords", "wordnet", "omw-1.4"):
        nltk.download(pkg, quiet=True)
    stop, lem = set(stopwords.words("english")), WordNetLemmatizer()

    def preprocess_text(text: str) -> str:
        text = re.sub(r"<[^>]+>", " ", text.lower())
        text = re.sub(r"[^a-z\s]", " ", text)
        return " ".join(lem.lemmatize(t) for t in text.split() if t not in stop and len(t) > 1)

    return preprocess_text


def build_vocab(train_texts):
    freq = Counter()
    for t in train_texts:
        freq.update(t.split())
    vocab = ["<PAD>", "<UNK>"] + [w for w, _ in freq.most_common(MAX_VOCAB - 2)]
    return {w: i for i, w in enumerate(vocab)}


def tokenise(text, word2idx):
    return [word2idx.get(tok, 1) for tok in text.split()[:MAX_LEN]]


class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2, dropout=0.3, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(self.dropout(self.embedding(x)), lengths.cpu(),
                                                   batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        return self.fc(self.dropout(torch.cat([hidden[-2], hidden[-1]], dim=1))).squeeze(1)


class CNNBiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_filters=64, kernel_sizes=(2, 3, 4), hidden_dim=256,
                 num_layers=2, dropout=0.3, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes])
        self.lstm = nn.LSTM(num_filters * len(kernel_sizes), hidden_dim, num_layers=num_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x, lengths):
        e = self.dropout(self.embedding(x)).permute(0, 2, 1)
        outs = [torch.relu(conv(e)) for conv in self.convs]
        n = min(c.size(2) for c in outs)
        feat = torch.cat([c[:, :, :n] for c in outs], dim=1).permute(0, 2, 1)
        _, (hidden, _) = self.lstm(feat)
        return self.fc(self.dropout(torch.cat([hidden[-2], hidden[-1]], dim=1))).squeeze(1)
