import torch
import torch.nn as nn
import torch.nn.functional as F
import tokenmonster
from models import WaifuLMUwU

tokenizer = tokenmonster.load("englishcode-16000-balanced-v1")
model = WaifuLMUwU(vocab_size=len(tokenizer), max_seq_length=1024)

model.load_state_dict(torch.load('trained_model.pth'))
model.eval()

start_tokens = torch.tensor([[0]], dtype=torch.long)

with torch.no_grad():
    idx = start_tokens
    for _ in range(200):
        # Since we used cce loss, the output logits are slightly different here
        # First we get hidden states and then mul by classifier weights (T)
        hidden_states, classifier_weights = model(idx)
        logits = torch.matmul(hidden_states[:, -1:], classifier_weights.t())
        idx_next = torch.multinomial(F.softmax(logits.squeeze(1), dim=-1), num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

text = tokenizer.decode(idx[0].tolist())
print(text)