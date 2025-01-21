import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader
import tokenmonster
from tqdm import tqdm
import math, os, sys, json, glob
from torch.optim.lr_scheduler import CosineAnnealingLR
from distributed_shampoo import AdamGraftingConfig, DistributedShampoo
from cut_cross_entropy import linear_cross_entropy

from models.model import WaifuLMUwU
from utils.trainutils import count_parameters_layerwise

class JSONLDataset(Dataset):
    def __init__(self, directory_path, tokenizer, seq_length=1024, text_key="text", max_files=None):
        self.seq_length = seq_length
        tokens = []
        
        for i, file in enumerate(glob.glob(os.path.join(directory_path, "*.jsonl"))):
            if max_files and i >= max_files:
                break
                
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        text = json.loads(line)[text_key]
                        if len(text) >= 100:
                            tokens.extend(tokenizer.tokenize(text))
                    except:
                        continue
        
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        total_seqs = len(self.tokens) // seq_length
            
        trimmed_length = total_seqs * seq_length
        self.sequences = self.tokens[:trimmed_length].view(-1, seq_length)
        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        return self.sequences[idx]

def train_model(model, train_loader, optimizer, device, epochs=5):
    model.train()
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda")
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-5 
    )
    
    model = torch.compile(
        model,
        backend='inductor',
        dynamic=False,
        fullgraph=True,
        options={
            "epilogue_fusion": True,
            "max_autotune": True,
        }
    )
    
    for epoch in range(epochs):
        running_loss = 0.0
        total_batches = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch_idx, data in enumerate(progress_bar):
            data = data.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.autocast(device_type='cuda', dtype=torch.float32):
                hidden_states, classifier_weights = model(data)
                
                loss = linear_cross_entropy(
                    hidden_states,
                    classifier_weights,
                    data,
                    shift=True,
                    reduction="mean"  # Look more into this
                )
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Update metrics - just add the loss itself
            running_loss += loss.item()
            total_batches += 1
            avg_loss = running_loss / total_batches
            perplexity = math.exp(min(avg_loss, 100))

            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'ppl': f'{perplexity:.2f}'
            })

            if batch_idx % 100 == 0:
                print(f'\nBatch {batch_idx}/{len(train_loader)}: '
                      f'Loss: {avg_loss:.4f}, '
                      f'Perplexity: {perplexity:.2f}, '
                      f'Batches Processed: {total_batches}')

        epoch_loss = running_loss / total_batches
        epoch_ppl = math.exp(min(epoch_loss, 100))
        print(f'\nEpoch {epoch+1} Summary:')
        print(f'Average Loss: {epoch_loss:.4f}')
        print(f'Perplexity: {epoch_ppl:.2f}')
        print(f'Total Batches Processed: {total_batches}\n')
        
def main():
    BATCH_SIZE = 16
    SEQ_LENGTH = 1024
    EPOCHS = 3
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = tokenmonster.load("englishcode-16000-balanced-v1")

    model = WaifuLMUwU(
        vocab_size=len(tokenizer),
        max_seq_length=SEQ_LENGTH
    ).to(DEVICE)

    dataset = JSONLDataset(
        directory_path="./Data_big",
        tokenizer=tokenizer,
        seq_length=SEQ_LENGTH,
        text_key="text",
        max_files=None
    )
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    optimizer = DistributedShampoo(
        model.parameters(),
        lr=0.0001,
        betas=(0.9, 0.999),
        epsilon=1e-12,
        weight_decay=1e-05,
        max_preconditioner_dim=2048,
        precondition_frequency=100,
        start_preconditioning_step=250,
        use_decoupled_weight_decay=False,
        grafting_config=AdamGraftingConfig(
            beta2=0.999,
            epsilon=1e-12,
        ),
    )
    
    print("*"*100)
    torch.set_float32_matmul_precision('high')
    
    count_parameters_layerwise(model)

    train_model(model, train_loader, optimizer, DEVICE, EPOCHS)

    torch.save(model.state_dict(), 'trained_model.pth')

if __name__ == "__main__":
    main()