import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from datetime import datetime

# Import demo model and utils (relative imports for package execution)
from .models.demo_model import DemoModel
from .pipeline.qc import quality_control
from .pipeline.alignment import align_sequence
from .pipeline.mutation import identify_mutations
from .pipeline.prioritization import prioritize_mutations

# Load configuration
from .config import load_config
config = load_config()

# Synthetic DNA generator
def random_dna(length: int) -> str:
    return ''.join(np.random.choice(list('ACGT'), size=length))

def generate_synthetic_dataset(num_samples: int = 200):
    min_len = config.get("min_seq_len", 50)
    max_len = config.get("max_seq_len", 5000)
    sequences = []
    labels = []
    for _ in range(num_samples):
        seq_len = np.random.randint(min_len, max_len + 1)
        seq = random_dna(seq_len)
        sequences.append(seq)
        # Random label index
        labels.append(np.random.randint(0, config.get("num_classes", 4)))
    return sequences, labels

def collate_batch(seqs, labels):
    # Convert to tensors compatible with demo model forward
    batch_mut_pos = []
    batch_mut_type = []
    for seq in seqs:
        # QC (should always pass for synthetic data)
        clean_seq = quality_control(seq)
        # Alignment to reference (use demo alignment)
        align_stats = align_sequence(clean_seq)
        # Load reference
        ref_path = Path(__file__).resolve().parent / "data" / "references.fasta"
        with open(ref_path, "r") as f:
            ref_seq = ''.join([l.strip() for l in f if not l.startswith('>')])
        pos_tensor, type_tensor = identify_mutations(clean_seq, ref_seq)
        batch_mut_pos.append(pos_tensor)
        batch_mut_type.append(type_tensor)
    # Pad to max mutations in batch (simple list of tensors, model handles empty)
    return batch_mut_pos, batch_mut_type, torch.tensor(labels, dtype=torch.long)

def train():
    device = torch.device('cpu')
    model = DemoModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Prepare synthetic data
    sequences, labels = generate_synthetic_dataset(num_samples=200)
    # Simple train/val split (80/20)
    split_idx = int(0.8 * len(sequences))
    train_seqs, val_seqs = sequences[:split_idx], sequences[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]

    epochs = config.get("train_epochs", 5)
    batch_size = config.get("batch_size", 16)

    # For logging
    log_path = Path(__file__).resolve().parents[1] / "outputs" / "train_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log_file:
        log_file.write('epoch,train_loss,train_acc,val_loss,val_acc\n')

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        correct = 0
        total = 0
        # Simple mini‑batch loop
        for i in range(0, len(train_seqs), batch_size):
            batch_seqs = train_seqs[i:i+batch_size]
            batch_labels = train_labels[i:i+batch_size]
            mut_pos, mut_type, lbl_tensor = collate_batch(batch_seqs, batch_labels)
            optimizer.zero_grad()
            # Forward pass per sample (demo model expects individual seq string)
            probs = []
            for seq_str, mp, mt in zip(batch_seqs, mut_pos, mut_type):
                prob = model(seq_str, mp.unsqueeze(0), mt.unsqueeze(0))
                probs.append(prob)
            probs = torch.cat(probs, dim=0)
            loss = criterion(probs, lbl_tensor)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            preds = probs.argmax(dim=1)
            correct += (preds == lbl_tensor).sum().item()
            total += lbl_tensor.size(0)
        train_loss = np.mean(epoch_losses)
        train_acc = correct / total

        # Validation
        model.eval()
        val_losses = []
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for i in range(0, len(val_seqs), batch_size):
                batch_seqs = val_seqs[i:i+batch_size]
                batch_labels = val_labels[i:i+batch_size]
                mut_pos, mut_type, lbl_tensor = collate_batch(batch_seqs, batch_labels)
                probs = []
                for seq_str, mp, mt in zip(batch_seqs, mut_pos, mut_type):
                    prob = model(seq_str, mp.unsqueeze(0), mt.unsqueeze(0))
                    probs.append(prob)
                probs = torch.cat(probs, dim=0)
                loss = criterion(probs, lbl_tensor)
                val_losses.append(loss.item())
                preds = probs.argmax(dim=1)
                val_correct += (preds == lbl_tensor).sum().item()
                val_total += lbl_tensor.size(0)
        val_loss = np.mean(val_losses)
        val_acc = val_correct / val_total
        # Log
        with open(log_path, "a") as log_file:
            log_file.write(f"{epoch},{train_loss:.4f},{train_acc:.4f},{val_loss:.4f},{val_acc:.4f}\n")
        print(f"Epoch {epoch}/{epochs} - Train loss {train_loss:.4f}, acc {train_acc:.4f} - Val loss {val_loss:.4f}, acc {val_acc:.4f}")

    # Save checkpoint
    ckpt_path = Path(__file__).resolve().parents[1] / "outputs" / "model_demo.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "epoch": epochs,
    }, ckpt_path)
    print(f"Training complete. Model saved to {ckpt_path}")

if __name__ == "__main__":
    train()
