import os
import torch
import torch.nn as nn
import torch.optim as optim
from .data_loader import get_data_loader  # 确保 get_data_loader 内部用了 Resize(64)

def train_model(
    model,
    epochs: int = 12,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = None,
    out_dir: str = "./artifacts",
    batch_size: int = 64,
):

    os.makedirs(out_dir, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader = get_data_loader(train=True, batch_size=batch_size)
    val_loader   = get_data_loader(train=False, batch_size=batch_size)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_acc = 0.0
    for ep in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        total, correct, tot_loss = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            tot_loss += loss.item() * y.size(0)
            total += y.size(0)
            correct += (logits.argmax(1) == y).sum().item()

        tr_loss = tot_loss / total
        tr_acc = correct / total

        # ---- Eval ----
        model.eval()
        v_tot, v_hit = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                v_tot += y.size(0)
                v_hit += (pred == y).sum().item()
        val_acc = v_hit / v_tot

        print(f"Epoch {ep}/{epochs}  loss={tr_loss:.4f}  train_acc={tr_acc:.3f}  val_acc={val_acc:.3f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"{out_dir}/model.pt")

    labels = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
    with open(f"{out_dir}/labels.txt", "w") as f:
        f.write("\n".join(labels))

    print(f"Best val_acc={best_acc:.3f}. Weights saved to {out_dir}/model.pt")
