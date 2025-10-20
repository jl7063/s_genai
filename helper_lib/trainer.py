import torch
import torch.nn as nn
import torch.optim as optim
from .data_loader import get_data_loader

def train_model(model, epochs=8, lr=1e-3, device="cpu", out_dir="./artifacts"):
    import os
    os.makedirs(out_dir, exist_ok=True)

    train_loader = get_data_loader(train=True)
    val_loader   = get_data_loader(train=False)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    best_acc = 0.0
    for ep in range(1, epochs+1):
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
        tr_loss = tot_loss/total
        tr_acc = correct/total

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