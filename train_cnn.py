from helper_lib.model import get_model
from helper_lib.trainer import train_model

if __name__ == "__main__":
    model = get_model("AssignmentCNN", num_classes=10)
    train_model(model, epochs=12, lr=1e-3, batch_size=64)
