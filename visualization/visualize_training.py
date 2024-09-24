# visualization/visualize_training.py

import torch
import matplotlib.pyplot as plt

def visualize_training():
    # Load training losses
    train_losses = torch.load('visualization/train_losses.pt')

    # Plot Loss Curve
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('visualization/training_loss_curve.png')
    plt.close()
    print("Training loss curve saved to visualization/training_loss_curve.png")

if __name__ == '__main__':
    visualize_training()
