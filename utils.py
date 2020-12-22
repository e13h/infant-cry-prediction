import os
import subprocess
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchaudio

from network import RNN


GOOGLE_COLAB_CHECKPOINT_DIR = 'drive/MyDrive/Colab Notebooks/infant_cry_checkpoints/'


def save_checkpoint(model, dir, path='checkpoint.pth'):
    dirpath = os.path.dirname(path)
    if dirpath and not os.path.exists(dir + dirpath):
        os.makedirs(dirpath)

    checkpoint = {
        'hidden_size': model.hidden_size,
        'output_size': model.output_size,
        'n_layers': model.n_layers,
        'batch_size': model.batch_size,
        'bidirectional': model.bidirectional,
        'state_dict': model.state_dict(),
        'num_epochs_trained': model.num_epochs_trained
    }
    torch.save(checkpoint, os.path.join(dir, path))


def load_checkpoint(filename, dir=GOOGLE_COLAB_CHECKPOINT_DIR):
    checkpoint = torch.load(os.path.join(dir, filename))
    model = RNN(
        hidden_size=checkpoint['hidden_size'],
        output_size=checkpoint['output_size'],
        n_layers=checkpoint['n_layers'],
        batch_size=checkpoint['batch_size'],
        bidirectional=checkpoint['bidirectional']
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.num_epochs_trained = checkpoint['num_epochs_trained']
    
    return model


def evaluate(model, test_data_loader, criterion, device, bar):
    model.eval()
    bar.set_description('evaluating')
    losses = []
    accuracies = []
    preds = []
    with torch.no_grad():
        for i, (waveform, target) in enumerate(test_data_loader):
            waveform = waveform.to(device)
            hidden = None

            output, hidden = model(waveform, hidden)
            target_ = torch.empty(output.shape[0], dtype=torch.long).fill_(target[0]).to(device)

            losses.append(criterion(output.squeeze(), target_).item())
            accuracies.append(1 if output.mean(axis=0).argmax().item() == target.item() else 0)
            preds.append(output.mean(axis=0).argmax().item())
            bar.update()
    model.train()
    return losses, accuracies, preds


## train a waveform sequence at a time
def train(train_data_loader, test_data_loader, num_epochs, model, optimizer, criterion, device, checkpoint_dir=GOOGLE_COLAB_CHECKPOINT_DIR, save=True):
    bar = tqdm(total=num_epochs*(len(train_data_loader)+len(test_data_loader)), position=0, leave=True, desc='training')
    eval_losses, eval_accuracies, eval_preds = [], [], []
    train_losses, train_accuracies = [[] for x in range(num_epochs)], [[] for x in range(num_epochs)]
    for epoch in range(num_epochs):
        for i, (waveform, target) in enumerate(train_data_loader):
            optimizer.zero_grad()
            hidden = None
            waveform = waveform.to(device)
            
            output, hidden = model(waveform, hidden)
            target_ = torch.empty(output.shape[0], dtype=torch.long).fill_(target[0]).to(device)
            
            loss = criterion(output.squeeze(), target_)
            train_losses[epoch].append(loss.item())
            train_accuracies[epoch].append(1 if output.mean(axis=0).argmax().item() == target.item() else 0)
            loss.backward()
            # if i % 50 == 0:
            bar.update()
            optimizer.step()
        eval_loss, eval_accuracy, eval_pred = evaluate(model, test_data_loader, criterion, device, bar)
        eval_losses.append(eval_loss)
        eval_accuracies.append(eval_accuracy)
        eval_preds.append(eval_pred)
        bar.set_description(f'epoch {epoch+1}/{num_epochs}, loss: {np.array(eval_loss).mean():.3f}')
        model.num_epochs_trained += 1
        if save is True:
            save_checkpoint(model, checkpoint_dir, path=f'checkpoint-{model.num_epochs_trained}_epochs.pth')

    return train_losses, eval_losses, train_accuracies, eval_accuracies, eval_preds


def plot_loss(train_losses=None, test_losses=None):
    if train_losses:
        plt.plot(range(1, len(train_losses)+1), np.array(train_losses).mean(axis=1), label='Train loss')
    if test_losses:
        plt.plot(range(1, len(test_losses)+1), np.array(test_losses).mean(axis=1), label='Test loss')
    plt.xlabel('Time (epochs)')
    plt.ylabel('Loss')
    plt.ylim(0)
    plt.title('Loss Over Time')
    plt.legend()
    plt.show()


def plot_accuracy(train_accuracies=None, test_accuracies=None):
    if train_accuracies:
        plt.plot(range(1, len(train_accuracies)+1), np.array(train_accuracies).mean(axis=1), label='Train accuracy')
    if test_accuracies:
        plt.plot(range(1, len(test_accuracies)+1), np.array(test_accuracies).mean(axis=1), label='Test accuracy')
    plt.xlabel('Time (epochs)')
    plt.ylabel('Accuracy')
    plt.ylim((0, 1.1))
    plt.title('Accuracy Over Time')
    plt.legend()
    plt.show()


def plot_waveform(waveform):
    plt.plot(waveform.t().numpy())
    plt.show()


def convert_file_to_wav(path):
    new_path = '.'.join(path.split('.')[:-1]) + '.wav'
    subprocess.run(['ffmpeg', '-y', '-i', path, new_path, '-hide_banner'])
    return new_path


def resample(waveform, original_rate, new_rate):
    resampler = torchaudio.transforms.Resample(original_rate, new_rate)
    return resampler(waveform), new_rate


def open_audio_file(path, target_rate=8000):
    if path[-4:] != '.wav':
        path = convert_file_to_wav(path)
    waveform, sample_rate = torchaudio.load(path)
    if sample_rate != target_rate:
        return resample(waveform, sample_rate, target_rate)
    return waveform, sample_rate


def test_a_waveform(waveform, model, target_decoding):
    waveform = waveform.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    output_dist, _ = model(waveform.unsqueeze(0))
    output = output_dist.mean(axis=0).argmax().item()
    output_label = target_decoding[output]
    output_dist = output_dist.mean(axis=0).squeeze().cpu().detach().numpy()
    output_dist_mapping = {label: output_dist[i] for i, label in target_decoding.items()}
    return output_label, output_dist_mapping
