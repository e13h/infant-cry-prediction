import gc
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import pickle

from dataset import DonateACryDataset
from network import RNN
from utils import load_checkpoint, train, open_audio_file, plot_waveform, test_a_waveform


CHECKPOINT_DIR = './checkpoints/'


clean_dirs = ['./donateacry-corpus/donateacry_corpus_cleaned_and_updated_data/']
clean_train_data = DonateACryDataset(clean_dirs, train=True, drop_hungry=0.75)
clean_test_data = DonateACryDataset(clean_dirs, train=False)

assert clean_train_data.target_decoding == clean_test_data.target_decoding
assert torch.cuda.is_available()


def run(checkpoint=None, dir=CHECKPOINT_DIR):
    gc.collect()
    batch_size = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # torch.autograd.set_detect_anomaly(True)
    clean_train_loader = DataLoader(clean_train_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    clean_test_loader = DataLoader(clean_test_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    if checkpoint:
        model = load_checkpoint(checkpoint, dir=dir).to(device)
    else:
        model = RNN(hidden_size=80, output_size=5, n_layers=2, batch_size=batch_size, bidirectional=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = torch.nn.CrossEntropyLoss()

    losses = train(clean_train_loader, clean_test_loader, 500, model, optimizer, loss, device, checkpoint_dir=CHECKPOINT_DIR)
    return model, losses

model, metrics = run()
train_losses, test_losses, train_accuracies, test_accuracies, test_preds = metrics

print('num epochs trained', model.num_epochs_trained)
print('micro:', f1_score(clean_test_data.data.target, test_preds[-1], average='micro'))
print('macro:', f1_score(clean_test_data.data.target, test_preds[-1], average='macro'))
print('distribution of preds:', np.unique(test_preds, return_counts=True))

with open('metrics.pkl', 'wb') as f:
    pickle.dump(metrics, f)
