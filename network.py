import torch
import torchaudio

class RNN(torch.nn.Module):
    def __init__(self, hidden_size, output_size, n_layers, batch_size=1, bidirectional=True):
        super(RNN, self).__init__()
        self.__dict__.update(locals())

        self.embedding = torchaudio.transforms.MFCC(sample_rate=8000, melkwargs={'pad': 100})
        self.gru = torch.nn.GRU(self.hidden_size, self.hidden_size, self.n_layers, bidirectional=bidirectional)
        gru_output_size = self.hidden_size * 2 if bidirectional else self.hidden_size
        self.out = torch.nn.Linear(gru_output_size, self.output_size)
        self.num_epochs_trained = 0
    
    def forward(self, waveform, hidden=None):
        windowed_waveform = torch.cat([self.embedding(t).reshape(1, 1, -1) for t in self.window_generator(waveform)])
        windowed_waveform.requires_grad_(True)
        output, hidden = self.gru(windowed_waveform, hidden)
        # output = torch.nn.functional.relu(self.out(output))
        output = self.out(output)
        return output, hidden
    
    def init_hidden(self, device):
        return torch.zeros(self.n_layers * 2, 1, self.hidden_size, device=device)
    
    @staticmethod
    def window_generator(waveform: torch.Tensor, sample_rate=8000, length=25, offset=10) -> torch.Tensor:
        time = waveform.shape[-1] / sample_rate * 1000  # convert to milliseconds
        for i in range(0, int(time), offset):
            yield waveform.clone().detach()[:, :, i:i+length].requires_grad_(True)
