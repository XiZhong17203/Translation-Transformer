import preprocessor
import torch
from torch.utils.data import DataLoader

class MyDataset:
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def collate_fn(self, batch):
        src = [item['src'] for item in batch]
        tgt_in = [item['tgt_in'] for item in batch]
        tgt_out = [item['tgt_out'] for item in batch]
        src_max = max(len(s) for s in src)
        tgt_max = max(len(t) for t in tgt_in)
        
        src_padded = [s + [0] * (src_max - len(s)) for s in src]
        tgt_in_padded = [t + [0] * (tgt_max - len(t)) for t in tgt_in]
        tgt_out_padded = [t + [0] * (tgt_max - len(t)) for t in tgt_out]
        
        return (
            torch.tensor(src_padded, dtype=torch.long),
            torch.tensor(tgt_in_padded, dtype=torch.long),
            torch.tensor(tgt_out_padded, dtype=torch.long)
        )

def load_dataset(text, rebuild=False):
    data = preprocessor.preprocess_text(text, rebuild)
    return MyDataset(data)