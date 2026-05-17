import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import *
from dataset import load_dataset
from trans_method import *

def train(
    model,
    dataloader,
    val_dataloader,
    optimizer, 
    criterion,
    device,
    lr_scheduler=None
    ):
    best_val_loss = 10
    
    for epoch in range(config['num_epochs']):
        model.train()
        total_loss = 0
        for src, tgt_in, tgt_out in dataloader:
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
            optimizer.zero_grad()
            loss = 0
            # [batch_size, 1, 1, seq_len_k]
            src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
            tgt_mask = (tgt_in != 0).unsqueeze(1).unsqueeze(2)
            tgt_mask = (tgt_mask,(torch.tril(torch.ones(tgt_in.size(1), tgt_in.size(1), device=device)).bool()).unsqueeze(0).unsqueeze(1))
            
            output, _ = model(src, tgt_in, src_mask, tgt_mask)
            
            # [batch_size * seq_len, cn_vocab_size], [batch_size * seq_len]
            loss += criterion(output.view(-1, output.size(-1)), tgt_out.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for src, tgt_in, tgt_out in val_dataloader:
                src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
                src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
                tgt_mask = (tgt_in != 0).unsqueeze(1).unsqueeze(2)
                tgt_mask = (tgt_mask,(torch.tril(torch.ones(tgt_in.size(1), tgt_in.size(1), device=device)).bool()).unsqueeze(0).unsqueeze(1))
                output, _ = model(src, tgt_in, src_mask, tgt_mask)
                loss = criterion(output.view(-1, output.size(-1)), tgt_out.view(-1))
                val_loss += loss.item()
        if val_loss/len(val_dataloader) < best_val_loss:
            best_val_loss = val_loss/len(val_dataloader)
            print(f'\t New best model with val loss {best_val_loss:.4f}, saving model...')
            model_save(model, os.path.join(BASE_DIR, 'model_path', 'best_model.pth'))
            print(f'\t Best model saved with val loss {best_val_loss:.4f}')
        if lr_scheduler:
            lr_scheduler.step(val_loss/len(val_dataloader))
        
        print(
            f'Epoch {epoch+1}/{config["num_epochs"]}, Loss: {total_loss/len(dataloader):.4f}, '
            f'Val Loss: {val_loss/len(val_dataloader):.4f}'
        )

    # model_save(model, os.path.join(BASE_DIR, 'model_path', 'model.pth'))
    
def model_save(model, path):
    torch.save(model.state_dict(), path)
    
def model_load(model, path):
    model.load_state_dict(torch.load(path, weights_only=True))

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = load_dataset('train')
    
    val_dataset = load_dataset('validation')
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=train_dataset.collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=val_dataset.collate_fn)
    
    model = Transformer().to(device)
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=0,label_smoothing=0.1)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    train(model, train_dataloader, val_dataloader, optimizer, criterion, device, lr_scheduler)