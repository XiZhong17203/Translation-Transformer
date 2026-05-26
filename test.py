import os
import torch
from model import Transformer
from train import model_load
from dataset import load_dataset, MyDataset
from torch.utils.data import DataLoader
from trans_method import *

def test(model, dataloader, device):
    model.eval()
    ngram = 1
    bleu = 0
    translation_sentences = []
    ground_truth_sentences = []
    
    with torch.no_grad():
        for src, _, tgt_out in dataloader:
            src = src.to(device)
            translation = translate(model, src, word2int_en, word2int_cn, int2word_cn, device, test_mode=True, use_beam=True, beam_size=8, alpha=0.6)
            
            for tr, gt in zip(translation, tgt_out):
                tr = [x for x in tr if x not in (config['pad_token_id'], config['bos_token_id'], config['eos_token_id'])]
                gt = [x.item() for x in gt if x.item() not in (config['pad_token_id'], config['bos_token_id'], config['eos_token_id'])]
                translation_sentences.append(tr)
                ground_truth_sentences.append(gt)
    for ngram in range(1, 5):
        bleu = BLEU_n(translation_sentences, ground_truth_sentences, ngram)
        print(f'BLEU-{ngram} Score: {bleu:.4f}')
    

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Transformer().to(device)
    # model_load(model, os.path.join(BASE_DIR, 'model_path', 'model_test.pth'))
    model_load(model, os.path.join(BASE_DIR, 'model_path', 'best_model_infoNCE.pth'))
    test_sentence = "you have changed so much that i can hardly recognize you ."
    test_ground_truth = "你 变 了 那么 多 ， 以至于 我 几乎 认不出 你 了 。"
    translation = translate(model, test_sentence, word2int_en, word2int_cn, int2word_cn, device, use_beam=True, beam_size=8, alpha=0.6)
    print(f'Source: {test_sentence}')
    print(f'Predict: {translation}')
    print(f'Ground Truth: {test_ground_truth}')
    
    # train_dataset = load_dataset('train')
    # slice_traindataset = MyDataset(train_dataset.data[:3000])
    # test_loader = DataLoader(slice_traindataset, batch_size=config['batch_size'], shuffle=True, collate_fn=slice_traindataset.collate_fn)
    
    test_dataset = load_dataset('test')
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=test_dataset.collate_fn)
    test(model, test_loader, device)
    