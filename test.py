import os
import yaml
import json
import math
import torch
from model import Transformer
from train import model_load
from collections import Counter
from dataset import load_dataset
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
config = yaml.load(open(os.path.join(BASE_DIR, 'config.yaml'), 'r'), Loader=yaml.FullLoader)

word2int_cn = json.load(open(os.path.join(BASE_DIR, 'cmn-eng-simple', 'word2int_cn.json'), 'r', encoding='utf-8'))
word2int_en = json.load(open(os.path.join(BASE_DIR, 'cmn-eng-simple', 'word2int_en.json'), 'r', encoding='utf-8'))
int2word_cn = json.load(open(os.path.join(BASE_DIR, 'cmn-eng-simple', 'int2word_cn.json'), 'r', encoding='utf-8'))

def translate(model, src_sentence, src_vocab, tgt_vocab, int2word, device, test_mode=False):
    model.eval()
    if test_mode:
        # [batch_size, seq_len]
        src_indices = src_sentence
        src_tensor = src_indices.to(device)
        
        tgt_indices = torch.full((src_tensor.size(0), 1), tgt_vocab['<BOS>'], dtype=torch.long).to(device)
        tgt_tensor = tgt_indices.to(device)
    else:
        # [1, seq_len]
        src_sentence = ['<BOS>'] + src_sentence.split() + ['<EOS>']
        src_indices = [src_vocab.get(token, src_vocab.get('<UNK>')) for token in src_sentence]
        src_tensor = torch.tensor(src_indices).unsqueeze(0).to(device)
        
        tgt_indices = [tgt_vocab['<BOS>']]
        tgt_tensor = torch.tensor(tgt_indices).unsqueeze(0).to(device)

    finished = torch.zeros(src_tensor.size(0), dtype=torch.bool).to(device)
    for _ in range(config['max_len']-1):
        src_mask = (src_tensor != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt_tensor != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt_mask,(torch.tril(torch.ones(tgt_tensor.size(1), tgt_tensor.size(1), device=device)).bool()).unsqueeze(0).unsqueeze(1))
        output, _ = model(src_tensor, tgt_tensor, src_mask, tgt_mask)
        
        next_tokens = output.argmax(dim=-1)[:, -1]
        next_tokens = next_tokens.masked_fill(finished, 0)
        finished = finished | (next_tokens == tgt_vocab['<EOS>'])
        if finished.all():
            break
        tgt_tensor = torch.cat([tgt_tensor, next_tokens.unsqueeze(1)], dim=1)
    if test_mode:
        return tgt_tensor[:, 1:].tolist()
    return ' '.join([int2word[str(idx)] for idx in tgt_tensor[0, 1:].tolist()])


def get_ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def BLEU_n(translation, ground_truth, ngram=4):
    translation_tokens = translation
    ground_truth_tokens = ground_truth
    precision = []
    for n in range(1, ngram + 1):
        total_match = 0
        total_pred = 0
        for tr, gt in zip(translation_tokens, ground_truth_tokens):
            pred_ngrams = Counter(get_ngrams(tr, n))
            gt_ngrams = Counter(get_ngrams(gt, n))
            overlap = pred_ngrams & gt_ngrams
            total_match += sum(overlap.values())
            total_pred += sum(pred_ngrams.values())
        precision.append(total_match / (total_pred + 1e-8))

    if min(precision) == 0:
        return 0.0
    
    mean = math.exp(sum(math.log(p) for p in precision) / ngram)
    pred_len = sum(len(tr) for tr in translation_tokens)
    gt_len = sum(len(gt) for gt in ground_truth_tokens)
    BP = math.exp(1 - gt_len / pred_len) if pred_len < gt_len else 1
    return BP * mean

def test(model, dataloader, device):
    model.eval()
    ngram = 1
    bleu = 0
    translation_sentences = []
    ground_truth_sentences = []
    
    with torch.no_grad():
        for src, _, tgt_out in dataloader:
            src = src.to(device)
            translation = translate(model, src, word2int_en, word2int_cn, int2word_cn, device, test_mode=True)
            
            for tr, gt in zip(translation, tgt_out):
                tr = [x for x in tr if x!=0]
                gt = [x.item() for x in gt if x!=0 and x!=word2int_cn['<EOS>']]
                translation_sentences.append(tr)
                ground_truth_sentences.append(gt)
    for ngram in range(1, 5):
        bleu = BLEU_n(translation_sentences, ground_truth_sentences, ngram)
        print(f'BLEU-{ngram} Score: {bleu:.4f}')
    

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Transformer().to(device)
    model_load(model, os.path.join(BASE_DIR, 'model_path', 'best_model_infoNCE.pth'))
    
    test_sentence = "can you wake me up at 7:00 tomorrow ?"
    test_ground_truth = "你 明天 早上 七点 可不可以 叫 我 起床 。"
    translation = translate(model, test_sentence, word2int_en, word2int_cn, int2word_cn, device)
    print(f'Source: {test_sentence}')
    print(f'Predict: {translation}')
    print(f'Ground Truth: {test_ground_truth}')
    test_dataset = load_dataset('test')
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=test_dataset.collate_fn)
    test(model, test_loader, device)
    