import os
import yaml
import json
import math
import torch
from collections import Counter
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
config = yaml.load(open(os.path.join(BASE_DIR, 'config.yaml'), 'r'), Loader=yaml.FullLoader)

word2int_cn = json.load(open(os.path.join(BASE_DIR, 'cmn-eng-simple', 'word2int_cn.json'), 'r', encoding='utf-8'))
word2int_en = json.load(open(os.path.join(BASE_DIR, 'cmn-eng-simple', 'word2int_en.json'), 'r', encoding='utf-8'))
int2word_cn = json.load(open(os.path.join(BASE_DIR, 'cmn-eng-simple', 'int2word_cn.json'), 'r', encoding='utf-8'))

def make_src_mask(src_tensor):
    return (src_tensor != config['pad_token_id']).unsqueeze(1).unsqueeze(2)

def make_tgt_mask(tgt_tensor):
    tgt_pad = (tgt_tensor != config['pad_token_id']).unsqueeze(1).unsqueeze(2)
    tgt_sub = torch.tril(
        torch.ones(tgt_tensor.size(1), tgt_tensor.size(1), device=tgt_tensor.device)
    ).bool().unsqueeze(0).unsqueeze(1)
    return (tgt_pad, tgt_sub)

def length_penalty(length, alpha=0.6):
    return ((5 + length) / 6) ** alpha

def beam_search_batch(
    model, 
    src_tensor,
    beam_size=5,
    max_len=None,
    alpha=0.6
    ):
    if max_len is None:
        max_len = config['max_len']
    bos_id = config['bos_token_id']
    eos_id = config['eos_token_id']

    batch_size = src_tensor.size(0)
    device = src_tensor.device
    vocab_size = config['cn_vocab_size']

    src_mask = make_src_mask(src_tensor)
    beam_scores = torch.full((batch_size, beam_size), float('-inf'), device=device)
    beam_scores[:, 0] = 0.0
    beam_tokens = torch.full((batch_size, beam_size, 1), bos_id, dtype=torch.long, device=device)
    finished = torch.zeros((batch_size, beam_size), dtype=torch.bool, device=device)

    for _ in range(max_len - 1):
        flat_tokens = beam_tokens.view(batch_size * beam_size, -1)
        src_expanded = src_tensor.unsqueeze(1).repeat(1, beam_size, 1).view(batch_size * beam_size, -1)
        src_mask_expanded = src_mask.unsqueeze(1).repeat(1, beam_size, 1, 1, 1).view(batch_size * beam_size, 1, 1, -1)

        tgt_mask = make_tgt_mask(flat_tokens)
        output, _ = model(src_expanded, flat_tokens, src_mask_expanded, tgt_mask)
        log_probs = F.log_softmax(output[:, -1, :], dim=-1)

        finished_flat = finished.view(-1)
        if finished_flat.any():
            log_probs[finished_flat] = float('-inf')
            log_probs[finished_flat, eos_id] = 0.0

        scores = beam_scores.view(-1, 1) + log_probs
        scores = scores.view(batch_size, beam_size * vocab_size)

        topk_scores, topk_indices = torch.topk(scores, beam_size, dim=-1)
        next_beam = topk_indices // vocab_size
        next_tokens = topk_indices % vocab_size

        gathered = torch.gather(
            beam_tokens,
            1,
            next_beam.unsqueeze(-1).expand(-1, -1, beam_tokens.size(-1))
        )
        beam_tokens = torch.cat([gathered, next_tokens.unsqueeze(-1)], dim=-1)
        beam_scores = topk_scores
        finished = torch.gather(finished, 1, next_beam) | (next_tokens == eos_id)

        if finished.all():
            break

    seq_len = beam_tokens.size(-1)
    eos_mask = (beam_tokens == eos_id)
    lengths = torch.full((batch_size, beam_size), seq_len, device=device)
    if eos_mask.any():
        idx = torch.arange(seq_len, device=device).view(1, 1, -1).expand_as(beam_tokens)
        first_eos = torch.where(eos_mask, idx, torch.full_like(idx, seq_len))
        lengths = first_eos.min(dim=-1).values

    final_scores = beam_scores / length_penalty(lengths.float(), alpha)
    best = final_scores.argmax(dim=-1)
    best_tokens = beam_tokens[torch.arange(batch_size, device=device), best]
    return best_tokens

def translate(
    model,
    src_sentence,
    src_vocab,
    tgt_vocab,
    int2word,
    device,
    test_mode=False,
    use_beam=False,
    beam_size=5,
    alpha=0.6,
):
    model.eval()
    if test_mode:
        # [batch_size, seq_len]
        src_indices = src_sentence
        src_tensor = src_indices.to(device)

        if use_beam:
            tokens = beam_search_batch(
                model,
                src_tensor,
                beam_size=beam_size,
                alpha=alpha,
            )
            results = []
            for seq in tokens.tolist():
                results.append(seq[1:])
            return results

        tgt_indices = torch.full((src_tensor.size(0), 1), tgt_vocab['<BOS>'], dtype=torch.long).to(device)
        tgt_tensor = tgt_indices.to(device)
    else:
        # [1, seq_len]
        src_sentence = ['<BOS>'] + src_sentence.split() + ['<EOS>']
        src_indices = [src_vocab.get(token, src_vocab.get('<UNK>')) for token in src_sentence]
        src_tensor = torch.tensor(src_indices).unsqueeze(0).to(device)

        if use_beam:
            tokens = beam_search_batch(
                model,
                src_tensor,
                beam_size=beam_size,
                alpha=alpha,
            )[0].tolist()
            tokens = [t for t in tokens[1:] if t != tgt_vocab['<EOS>']]
            return ' '.join([int2word[str(idx)] for idx in tokens])

        tgt_indices = [tgt_vocab['<BOS>']]
        tgt_tensor = torch.tensor(tgt_indices).unsqueeze(0).to(device)

    finished = torch.zeros(src_tensor.size(0), dtype=torch.bool).to(device)
    for _ in range(config['max_len']-1):
        src_mask = make_src_mask(src_tensor)
        tgt_mask = make_tgt_mask(tgt_tensor)
        output, _ = model(src_tensor, tgt_tensor, src_mask, tgt_mask)
        
        next_tokens = output.argmax(dim=-1)[:, -1]
        next_tokens = next_tokens.masked_fill(finished, 0)
        finished = finished | (next_tokens == tgt_vocab['<EOS>'])
        if finished.all():
            break
        tgt_tensor = torch.cat([tgt_tensor, next_tokens.unsqueeze(1)], dim=1)
    if test_mode:
        return tgt_tensor[:, 1:].tolist()
    tokens = [idx for idx in tgt_tensor[0, 1:].tolist() if idx != tgt_vocab['<EOS>']]
    return ' '.join([int2word[str(idx)] for idx in tokens])

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