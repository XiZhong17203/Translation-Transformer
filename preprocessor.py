import os
import json
mapping = {
    'train': 'train_processed.json',
    'test': 'test_processed.json',
    'validation': 'validation_processed.json'
}
txt_mapping = {
    'train': 'training.txt',
    'test': 'testing.txt',
    'validation': 'validation.txt'
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

word2id_cn = json.load(open(os.path.join(BASE_DIR, 'cmn-eng-simple', 'word2int_cn.json'), 'r', encoding='utf-8'))
word2id_en = json.load(open(os.path.join(BASE_DIR, 'cmn-eng-simple', 'word2int_en.json'), 'r', encoding='utf-8'))

def preprocess_text(text, rebuild=False):
    save_path = mapping[text]
    txt_path = txt_mapping[text]
    
    if os.path.exists(os.path.join(BASE_DIR, 'cache', save_path)) and not rebuild:
        with open(os.path.join(BASE_DIR, 'cache', save_path), 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    data = []
    with open(os.path.join(BASE_DIR, 'cmn-eng-simple', txt_path), 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            en, cn = line.split('\t')
            src_tokens = en.split()
            tgt_tokens = cn.split()
            
            src = ['<BOS>'] + src_tokens + ['<EOS>']
            tgt_in = ['<BOS>'] + tgt_tokens
            tgt_out = tgt_tokens + ['<EOS>']
            
            src_ids = [word2id_en.get(token, word2id_en.get('<UNK>')) for token in src]
            tgt_in_ids = [word2id_cn.get(token, word2id_cn.get('<UNK>')) for token in tgt_in]
            tgt_out_ids = [word2id_cn.get(token, word2id_cn.get('<UNK>')) for token in tgt_out]

            data.append({
                'src': src_ids,
                'tgt_in': tgt_in_ids,
                'tgt_out': tgt_out_ids
            })

    with open(os.path.join(BASE_DIR, 'cache', save_path), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)

    return data