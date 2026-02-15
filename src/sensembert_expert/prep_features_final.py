"""Created by Noas Shaalan."""

import json
import os
import torch
import numpy as np
import re
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = '/home/finisher-pc/Documents/NLU_final'
SENSEMBERT_FILE = os.path.join(BASE_DIR, 'sensembert/sensembert_EN_supervised.txt')
DATA_DIR = os.path.join(BASE_DIR, 'ambistory-main')
DEV_FILE = os.path.join(DATA_DIR, 'dev.json')
TRAIN_FILE = os.path.join(DATA_DIR, 'train.json')
TEST_FILE = os.path.join(DATA_DIR, 'test.json')
OUTPUT_FILE = os.path.join(SCRIPT_DIR, 'sensembert_context_features.pt')

lemmatizer = WordNetLemmatizer()

def load_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def get_required_sense_keys(datasets):
    """
    Fucntion extracts the word sense keys from wordnet for each
    homonym in the dataset
    """
    required_keys = set()
    for ds in datasets:
        for entry in ds.values():
            homonym = entry.get('homonym', '').lower()
            if not homonym:
                continue
            for synset in wn.synsets(homonym):
                for lemma in synset.lemmas():
                    required_keys.add(lemma.key())
    return required_keys

def load_sensembert_vectors(filepath, required_keys):
    """
    Function loads the vectors from sensembert file using
    the required word sense keys from our homonyms
    """
    vectors = {} 
    with open(filepath, 'r', encoding='utf-8') as f:
        header = f.readline()
        for line in tqdm(f):
            parts = line.split(' ')
            key = parts[0]
            if key in required_keys:
                try:
                    vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                    vectors[key] = vec
                except ValueError:
                    continue
                    
    print(f"Loaded {len(vectors)} vectors from SensEmBERT")
    return vectors

def find_matching_sense_vector(homonym, judged_meaning, vectors):
    """
    Function finds the synset by matching its definition and return the existing sense vector
    """
    meaning_norm = judged_meaning.lower().strip()
    synsets = wn.synsets(homonym.lower())
    if not synsets:
        synsets = wn.synsets(lemmatizer.lemmatize(homonym.lower()))
        
    for synset in synsets:
        if synset.definition().lower().strip() == meaning_norm:
            for lemma in synset.lemmas():
                if lemma.key() in vectors:
                    return vectors[lemma.key()]
                    
    return None

def get_bert_embedding(text, target_word, tokenizer, model, device):
    """Function Uses same BERT model as SensEmBERT to embed the context from dataset"""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(device)
    input_ids = inputs['input_ids'][0]
    
    for word in [target_word, target_word.lower()]:
        target_ids = tokenizer.encode(word, add_special_tokens=False)
        if not target_ids: 
            continue
        
        matches = (input_ids == target_ids[0]).nonzero(as_tuple=True)[0]
        if len(matches) > 0:
            token_idx = matches[0].item()
            with torch.no_grad():
                outputs = model(**inputs)
                return outputs.last_hidden_state[0, token_idx, :].cpu().numpy()
    return None

def create_features(u, v):
    """Constructures features (concat, product, abs difference) use to train MLP"""
    # Double (BERT-Large 1024) dimensons to match SensEmBERT (2048)
    if len(u) == 1024:
        u = np.concatenate([u, u])
        
    concat = np.concatenate([u, v])
    product = u * v
    diff = np.abs(u - v)
    return np.concatenate([concat, product, diff])

def extract_features(dataset, vectors, tokenizer, model, device, explode_choices=False):
    """Extracts features from the dataset for training MLP"""
    features, labels = [], []
    model.eval()
    
    for entry in tqdm(dataset.values()):
        if 'sentence' not in entry or 'homonym' not in entry: 
            continue
            
        context = f"{entry.get('precontext', '')} {entry['sentence']} {entry.get('ending', '')}".strip()
        ratings = [float(c) for c in entry['choices']] if (explode_choices and 'choices' in entry) else [float(entry.get('average', 0.0))]
        
        target_vec = find_matching_sense_vector(entry['homonym'], entry.get('judged_meaning', ''), vectors)
        bert_emb = get_bert_embedding(context, entry['homonym'], tokenizer, model, device)
        
        if target_vec is not None and bert_emb is not None:
            rich_vec = create_features(bert_emb, target_vec)
        else:
            rich_vec = np.zeros(8192, dtype=np.float32)
            
        for r in ratings:
            features.append(rich_vec)
            labels.append(r)
        
    return np.array(features), np.array(labels)

def main():
    print("Loading datasets...")
    dev_data = load_data(DEV_FILE)
    train_data = load_data(TRAIN_FILE)
    test_data = load_data(TEST_FILE)
    
    # get wordnet sense keys from datasets
    required_keys = get_required_sense_keys([dev_data, train_data, test_data])
    
    # Load SensEmBERT
    vectors = load_sensembert_vectors(SENSEMBERT_FILE, required_keys)
    
    # Setup BERT
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
    model = BertModel.from_pretrained('bert-large-cased').to(device)
    
    print("Extracting features from Train Set")
    train_feats, train_labels = extract_features(train_data, vectors, tokenizer, model, device, explode_choices=True)
    
    print("Extracting features from Dev Set")
    dev_feats, dev_labels = extract_features(dev_data, vectors, tokenizer, model, device, explode_choices=False)

    print("Extracting features from Test Set")
    test_feats, test_labels = extract_features(test_data, vectors, tokenizer, model, device, explode_choices=False)
    
    torch.save({
        'train_features': torch.tensor(train_feats, dtype=torch.float32),
        'train_labels': torch.tensor(train_labels, dtype=torch.float32),
        'dev_features': torch.tensor(dev_feats, dtype=torch.float32),
        'dev_labels': torch.tensor(dev_labels, dtype=torch.float32),
        'test_features': torch.tensor(test_feats, dtype=torch.float32),
        'test_labels': torch.tensor(test_labels, dtype=torch.float32)
    }, OUTPUT_FILE)
    
if __name__ == "__main__":
    main()
