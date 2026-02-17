'''
Created by Noas Shaalan,

This script is used to prepare the training features for the ARES expert model,

BEFORE USING THIS SCRIPT
1. You shouhld have the pre-trained vectors installed from sapeinza NLP lab: https://nlp.uniroma1.it/sensembert/

'''

import os
import sys
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from data_utils import load_dataset, expand_annotator_samples
from expert_dataset import build_context

BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))  # repo root
ARES_FILE = os.path.join(BASE_DIR, 'ares/ares_embedding/ares_bert_large.txt')
DATA_DIR = os.path.join(BASE_DIR, 'ambistory-main')
DEV_FILE = os.path.join(DATA_DIR, 'dev.json')
TRAIN_FILE = os.path.join(DATA_DIR, 'train.json')
TEST_FILE = os.path.join(DATA_DIR, 'test.json')
OUTPUT_FILE = os.path.join(SCRIPT_DIR, 'ares_context_features.pt')

lemmatizer = WordNetLemmatizer()

def get_required_sense_keys(records):
    """Identify judged meaning sense key from wordnet to extract.
    input: records - list of records from the dataset
    output: set of sense keys
    """
    required_keys = set()
    for entry in records:
        homonym = entry.get('homonym', '').lower()
        if not homonym:
            continue
        for synset in wn.synsets(homonym):
            for lemma in synset.lemmas():
                required_keys.add(lemma.key())
    return required_keys

def load_ares_vectors(filepath, required_keys):
    """Load only necessary ARES vectors into memory for features extraction.
    input: filepath - path to the ARES vectors file
    output: dictionary of sense keys to vectors
    """
    print(f"Loading ARES subset for {len(required_keys)} potential sense keys...")
    vectors = {} 
    with open(filepath, 'r', encoding='utf-8') as f:
        f.readline()  # Skip header
        for line in f:
            parts = line.split(' ')
            key = parts[0]
            if key in required_keys:
                try:
                    vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                    vectors[key] = vec
                except ValueError:
                    continue
    print(f"Loaded {len(vectors)} vectors from ARES.")
    return vectors

def find_matching_sense_vector(homonym, judged_meaning, ares_vectors):
    """Find the sense key (from wordnet) matching the judged definition string in ares vectors
    input: homonym - the word to find the sense key for
    output: sense key
    """
    meaning_norm = judged_meaning.lower().strip()
    synsets = wn.synsets(homonym.lower())
    if not synsets:
        synsets = wn.synsets(lemmatizer.lemmatize(homonym.lower()))
        
    for synset in synsets:
        if synset.definition().lower().strip() == meaning_norm:
            for lemma in synset.lemmas():
                if lemma.key() in ares_vectors:
                    return ares_vectors[lemma.key()]
    return None

def get_bert_embedding(text, target_word, tokenizer, model, device):
    """Extract BERT embedding for the target word in context.
    input: text - context sentence
    output: BERT embedding
    """
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(device)
    input_ids = inputs['input_ids'][0]
    
    for word in [target_word, target_word.lower()]:
        target_ids = tokenizer.encode(word, add_special_tokens=False)
        if not target_ids: continue
        
        matches = (input_ids == target_ids[0]).nonzero(as_tuple=True)[0]
        if len(matches) > 0:
            token_idx = matches[0].item()
            with torch.no_grad():
                outputs = model(**inputs)
                return outputs.last_hidden_state[0, token_idx, :].cpu().numpy()
    return None

def create_features(u, v):
    """Combine context and sense vectors into feature vector for training.
    input: u - BERT embedding
    output: feature vector
    """
    if len(u) == 1024:
        u = np.concatenate([u, u])  # 1024 to 2048 to match ARES
        
    concat = np.concatenate([u, v])
    product = u * v
    diff = np.abs(u - v)
    return np.concatenate([concat, product, diff])

def extract_features(records, ares_vectors, tokenizer, model, device):
    """Generate features for all samples in our dataset.
    input: records - list of records from the dataset
    output: features - list of feature vectors
    labels - list of labels
    """
    features, labels = [], []
    model.eval()
    
    for entry in records:
        if 'sentence' not in entry or 'homonym' not in entry: continue
            
        context = build_context(entry)
        
        target_vec = find_matching_sense_vector(entry['homonym'], entry.get('judged_meaning', ''), ares_vectors)
        bert_emb = get_bert_embedding(context, entry['homonym'], tokenizer, model, device)
        
        if target_vec is not None and bert_emb is not None:
            rich_vec = create_features(bert_emb, target_vec)
        else:
            rich_vec = np.zeros(8192, dtype=np.float32)
            
        features.append(rich_vec)
        labels.append(float(entry.get('average', 0.0)))
        
    return np.array(features), np.array(labels)

def main():
    dev_records = load_dataset(DEV_FILE)
    train_records = load_dataset(TRAIN_FILE)
    test_records = load_dataset(TEST_FILE)
    
    train_records_expanded = expand_annotator_samples(train_records, label_key='average', choices_key='choices')
    
    required_keys = get_required_sense_keys(dev_records + train_records + test_records)
    print(f"Identified {len(required_keys)} target sense keys.")
    
    ares_vectors = load_ares_vectors(ARES_FILE, required_keys)
    
    print("Loading BERT-Large-Cased...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
    model = BertModel.from_pretrained('bert-large-cased').to(device)
    
    print(f"Extracting [Train] features ({len(train_records_expanded)} samples)...")
    train_feats, train_labels = extract_features(train_records_expanded, ares_vectors, tokenizer, model, device)
    
    print(f"Extracting [Dev] features ({len(dev_records)} samples)...")
    dev_feats, dev_labels = extract_features(dev_records, ares_vectors, tokenizer, model, device)

    print(f"Extracting [Test] features ({len(test_records)} samples)...")
    test_feats, test_labels = extract_features(test_records, ares_vectors, tokenizer, model, device)
    
    print(f"Saving features to {OUTPUT_FILE}...")
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
