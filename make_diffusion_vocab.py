from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFKC
from tokenizers.processors import TemplateProcessing

import os
import re
import pickle
import unicodedata
from tqdm import tqdm
from w3lib.html import remove_tags
from dataclasses import dataclass
# Hugging Face Tokenizer imports
from tokenizers import Tokenizer, normalizers, pre_tokenizers, processors
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer

# --- CONFIGURATION PATHS ---
INPUT_DATA_DIR = 'txt/en'
OUTPUT_TRAIN_DIR = 'txt/train_data.pkl'
OUTPUT_TEST_DIR = 'txt/test_data.pkl'
OUTPUT_VOCAB_PATH = 'txt/diffusion_vocab.json'

# Ensure the directories exist
os.makedirs(INPUT_DATA_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_TRAIN_DIR), exist_ok=True)

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')

def normalize_string(s):
    # Normalize unicode characters
    s = unicode_to_ascii(s)
    # Remove XML/HTML tags
    s = remove_tags(s)
    # Add white space before !.?
    s = re.sub(r'([!.?])', r' \1', s)
    # Strip everything except letters and base punctuation
    s = re.sub(r'[^a-zA-Z.!?]+', r' ', s)
    s = re.sub(r'\s+', r' ', s)
    return s.strip().lower()

def cut_data(cleaned, min_length=4, max_length=30):
    cutted_lines = []
    for line in cleaned:
        length = len(line.split())
        if min_length < length < max_length:
            cutted_lines.append(line)
    return cutted_lines


# --- LOAD AND PREPROCESS ---
sentences = []
print('Processing raw files...')
for fn in tqdm(os.listdir(INPUT_DATA_DIR)):
    if not fn.endswith('.txt'): 
        continue
    with open(os.path.join(INPUT_DATA_DIR, fn), 'r', encoding='utf8') as f:
        raw_data = f.read()
    
    file_sentences = raw_data.strip().split('\n')
    cleaned_sentences = [normalize_string(s) for s in file_sentences]
    cleaned_sentences = cut_data(cleaned_sentences)
    sentences.extend(cleaned_sentences)

# Deduplicate
sentences = list(dict.fromkeys(sentences))
print(f"\nSuccessfully loaded {len(sentences)} unique sentences.")
print("Sample clean text:", sentences[:3])


# 1. Initialize WordLevel Model
tokenizer = Tokenizer(WordLevel(unk_token="<UNK>"))

# 2. Add Normalization & Pre-tokenization Rules
tokenizer.normalizer = normalizers.Sequence([
    normalizers.NFKD(),
    normalizers.Lowercase()
])

tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
    pre_tokenizers.WhitespaceSplit(),
    pre_tokenizers.Punctuation()
])

# 3. Train the Tokenizer
# The special tokens array maps them precisely to indices 0, 1, 2, 3 as in your original SPECIAL_TOKENS dict
trainer = WordLevelTrainer(
    min_frequency=1,
    special_tokens=["<PAD>", "<START>", "<END>", "<UNK>"]
)

print("Training Tokenizer...")
tokenizer.train_from_iterator(sentences, trainer=trainer)

# 4. Attach the Template Post-Processor (Inserts <START> and <END> dynamically)
start_id = tokenizer.token_to_id("<START>")
end_id = tokenizer.token_to_id("<END>")

tokenizer.post_processor = processors.TemplateProcessing(
    single="<START> $A <END>",
    special_tokens=[
        ("<START>", start_id),
        ("<END>", end_id),
    ],
)

# 5. Save the trained model config
tokenizer.save(OUTPUT_VOCAB_PATH)
print(f"Tokenizer saved to: {OUTPUT_VOCAB_PATH}")