# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 09:47:54 2020

@author: HQ Xie
utils.py
"""
import os 
import math
import torch
import time
import torch.nn as nn
import numpy as np
from w3lib.html import remove_tags
from nltk.translate.bleu_score import sentence_bleu
from models.mutual_info import sample_batch, mutual_information
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BleuScore():
    def __init__(self, w1, w2, w3, w4):
        self.w1 = w1 # 1-gram weights
        self.w2 = w2 # 2-grams weights
        self.w3 = w3 # 3-grams weights
        self.w4 = w4 # 4-grams weights
    
    def compute_blue_score(self, real, predicted):
        score = []
        for (sent1, sent2) in zip(real, predicted):
            sent1 = remove_tags(sent1).split()
            sent2 = remove_tags(sent2).split()
            score.append(sentence_bleu([sent1], sent2, 
                          weights=(self.w1, self.w2, self.w3, self.w4)))
        return score
            

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        # 将数组全部填充为某一个值
        true_dist.fill_(self.smoothing / (self.size - 2)) 
        # 按照index将input重新排列 
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence) 
        # 第一行加入了<strat> 符号，不需要加入计算
        true_dist[:, self.padding_idx] = 0 #
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)


class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        self._weight_decay = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        weight_decay = self.weight_decay()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
            p['weight_decay'] = weight_decay
        self._rate = rate
        self._weight_decay = weight_decay
        # update weights
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
            
        # if step <= 3000 :
        #     lr = 1e-3
            
        # if step > 3000 and step <=9000:
        #     lr = 1e-4
             
        # if step>9000:
        #     lr = 1e-5
         
        lr = self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
  
        return lr
    

        # return lr
    
    def weight_decay(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
            
        if step <= 3000 :
            weight_decay = 1e-3
            
        if step > 3000 and step <=9000:
            weight_decay = 0.0005
             
        if step>9000:
            weight_decay = 1e-4

        weight_decay =   0
        return weight_decay

            
class SeqtoText:
    def __init__(self, vocb_dictionary, end_idx):
        self.reverse_word_map = dict(zip(vocb_dictionary.values(), vocb_dictionary.keys()))
        self.end_idx = end_idx
        
    def sequence_to_text(self, list_of_indices):
        # Looking up words in dictionary
        words = []
        for idx in list_of_indices:
            if idx == self.end_idx:
                break
            else:
                words.append(self.reverse_word_map.get(idx))
        words = ' '.join(words)
        return(words) 


class Channels():

    def AWGN(self, Tx_sig, n_var):
        Rx_sig = Tx_sig + torch.normal(0, n_var, size=Tx_sig.shape).to(device)
        return Rx_sig

    def Rayleigh(self, Tx_sig, n_var):
        shape = Tx_sig.shape
        H_real = torch.normal(0, math.sqrt(1/2), size=[1]).to(device)
        H_imag = torch.normal(0, math.sqrt(1/2), size=[1]).to(device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)
        Rx_sig = self.AWGN(Tx_sig, n_var)
        # Channel estimation
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)

        return Rx_sig

    def Rician(self, Tx_sig, n_var, K=1):
        shape = Tx_sig.shape
        mean = math.sqrt(K / (K + 1))
        std = math.sqrt(1 / (K + 1))
        H_real = torch.normal(mean, std, size=[1]).to(device)
        H_imag = torch.normal(mean, std, size=[1]).to(device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)
        Rx_sig = self.AWGN(Tx_sig, n_var)
        # Channel estimation
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)

        return Rx_sig

def initNetParams(model):
    '''Init net parameters.'''
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
         
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    # 产生下三角矩阵
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask)

    
def create_masks(src, trg, padding_idx):

    src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor) #[batch, 1, seq_len]

    trg_mask = (trg == padding_idx).unsqueeze(-2).type(torch.FloatTensor) #[batch, 1, seq_len]
    look_ahead_mask = subsequent_mask(trg.size(-1)).type_as(trg_mask.data)
    combined_mask = torch.max(trg_mask, look_ahead_mask)
    
    return src_mask.to(device), combined_mask.to(device)

def loss_function(x, trg, padding_idx, criterion):
    
    loss = criterion(x, trg)
    mask = (trg != padding_idx).type_as(loss.data)
    # a = mask.cpu().numpy()
    loss *= mask
    
    return loss.mean()

def PowerNormalize(x):
    
    x_square = torch.mul(x, x)
    power = torch.mean(x_square).sqrt()
    if power > 1:
        x = torch.div(x, power)
    
    return x


def SNR_to_noise(snr):
    snr = 10 ** (snr / 10)
    noise_std = 1 / np.sqrt(2 * snr)

    return noise_std


class SemanticSimilarityLoss(nn.Module):
    """Semantic similarity loss using a frozen Sentence-BERT model.

    Encodes the *source* sentence and the *predicted* (reconstructed) sentence
    with a shared, frozen SBERT encoder, then returns the mean cosine distance
    (1 - cosine_similarity) across the batch so it can be minimised alongside
    the cross-entropy reconstruction loss.

    Args:
        model_name: Any ``sentence-transformers`` compatible model identifier.
            Defaults to ``'all-MiniLM-L6-v2'`` — small, fast, and accurate.
        device: torch device to place the SBERT model on.
    """

    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        device: torch.device = None,
    ):
        super().__init__()
        self._sbert_device = device or torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu'
        )
        self._sbert = SentenceTransformer(model_name, device=str(self._sbert_device))
        # Freeze all SBERT parameters — it is a *reference* encoder only.
        for param in self._sbert.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def _encode(self, sentences: list) -> torch.Tensor:
        """Return normalised sentence embeddings as a (B, D) tensor."""
        embeddings = self._sbert.encode(
            sentences,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            device=str(self._sbert_device),
        )
        return embeddings  # already on self._sbert_device

    def forward(
        self,
        src_sentences: list,
        pred_sentences: list,
    ) -> torch.Tensor:
        """Compute mean cosine distance between source and predicted sentences.

        Args:
            src_sentences:  List[str] of ground-truth / source sentences.
            pred_sentences: List[str] of model-reconstructed sentences.

        Returns:
            Scalar tensor: mean (1 - cosine_similarity) ∈ [0, 2].
        """
        src_emb  = self._encode(src_sentences)   # (B, D)
        pred_emb = self._encode(pred_sentences)  # (B, D)
        # cosine similarity is dot-product of unit vectors
        cos_sim  = (src_emb * pred_emb).sum(dim=-1)  # (B,)
        loss     = (1.0 - cos_sim).mean()
        return loss


def _decode_predictions(
    logits: torch.Tensor,
    idx_to_token: dict,
    pad_id: int,
    end_id: int,
) -> list:
    """Convert a (B, T, V) logit tensor to a list of decoded strings.

    Args:
        logits:      Raw logits from the model head — shape (B, T, V).
        idx_to_token: Mapping from token index to string token.
        pad_id:      Index used for padding tokens (skipped in output).
        end_id:      Index of the end/stop token (decoding stops here).

    Returns:
        List[str] of length B.
    """
    token_ids = logits.argmax(dim=-1)  # (B, T)
    sentences = []
    for seq in token_ids:
        tokens = []
        for idx in seq.tolist():
            if idx == end_id:
                break
            if idx != pad_id:
                tok = idx_to_token.get(idx, '')
                if tok:
                    tokens.append(tok)
        sentences.append(' '.join(tokens) if tokens else '<empty>')
    return sentences

def train_step(model, src, trg, n_var, pad, opt, criterion, channel, mi_net=None,
               sem_loss_fn=None, sem_weight=0.1, idx_to_token=None, end_id=None):
    model.train()

    channels = Channels()
    opt.zero_grad()

    if model.use_bert_encoder:
        # src is a dict: {'input_ids': ..., 'attention_mask': ...} (BERT tokens)
        # trg is also a dict with the same keys (auto-encoding: src == trg)
        src_input_ids = src['input_ids'].to(device)
        src_attn_mask = src['attention_mask'].to(device)       # 1=real, 0=pad
        trg_input_ids = trg['input_ids'].to(device)

        bert_pad_id = 0  # [PAD] token ID in BERT vocabulary
        trg_inp = trg_input_ids[:, :-1]   # teacher-forced decoder input
        trg_real = trg_input_ids[:, 1:]   # ground-truth shifted by one

        # look-ahead mask for decoder self-attention
        trg_mask = (trg_inp == bert_pad_id).unsqueeze(-2).type(torch.FloatTensor).to(device)
        look_ahead_mask = subsequent_mask(trg_inp.size(-1)).type_as(trg_mask.data)
        combined_mask = torch.max(trg_mask, look_ahead_mask)

        # src padding mask for decoder cross-attention (invert BERT convention)
        src_mask = (1 - src_attn_mask).unsqueeze(-2).type(torch.FloatTensor).to(device)

        enc_output = model.encoder(src_input_ids, attention_mask=src_attn_mask)
        channel_enc_output = model.channel_encoder(enc_output)
        Tx_sig = PowerNormalize(channel_enc_output)

        if channel == 'AWGN':
            Rx_sig = channels.AWGN(Tx_sig, n_var)
        elif channel == 'Rayleigh':
            Rx_sig = channels.Rayleigh(Tx_sig, n_var)
        elif channel == 'Rician':
            Rx_sig = channels.Rician(Tx_sig, n_var)
        else:
            raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

        channel_dec_output = model.channel_decoder(Rx_sig)
        dec_output = model.decoder(trg_inp, channel_dec_output, combined_mask, src_mask)
        pred = model.dense(dec_output)

        ntokens = pred.size(-1)
        loss = loss_function(pred.contiguous().view(-1, ntokens),
                             trg_real.contiguous().view(-1),
                             bert_pad_id, criterion)

        if mi_net is not None:
            mi_net.eval()
            joint, marginal = sample_batch(Tx_sig, Rx_sig)
            mi_lb, _, _ = mutual_information(joint, marginal, mi_net)
            loss = loss + 0.0009 * (-mi_lb)

        if sem_loss_fn is not None and sem_weight > 0.0 and idx_to_token is not None:
            # Decode source and predicted sentences for semantic comparison.
            # For the BERT encoder path we use the BERT vocab (idx_to_token must
            # be the BERT id→token map) and end_id should be the SEP token id.
            bert_end_id = end_id if end_id is not None else 102  # [SEP]
            # src_sentences: reconstruct from src_input_ids (skip [CLS]/[SEP]/[PAD])
            src_sentences  = _decode_predictions(
                # treat src_input_ids as a (B, T, 1) logit-like tensor via one-hot trick
                # instead, pass ids directly through a helper call
                torch.nn.functional.one_hot(
                    src_input_ids, num_classes=ntokens
                ).float(),
                idx_to_token, bert_pad_id, bert_end_id,
            )
            pred_sentences = _decode_predictions(pred, idx_to_token, bert_pad_id, bert_end_id)
            sem_loss = sem_loss_fn(src_sentences, pred_sentences)
            loss = loss + sem_weight * sem_loss

    else:
        trg_inp = trg[:, :-1]
        trg_real = trg[:, 1:]

        src_mask, look_ahead_mask = create_masks(src, trg_inp, pad)

        enc_output = model.encoder(src, src_mask)
        channel_enc_output = model.channel_encoder(enc_output)
        Tx_sig = PowerNormalize(channel_enc_output)

        if channel == 'AWGN':
            Rx_sig = channels.AWGN(Tx_sig, n_var)
        elif channel == 'Rayleigh':
            Rx_sig = channels.Rayleigh(Tx_sig, n_var)
        elif channel == 'Rician':
            Rx_sig = channels.Rician(Tx_sig, n_var)
        else:
            raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

        channel_dec_output = model.channel_decoder(Rx_sig)
        dec_output = model.decoder(trg_inp, channel_dec_output, look_ahead_mask, src_mask)
        pred = model.dense(dec_output)

        ntokens = pred.size(-1)
        loss = loss_function(pred.contiguous().view(-1, ntokens),
                             trg_real.contiguous().view(-1),
                             pad, criterion)

        if mi_net is not None:
            mi_net.eval()
            joint, marginal = sample_batch(Tx_sig, Rx_sig)
            mi_lb, _, _ = mutual_information(joint, marginal, mi_net)
            loss = loss + 0.0009 * (-mi_lb)

        if sem_loss_fn is not None and sem_weight > 0.0 and idx_to_token is not None:
            end_tok_id = end_id if end_id is not None else pad
            src_sentences  = _decode_predictions(
                torch.nn.functional.one_hot(src, num_classes=ntokens).float(),
                idx_to_token, pad, end_tok_id,
            )
            pred_sentences = _decode_predictions(pred, idx_to_token, pad, end_tok_id)
            sem_loss = sem_loss_fn(src_sentences, pred_sentences)
            loss = loss + sem_weight * sem_loss

    loss.backward()
    opt.step()

    return loss.item()



def train_mi(model, mi_net, src, n_var, padding_idx, opt, channel):
    mi_net.train()
    opt.zero_grad()
    channels = Channels()
    src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor).to(device)  # [batch, 1, seq_len]
    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    Tx_sig = PowerNormalize(channel_enc_output)

    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    joint, marginal = sample_batch(Tx_sig, Rx_sig)
    mi_lb, _, _ = mutual_information(joint, marginal, mi_net)
    loss_mine = -mi_lb

    loss_mine.backward()
    torch.nn.utils.clip_grad_norm_(mi_net.parameters(), 10.0)
    opt.step()

    return loss_mine.item()

def val_step(model, src, trg, n_var, pad, criterion, channel):
    channels = Channels()

    if model.use_bert_encoder:
        src_input_ids = src['input_ids'].to(device)
        src_attn_mask = src['attention_mask'].to(device)
        trg_input_ids = trg['input_ids'].to(device)

        bert_pad_id = 0
        trg_inp = trg_input_ids[:, :-1]
        trg_real = trg_input_ids[:, 1:]

        trg_mask = (trg_inp == bert_pad_id).unsqueeze(-2).type(torch.FloatTensor).to(device)
        look_ahead_mask = subsequent_mask(trg_inp.size(-1)).type_as(trg_mask.data)
        combined_mask = torch.max(trg_mask, look_ahead_mask)
        src_mask = (1 - src_attn_mask).unsqueeze(-2).type(torch.FloatTensor).to(device)

        enc_output = model.encoder(src_input_ids, attention_mask=src_attn_mask)
        channel_enc_output = model.channel_encoder(enc_output)
        Tx_sig = PowerNormalize(channel_enc_output)

        if channel == 'AWGN':
            Rx_sig = channels.AWGN(Tx_sig, n_var)
        elif channel == 'Rayleigh':
            Rx_sig = channels.Rayleigh(Tx_sig, n_var)
        elif channel == 'Rician':
            Rx_sig = channels.Rician(Tx_sig, n_var)
        else:
            raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

        channel_dec_output = model.channel_decoder(Rx_sig)
        dec_output = model.decoder(trg_inp, channel_dec_output, combined_mask, src_mask)
        pred = model.dense(dec_output)

        ntokens = pred.size(-1)
        loss = loss_function(pred.contiguous().view(-1, ntokens),
                             trg_real.contiguous().view(-1),
                             bert_pad_id, criterion)
    else:
        trg_inp = trg[:, :-1]
        trg_real = trg[:, 1:]

        src_mask, look_ahead_mask = create_masks(src, trg_inp, pad)

        enc_output = model.encoder(src, src_mask)
        channel_enc_output = model.channel_encoder(enc_output)
        Tx_sig = PowerNormalize(channel_enc_output)

        if channel == 'AWGN':
            Rx_sig = channels.AWGN(Tx_sig, n_var)
        elif channel == 'Rayleigh':
            Rx_sig = channels.Rayleigh(Tx_sig, n_var)
        elif channel == 'Rician':
            Rx_sig = channels.Rician(Tx_sig, n_var)
        else:
            raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

        channel_dec_output = model.channel_decoder(Rx_sig)
        dec_output = model.decoder(trg_inp, channel_dec_output, look_ahead_mask, src_mask)
        pred = model.dense(dec_output)

        ntokens = pred.size(-1)
        loss = loss_function(pred.contiguous().view(-1, ntokens),
                             trg_real.contiguous().view(-1),
                             pad, criterion)

    return loss.item()

    
def greedy_decode(model, src, n_var, max_len, padding_idx, start_symbol, channel,
                  attention_mask=None):
    """Greedy (argmax) decoder.

    For the BERT encoder path, ``src`` contains BERT input_ids and
    ``attention_mask`` must be provided.  ``padding_idx`` and
    ``start_symbol`` should use BERT token IDs in that case
    (pad=0, start=[CLS]=101).

    For the custom encoder path the original behaviour is preserved.
    """
    channels = Channels()

    if model.use_bert_encoder:
        src_input_ids = src.to(device)
        src_attn_mask = attention_mask.to(device) if attention_mask is not None else None
        src_mask = None  # not needed for cross-attention mask derivation here
        if src_attn_mask is not None:
            src_mask = (1 - src_attn_mask).unsqueeze(-2).type(torch.FloatTensor).to(device)

        enc_output = model.encoder(src_input_ids, attention_mask=src_attn_mask)
    else:
        src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor).to(device)
        enc_output = model.encoder(src, src_mask)

    channel_enc_output = model.channel_encoder(enc_output)
    Tx_sig = PowerNormalize(channel_enc_output)

    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    memory = model.channel_decoder(Rx_sig)

    outputs = torch.ones(src.size(0), 1).fill_(start_symbol).type_as(src.data)

    for i in range(max_len - 1):
        trg_mask = (outputs == padding_idx).unsqueeze(-2).type(torch.FloatTensor)
        look_ahead_mask = subsequent_mask(outputs.size(1)).type(torch.FloatTensor)
        combined_mask = torch.max(trg_mask, look_ahead_mask)
        combined_mask = combined_mask.to(device)

        dec_output = model.decoder(outputs, memory, combined_mask, src_mask)
        pred = model.dense(dec_output)

        prob = pred[:, -1:, :]  # (batch_size, 1, vocab_size)
        _, next_word = torch.max(prob, dim=-1)
        outputs = torch.cat([outputs, next_word], dim=1)

    return outputs
