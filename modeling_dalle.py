"""
 coding=utf-8
 Copyright 2018, Antonio Mendoza Hao Tan, Mohit Bansal
 Adapted From Facebook Inc, Detectron2

 Adapted from https://github.com/kuprel/min-dalle
 
 Copyright 2022, Ontocord, LLC
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.import copy
 """
import os
import numpy
from PIL import Image
from torch import LongTensor
import torch
from torch import nn
import json
import requests
from typing import Iterator
from typing import List
import torch
from torch import nn, BoolTensor, FloatTensor, LongTensor
from typing import Tuple, List
from torch import nn, LongTensor, FloatTensor, BoolTensor
from math import inf
from typing import List, Tuple
from .vqgan_detokenizer import * 


from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig

MIN_DALLE_REPO = 'https://huggingface.co/kuprel/min-dalle/resolve/main/'


IMAGE_TOKEN_COUNT = 256
BLANK_TOKEN = 6965


class TextTokenizer:
    def __init__(self, vocab: dict, merges: List[str]):
        self.token_from_subword = vocab
        pairs = [tuple(pair.split()) for pair in merges]
        self.rank_from_pair = dict(zip(pairs, range(len(pairs))))

    def tokenize(self, text: str, is_verbose: bool = False, max_len=64) -> List[int]:
        sep_token = self.token_from_subword['</s>']
        cls_token = self.token_from_subword['<s>']
        unk_token = self.token_from_subword['<unk>']
        if True:
          text = text.lower().encode("ascii", errors="ignore").decode()
          tokens = [
              self.token_from_subword.get(subword, unk_token)
              for word in text.split(" ") if len(word) > 0
              for subword in self.get_byte_pair_encoding(word, is_verbose)
          ]
          tokens = [cls_token] + tokens + [sep_token]
        if len(tokens) > max_len: 
            tokens = tokens[:max_len]
        if is_verbose: print("text tokens", tokens)
        text_tokens = numpy.ones((2, 64), dtype=numpy.int32)
        text_tokens[0, :2] = [tokens[0], tokens[-1]]
        text_tokens[1, :len(tokens)] = tokens

        return torch.tensor(text_tokens).to(torch.long)
        


    def get_byte_pair_encoding(self, word: str, is_verbose: bool) -> List[str]:
        def get_pair_rank(pair: Tuple[str, str]) -> int:
            return self.rank_from_pair.get(pair, inf)

        subwords = [chr(ord(" ") + 256)] + list(word)
        while len(subwords) > 1:
            pairs = list(zip(subwords[:-1], subwords[1:]))
            pair_to_merge = min(pairs, key=get_pair_rank)
            if pair_to_merge not in self.rank_from_pair: break
            i = pairs.index(pair_to_merge)
            subwords = (
                (subwords[:i] if i > 0 else []) + 
                [subwords[i] + subwords[i + 1]] + 
                (subwords[i + 2:] if i + 2 < len(subwords) else [])
            )

        if is_verbose: print(subwords)
        return subwords

class GLU(nn.Module):
    def __init__(self, count_in_out, count_middle):
        super().__init__()
        self.gelu = nn.GELU()
        self.ln0 = nn.LayerNorm(count_in_out)
        self.ln1 = nn.LayerNorm(count_middle)
        self.fc0 = nn.Linear(count_in_out, count_middle, bias=False)
        self.fc1 = nn.Linear(count_in_out, count_middle, bias=False)
        self.fc2 = nn.Linear(count_middle, count_in_out, bias=False)
    
    def forward(self, z: FloatTensor) -> FloatTensor:
        z = self.ln0.forward(z)
        w = self.fc0.forward(z)
        w = self.gelu.forward(w)
        v = self.fc1.forward(z)
        z = self.ln1.forward(w * v)
        z = self.fc2.forward(z)
        return z


class AttentionBase(nn.Module):
    def __init__(self, head_count: int, embed_count: int):
        super().__init__()
        self.head_count = head_count
        self.embed_count = embed_count

        self.k_proj = nn.Linear(embed_count, embed_count, bias=False)
        self.v_proj = nn.Linear(embed_count, embed_count, bias=False)
        self.q_proj = nn.Linear(embed_count, embed_count, bias=False)
        self.out_proj = nn.Linear(embed_count, embed_count, bias=False)
        self.register_buffer('one', torch.ones((1, 1)))
    
    def forward(
        self,
        keys: FloatTensor,
        values: FloatTensor,
        queries: FloatTensor,
        attention_mask: BoolTensor
    ) -> FloatTensor:
        keys = keys.reshape(keys.shape[:2] + (self.head_count, -1))
        values = values.reshape(values.shape[:2] + (self.head_count, -1))
        queries = queries.reshape(queries.shape[:2] + (self.head_count, -1))
        queries /= queries.shape[-1] ** 0.5

        attention_bias = torch.where(
            attention_mask,
            self.one * 0,
            self.one * (-torch.inf),
        )
        attention_weights: FloatTensor = torch.einsum(
            'bqhc,bkhc->bhqk',
            queries, 
            keys
        )
        attention_weights += attention_bias[:, None, None, :]
        attention_weights = torch.softmax(attention_weights, -1)
        attention_output: FloatTensor = torch.einsum(
            "bhqk,bkhc->bqhc",
            attention_weights, 
            values
        )
        shape = attention_output.shape[:2] + (self.embed_count,)
        attention_output = attention_output.reshape(shape)
        attention_output = self.out_proj.forward(attention_output)
        return attention_output


class EncoderSelfAttention(AttentionBase):
    def forward(
        self,
        encoder_state: FloatTensor,
        attention_mask: BoolTensor
    ) -> FloatTensor:
        keys = self.k_proj.forward(encoder_state)
        values = self.v_proj.forward(encoder_state)
        queries = self.q_proj.forward(encoder_state)
        return super().forward(keys, values, queries, attention_mask)


class EncoderLayer(nn.Module):
    def __init__(self, embed_count: int, head_count: int, glu_embed_count: int):
        super().__init__()
        self.pre_self_attn_layer_norm = nn.LayerNorm(embed_count)
        self.self_attn = EncoderSelfAttention(head_count, embed_count)
        self.self_attn_layer_norm = nn.LayerNorm(embed_count)
        self.glu = GLU(embed_count, glu_embed_count)
    
    def forward(
        self,
        encoder_state: FloatTensor,
        attention_mask: BoolTensor
    ) -> FloatTensor:
        residual = encoder_state
        encoder_state = self.pre_self_attn_layer_norm.forward(encoder_state)
        encoder_state = self.self_attn.forward(encoder_state, attention_mask)
        encoder_state = self.self_attn_layer_norm.forward(encoder_state)
        encoder_state = residual + encoder_state
        residual = encoder_state
        encoder_state = self.glu.forward(encoder_state)
        encoder_state = residual + encoder_state
        return encoder_state


class DalleBartEncoder(nn.Module):
    def __init__(
        self,
        layer_count: int,
        embed_count: int,
        attention_head_count: int,
        text_vocab_count: int,
        text_token_count: int,
        glu_embed_count: int
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(text_vocab_count, embed_count)
        self.embed_positions = nn.Embedding(text_token_count, embed_count)
        self.layers: List[EncoderLayer] = nn.ModuleList([
            EncoderLayer(
                embed_count = embed_count,
                head_count = attention_head_count,
                glu_embed_count = glu_embed_count
            ) 
            for _ in range(layer_count)
        ])
        self.layernorm_embedding = nn.LayerNorm(embed_count)
        self.final_ln = nn.LayerNorm(embed_count)
        token_indices = torch.arange(text_token_count).to(torch.long)
        self.register_buffer('token_indices', token_indices)

    def forward(self, text_tokens: LongTensor) -> FloatTensor:
        attention_mask = text_tokens.not_equal(1)
        pose_tokens = self.token_indices[None][[0] * text_tokens.shape[0]]
        encoder_state = (
            self.embed_tokens.forward(text_tokens) +
            self.embed_positions.forward(pose_tokens)
        )
        encoder_state = self.layernorm_embedding.forward(encoder_state)
        for layer in self.layers:
            encoder_state = layer.forward(encoder_state, attention_mask)
        encoder_state = self.final_ln.forward(encoder_state)
        return encoder_state

class DecoderCrossAttention(AttentionBase):
    def forward(
        self,
        decoder_state: FloatTensor,
        encoder_state: FloatTensor,
        attention_mask: BoolTensor
    ) -> FloatTensor:
        keys = self.k_proj.forward(encoder_state)
        values = self.v_proj.forward(encoder_state)
        queries = self.q_proj.forward(decoder_state)
        return super().forward(keys, values, queries, attention_mask)


class DecoderSelfAttention(AttentionBase):
    def __init__(self, head_count: int, embed_count: int):
        super().__init__(head_count, embed_count)
        token_indices = torch.arange(IMAGE_TOKEN_COUNT)
        self.register_buffer('token_indices', token_indices)

    def forward(
        self, 
        decoder_state: FloatTensor,
        attention_state: FloatTensor,
        token_index: LongTensor
    ) -> Tuple[FloatTensor, FloatTensor]:
        keys = self.k_proj.forward(decoder_state)
        values = self.v_proj.forward(decoder_state)
        queries = self.q_proj.forward(decoder_state)
        attn_mask = self.token_indices < token_index + 1
        attn_mask = attn_mask[None][[0] * decoder_state.shape[0]]
        attn_state_new = torch.cat([keys, values]).to(attention_state.dtype)
        attention_state[:, token_index] = attn_state_new
        batch_count = decoder_state.shape[0]
        keys = attention_state[:batch_count]
        values = attention_state[batch_count:]
        #print (keys.dtype, values.dtype, queries.dtype, attn_mask.dtype)
        decoder_state = super().forward(keys, values, queries, attn_mask)
        return decoder_state, attention_state


class DecoderLayer(nn.Module):
    def __init__(
        self, 
        head_count: int, 
        embed_count: int,
        glu_embed_count: int
    ):
        super().__init__()
        self.pre_self_attn_layer_norm = nn.LayerNorm(embed_count)
        self.self_attn = DecoderSelfAttention(head_count, embed_count)
        self.self_attn_layer_norm = nn.LayerNorm(embed_count)
        self.pre_encoder_attn_layer_norm = nn.LayerNorm(embed_count)
        self.encoder_attn = DecoderCrossAttention(head_count, embed_count)
        self.encoder_attn_layer_norm = nn.LayerNorm(embed_count)
        self.glu = GLU(embed_count, glu_embed_count)


    def forward(
        self,
        decoder_state: FloatTensor,
        encoder_state: FloatTensor,
        attention_state: FloatTensor,
        attention_mask: BoolTensor,
        token_index: LongTensor
    ) -> Tuple[FloatTensor, FloatTensor]:
        # Self Attention
        residual = decoder_state
        decoder_state = self.pre_self_attn_layer_norm.forward(decoder_state)
        decoder_state, attention_state = self.self_attn.forward(
            decoder_state,
            attention_state,
            token_index
        )
        decoder_state = self.self_attn_layer_norm.forward(decoder_state)
        decoder_state = residual + decoder_state

        # Cross Attention
        residual = decoder_state
        decoder_state = self.pre_encoder_attn_layer_norm.forward(decoder_state)
        decoder_state = self.encoder_attn.forward(
            decoder_state,
            encoder_state,
            attention_mask
        )
        decoder_state = self.encoder_attn_layer_norm.forward(decoder_state)
        decoder_state = residual + decoder_state

        # Feed forward
        residual = decoder_state
        decoder_state = self.glu.forward(decoder_state)
        decoder_state = residual + decoder_state

        return decoder_state, attention_state


class DalleBartDecoder(nn.Module):
    def __init__(
        self,
        image_vocab_count: int,
        embed_count: int,
        attention_head_count: int,
        glu_embed_count: int,
        layer_count: int,
        start_token: int
    ):
        super().__init__()
        self.layer_count = layer_count
        self.embed_count = embed_count
        self.embed_tokens = nn.Embedding(image_vocab_count + 1, embed_count)
        self.embed_positions = nn.Embedding(IMAGE_TOKEN_COUNT, embed_count)
        self.layers: List[DecoderLayer] = nn.ModuleList([
            DecoderLayer(
                attention_head_count,
                embed_count,
                glu_embed_count
            ) 
            for _ in range(layer_count)
        ])
        self.layernorm_embedding = nn.LayerNorm(embed_count)
        self.final_ln = nn.LayerNorm(embed_count)
        self.lm_head = nn.Linear(embed_count, image_vocab_count + 1, bias=False)
        zero_prob = torch.zeros([1])
        token_indices = torch.arange(IMAGE_TOKEN_COUNT)
        start_token = torch.tensor([start_token]).to(torch.long)
        self.register_buffer('zero_prob', zero_prob)
        self.register_buffer('token_indices', token_indices)
        self.register_buffer('start_token', start_token)

    def forward(
       self,
        log2_k: int,
        log2_supercondition_factor: int,
        attention_mask: BoolTensor,
        encoder_state: FloatTensor,
        attention_state: FloatTensor,
        prev_tokens: LongTensor,
        token_index: LongTensor
    ) -> Tuple[FloatTensor, FloatTensor]:
        image_count = encoder_state.shape[0] // 2
        token_index_batched = token_index[[0] * image_count * 2]
        prev_tokens = prev_tokens[list(range(image_count)) * 2]
        decoder_state = self.embed_tokens.forward(prev_tokens)
        decoder_state += self.embed_positions.forward(token_index_batched)
        decoder_state = self.layernorm_embedding.forward(decoder_state)
        decoder_state = decoder_state[:, None]
        for i in range(self.layer_count):
            decoder_state, attention_state[i] = self.layers[i].forward(
                decoder_state,
                encoder_state,
                attention_state[i],
                attention_mask,
                token_index
            )
        decoder_state = self.final_ln(decoder_state)
        logits = self.lm_head(decoder_state)
        return logits, decoder_state

    def decode_step(
        self,
        log2_k: int,
        log2_supercondition_factor: int,
        attention_mask: BoolTensor,
        encoder_state: FloatTensor,
        attention_state: FloatTensor,
        prev_tokens: LongTensor,
        token_index: LongTensor
    ) -> Tuple[FloatTensor, FloatTensor]:
        image_count = encoder_state.shape[0] // 2
        token_index_batched = token_index[[0] * image_count * 2]
        prev_tokens = prev_tokens[list(range(image_count)) * 2]
        decoder_state = self.embed_tokens.forward(prev_tokens)
        decoder_state += self.embed_positions.forward(token_index_batched)
        decoder_state = self.layernorm_embedding.forward(decoder_state)
        decoder_state = decoder_state[:, None]
        for i in range(self.layer_count):
            decoder_state, attention_state[i] = self.layers[i].forward(
                decoder_state,
                encoder_state,
                attention_state[i],
                attention_mask,
                token_index
            )
        decoder_state = self.final_ln(decoder_state)
        logits = self.lm_head(decoder_state)
        #print (decoder_state.dtype, logits.dtype, self.zero_prob)
        a = 2 ** log2_supercondition_factor
        logits: FloatTensor = (
            logits[:image_count, -1] * (1 - a) + 
            logits[image_count:, -1] * a
        )

        top_logits, _ = logits.topk(2 ** log2_k, dim=-1)
        probs = torch.where(
            logits < top_logits[:, [-1]],
            self.zero_prob.to(logits.dtype),
            torch.exp(logits - top_logits[:, [0]])
        )
        return probs, attention_state


    def decode_row(
        self,
        row_index: int,
        log2_k: int,
        log2_supercondition_factor: int,
        encoder_state: FloatTensor,
        attention_mask: BoolTensor,
        attention_state: FloatTensor,
        image_tokens_sequence: LongTensor
    ) -> Tuple[FloatTensor, LongTensor]:
        for col_index in range(16):
            i = 16 * row_index + col_index
            probs, attention_state = self.decode_step(
                log2_k = log2_k,
                log2_supercondition_factor = log2_supercondition_factor,
                attention_mask = attention_mask,
                encoder_state = encoder_state,
                attention_state = attention_state,
                prev_tokens = image_tokens_sequence[:, i],
                token_index = self.token_indices[[i]]
            )
            image_tokens_sequence[:, i + 1] = torch.multinomial(probs, 1)[:, 0]

        return attention_state, image_tokens_sequence

    
    def decode_initial(
        self,
        seed: int,
        image_count: int,
        text_tokens: LongTensor,
        encoder_state: FloatTensor
    ) -> Tuple[FloatTensor, FloatTensor, FloatTensor, LongTensor]:
        expanded_indices = [0] * image_count + [1] * image_count
        text_tokens = text_tokens[expanded_indices]
        encoder_state = encoder_state[expanded_indices]
        attention_mask = text_tokens.not_equal(1)

        attention_state_shape = (
            self.layer_count,
            image_count * 4,
            IMAGE_TOKEN_COUNT,
            self.embed_count
        )
        attention_state = torch.zeros(attention_state_shape).to(encoder_state.dtype)
        image_tokens_sequence = torch.full(
            (image_count, IMAGE_TOKEN_COUNT + 1), 
            BLANK_TOKEN,
            dtype=torch.long
        )
        attention_state = attention_state.to(self.embed_tokens.weight.device)
        image_tokens_sequence = image_tokens_sequence.to(self.embed_tokens.weight.device)
        
        image_tokens_sequence[:, 0] = self.start_token[0]

        if seed > 0: torch.manual_seed(seed)

        return encoder_state, attention_mask, attention_state, image_tokens_sequence



class DalleConfig(PretrainedConfig):
    model_type = "dalle"
    def __init__(
        self,
        **kwargs
    ):
        if 'is_encoder_decoder' not in kwargs:
          kwargs['is_encoder_decoder'] = True
        super().__init__(
            **kwargs,
        )

class DalleModel (PreTrainedModel):
    config_class = DalleConfig
    base_model_prefix = "model"
    
    def __init__(
        self,
        config: DalleConfig=None,
        models_root: str = None,
        is_mega: bool = False, 
        is_verbose = False
    ):
        if config is None:
          config = DalleConfig()
        super().__init__(config)
        self.config = config
        #TODO - move the defaults up to DalleConfig
        is_mega = config.is_mega = self.is_mega = config.is_mega if config is not None and  hasattr(config, 'is_mega') else is_mega
        config.text_token_count = self.text_token_count = config.text_token_count if config is not None and  hasattr(config, 'text_token_count')  else 64
        config.layer_count = self.layer_count = config.layer_count if config is not None and  hasattr(config, 'layer_count') else  (24 if is_mega else 12)
        config.attention_head_count = self.attention_head_count = config.encoder_attention_heads if config is not None and  hasattr(config, 'encoder_attention_heads') else (32 if is_mega else 16)
        config.embed_count = self.embed_count = config.embed_count if config is not None and  hasattr(config, 'embed_count') else (2048 if is_mega else 1024)
        config.glu_embed_count = self.glu_embed_count = config.glu_embed_count if config is not None and  hasattr(config, 'glu_embed_count') else (4096 if is_mega else 2730)
        config.text_vocab_count = self.text_vocab_count = config.text_vocab_count if config is not None and  hasattr(config, 'text_vocab_count') else (50272 if is_mega else 50264)
        config.image_vocab_count = self.image_vocab_count = config.image_vocab_count if config is not None and  hasattr(config, 'image_vocab_count') else (16415 if is_mega else 16384)
        #legacy mode loading
        if models_root:
          model_name = 'dalle_bart_{}'.format('mega' if is_mega else 'mini')
          dalle_path = os.path.join(models_root, model_name)
          if not os.path.exists(dalle_path): os.makedirs(dalle_path)
          self.vocab_path = os.path.join(dalle_path, 'vocab.json')
          self.merges_path = os.path.join(dalle_path, 'merges.txt')
          vqgan_path = os.path.join(models_root, 'vqgan')
          if not os.path.exists(vqgan_path): os.makedirs(vqgan_path)
          self.encoder_params_path = os.path.join(dalle_path, 'encoder.pt')
          self.decoder_params_path = os.path.join(dalle_path, 'decoder.pt')
          self.detoker_params_path = os.path.join(vqgan_path, 'detoker.pt')
          self.init_tokenizer(is_verbose=is_verbose)
          self.init_encoder(is_verbose=is_verbose)
          self.init_decoder(is_verbose=is_verbose)
          self.init_detokenizer(is_verbose=is_verbose)
          config.vocab = self.tokenizer.token_from_subword
          config.merges = [" ".join(a) for a in self.tokenizer.rank_from_pair.keys()]
        else:
          #in the HF compatibale mode, we do not use the custom tokenizer. Instead we expect tensors to be passed. 
           self.tokenizer = TextTokenizer(config.vocab, config.merges)
           self.encoder = DalleBartEncoder(
               attention_head_count = self.attention_head_count,
               embed_count = self.embed_count,
               glu_embed_count = self.glu_embed_count,
               text_token_count = self.text_token_count,
               text_vocab_count = self.text_vocab_count,
               layer_count = self.layer_count
           )
           self.decoder = DalleBartDecoder(
               image_vocab_count = self.image_vocab_count,
               attention_head_count = self.attention_head_count,
               embed_count = self.embed_count,
               glu_embed_count = self.glu_embed_count,
               layer_count = self.layer_count,
               start_token = self.image_vocab_count
           )
           self.detokenizer = VQGanDetokenizer().eval()
           # Initialize weights and apply final processing
           self.post_init()
       

    def download_tokenizer(self, is_verbose=False):
        if is_verbose: print("downloading tokenizer params")
        suffix = '' if self.is_mega else '_mini'
        vocab = requests.get(MIN_DALLE_REPO + 'vocab{}.json'.format(suffix))
        merges = requests.get(MIN_DALLE_REPO + 'merges{}.txt'.format(suffix))
        with open(self.vocab_path, 'wb') as f: f.write(vocab.content)
        with open(self.merges_path, 'wb') as f: f.write(merges.content)


    def download_encoder(self, is_verbose=False):
        if is_verbose: print("downloading encoder params")
        suffix = '' if self.is_mega else '_mini'
        params = requests.get(MIN_DALLE_REPO + 'encoder{}.pt'.format(suffix))
        with open(self.encoder_params_path, 'wb') as f: f.write(params.content)


    def download_decoder(self, is_verbose=False):
        if is_verbose: print("downloading decoder params")
        suffix = '' if self.is_mega else '_mini'
        params = requests.get(MIN_DALLE_REPO + 'decoder{}.pt'.format(suffix))
        with open(self.decoder_params_path, 'wb') as f: f.write(params.content)
    


    def init_tokenizer(self, is_verbose=False):
        is_downloaded = os.path.exists(self.vocab_path)
        is_downloaded &= os.path.exists(self.merges_path)
        if not is_downloaded: self.download_tokenizer(is_verbose)
        if is_verbose: print("intializing TextTokenizer")
        with open(self.vocab_path, 'r', encoding='utf8') as f:
            vocab = json.load(f)
        with open(self.merges_path, 'r', encoding='utf8') as f:
            merges = f.read().split("\n")[1:-1]
        
        self.tokenizer = TextTokenizer(vocab, merges)


    def init_encoder(self, is_verbose=False):
        is_downloaded = os.path.exists(self.encoder_params_path)
        if not is_downloaded: self.download_encoder(is_verbose)
        if is_verbose: print("initializing DalleBartEncoder")
        self.encoder = DalleBartEncoder(
            attention_head_count = self.attention_head_count,
            embed_count = self.embed_count,
            glu_embed_count = self.glu_embed_count,
            text_token_count = self.text_token_count,
            text_vocab_count = self.text_vocab_count,
            layer_count = self.layer_count
        ).eval()
        params = torch.load(self.encoder_params_path)
        self.encoder.load_state_dict(params, strict=False)
        del params


    def init_decoder(self, is_verbose=False):
        is_downloaded = os.path.exists(self.decoder_params_path)
        if not is_downloaded: self.download_decoder(is_verbose)
        if is_verbose: print("initializing DalleBartDecoder")
        self.decoder = DalleBartDecoder(
            image_vocab_count = self.image_vocab_count,
            attention_head_count = self.attention_head_count,
            embed_count = self.embed_count,
            glu_embed_count = self.glu_embed_count,
            layer_count = self.layer_count,
            start_token = self.image_vocab_count
        )
        params = torch.load(self.decoder_params_path)
        self.decoder.load_state_dict(params, strict=False)
        del params
        

    def download_detokenizer(self, is_verbose=False):
        if is_verbose: print("downloading detokenizer params")
        params = requests.get(MIN_DALLE_REPO + 'detoker.pt')
        with open(self.detoker_params_path, 'wb') as f: f.write(params.content)

    def init_detokenizer(self, is_verbose=False):
        is_downloaded = os.path.exists(self.detoker_params_path)
        if not is_downloaded: self.download_detokenizer()
        if is_verbose: print("initializing VQGanDetokenizer")
        self.detokenizer = VQGanDetokenizer().eval()
        params = torch.load(self.detoker_params_path)
        self.detokenizer.load_state_dict(params)
        del params
        

    def image_from_tokens(
        self,
        grid_size: int,
        image_tokens: LongTensor,
        is_verbose: bool = False
    ) -> Image.Image:
        #if torch.cuda.is_available(): torch.cuda.empty_cache()
        if is_verbose: print("detokenizing image")
        images = self.detokenizer.forward(image_tokens).to(torch.uint8)
        images = images.reshape([grid_size] * 2 + list(images.shape[1:]))
        image = images.flatten(1, 2).transpose(0, 1).flatten(1, 2)
        image = Image.fromarray(image.to('cpu').detach().numpy())
        return image

    def forward(
        self,
        input_ids: LongTensor,
        decoder_ids: LongTensor,
        attention_mask: BoolTensor,
    ) -> Tuple[FloatTensor, FloatTensor, FloatTensor]:
      encoder_state = self.encoder(input_ids)
      logits, decoder_state = self.decoder(encoder_state, decoder_ids)
      return (logits, encoder_state, decoder_state)

    def generate_image_or_tokens(
        self, 
        text: str, 
        seed: int,
        grid_size: int,
        log2_mid_count: int,
        log2_k: int = 6,
        log2_supercondition_factor: int = 3,
        is_verbose: bool = False,
        token_output = True,
        image_output=False
    ) :
        #TODO - convert to using tokenizer with masks
        assert(log2_mid_count in range(5))
        if is_verbose: print("tokenizing text")
        if type(text) is str:
          text_tokens = self.tokenizer.tokenize(text, is_verbose=is_verbose, max_len=self.text_token_count)
        else:
          text_tokens = text
        text_tokens = text_tokens.to(self.encoder.embed_tokens.weight.device)
        if is_verbose: print("encoding text tokens")
        encoder_state = self.encoder.forward(text_tokens)
        
        encoder_state, attention_mask, attention_state, image_tokens = ( 
                self.decoder.decode_initial(
                    seed, 
                    grid_size ** 2, 
                    text_tokens, 
                    encoder_state
                )
            )

        row_count = 16
        for row_index in range(row_count):
            if is_verbose: 
                print('sampling row {} of {}'.format(row_index + 1, row_count))
            if True: # with torch.cuda.amp.autocast(dtype=self.dtype):
                attention_state, image_tokens = self.decoder.decode_row(
                    row_index,
                    log2_k,
                    log2_supercondition_factor,
                    encoder_state,
                    attention_mask,
                    attention_state,
                    image_tokens
                )
            if True: # with torch.cuda.amp.autocast(dtype=torch.float32):
                if ((row_index + 1) * (2 ** log2_mid_count)) % row_count == 0:
                    tokens = image_tokens[:, 1:]
                    if token_output and not image_output:
                      yield image_tokens
                    elif image_output and not token_output:
                      yield self.image_from_tokens(grid_size, tokens, is_verbose)
                    else:
                      yield image_tokens, self.image_from_tokens(grid_size, tokens, is_verbose)
                      


    def generate(
        self, 
        text: str,
        seed: int = -1,
        grid_size: int = 1,
        log2_k: int = 6,
        log2_supercondition_factor: int = 5,
        is_verbose: bool = False,
        token_output = True,
        image_output = False
    ):
        log2_mid_count = 0
        image_or_token_stream = self.generate_image_or_tokens(
            text,
            seed,
            grid_size,
            log2_mid_count,
            log2_k,
            log2_supercondition_factor,
            is_verbose,
            token_output,
            image_output
        )
        return next(image_or_token_stream)
