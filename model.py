import math
import logging
import torch
import torch.nn as nn
from transformers import BertModel

def masked_softmax(vec, mask, dim=1):
    masked_vec = vec * mask.float()
    max_vec = torch.max(masked_vec, dim=dim, keepdim=True)[0]
    exps = torch.exp(masked_vec-max_vec)
    masked_exps = exps * mask.float()
    masked_sums = masked_exps.sum(dim, keepdim=True)
    zeros = (masked_sums == 0)
    masked_sums += zeros.float()
    return masked_exps/masked_sums


def masked_argmax(vec, mask, dim, keepdim=False):
    vec_rank = torch.argsort(torch.argsort(vec, dim=dim), dim=dim) + 1
    masked_vec_rank = vec_rank * mask.float()
    return torch.argmax(masked_vec_rank, dim=dim, keepdim=keepdim)


def masked_sum(vec, mask, dim, keepdim=False):
    return torch.sum(vec * mask.float(), dim=dim, keepdim=keepdim)


def masked_mean(vec, mask, dim, keepdim=False):
    return masked_sum(vec, mask, dim, keepdim) / torch.sum(mask.float(), dim=dim, keepdim=keepdim)


class Bert(nn.Module):
    def __init__(self, vocab_size=30522, embed_dim=768, padding_idx=0, max_length=512, drop_prob=0.1,
                 n_head=12, k_dim=64, v_dim=64, feedforward_dim=3072, n_layer=12):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.position_embeddings = nn.Embedding(max_length, embed_dim, padding_idx=padding_idx)
        self.token_type_embeddings = nn.Embedding(2, embed_dim)
        self.embedding_norm = nn.LayerNorm(embed_dim, eps=1e-12)
        self.encoder = nn.ModuleList([
            nn.ModuleDict({
                "query": nn.Linear(embed_dim, n_head * k_dim),
                "key": nn.Linear(embed_dim, n_head * k_dim),
                "value": nn.Linear(embed_dim, n_head * v_dim),
                "out": nn.Linear(n_head * v_dim, embed_dim),
                "norm": nn.LayerNorm(embed_dim, eps=1e-12),
                "ff": nn.Sequential(
                    nn.Linear(embed_dim, feedforward_dim),
                    nn.GELU(),
                    nn.Linear(feedforward_dim, embed_dim),
                ),
                "ff_norm": nn.LayerNorm(embed_dim, eps=1e-12)
            })
            for _ in range(n_layer)
        ])
        self.dropout = nn.Dropout(p=drop_prob)
        self.padding_idx = padding_idx
        self.n_head = n_head
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.embed_dim = embed_dim

    def forward(self, input_ids, attention_mask):
        batch, seq_len = input_ids.shape
        with torch.no_grad():
            h = self.forward_embedding(input_ids, batch, seq_len)
        for i, module_dict in enumerate(self.encoder):
            h = self.forward_layer(h, attention_mask, module_dict, batch, seq_len)
        return h

    def forward_embedding(self, input_ids, batch, seq_len):
        word = self.word_embeddings(input_ids)
        position_ids = torch.arange(0, seq_len, device=input_ids.device)
        position_ids = position_ids[None, :].expand(batch, -1)
        position = self.position_embeddings(position_ids)
        token_type_ids = torch.zeros_like(position_ids)
        token_type = self.token_type_embeddings(token_type_ids)
        #h = self.dropout(self.embedding_norm(word + position + token_type))
        h = self.embedding_norm(word + position + token_type)
        return h
        
    def _multiheadattn(self, h, attention_mask, module_dict, batch, seq_len):
        q = module_dict["query"](h).view(batch, seq_len, self.n_head, self.k_dim)
        k = module_dict["key"](h).view(batch, seq_len, self.n_head, self.k_dim)
        v = module_dict["value"](h).view(batch, seq_len, self.n_head, self.v_dim)
        q = q.permute(0, 2, 1, 3)  # [b, h, l, d]
        k = k.permute(0, 2, 3, 1)  # [b, h, d, l]
        a = torch.matmul(q, k) / math.sqrt(self.k_dim)                  # [b, h, l, l]
        a = masked_softmax(a, attention_mask[:, None, None, :], dim=3)  # [b, h, l, l]
        o = torch.matmul(a, v.permute(0, 2, 1, 3)).permute(0, 2, 1, 3).contiguous()
        # [b, h, l, l] x [b, h, l, d] = [b, h, l, d] -> [b, l, h, d]
        o = o.view(batch, seq_len, -1)          # [b, l, h*d]
        o = self.dropout(module_dict["out"](o)) # [b, l, h*d]
        logging.debug("The norm of vector before multihead attention: %.4f" % torch.norm(h, dim=2).mean())
        logging.debug("The norm of vector after multihead attention: %.4f" % torch.norm(o, dim=2).mean())
        h = module_dict["norm"](h + o)
        return h

    def forward_layer(self, h, attention_mask, module_dict, batch, seq_len):
        h = self._multiheadattn(h, attention_mask, module_dict, batch, seq_len)
        h = module_dict["ff_norm"](h + self.dropout(module_dict["ff"](h)))
        return h

    def load(self):
        model = BertModel.from_pretrained("bert-base-uncased")
        self.word_embeddings.load_state_dict(model.embeddings.word_embeddings.state_dict())
        self.position_embeddings.load_state_dict(model.embeddings.position_embeddings.state_dict())
        self.token_type_embeddings.load_state_dict(model.embeddings.token_type_embeddings.state_dict())
        self.embedding_norm.load_state_dict(model.embeddings.LayerNorm.state_dict())
        for t, f in zip(self.encoder, model.encoder.layer):
            t["query"].load_state_dict(f.attention.self.query.state_dict())
            t["key"].load_state_dict(f.attention.self.key.state_dict())
            t["value"].load_state_dict(f.attention.self.value.state_dict())
            t["out"].load_state_dict(f.attention.output.dense.state_dict())
            t["norm"].load_state_dict(f.attention.output.LayerNorm.state_dict())
            t["ff"][0].load_state_dict(f.intermediate.dense.state_dict())
            t["ff"][2].load_state_dict(f.output.dense.state_dict())
            t["ff_norm"].load_state_dict(f.output.LayerNorm.state_dict())


class TMix(nn.Module):
    def __init__(self, embedding_model, mixup_layer=0):
        super().__init__()
        self.embedding_model = embedding_model
        self.mixup_layer = mixup_layer

    def forward(self, input_ids, attention_mask, mixup_indices=None, lambda_=None):
        batch, seq_len = input_ids.shape
        with torch.no_grad():
            h = self.embedding_model.forward_embedding(input_ids, batch, seq_len)
        for module_dict in self.embedding_model.encoder[:self.mixup_layer]:
            h = self.embedding_model.forward_layer(h, attention_mask, module_dict, batch, seq_len)
        if mixup_indices is not None:
            h = lambda_ * h + (1 - lambda_) * h[mixup_indices]
        for module_dict in self.embedding_model.encoder[self.mixup_layer:]:
            h = self.embedding_model.forward_layer(h, attention_mask, module_dict, batch, seq_len)
        return h


class ProposedMix(nn.Module):
    def __init__(self, embedding_model, mixup_layer=0):
        super().__init__()
        self.embedding_model = embedding_model
        self.mixup_layer = mixup_layer
        self.memory_k = nn.Parameter(torch.zeros(10, self.embedding_model.embed_dim),
                                     False)
        self.memory_v = nn.Parameter(torch.zeros(10, self.embedding_model.embed_dim),
                                     False)

    def forward(self, input_ids, attention_mask, idx=None, mixup_indices=None, lambda_=None):
        batch, seq_len = input_ids.shape
        with torch.no_grad():
            h = self.embedding_model.forward_embedding(input_ids, batch, seq_len)
        for module_dict in self.embedding_model.encoder[:self.mixup_layer]:
            h = self.embedding_model.forward_layer(h, attention_mask, module_dict, batch, seq_len)
        if mixup_indices is not None:
            if idx is not None:
                h = torch.gather(h, dim=1, index=idx[:, :, None].expand(-1, -1, h.shape[2]))
                attention_mask = torch.gather(attention_mask, dim=1, index=idx)
            mix_h = torch.where(attention_mask[:, :, None], h[mixup_indices], h)
            h = lambda_ * h + (1 - lambda_) * mix_h
        for module_dict in self.embedding_model.encoder[self.mixup_layer:]:
            h = self.embedding_model.forward_layer(h, attention_mask, module_dict, batch, seq_len)
        return h


def print_gradient(grad):
    wandb.log({"Gamma gradient": grad.detach()})
    #print(grad)


class AdaMix(nn.Module):
    def __init__(self, embedding_model, mixup_layer=0):
        super().__init__()
        self.embedding_model = embedding_model
        self.policy_region_generator = nn.Sequential(
            nn.Linear(2*self.embedding_model.embed_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 3),
            nn.Softmax())
        self.mixup_layer = mixup_layer

    def forward(self, input_ids, attention_mask, mixup_indices=None, eps=None):
        batch, seq_len = input_ids.shape
        with torch.no_grad():
            h = self.embedding_model.forward_embedding(input_ids, batch, seq_len)
        for module_dict in self.embedding_model.encoder[:self.mixup_layer]:
            h = self.embedding_model.forward_layer(h, attention_mask, module_dict, batch, seq_len)
        if mixup_indices is not None:
            # Policy Region Generator
            sentence_h = h.mean(dim=1)
            policy_region = self.policy_region_generator(
                torch.cat((sentence_h, sentence_h[mixup_indices]), dim=1))  # [B, 3]
            gamma = policy_region[:, 1] * eps + policy_region[:, 0]
            mix_h = gamma[:, None, None] * h + (1 - gamma)[:, None, None] * h[mixup_indices]
            mix_h = torch.where(
                attention_mask[:, :, None] & attention_mask[mixup_indices, :, None],
                mix_h, h)

        for module_dict in self.embedding_model.encoder[self.mixup_layer:]:
            h = self.embedding_model.forward_layer(h, attention_mask, module_dict, batch, seq_len)
            if mixup_indices is not None:
                mix_h = self.embedding_model.forward_layer(mix_h, attention_mask, module_dict, batch, seq_len)

        if mixup_indices is None:
            return h
        else:
            return h, mix_h, gamma

    def predict(self, input_ids, attention_mask):
        return super().forward(input_ids=input_ids, attention_mask=attention_mask)


class SentenceClassificationModel(nn.Module):
    def __init__(self, embedding_model, n_class):
        super().__init__()
        self.embedding_model = embedding_model
        self.classifier = nn.Linear(embedding_model.embed_dim, n_class)

    def forward(self, input_ids, attention_mask):
        h = self.embedding_model(input_ids, attention_mask)
        self.h = h
        return self.classifier(masked_mean(h, attention_mask[:, :, None], dim=1))

    def load(self):
        self.embedding_model.load()


class TMixSentenceClassificationModel(nn.Module):
    def __init__(self, mix_model, n_class):
        super().__init__()
        self.mix_model = mix_model
        self.classifier = nn.Linear(self.mix_model.embedding_model.embed_dim, n_class)

    def forward(self, input_ids, attention_mask, mixup_indices=None, lambda_=None):
        h = self.mix_model(input_ids, attention_mask, mixup_indices=mixup_indices, lambda_=lambda_)
        self.h = h
        return self.classifier(masked_mean(h, attention_mask[:, :, None], dim=1))

    def load(self):
        self.mix_model.embedding_model.load()


class ProposedMixSentenceClassificationModel(nn.Module):
    def __init__(self, mix_model, n_class):
        super().__init__()
        self.mix_model = mix_model
        self.classifier = nn.Linear(self.mix_model.embedding_model.embed_dim, n_class)

    def forward(self, input_ids, attention_mask, idx=None, mixup_indices=None, lambda_=None):
        h = self.mix_model(input_ids, attention_mask, idx=idx, mixup_indices=mixup_indices, lambda_=lambda_)
        self.h = h
        if idx is not None:
            attention_mask = torch.gather(attention_mask, dim=1, index=idx)
        return self.classifier(masked_mean(h, attention_mask[:, :, None], dim=1))

    def load(self):
        self.mix_model.embedding_model.load()


class AdaMixSentenceClassificationModel(nn.Module):
    def __init__(self, mix_model, n_class):
        super().__init__()
        self.mix_model = mix_model
        self.classifier = nn.Sequential(
            nn.Linear(self.mix_model.embedding_model.embed_dim, 128),
            nn.Tanh(),
            nn.Linear(128, n_class)
        )
        self.intrusion_classifier = nn.Sequential(
            nn.Linear(self.mix_model.embedding_model.embed_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 2)
        )

    def forward(self, input_ids, attention_mask, mixup_indices=None, eps=None):
        if mixup_indices is None:
            h = self.mix_model(input_ids, attention_mask)
            return self.classifier(torch.mean(h, dim=1))
        else:
            h, mix_h, gamma = self.mix_model(input_ids, attention_mask, mixup_indices, eps)
            out = self.classifier(masked_mean(h, attention_mask[:, :, None], dim=1))
            mix_out = self.classifier(masked_mean(mix_h, attention_mask[:, :, None], dim=1))
            return out, mix_out, gamma
        
    def predict_intrusion(self, input_ids, attention_mask, mixup_indices, eps):
        h, mix_h, gamma = self.mix_model(input_ids, attention_mask, mixup_indices, eps)
        # Intrusion Discriminator
        intr = self.intrusion_classifier(masked_mean(h, attention_mask[:, :, None], dim=1))
        mix_intr = self.intrusion_classifier(masked_mean(mix_h, attention_mask[:, :, None], dim=1))
        return intr, mix_intr

    def load(self):
        self.mix_model.embedding_model.load()


def create_model(vocab_size=30522, embed_dim=768, padding_idx=0, drop_prob=0.1, n_head=12,
                 k_dim=64, v_dim=64, feedforward_dim=3072, n_layer=12, augment='none', 
                 mixup_layer=3, n_class=4):
    embedding_model = Bert(vocab_size=vocab_size, embed_dim=embed_dim, padding_idx=padding_idx,
                           drop_prob=drop_prob, n_head=n_head, k_dim=k_dim, v_dim=v_dim,
                           feedforward_dim=feedforward_dim, n_layer=n_layer)
    if augment == "none":
        model = SentenceClassificationModel(embedding_model, n_class)
    elif augment == "proposed":
        embedding_model = ProposedMix(embedding_model, mixup_layer=mixup_layer)
        model = ProposedMixSentenceClassificationModel(embedding_model, n_class)
    elif augment == "tmix":
        embedding_model = TMix(embedding_model, mixup_layer=mixup_layer)
        model = TMixSentenceClassificationModel(embedding_model, n_class)
    elif augment == "adamix":
        embedding_model = AdaMix(embedding_model, mixup_layer=mixup_layer)
        model = AdaMixSentenceClassificationModel(embedding_model, n_class)
    else:
        raise AttributeError("Invalid augment")
    return model
