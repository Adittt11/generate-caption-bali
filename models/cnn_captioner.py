"""CNN encoders with Transformer or non-attention LSTM caption decoders."""

import torch
from torch import nn
from torchvision import models, transforms


SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")


class CNNEncoder(nn.Module):
    def __init__(self, backbone="resnet50", pretrained=True):
        super().__init__()
        self.backbone_name = backbone
        if backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            network = models.resnet50(weights=weights)
            self.features = nn.Sequential(*list(network.children())[:-2])
            self.feature_dim = network.fc.in_features
        elif backbone == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            network = models.efficientnet_b0(weights=weights)
            self.features = network.features
            self.feature_dim = network.classifier[1].in_features
        else:
            raise ValueError(f"Backbone tidak didukung: {backbone}")

        self.freeze()

    def freeze(self):
        for parameter in self.features.parameters():
            parameter.requires_grad = False

    def unfreeze_last_block(self):
        self.freeze()
        for parameter in self.features[-1].parameters():
            parameter.requires_grad = True

    def forward(self, images):
        return self.features(images).flatten(2).transpose(1, 2)


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, feature_dim, d_model=512, layers=4, heads=8, max_length=16):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(max_length, d_model)
        self.feature_projection = nn.Linear(feature_dim, d_model)
        self.image_position_embedding = nn.Parameter(torch.zeros(1, d_model, 7, 7))
        nn.init.trunc_normal_(self.image_position_embedding, std=0.02)
        block = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(block, num_layers=layers)
        self.output = nn.Linear(d_model, vocab_size)
        self.max_length = max_length

    def project_image_grid(self, memory):
        grid_size = int(memory.shape[1] ** 0.5)
        visual = self.feature_projection(memory)
        visual = visual.transpose(1, 2).reshape(memory.shape[0], -1, grid_size, grid_size)
        positions = self.image_position_embedding
        if positions.shape[-2:] != (grid_size, grid_size):
            positions = torch.nn.functional.interpolate(
                positions, size=(grid_size, grid_size), mode="bilinear", align_corners=False
            )
        return (visual + positions).flatten(2).transpose(1, 2)

    def forward(self, memory, tokens):
        length = tokens.shape[1]
        positions = torch.arange(length, device=tokens.device).unsqueeze(0)
        target = self.token_embedding(tokens) + self.position_embedding(positions)
        target_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=tokens.device), diagonal=1
        )
        decoded = self.decoder(
            target,
            self.project_image_grid(memory),
            tgt_mask=target_mask,
            tgt_key_padding_mask=tokens.eq(0),
        )
        return self.output(decoded)


class LSTMDecoder(nn.Module):
    """CNN-LSTM baseline conditioned on pooled image features, without attention."""

    def __init__(self, vocab_size, feature_dim, d_model=512, layers=2):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.image_projection = nn.Linear(feature_dim, d_model)
        self.initial_hidden = nn.Linear(d_model, d_model * layers)
        self.initial_cell = nn.Linear(d_model, d_model * layers)
        self.layers = layers
        self.lstm = nn.LSTM(d_model * 2, d_model, num_layers=layers, batch_first=True, dropout=0.1)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, memory, tokens):
        image = self.image_projection(memory.mean(dim=1))
        batch_size = image.shape[0]
        hidden = self.initial_hidden(image).view(batch_size, self.layers, -1).transpose(0, 1).contiguous()
        cell = self.initial_cell(image).view(batch_size, self.layers, -1).transpose(0, 1).contiguous()
        words = self.token_embedding(tokens)
        repeated_image = image.unsqueeze(1).expand(-1, words.shape[1], -1)
        decoded, _ = self.lstm(torch.cat((words, repeated_image), dim=-1), (hidden, cell))
        return self.output(decoded)


class ImageCaptioner(nn.Module):
    def __init__(self, vocab_size, backbone="resnet50", decoder="transformer",
                 pretrained=True, d_model=512, decoder_layers=4, attention_heads=8,
                 max_length=16):
        super().__init__()
        self.backbone_name = backbone
        self.decoder_name = decoder
        self.encoder = CNNEncoder(backbone, pretrained=pretrained)
        if decoder == "transformer":
            self.decoder = TransformerDecoder(
                vocab_size, self.encoder.feature_dim, d_model, decoder_layers,
                attention_heads, max_length,
            )
        elif decoder == "lstm":
            self.decoder = LSTMDecoder(vocab_size, self.encoder.feature_dim, d_model, decoder_layers)
        else:
            raise ValueError(f"Decoder tidak didukung: {decoder}")

    def forward(self, images, tokens):
        return self.decoder(self.encoder(images), tokens)

    def next_logits(self, images, tokens):
        return self.forward(images, tokens)[:, -1, :]


def beam_search(model, image, word_to_id, id_to_word, device, beam_size=3,
                min_words=10, max_words=15):
    """Generate one caption with autoregressive beam search."""
    model.eval()
    image = image.unsqueeze(0).to(device)
    bos, eos = word_to_id["<bos>"], word_to_id["<eos>"]
    beams = [([bos], 0.0, False)]
    with torch.inference_mode():
        for _ in range(max_words + 1):
            candidates = []
            for tokens, score, finished in beams:
                if finished:
                    candidates.append((tokens, score, True))
                    continue
                token_tensor = torch.tensor([tokens], device=device)
                log_probs = torch.log_softmax(model.next_logits(image, token_tensor)[0], dim=-1)
                log_probs[word_to_id["<pad>"]] = -torch.inf
                log_probs[bos] = -torch.inf
                if len(tokens) - 1 < min_words:
                    log_probs[eos] = -torch.inf
                values, indices = torch.topk(log_probs, min(beam_size, log_probs.numel()))
                for value, index in zip(values.tolist(), indices.tolist()):
                    new_tokens = tokens + [index]
                    candidates.append((new_tokens, score + value, index == eos))
            beams = sorted(
                candidates,
                key=lambda item: item[1] / max(len(item[0]) - 1, 1) ** 0.7,
                reverse=True,
            )[:beam_size]
            if all(item[2] for item in beams):
                break

    best = beams[0][0][1:]
    words = [id_to_word[str(token)] for token in best if token != eos]
    return " ".join(words[:max_words]).strip()


def image_transform(image_size=224, augment=False):
    operations = [transforms.Resize((image_size, image_size))]
    if augment:
        operations += [transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.1, 0.1, 0.1)]
    operations += [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
    return transforms.Compose(operations)
