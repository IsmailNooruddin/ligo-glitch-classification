"""
Module: vlm_models.py
Purpose: Step 16 (Novelty 2) — Vision-Language Model (OpenAI CLIP ViT-B/16) Wrapper,
         Domain-Engineered LIGO Physics Prompts, Text-Initialized Fine-Tuning Adapter,
         and Frozen-Embedding MLP Probe.
Author: L4 Group 23 (Ismail Nooruddin, Munir Tariq, Ramsha Arshad)

Key Design Features:
- Uses `openai/clip-vit-base-patch16` (~149.6M total params: ViT-B/16 vision encoder +
  Transformer text encoder in a shared 512-dim L2-normalized multimodal space), enabling
  a direct apples-to-apples comparison against Step 15's ImageNet-pretrained `ViT-B/16`.
- Uses version-safe Hugging Face calls (`vision_model(...).pooler_output -> visual_projection`
  and `text_model(...).pooler_output -> text_projection`) compatible across transformers v4 & v5.
- Provides both Basic class-name prompts and Physics-Enriched Gravity Spy prompt ensembles
  (4 domain-specific prompts averaged per class in L2-normalized CLIP text space).
- Supports offline `--smoke-test` verification via `use_mock=True` (never triggered in full runs).
"""

import hashlib
import math
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


CLIP_MODEL_ID = "openai/clip-vit-base-patch16"

# Official OpenAI CLIP channel normalization constants
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# Human-readable names for the 14 shared Gravity Spy glitch classes
CLASS_READABLE_NAMES = {
    "Blip": "Blip",
    "Chirp": "Chirp",
    "Extremely_Loud": "Extremely Loud",
    "Koi_Fish": "Koi Fish",
    "Low_Frequency_Burst": "Low Frequency Burst",
    "Low_Frequency_Lines": "Low Frequency Lines",
    "No_Glitch": "No Glitch",
    "None_of_the_Above": "None of the Above",
    "Power_Line": "Power Line",
    "Repeating_Blips": "Repeating Blips",
    "Scattered_Light": "Scattered Light",
    "Scratchy": "Scratchy",
    "Tomte": "Tomte",
    "Violin_Mode": "Violin Mode",
}

# Basic zero-shot prompt per class (ablation baseline for prompt engineering)
BASIC_PROMPTS = {
    cls: [f"a LIGO gravitational-wave spectrogram showing a {readable} glitch"]
    for cls, readable in CLASS_READABLE_NAMES.items()
}

# Domain-engineered Gravity Spy physics & morphology prompt ensemble (4 prompts per class)
PHYSICS_PROMPT_ENSEMBLE = {
    "Blip": [
        "a time-frequency spectrogram showing a single brief teardrop-shaped vertical spike glitch in the center",
        "a LIGO spectrogram of a Blip glitch: one short narrow vertical burst lasting less than a tenth of a second",
        "a spectrogram with one isolated vertical flame-like impulse centered at zero seconds against dark noise",
        "a gravitational-wave spectrogram featuring a solitary short vertical teardrop transient spike",
    ],
    "Chirp": [
        "a time-frequency spectrogram showing an upward sweeping curved chirp track rising rapidly in frequency over time",
        "a LIGO spectrogram of a gravitational-wave Chirp signal curving steeply upward from low to high frequency",
        "a spectrogram with a bright thin J-shaped upward-sweeping binary black hole inspiral curve",
        "a spectrogram depicting a narrow rising exponential frequency sweep accelerating upward",
    ],
    "Extremely_Loud": [
        "a time-frequency spectrogram completely flooded and saturated with massive bright energy across all frequencies",
        "a LIGO spectrogram of an Extremely Loud glitch blinding the entire central frame with intense brightness",
        "an overwhelmingly bright high-energy spectrogram dominated by a huge blinding blast covering low and high frequencies",
        "a saturated gravitational-wave spectrogram overwhelmed by an enormous bright central explosion of energy",
    ],
    "Koi_Fish": [
        "a time-frequency spectrogram showing a tall bright vertical central spike with wide low-frequency fins on both sides",
        "a LIGO spectrogram of a Koi Fish glitch shaped like a fish head with a broad base and tall vertical body",
        "a spectrogram featuring a strong vertical burst with symmetrical pectoral-fin flares near the bottom",
        "a high-energy vertical spectrogram transient with wide triangular wings extending horizontally at low frequency",
    ],
    "Low_Frequency_Burst": [
        "a time-frequency spectrogram showing a compact bright cloud or mound of energy strictly at the very bottom low frequencies",
        "a LIGO spectrogram of a Low Frequency Burst localized as a single glowing blob along the bottom edge",
        "a spectrogram with a short bright low-frequency hump confined below 64 Hz at the bottom center",
        "a gravitational-wave spectrogram containing an isolated glowing patch at the lowest frequency band",
    ],
    "Low_Frequency_Lines": [
        "a time-frequency spectrogram showing persistent flat horizontal stripes running across the very bottom low-frequency edge",
        "a LIGO spectrogram of Low Frequency Lines appearing as continuous narrow horizontal bands near the bottom",
        "a spectrogram with long steady horizontal streaks extending left to right along the lowest frequencies",
        "a gravitational-wave spectrogram dominated by flat horizontal low-frequency noise lines spanning the width",
    ],
    "No_Glitch": [
        "a uniform smooth background noise spectrogram with no bright glitch pattern or distinct transient shape",
        "a LIGO spectrogram of No Glitch showing only faint even random background speckle across the entire image",
        "an empty quiet time-frequency spectrogram containing purely flat featureless background noise",
        "a dark uniform gravitational-wave spectrogram with no localized bright artifact or line",
    ],
    "None_of_the_Above": [
        "an unusual anomalous unclassified noise artifact in a time-frequency spectrogram with irregular atypical geometry",
        "a LIGO spectrogram of None of the Above showing a rare irregular transient that does not match standard glitch shapes",
        "a peculiar non-standard gravitational-wave spectrogram glitch with asymmetric complex scattered structure",
        "an atypical outlier transient pattern in a scientific time-frequency spectrogram",
    ],
    "Power_Line": [
        "a time-frequency spectrogram showing a sharp thin straight horizontal line across the frame at 60 Hz",
        "a LIGO spectrogram of a Power Line glitch appearing as a crisp narrow horizontal stripe in the lower-middle band",
        "a spectrogram with a single flat narrowband electrical mains frequency horizontal line spanning across time",
        "a gravitational-wave spectrogram featuring a razor-thin continuous horizontal line at constant low-mid frequency",
    ],
    "Repeating_Blips": [
        "a time-frequency spectrogram showing multiple short vertical blip spikes repeating in a row across time",
        "a LIGO spectrogram of Repeating Blips appearing as a periodic train of several vertical teardrop spikes",
        "a spectrogram with a sequence of closely spaced vertical bursts repeating side by side horizontally",
        "a gravitational-wave spectrogram containing several recurring short vertical spikes lined up in time",
    ],
    "Scattered_Light": [
        "a time-frequency spectrogram showing smooth curved rainbow-like arches or rounded hills in the lower half",
        "a LIGO spectrogram of Scattered Light appearing as stacked wavy curved arches rising and falling smoothly",
        "a spectrogram with repeating semicircular dome-shaped arches caused by optical light scattering at low frequencies",
        "a gravitational-wave spectrogram featuring gentle smooth parabolic arcs and concentric curved waves",
    ],
    "Scratchy": [
        "a time-frequency spectrogram showing a dense jagged cluster of many sharp closely packed high-frequency spikes",
        "a LIGO spectrogram of a Scratchy glitch resembling rapid jagged scratch marks or serrated mountain peaks",
        "a spectrogram filled with a noisy horizontal band of rapid choppy vertical needles across middle frequencies",
        "a gravitational-wave spectrogram exhibiting a dense spiky comb-like band of rapid fluctuations",
    ],
    "Tomte": [
        "a time-frequency spectrogram showing a short triangular pyramid or cone-shaped burst pointed at the top",
        "a LIGO spectrogram of a Tomte glitch shaped like a broad flat-bottomed triangle or hat centered in time",
        "a spectrogram with a single short conical mountain-shaped spike wider at the low-frequency base than the tip",
        "a gravitational-wave spectrogram containing a bright short triangular impulse reminiscent of a gnome hat",
    ],
    "Violin_Mode": [
        "a time-frequency spectrogram showing a bright narrow horizontal harmonic resonance line high up near 500 Hz",
        "a LIGO spectrogram of a Violin Mode harmonic appearing as a sharp thin horizontal dash in the upper frequency region",
        "a spectrogram with a concentrated high-frequency narrowband horizontal line from mirror suspension fiber ringing",
        "a gravitational-wave spectrogram featuring a crisp bright horizontal streak in the upper half of the frequency axis",
    ],
}


def _get_viridis_cmap():
    """Returns Matplotlib's viridis colormap safely across Matplotlib 3.5 through 3.11+."""
    try:
        return matplotlib.colormaps["viridis"]
    except (AttributeError, KeyError):
        import matplotlib.pyplot as plt
        return plt.get_cmap("viridis")


def grayscale_to_viridis_rgb(x_gray):
    """
    Converts grayscale spectrograms in [0, 1] of shape (N, 1, H, W) or (N, H, W)
    into Viridis RGB arrays of shape (N, 3, H, W) float32 in [0, 1].
    Useful for zero-shot CLIP evaluation since scientific spectrograms on the web
    are predominantly rendered in Viridis colormaps.
    """
    arr = np.asarray(x_gray, dtype=np.float32)
    if arr.ndim == 4:
        if arr.shape[1] != 1:
            raise ValueError(f"Expected 1-channel input for Viridis mapping, got shape {arr.shape}")
        arr = arr[:, 0, :, :]
    elif arr.ndim != 3:
        raise ValueError(f"Expected 3D or 4D input, got shape {arr.shape}")

    arr = np.clip(arr, 0.0, 1.0)
    cmap = _get_viridis_cmap()
    # Process in chunks to keep peak memory small
    out = np.empty((arr.shape[0], 3, arr.shape[1], arr.shape[2]), dtype=np.float32)
    chunk = 256
    for i in range(0, arr.shape[0], chunk):
        rgba = cmap(arr[i : i + chunk])  # (b, H, W, 4) float64
        out[i : i + chunk] = np.transpose(rgba[..., :3].astype(np.float32), (0, 3, 1, 2))
    return out


class _MockCLIPCore(nn.Module):
    """
    Lightweight deterministic CLIP-compatible core ONLY for offline `--smoke-test` when
    Hugging Face `transformers` is not installed locally. Never used in full runs.
    """
    def __init__(self, embed_dim=512):
        super().__init__()
        self.embed_dim = embed_dim
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, embed_dim),
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(20.0), dtype=torch.float32))

    def encode_images_raw(self, pixel_values):
        return self.vision_encoder(pixel_values)

    def encode_texts_raw(self, text_list, device):
        vecs = []
        for t in text_list:
            digest = hashlib.sha256(t.encode("utf-8")).digest()
            seed = int.from_bytes(digest[:4], "little")
            rng = np.random.RandomState(seed)
            vecs.append(rng.randn(self.embed_dim).astype(np.float32))
        return torch.from_numpy(np.stack(vecs, axis=0)).to(device)


class CLIPVisionTextWrapper(nn.Module):
    """
    Unified wrapper around Hugging Face `openai/clip-vit-base-patch16` providing:
    - GPU-accelerated bicubic resizing to (224, 224) + exact OpenAI CLIP normalization
    - L2-normalized 512-dim image embeddings (`encode_images`)
    - L2-normalized 512-dim text prompt embeddings & class prototypes (`build_class_text_prototypes`)
    - Version-safe forward calls compatible across transformers v4.x and v5.x.
    """
    def __init__(self, model_id=CLIP_MODEL_ID, input_size=(224, 224), use_mock=False):
        super().__init__()
        self.model_id = model_id
        self.input_size = input_size
        self.use_mock = use_mock

        mean = torch.tensor(CLIP_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(CLIP_STD, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("clip_mean", mean)
        self.register_buffer("clip_std", std)

        if use_mock:
            self.mock_core = _MockCLIPCore(embed_dim=512)
            self.clip = None
            self.tokenizer = None
            self.embed_dim = 512
        else:
            try:
                from transformers import CLIPModel, CLIPTokenizer
            except ImportError as exc:
                raise ImportError(
                    "Hugging Face `transformers` is required to run real CLIP ViT-B/16. "
                    "Install via `pip install transformers` or run on Google Colab."
                ) from exc

            self.tokenizer = CLIPTokenizer.from_pretrained(model_id)
            self.clip = CLIPModel.from_pretrained(model_id)
            self.embed_dim = int(self.clip.config.projection_dim)
            self.mock_core = None

    def _preprocess(self, x):
        """Ensures 3 channels, bicubic resizes to (224, 224), clamps [0,1], and applies CLIP norm."""
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] != 3:
            raise ValueError(f"Expected 1 or 3 channels, got shape {tuple(x.shape)}")

        if x.shape[2:] != self.input_size:
            x = F.interpolate(x, size=self.input_size, mode="bicubic", align_corners=False)
        x = x.clamp(0.0, 1.0)
        x = (x - self.clip_mean) / self.clip_std
        return x

    def encode_images(self, x, normalize=True):
        """Returns (B, 512) CLIP visual embeddings (L2-normalized by default)."""
        px = self._preprocess(x)
        if self.use_mock:
            emb = self.mock_core.encode_images_raw(px)
        else:
            vision_out = self.clip.vision_model(pixel_values=px)
            emb = self.clip.visual_projection(vision_out.pooler_output)
        if normalize:
            emb = F.normalize(emb.float(), dim=-1)
        return emb

    @torch.no_grad()
    def encode_texts(self, text_list, device=None, normalize=True):
        """Returns (M, 512) CLIP text embeddings (L2-normalized by default)."""
        if device is None:
            device = self.clip_mean.device
        if self.use_mock:
            emb = self.mock_core.encode_texts_raw(text_list, device=device)
        else:
            tokens = self.tokenizer(
                text_list,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(device)
            text_out = self.clip.text_model(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
            )
            emb = self.clip.text_projection(text_out.pooler_output)
        if normalize:
            emb = F.normalize(emb.float(), dim=-1)
        return emb

    @torch.no_grad()
    def build_class_text_prototypes(self, class_names, prompt_mode="physics", device=None):
        """
        Builds L2-normalized class text prototype matrix of shape (C, 512).
        - prompt_mode='basic'   : Single class-name prompt per class
        - prompt_mode='physics' : 4 domain-crafted Gravity Spy physics prompts averaged per class
        """
        prompt_bank = BASIC_PROMPTS if prompt_mode == "basic" else PHYSICS_PROMPT_ENSEMBLE
        if device is None:
            device = self.clip_mean.device

        prototypes = []
        for cls_name in class_names:
            if cls_name not in prompt_bank:
                raise KeyError(f"Class '{cls_name}' missing from prompt_bank '{prompt_mode}'.")
            prompts = prompt_bank[cls_name]
            emb = self.encode_texts(prompts, device=device, normalize=True)  # (K, 512)
            proto = F.normalize(emb.mean(dim=0, keepdim=True), dim=-1)       # (1, 512)
            prototypes.append(proto)
        return torch.cat(prototypes, dim=0)  # (C, 512)

    def get_logit_scale(self):
        """Returns clamped temperature scale exp(logit_scale) <= 100.0."""
        raw = self.mock_core.logit_scale if self.use_mock else self.clip.logit_scale
        return raw.exp().clamp(max=100.0)

    def vision_parameters(self):
        """Returns trainable parameters belonging strictly to the vision encoder + visual projection."""
        if self.use_mock:
            return list(self.mock_core.vision_encoder.parameters())
        return list(self.clip.vision_model.parameters()) + list(self.clip.visual_projection.parameters())

    def freeze_text_encoder(self):
        """Freezes the text encoder parameters so end-to-end fine-tuning focuses solely on visual adaptation."""
        if not self.use_mock and self.clip is not None:
            for p in self.clip.text_model.parameters():
                p.requires_grad = False
            for p in self.clip.text_projection.parameters():
                p.requires_grad = False
            self.clip.logit_scale.requires_grad = False


class CLIPTextInitializedAdapter(nn.Module):
    """
    Modern Prompt-Initialized Vision-Language Classifier (WiSE-FT / Prompt-Init paradigm):
    - Initializes the cosine classification head directly from CLIP's physics text prototypes,
      so at Epoch 0 the model starts as the exact zero-shot Vision-Language classifier!
    - Fine-tunes the CLIP vision encoder gently (`backbone_lr=1.5e-5`) alongside the cosine
      class prototypes (`head_lr=5e-4`) and learnable temperature scale (`<= 100.0`).
    """
    def __init__(self, vlm_wrapper, text_prototypes, init_scale=25.0):
        super().__init__()
        self.vlm = vlm_wrapper
        self.vlm.freeze_text_encoder()
        self.embed_dim = int(text_prototypes.shape[1])
        self.num_classes = int(text_prototypes.shape[0])

        # Learnable cosine class prototypes initialized from normalized physics text embeddings
        self.class_prototypes = nn.Parameter(F.normalize(text_prototypes.detach().float().clone(), dim=-1))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(float(init_scale)), dtype=torch.float32))

    def extract_embeddings(self, x):
        """Returns L2-normalized fine-tuned 512-dim CLIP image embeddings."""
        return self.vlm.encode_images(x, normalize=True)

    def forward(self, x):
        """Computes scaled cosine similarity logits of shape (B, num_classes)."""
        img_emb = self.extract_embeddings(x)
        proto = F.normalize(self.class_prototypes, dim=-1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        return scale * (img_emb @ proto.T)

    def get_param_groups(self, backbone_lr=1.5e-5, head_lr=5e-4, weight_decay=1e-2):
        """Returns differential AdamW param groups for gentle CLIP vision adaptation."""
        return [
            {"params": self.vlm.vision_parameters(), "lr": backbone_lr, "weight_decay": weight_decay},
            {"params": [self.class_prototypes], "lr": head_lr, "weight_decay": weight_decay},
            {"params": [self.logit_scale], "lr": head_lr, "weight_decay": 0.0},
        ]


class FrozenCLIPMLPProbe(nn.Module):
    """Lightweight 2-layer MLP classifier trained on frozen 512-dim CLIP embeddings."""
    def __init__(self, input_dim=512, hidden_dim=256, num_classes=14, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def count_trainable_parameters(model):
    """Returns count of trainable parameters in a PyTorch module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    class_list = sorted(CLASS_READABLE_NAMES.keys())
    assert len(class_list) == 14, f"Expected 14 classes, got {len(class_list)}"
    for c in class_list:
        assert c in BASIC_PROMPTS and len(BASIC_PROMPTS[c]) == 1
        assert c in PHYSICS_PROMPT_ENSEMBLE and len(PHYSICS_PROMPT_ENSEMBLE[c]) == 4

    # Verify Viridis converter
    dummy_gray = np.random.rand(4, 1, 140, 170).astype(np.float32)
    vir_rgb = grayscale_to_viridis_rgb(dummy_gray)
    assert vir_rgb.shape == (4, 3, 140, 170) and 0.0 <= vir_rgb.min() <= vir_rgb.max() <= 1.0

    # Verify Mock CLIP wrapper + Text-Init Adapter + MLP Probe
    vlm = CLIPVisionTextWrapper(use_mock=True)
    protos = vlm.build_class_text_prototypes(class_list, prompt_mode="physics")
    assert protos.shape == (14, 512)

    adapter = CLIPTextInitializedAdapter(vlm, protos)
    x = torch.rand(2, 3, 140, 170)
    logits = adapter(x)
    assert logits.shape == (2, 14)
    logits.sum().backward()

    probe = FrozenCLIPMLPProbe(input_dim=512, hidden_dim=256, num_classes=14)
    p_out = probe(torch.randn(4, 512))
    assert p_out.shape == (4, 14)
    print(f"[OK] vlm_models.py unit tests passed! Adapter trainable params (mock)={count_trainable_parameters(adapter):,}")
