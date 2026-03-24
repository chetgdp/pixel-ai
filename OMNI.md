# OMNI: Omni-Modal Neural Input to One Pixel

## The Problem

pixel.ai maps arbitrary byte input to a single RGBA pixel (4 bytes). The forward pass works — 10M i8 parameters running in ~2ms via WebGL2 fragment shader hacking. But the network has no understanding of what the bytes mean. "tree" and "oak" produce completely unrelated colors because the input stage (fold_bytes) is a lossy XOR hash that destroys semantic relationships.

The question: can a tiny network running in a browser learn to map any byte sequence to a meaningful color?

## Prior Art

### Nous Research — Large Information Models

Source: Bowen Peng (chief scientist, Nous Research) live presentation. No published paper yet.

Bowen coined the term "Large Information Model" (LIM) — a tribute to LLMs, but modeling raw information (bytes) instead of language. The thesis: language is not necessarily a precursor to intelligence. Modeling raw information from the universe could unlock capabilities we haven't considered — DNA generation from drawings, robot control via digital signals, multiplayer game agents via raw network packets, discovering compression algorithms via RL.

The experiment: a 600M parameter model trained on raw BMP file bytes. Vocab = 256 (one per byte value) + a few special tokens. No tokenizer. They tried ~20 models before one worked. Dataset: DataComp Common Pool, 300GB subset. Images resized to ~48px (~8KB each). Pre-trained at 32K context (4 images per data sample). Trained 1 day on 64 B200s.

Results: the model generates valid BMP files with correct headers and coherent (blurry, early-DALL-E-quality) images. Non-cherrypicked samples show recognizable buildings, people, cars, gyms. It can do file completion (truncate a BMP at the middle, let it finish) and conditional generation (give it a wide-format BMP header, it generates wide landscape-like images; tall headers produce phone-photo-like portraits). They verified generations are not memorized from the training set.

**Lighthouse Attention** — a new hierarchical attention mechanism, paper "coming soonish" (Bowen's words). Compatible with Flash Attention 2, giving 4x training speed vs full attention. Designed so different parts of the model attend at different granularities — dense attention for text, sparser attention for images/video/long context. Still unpublished. The concept exists under other names in the literature: MEGABYTE (Meta, 2023), MBLM (IBM, 2025), BLT's dynamic patching.

Bowen identified four bottlenecks for information modeling:
1. **Wildly different granularities** — text is info-dense, speech is megabytes of sparse waveform. Cross-entropy loss is inefficient for sparse modalities.
2. **Huge context lengths** — not just quadratic attention. The hard part is doing both long and short context well simultaneously. Linear attention might work but could hurt short-context performance.
3. **Slow inference** — autoregressive byte-by-byte through the full model. Solution: MoE-like routing (go through a subset per byte, not the full model).
4. **Generation order** — not everything is left-to-right. Drawings start with face/body, not top-left pixel. Any-order autoregressive and diffusion are being explored, but diffusion struggles with discrete tokens.

On semantic representations (from Q&A): Bowen believes they emerge internally at scale without explicit extraction. "If you scale big enough the model would be able to kind of have this internal representation of buildings... it's just not extractable... it's completely smeared between five six bytes." This is the "scale through it" philosophy.

Nous's published byte-level work: "Distilling Token-Trained Models into Byte-Level Models" (arXiv:2602.01007, Feb 2026) — converting existing token-trained LLMs to byte-level via 2-stage curriculum (Progressive Knowledge Distillation + Byte-Level SFT), retaining >92% of teacher performance using ~125B bytes.

None of the four bottlenecks apply to pixel.ai — our output is 4 bytes, one forward pass, no generation.

### Meta — Byte Latent Transformer (BLT)

arXiv:2412.09871, accepted ACL 2025.

Three-module architecture:
- **Local Encoder** — lightweight byte-level Transformer with local windowed attention. Each byte embedding is augmented with **hash n-gram features (n=3..8)**: for each byte, n-grams from preceding bytes are hashed into a fixed-size embedding table and summed. Cross-attention pools byte representations into patch representations.
- **Latent Transformer** — the heavy component. Standard autoregressive Transformer over the (much shorter) patch sequence. This is where most FLOPs go.
- **Local Decoder** — reverse cross-attention. Patches as keys/values, bytes as queries. Predicts next byte.

**Entropy-based dynamic patching**: a small pre-trained byte LM predicts next-byte entropy. High-entropy positions (hard to predict, e.g., first letter of a new word) become patch boundaries. Low-entropy sequences ("tion" after "na") get grouped into long patches. The model allocates more compute to surprising regions.

Scale: first FLOP-controlled scaling study up to **8B params, 4T training bytes**. Matches Llama 3 with up to **50% fewer inference FLOPs**.

The learned compression (patching) is essentially what our fold_bytes does statically — BLT learns it, we hardcode it. BLT's hash n-gram features (n=3..8) are directly applicable to improving fold_bytes — this is the most actionable technique from the literature.

### The Byte-Level Field (2024-2026)

| Model | Org | Params | Key Innovation | Reference |
|-------|-----|--------|---------------|-----------|
| BLT | Meta | up to 8B | Entropy-based dynamic patching, hash n-gram embeddings | arXiv:2412.09871 |
| EvaByte | HKU/SambaNova | 6.5B | EVA attention + multibyte prediction, rivals tokenized LMs with 5x less data, 2x faster decoding | github.com/OpenEvaByte/evabyte |
| Bolmo | Allen AI | 1B, 7B | "Byteification" via distillation from OLMo 3, <1% pretraining budget. Bolmo 7B beats BLT 7B by +16.5% on STEM | arXiv:2512.15586 |
| ByteFlow | — | 600M, 1.3B | Learned segmentation via coding-rate criterion (information-theoretic). Top-K selection for adaptive boundaries without dynamic control flow | arXiv:2603.03583 |
| MBLM | IBM | varies | Unlimited hierarchy depth, hybrid Mamba+Transformer, 5M byte contexts on 1 GPU. First byte-level VQA results | arXiv:2502.14553 |
| bGPT | Microsoft/ZJU | ~110M | Hierarchical Transformer on raw binary. BMP, MIDI, CPU ops. 0.0011 bpb on ABC-to-MIDI, >99.99% CPU sim accuracy | arXiv:2402.19155 |
| MambaByte | — | varies | SSM (Mamba) on raw bytes, no patching needed | COLM 2024 |
| SpaceByte | — | up to 800M | Global Transformer blocks only at word boundaries (spaces) | NeurIPS 2024 |
| ByT5 | Google | 300M-12.9B | Direct byte-to-byte T5 | arXiv:2105.13626 |

The field has reached an inflection point. Multiple groups have independently shown byte-level models matching tokenized counterparts. The emerging consensus: with hierarchical encoding, learned compression, and efficient attention, byte-level models are not a compromise — they match or exceed tokenized models while gaining robustness, simplicity, and universality.

## Our Advantage

pixel.ai sidesteps the hardest problems in omni-modal modeling:

| Problem | Omni-modal (Nous etc.) | pixel.ai |
|---------|----------------------|----------|
| Output complexity | Generate entire files, correct headers, variable length | 4 bytes. One forward pass. |
| Generation order | Left-to-right? Diffusion? Any-order? | N/A — no sequential generation |
| Inference speed | Autoregressive, one byte at a time through entire model | Single forward pass, ~2ms |
| Context length | Megabytes of output context | Zero output context |
| Architecture | Need hierarchical attention, MoE, etc. | Simple MLP works |

The only hard problem we share: learning meaningful representations from raw byte input.

### Is 10M Params Enough?

10M is on the extreme low end for *discovering* byte-level semantics from scratch. The smallest published model with documented semantic success is ByT5-Small at 300M (30x larger). bGPT at 110M learned binary format structure but not open-ended semantics. The "Super Tiny Language Models" paper (arXiv:2405.14159) studies 10-100M param byte-level models and finds pre-training is critical even at tiny scale.

**However**, pixel.ai's task is radically simpler than language modeling. We don't need the network to "understand" language — we need it to learn associations like "ocean" → blue. This is closer to classification/regression (mapping inputs to ~100-1000 color clusters) than to language generation.

The bottleneck is the fold, not network capacity. With the current XOR hash destroying all semantic structure, even 100M params in the MLP wouldn't help — "tree" and "oak" still hash to unrelated vectors. With a good fold (learned or n-gram-based), the existing 10M MLP should be enough for text-to-color.

## Four Problems to Solve

### 1. The Fold Problem (Representation)

Current state: `fold_bytes` is a deterministic XOR hash that compresses arbitrary-length input into a fixed [i8; 4096] vector. It preserves no semantic information — "tree" and "oak" hash to completely unrelated vectors.

The golden ratio constant `0x9E3779B9` (floor(2^32 / phi)) gives good dispersion of sequential keys via Fibonacci hashing (Knuth, TAOCP Vol. 3), but the overall fold still destroys semantic structure because: (1) byte values are added then XORed with a position hash — partially canceling each other's information, (2) the modular reduction `state[i] % 128` loses most accumulated information, (3) no local structure (n-grams) is captured. The constant itself is fine for position differentiation; the problem is the overall fold design.

What we need: similar inputs should produce similar representations. Options ranked by complexity:

**Tier 1: N-gram fold with random projection (zero parameters, no training)**

Replace the XOR hash with a SimHash-style random projection from byte n-gram space. For each byte bigram/trigram in the input and each output position, compute a deterministic +1/-1 sign via hash and accumulate. This gives cosine-similarity preservation: "tree" and "trees" (sharing 3/4 bigrams) produce similar vectors.

This is what BLT's hash n-gram features do (minus the learned embeddings), and what fastText's subword embeddings approximate. SimHash (Charikar, 2002) and the Johnson-Lindenstrauss lemma guarantee that random projections from n-gram space preserve pairwise distances. Implementation complexity is comparable to current `fold_bytes`.

With STATE_SIZE=4096 and byte bigrams (256^2 = 65,536 possible), hashing into 4096 buckets gives ~16 bigrams per bucket — reasonable collision rate. Trigrams (256^3 = 16.7M possible) give better discrimination at the same bucket count.

**Tier 2: Byte embedding + 1D conv + pool (~20-150K parameters)**

256 learnable byte embeddings of dimension d (e.g., d=16): 4,096 params. Two 1D conv layers (kernel=3, 16→32→32 channels): ~2,000 params. Global average pool to fixed size, linear projection to STATE_SIZE: ~130K params. Total: under 150K parameters.

This is the fastText approach at the byte level. The conv layers learn local byte patterns (effectively learned n-grams). Trains jointly with the main network via backprop. Can be quantized to i8 after training. A single conv layer with kernel=3 adds minimal parameters but gives order-sensitivity — distinguishing "red" from "der" (same letters, different order).

References: Charformer's GBST (arXiv:2106.12672) learns "soft tokenization" from bytes end-to-end. CANINE (arXiv:2103.06874) uses strided convolution to downsample character sequences at 121M total params, but the downsampling component itself is tiny. BLT's local encoder is deliberately kept small (roughly doubles when the main model grows 20x). The fold does not need to be large.

**Tier 3: Architecture redesign (reduce fold, redistribute params)**

The first layer (4096→2048) holds 78% of all parameters (8.39M). If the fold output is 256, the first layer becomes 256→2048 = 524K params, freeing ~7.9M for redistribution into more hidden layers, wider intermediate representations, or the learnable fold itself.

**Option B: Scale through it** — the Nous approach. Bowen's position: "if you scale big enough the model would have this internal representation." Works at 600M+ (images), unclear at 10M. Without any fold improvement, rough estimate: 50-100M minimum to learn character-level patterns, based on CANINE (121M) and ByT5-Small (300M) results scaled down for the simpler task. Impractical for a browser-based 2ms inference target.

**Contrastive regularization**: Train the learnable fold with a combined loss — primary task loss (color prediction) plus a contrastive/triplet regularizer on fold outputs. This ensures the fold space is organized semantically. The training data generator can produce similarity pairs cheaply: ("tree", "oak", similar), ("tree", "airplane", dissimilar). This is deep supervised hashing (Salakhutdinov & Hinton, 2009; Wang et al., 2016).

### 2. CRITICAL: isigmoid Gradient Flow

**Training cannot work with the current activation scale parameters.** This must be addressed before any training infrastructure.

The activation `isigmoid(x, s) = (x * 127) / (|x| + s)` is a scaled softsign. Its derivative: `d/dx = (127 * s) / (|x| + s)^2`. The maximum derivative (at x=0) is `127/s`.

With `s = input_size` per layer:

| Layer | Scale (s) | Max d/dx | Note |
|-------|-----------|----------|------|
| 1 (4096→2048) | 4096 | 0.031 | |
| 2 (2048→1024) | 2048 | 0.062 | |
| 3 (1024→256) | 1024 | 0.124 | |
| 4 (256→64) | 256 | 0.496 | |
| 5 (64→16) | 64 | 1.984 | only layer with max derivative > 1 |
| output (16→4) | 4096 | 0.031 | scale = HIDDEN5_SIZE^3 |

**Product of max derivatives (best case, all activations at zero): ~1.46 x 10^-8**

Even in the absolute best case, gradients from the output to layer 1 are attenuated by a factor of 10 billion. In practice it's worse — for non-zero activations, |x| grows and the (|x| + s)^2 denominator grows quadratically.

For comparison: standard sigmoid's max derivative is 0.25. Across 6 layers: 0.25^6 = 0.000244. Already considered problematic, but four orders of magnitude better than isigmoid with current scales.

Softsign normally mitigates vanishing gradients better than sigmoid because it converges polynomially (1/x^2) rather than exponentially (e^-x) — the tails decay more slowly, so neurons are less likely to saturate. But this advantage is completely negated when the scale parameter pushes the max derivative far below 1.

**Mitigations:**
- **Reduce scale during training**: use `s = sqrt(input_size)` instead of `s = input_size`. Layer 1 goes from max derivative 0.031 → 1.98. Product across all layers becomes tractable.
- **Separate scales for training vs inference**: train with small s for gradient flow, deploy with current s values for the quantized inference behavior.
- **Skip/residual connections**: let gradients bypass layers. Not trivial with the texture pipeline but doable.
- **Per-layer gradient scaling**: manual gradient multipliers to counteract attenuation.

### 3. Training Data

We need (input, target_color) pairs. The research shows concrete datasets exist and LLMs can generate quality training data.

**Existing datasets:**

| Dataset | Size | Content | Source |
|---------|------|---------|--------|
| XKCD Color Survey | 954 named colors | Human-consensus hex values from 222,500 participants | xkcd.com/color/rgb/ |
| PAT (Palette-and-Text) | 10,183 pairs | Text phrase → 5-color palette. 4,312 unique words. ECCV 2018 | github.com/awesome-davian/Text2Colors |
| UW-71 | 71 colors x N concepts | Human association ratings across CIELAB space. 720 participants | github.com/SchlossVRL/sem_disc_theory |
| CSS Named Colors | 140 colors | Perfectly consistent ground truth | W3C spec |
| Color-Pedia | varies | Color descriptions and values | huggingface.co/datasets/boltuix/color-pedia |
| MuCED | 2,634 pairs | Expert-validated music → palette pairs (cross-modal) | ACM MM 2025 |

**Color semantics research** (Schloss, 2024, "Color Semantics in Human Cognition"):
- People can reliably judge association strength between any concept and any color, regardless of concreteness.
- Any concept elicits a **distribution** of association strengths across color space, not a single color. "Ocean" peaks at blues but has nonzero weight on greens, grays.
- Concrete nouns ("ocean", "fire") have highest cross-person consistency. Even congenitally blind people show similar associations, suggesting encoding in language structure, not just visual experience.
- Abstract concepts ("anger", "peace") show significant consistency but wider distributions. Red-love, yellow-joy, black-sadness are near-universal across 30 nations (Jonauskaite et al., 2020).

**LLM-generated data quality**: GPT-4 color-concept associations correlate with human ratings comparably to image-statistics methods (Mukherjee et al., 2024, arXiv:2406.17781). Performance varies by concept concreteness and "peakiness" of the color distribution. Known Western-centric bias. GPT-4o achieves ~50% accuracy on predicting the single most-voted color for a concept (Takase et al., 2025) — decent but not perfect.

**Recommended dataset construction:**
1. **Validation set (~1k, never train on these)**: CSS 140 colors + XKCD 954 colors. Unambiguous human-consensus mappings.
2. **Tier 1 (~10k)**: LLM-generated with spot-checking. Prompt for "what single hex color best represents [concept]?" plus confidence.
3. **Tier 2 (~50k)**: LLM-generated broad coverage without manual review. Use with appropriate regularization.

**Training data scale**: For a 10M param MLP learning text→color, 10k-20k pairs is the practical sweet spot. The PAT dataset (10,183 pairs) was sufficient for Text2Colors (ECCV 2018). The i8 quantization acts as strong implicit regularization (limited precision constrains the hypothesis space). Research on overparameterized networks (Holmdk, 2020) shows they can still generalize even with more parameters than data points, especially with early stopping.

### 4. Loss Function

**Use L2 loss in CIELAB space, not MSE on RGBA.**

CIELAB (L\*a\*b\*) was specifically designed so Euclidean distance approximates perceived color difference. RGB is device-dependent with no perceptual uniformity — a distance of 10 in red is perceptually different from 10 in green. HSL is better but not perceptually uniform, and hue is circular (0 and 360 are the same color — naive MSE breaks at the wraparound).

The conversion from sRGB to CIELAB is fully differentiable:
```
sRGB [0,255] → normalize [0,1] → gamma expand → linear RGB
  → XYZ (3x3 matrix, D65 white) → CIELAB (cube root + linear segment near zero)
```

Every step uses standard differentiable operations. Handle alpha separately with MSE.

CIEDE2000 (Delta E 2000) is the gold standard for perceptual color difference, correcting remaining non-uniformities in CIELAB. A differentiable implementation exists (Zhao et al., CVPR 2020, github.com/ZhengyuZhao/PerC-Adversarial). But simple CIELAB L2 is likely sufficient — the improvement from CIEDE2000 is modest relative to the magnitude of errors a training network will produce.

## Training Infrastructure

### Phase 1: CPU Training in Rust (do this first)

Every signal says this is correct:
- **Debugging**: inspect every gradient, activation, weight update in Rust. Debugging shaders is miserable — no stepping, no printf, readPixels is a GPU sync point.
- **Speed is sufficient**: ~100-200ms per forward+backward step in native Rust. 10k samples = 15-30 minutes.
- **No precision gymnastics**: f64/f32 natively. No RGBA16F underflow, no loss scaling, no texture format support checks.
- **Flexibility**: easy to experiment with optimizers (Adam, SGD), loss functions (CIELAB, MSE), architecture changes.
- **Deploy is simple**: serialize i8 weights → upload to GPU textures. Existing WebGL2 inference pipeline unchanged.

**Quantization-Aware Training (QAT)**: maintain f32 shadow weights. Forward pass: quantize to i8 via `w_quant = clamp(round(w_float), -128, 127)`, compute in i32, activate. Backward pass: Straight-Through Estimator (STE) — pretend quantization was the identity function within the clamp range, zero gradient outside. This is standard industry practice (NVIDIA validated across classification, detection, language models).

### In-Browser Training: Skip WebGL2, Use WebGPU

WebGL2 backprop is proven possible — TensorFlow.js (formerly deeplearn.js) does it at scale, SimpNet (github.com/SCRN-VRC/SimpNet-Deep-Learning-in-a-Shader) implements trainable CNNs in fragment shaders. But it is painful:

- No scatter writes (fragment shaders write to fixed output locations)
- No atomics (gradient reduction requires multiple passes)
- No shared memory (no tiling optimizations for matrix multiplies)
- Texture feedback loop prohibition (mandatory ping-pong double buffering)
- ~18 draw calls per training step (forward + backward + update, 6 each)

**WebGPU is now the right target.** As of late 2025, all major desktop browsers ship WebGPU (Chrome, Edge, Firefox, Safari 26). Compute shaders eliminate the fragment-shader hackery:
- Storage buffers (read/write arbitrary data, no RGBA packing)
- Workgroup shared memory (up to 32KB, enables tiled matrix multiply — critical for the 4096x2048 layer 1)
- Atomics (gradient accumulation across a batch in a single dispatch)
- No feedback loop restriction (read/write with proper barriers, no double buffering needed)
- 3-10x faster than WebGL2 for compute-heavy tasks

**Float precision for gradients**: RGBA32F render targets require `EXT_color_buffer_float` in WebGL2 (widely supported on desktop, poor on mobile). RGBA16F is more portable but needs loss scaling to avoid gradient underflow (Micikevicius et al., ICLR 2018). With WebGPU, this is a non-issue — storage buffers natively support f32.

**Memory budget for double-buffered training** (WebGL2 path if needed): f32 shadow weights x2 (~82MB) + i8 inference weights (~10MB) = ~93MB. Well within desktop GPU limits but tight on mobile.

### RLHF-Lite: Binary Feedback

Pure like/dislike with no target color **will not converge in reasonable time**. The signal is 1 bit for 4 output dimensions — the model can't tell which channel was wrong. REINFORCE with binary rewards has extremely high variance, and the credit assignment problem is severe.

**What works instead (after supervised pre-training):**
1. **Thumbs up** → use current output as target, do 1 SGD step with small learning rate. Full 4D target for free.
2. **Thumbs down** → show 4 random perturbations, user picks best → DPO-style (arXiv:2305.18290) update toward chosen variant.
3. **Optional**: color picker for "I wanted something like this" → full supervised signal.

This is how production RLHF works: supervised fine-tuning first, preference optimization to refine. Expect hundreds of A/B comparisons per concept for convergence with DPO, thousands with raw binary feedback.

## Multi-Modal Color Extraction

**Image → dominant color**: K-means clustering (k=5 in CIELAB space, take largest cluster centroid) is the practical choice. Median Cut (Heckbert, 1979) is the classic alternative. Quality is high for single-dominant-color images, ambiguous for complex scenes.

**Audio → color (synesthesia mapping)**: The most defensible deterministic mapping, grounded in research:
- **Pitch (fundamental frequency) → hue**: low=red, high=violet, logarithmic. Pitch-to-brightness is the most robust and cross-culturally consistent mapping (found in both synesthetes and non-synesthetes).
- **Energy/loudness → saturation**: quiet=desaturated, loud=vivid.
- **Spectral centroid → lightness**: dark timbre=dark, bright timbre=light.
- Reference: Music2Palette (ACM MM 2025) provides 2,634 expert-validated music-palette pairs.

**Cross-modal generalization**: won't work until the fold is fixed. Text and images hash to completely unrelated vectors through the XOR fold. The later layers can't learn that text-"ocean" and bytes-of-a-seascape should produce similar outputs because their input representations share zero structure. After a learned fold (Phase 2), cross-modal transfer requires contrastive training — this is essentially CLIP at massive scale. At 10M params, expect only simple associations.

## Revised Roadmap

### Phase 0: Fix isigmoid (prerequisite for all training)
- Training literally cannot work with current scale parameters (gradient product ~10^-8)
- Implement configurable scale: `s = sqrt(input_size)` or small constant for training
- Validate gradient flow: product of max derivatives should be > 10^-4 across all layers
- Separate training scales from inference scales if needed

### Phase 1: N-gram Fold (zero-parameter improvement)
- Replace XOR hash with SimHash-style bigram/trigram random projection
- No training needed, immediate improvement to semantic similarity preservation
- Validate: cosine similarity between fold("tree") and fold("trees") should be high

### Phase 2: Text Training (CPU)
- Generate 10k-20k (text, color) pairs via LLM, tiered by confidence
- Hold out XKCD 954 + CSS 140 as validation set
- Implement backprop in Rust (native, not WASM) with f32 shadow weights and STE
- Loss: L2 in CIELAB space + MSE on alpha
- Train, quantize to i8, deploy to GPU
- Validate: "tree" → green, "ocean" → blue, "fire" → red

### Phase 3: Learnable Fold
- Replace n-gram fold with byte embedding + 1D conv + pool (~20-150K params)
- Train jointly with the main network
- Add contrastive/triplet regularization on fold outputs
- Consider architecture redesign: smaller fold output, redistributed params
- Test: do semantically similar inputs produce similar colors?

### Phase 4: In-Browser Training (WebGPU)
- Implement training via WebGPU compute shaders (skip WebGL2 backprop)
- RLHF-lite: thumbs up = use as target, thumbs down = show perturbations + DPO
- Real-time weight updates visible in the visualization
- Keep WebGL2 for inference (broader compatibility)

### Phase 5: Multi-Modal
- Add image → dominant color training data (k-means in CIELAB)
- Add audio → synesthesia mapping
- Contrastive fold training for cross-modal alignment
- Test cross-modal generalization

### Phase 6: Scale
- Bigger network if needed
- More training data (50k+)
- Explore if the model learns any surprising cross-modal relationships

## Current Architecture

10.77M parameters, all i8:
```
input [u8] → fold_bytes → [i8; 4096]
  → hidden1: 4096 → 2048 (8.39M weights, 78% of model)
    → hidden2: 2048 → 1024 (2.10M weights)
      → hidden3: 1024 → 256 (262K weights)
        → hidden4: 256 → 64 (16K weights)
          → hidden5: 64 → 16 (1K weights)
            → output: 16 → 4 RGBA (64 weights)
```

Forward pass: ~2ms on GPU (WebGL2), ~30-50ms on CPU (WASM).
Model size: ~10.3MB (1 byte per parameter).
Activation: isigmoid — `(x * 127) / (|x| + scale)`, integer-friendly scaled softsign.
Inference: entirely in-browser, no server.

## Answered Questions

**Is 10M parameters enough to learn byte-level semantics for text?**
Not from scratch (smallest success: ByT5-Small at 300M). But for the simpler task of text→color with a good fold, likely yes. The bottleneck is the fold, not network capacity.

**Can the fold be learned efficiently?**
Yes. BLT, CANINE, Charformer all show the compression layer can be ~1-5% of total parameters. A 1D conv fold at ~20-150K params is the sweet spot. N-gram hashing (zero params) is a strong baseline.

**Does training on text-color generalize to image-color?**
Not until the fold is fixed. After a learned fold with contrastive cross-modal training, simple associations may transfer. Full cross-modal generalization requires much larger scale (CLIP-level).

**Is MSE on RGBA the right loss?**
No. L2 in CIELAB space is substantially better for perceptual color similarity. The conversion is fully differentiable. CIEDE2000 is the gold standard but likely overkill.

**Can RLHF (like/dislike) converge with binary feedback?**
Not from scratch. After supervised pre-training, a modified approach works: thumbs-up = use as target (full 4D signal), thumbs-down = show perturbations + pick best (DPO). Pure binary REINFORCE has too much variance.

**At what point does WebGL2 become the bottleneck?**
For inference: probably never at current scale. For training: immediately. WebGPU compute shaders are now the right target (all major desktop browsers support it as of late 2025). 3-10x faster, no fragment-shader hacking needed.

## Open Questions

- What is the optimal scale parameter for isigmoid during training? Is `sqrt(input_size)` enough, or should we use a learned/adaptive scale?
- Should we train the fold with a combined loss (color prediction + contrastive similarity), or train it in two stages?
- How small can the fold output be before we lose capacity? Is 256 sufficient, or do we need the full 4096?
- Does Bolmo's "byteification via distillation" approach work at 10M scale — can we distill a large model's text representations into a tiny fold?
- For the n-gram fold, what n-gram sizes give the best tradeoff? BLT uses n=3..8; is n=2..4 sufficient for short text?

## References

**Byte-level models:**
- BLT: arXiv:2412.09871, github.com/facebookresearch/blt
- EvaByte: github.com/OpenEvaByte/evabyte, huggingface.co/EvaByte/EvaByte
- Bolmo: arXiv:2512.15586, allenai.org/blog/bolmo
- ByteFlow: arXiv:2603.03583
- MBLM: arXiv:2502.14553, github.com/ai4sd/multiscale-byte-lm
- bGPT: arXiv:2402.19155, github.com/sanderwood/bgpt
- ByT5: arXiv:2105.13626
- MambaByte: arXiv:2401.13660
- SpaceByte: arXiv:2404.14408
- Nous distillation: arXiv:2602.01007
- Super Tiny LMs: arXiv:2405.14159
- Awesome-Byte-LLM: github.com/zjysteven/Awesome-Byte-LLM

**Color and perception:**
- Schloss 2024, "Color Semantics in Human Cognition": doi.org/10.1177/09637214231208189
- LLMs estimate color-concept associations: arXiv:2406.17781
- Universal color-emotion associations: doi.org/10.1177/0956797620948810
- PerC-Adversarial (differentiable CIEDE2000): github.com/ZhengyuZhao/PerC-Adversarial
- Text2Colors: github.com/awesome-davian/Text2Colors
- Color2Vec: doi.org/10.1145/3571816

**Training and hashing:**
- SimHash: Charikar, 2002
- FastText subword embeddings: aclanthology.org/Q17-1010
- Semantic Hashing: Salakhutdinov & Hinton, 2009
- Charformer GBST: arXiv:2106.12672
- CANINE: arXiv:2103.06874
- DPO: arXiv:2305.18290
- Mixed precision training: NVIDIA, ICLR 2018

**WebGL/WebGPU:**
- TensorFlow.js: github.com/tensorflow/tfjs
- SimpNet in shaders: github.com/SCRN-VRC/SimpNet-Deep-Learning-in-a-Shader
- GPT-2 in WebGL: nathan.rs/posts/gpu-shader-programming/
- WebGPU browser support: caniuse.com/webgpu

**Datasets:**
- XKCD Color Survey: xkcd.com/color/rgb/
- PAT: github.com/awesome-davian/Text2Colors
- UW-71: github.com/SchlossVRL/sem_disc_theory
- Music2Palette: arXiv:2507.04758
