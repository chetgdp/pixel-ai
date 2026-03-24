# OMNI: Omni-Modal Neural Input to One Pixel

## The Problem

pixel.ai maps arbitrary byte input to a single RGBA pixel (4 bytes). The forward pass works — 10M i8 parameters running in ~2ms via WebGL2 fragment shader hacking. But the network has no understanding of what the bytes mean. "tree" and "oak" produce completely unrelated colors because the input stage (fold_bytes) is a lossy XOR hash that destroys semantic relationships.

The question: can a tiny network running in a browser learn to map any byte sequence to a meaningful color?

## Prior Art

### Nous Research — Large Information Models
Bowen Peng at Nous Research presented a 600M parameter model trained on raw BMP file bytes. Vocab size = 256 (one per byte value), no tokenizer. The model learned to generate valid BMP files with correct headers and coherent images. Trained 1 day on 64 B200s, 300GB of images from DataComp Common Pool.

Key insight: model raw information, not language. Any data is just bytes. They developed "Lighthouse Attention" — hierarchical attention at different scales — to handle the granularity problem (text is dense info, images are sparse waveforms).

Their bottlenecks: slow byte-by-byte autoregressive generation, huge context lengths, generation order. None of these apply to pixel.ai — our output is 4 bytes, one forward pass, no generation.

### Meta — Byte Latent Transformer (BLT)
Operates on raw bytes, learns dynamic compression of byte sequences into patches. Matches tokenized models at 8B params. The learned compression is essentially what our fold_bytes does statically — BLT learns it, we hardcode it.

### General Trend
The field is moving toward tokenizer-free, byte-level models that treat all modalities uniformly. bGPT, MBLM, EvaByte, ByteFlow all demonstrate this. The consensus is that with enough scale and data, byte-level models can learn internal semantic representations without explicit tokenization.

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

## Three Problems to Solve

### 1. The Fold Problem (Representation)

Current state: `fold_bytes` is a deterministic XOR hash that compresses arbitrary-length input into a fixed [i8; 4096] vector. It preserves no semantic information — "tree" and "oak" hash to completely unrelated vectors.

What we need: similar inputs should produce similar representations. Options:

**A. Learnable fold** — replace the hardcoded XOR hash with a small network that learns to compress byte sequences. Could be a simple RNN or 1D convolution over the input bytes that outputs a fixed-size vector. Train the fold jointly with the main network.

**B. Scale through it** — the Nous approach. Don't fold at all, just make the network big enough to learn byte-level semantics directly. Works at 600M+ params, unclear if tractable at our scale (10M).

**C. Hybrid** — keep the fold for non-text inputs (images, binary), but add a learned text embedding path. Accept that different modalities might need different input pipelines, even if the network itself is shared.

**D. Byte n-gram features** — instead of folding to single byte positions, fold to byte bigram/trigram positions. "tr", "re", "ee" would be features, giving "tree" and "trees" overlapping representations. Larger state vector but preserves local structure.

### 2. Training Data Generation

We need (input, target_color) pairs. Different strategies per modality:

**Text to color (easiest start)**
- Use an LLM to generate pairs: "ocean" -> #1a5276ff, "sunset" -> #e74c3cff
- Can generate thousands of pairs cheaply
- Covers semantic understanding: "tree", "oak", "forest" all map to greens/browns
- Validation is intuitive — humans can judge if colors make sense

**Image to color**
- Dominant color extraction from images — trivial, well-solved
- Could also do average color, palette extraction, mood-based color
- Large datasets available (ImageNet, Common Crawl images)

**Audio to color**
- Synesthesia mapping — pitch to hue, volume to brightness
- Or just dominant frequency analysis
- Less obvious what "correct" means

**Binary/arbitrary bytes**
- File type → conventional color? (code → green terminal, PDF → red)
- Or accept that arbitrary bytes don't have a meaningful color — the model learns what it can from structure

Start with text. It has the clearest semantic-to-color mapping and the easiest data generation.

### 3. Training Infrastructure (WebGL2 Backprop)

The forward pass runs in WebGL2 via fragment shaders. Backprop needs:

**Gradient textures** — RGBA32F or RGBA16F render targets to store gradient accumulation. Float precision required for small gradient steps that i8 can't represent.

**Backward pass shaders** — for each layer, a shader that reads the next layer's gradients and computes this layer's gradients. The chain rule through isigmoid:

```
isigmoid(x, s) = (x * 127) / (|x| + s)
d/dx isigmoid = (127 * s) / (|x| + s)^2
```

**Weight update shader** — reads current i8 weights (as float) + float gradients, applies learning rate, quantizes back to i8, writes new weight texture.

**Double buffering** — weight textures are read during forward pass, written during update. Need two copies, swap after each step.

**Loss function** — MSE between output RGBA and target RGBA. Computed on the 1x1 output pixel.

The pipeline roughly triples in draw calls: forward pass (6 draws) + backward pass (6 draws) + weight update (6 draws). All still within WebGL2 fragment shader constraints.

Alternative: train on CPU (Rust), deploy weights to GPU for inference. Simpler, slower training, but avoids the WebGL2 backprop complexity entirely. Could use this to validate the approach before optimizing.

## Proposed Roadmap

### Phase 1: Text Training (CPU)
- Generate 10k+ (text, color) pairs via LLM
- Implement backprop in Rust (CPU) with f32 shadow weights
- Train on text pairs, quantize to i8, test on GPU
- Validate: "tree" → green, "ocean" → blue, "fire" → red

### Phase 2: Learnable Fold
- Replace fold_bytes with a small learnable compression network
- Train jointly with the main network
- Test: do semantically similar inputs produce similar colors?

### Phase 3: WebGL2 Backprop
- Implement gradient shaders for in-browser training
- Like/dislike button for live user feedback (RLHF-lite)
- Real-time weight updates visible in the visualization

### Phase 4: Multi-Modal
- Add image → dominant color training data
- Add audio → synesthesia mapping
- Test cross-modal generalization: does training on text help with images?

### Phase 5: Scale
- Bigger network (if needed)
- More training data
- WebGPU compute shaders for faster training
- Explore if the model learns any surprising cross-modal relationships

## Current Architecture

10.77M parameters, all i8:
```
input [u8] → fold_bytes → [i8; 4096]
  → hidden1: 4096 → 2048 (8.39M weights)
    → hidden2: 2048 → 1024 (2.10M weights)
      → hidden3: 1024 → 256 (262K weights)
        → hidden4: 256 → 64 (16K weights)
          → hidden5: 64 → 16 (1K weights)
            → output: 16 → 4 RGBA (64 weights)
```

Forward pass: ~2ms on GPU (WebGL2), ~30-50ms on CPU (WASM).
Model size: ~10.3MB (1 byte per parameter).
Activation: isigmoid — `(x * 127) / (|x| + scale)`, integer-friendly s-curve.
Inference: entirely in-browser, no server.

## Open Questions

- Is 10M parameters enough to learn byte-level semantics for text? Or do we need to scale up?
- Can the fold be learned efficiently, or should we just increase STATE_SIZE and eat the cost?
- Does training on text-color pairs generalize at all to image-color? Or are they completely separate learned mappings?
- Is MSE on RGBA the right loss, or should we train in HSL space where perceptual similarity is more linear?
- Can RLHF (like/dislike) actually converge with binary feedback and no target color?
- At what point does the WebGL2 hack become the bottleneck and we need WebGPU?
