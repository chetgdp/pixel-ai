use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
mod gpu_webgl2;

// fold to 4k then division by 2/4 all the way down
pub const STATE_SIZE:   usize = 4096;
pub const HIDDEN1_SIZE: usize = 2048;
pub const HIDDEN2_SIZE: usize = 1024;
pub const HIDDEN3_SIZE: usize = 256;
pub const HIDDEN4_SIZE: usize = 64;
pub const HIDDEN5_SIZE: usize = 16;
pub const OUTPUT_SIZE:  usize = 4; // RGBA

// we use i8 because it matches our u8 output, ez
pub struct Network {
    w1: [[i8; STATE_SIZE]; HIDDEN1_SIZE],
    b1: [i8; HIDDEN1_SIZE],
    w2: [[i8; HIDDEN1_SIZE]; HIDDEN2_SIZE],
    b2: [i8; HIDDEN2_SIZE],
    w3: [[i8; HIDDEN2_SIZE]; HIDDEN3_SIZE],
    b3: [i8; HIDDEN3_SIZE],
    w4: [[i8; HIDDEN3_SIZE]; HIDDEN4_SIZE],
    b4: [i8; HIDDEN4_SIZE],
    w5: [[i8; HIDDEN4_SIZE]; HIDDEN5_SIZE],
    b5: [i8; HIDDEN5_SIZE],
    w6: [[i8; HIDDEN5_SIZE]; OUTPUT_SIZE],
    b6: [i8; OUTPUT_SIZE],
}

// simple neural network
fn isigmoid(x: i32, scale: i32) -> i8 {
    ((x * 127) / (x.abs() + scale)) as i8
}

// f32 versions for training
fn isigmoid_f32(x: f32, scale: f32) -> f32 {
    (x * 127.0) / (x.abs() + scale)
}

// derivative: d/dx isigmoid = 127 * scale / (|x| + scale)^2
fn isigmoid_deriv(x: f32, scale: f32) -> f32 {
    let d = x.abs() + scale;
    127.0 * scale / (d * d)
}

// forward one layer in f32, returns (pre_activations, activations)
fn forward_layer(
    weights: &[f32], biases: &[f32], input: &[f32],
    out_size: usize, in_size: usize, scale: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut pre = Vec::with_capacity(out_size);
    let mut act = Vec::with_capacity(out_size);
    for i in 0..out_size {
        let mut sum = biases[i];
        for j in 0..in_size {
            sum += weights[i * in_size + j] * input[j];
        }
        pre.push(sum);
        act.push(isigmoid_f32(sum, scale));
    }
    (pre, act)
}

// backward one layer: compute gradients, update weights in place, return d_input
// updates weights immediately to avoid storing the full gradient matrix
fn backward_and_update(
    d_act: &[f32], pre: &[f32], input: &[f32],
    weights: &mut [f32], biases: &mut [f32],
    out_size: usize, in_size: usize, scale: f32, lr: f32,
) -> Vec<f32> {
    let mut d_input = vec![0.0f32; in_size];
    for i in 0..out_size {
        let d_pre = d_act[i] * isigmoid_deriv(pre[i], scale);
        biases[i] -= lr * d_pre;
        for j in 0..in_size {
            // propagate gradient using original weight, then update
            d_input[j] += d_pre * weights[i * in_size + j];
            weights[i * in_size + j] -= lr * d_pre * input[j];
        }
    }
    d_input
}

fn parse_hex_rgba(hex: &str) -> [u8; 4] {
    let hex = hex.trim_start_matches('#');
    let r = u8::from_str_radix(&hex[0..2], 16).unwrap_or(0);
    let g = u8::from_str_radix(&hex[2..4], 16).unwrap_or(0);
    let b = u8::from_str_radix(&hex[4..6], 16).unwrap_or(0);
    let a = if hex.len() >= 8 {
        u8::from_str_radix(&hex[6..8], 16).unwrap_or(255)
    } else { 255 };
    [r, g, b, a]
}

impl Network {
    // when we create a new network we want to init all the weights and biases at random
    // the seed we pass in is hardcoded, so the initial weights are always the same
    pub fn new(seed: u64) -> Self {
        let mut rng = seed;
        let mut next = || -> i8 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng >> 56) as i8
        };

        let mut w1 = [[0i8; STATE_SIZE]; HIDDEN1_SIZE];
        let mut b1 = [0i8; HIDDEN1_SIZE];
        for i in 0..HIDDEN1_SIZE {
            for j in 0..STATE_SIZE { w1[i][j] = next(); }
            b1[i] = next();
        }

        let mut w2 = [[0i8; HIDDEN1_SIZE]; HIDDEN2_SIZE];
        let mut b2 = [0i8; HIDDEN2_SIZE];
        for i in 0..HIDDEN2_SIZE {
            for j in 0..HIDDEN1_SIZE { w2[i][j] = next(); }
            b2[i] = next();
        }

        let mut w3 = [[0i8; HIDDEN2_SIZE]; HIDDEN3_SIZE];
        let mut b3 = [0i8; HIDDEN3_SIZE];
        for i in 0..HIDDEN3_SIZE {
            for j in 0..HIDDEN2_SIZE { w3[i][j] = next(); }
            b3[i] = next();
        }

        let mut w4 = [[0i8; HIDDEN3_SIZE]; HIDDEN4_SIZE];
        let mut b4 = [0i8; HIDDEN4_SIZE];
        for i in 0..HIDDEN4_SIZE {
            for j in 0..HIDDEN3_SIZE { w4[i][j] = next(); }
            b4[i] = next();
        }

        let mut w5 = [[0i8; HIDDEN4_SIZE]; HIDDEN5_SIZE];
        let mut b5 = [0i8; HIDDEN5_SIZE];
        for i in 0..HIDDEN5_SIZE {
            for j in 0..HIDDEN4_SIZE { w5[i][j] = next(); }
            b5[i] = next();
        }

        let mut w6 = [[0i8; HIDDEN5_SIZE]; OUTPUT_SIZE];
        let mut b6 = [0i8; OUTPUT_SIZE];
        for i in 0..OUTPUT_SIZE {
            for j in 0..HIDDEN5_SIZE { w6[i][j] = next(); }
            b6[i] = next();
        }

        Network { w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, w6, b6 }
    }

    pub fn forward(&self, state: &[i8; STATE_SIZE]) -> [u8; 4] {
        let mut h1 = [0i8; HIDDEN1_SIZE];
        for i in 0..HIDDEN1_SIZE {
            let mut sum = self.b1[i] as i32;
            for j in 0..STATE_SIZE {
                sum += self.w1[i][j] as i32 * state[j] as i32;
            }
            h1[i] = isigmoid(sum, STATE_SIZE as i32);
        }

        let mut h2 = [0i8; HIDDEN2_SIZE];
        for i in 0..HIDDEN2_SIZE {
            let mut sum = self.b2[i] as i32;
            for j in 0..HIDDEN1_SIZE {
                sum += self.w2[i][j] as i32 * h1[j] as i32;
            }
            h2[i] = isigmoid(sum, HIDDEN1_SIZE as i32);
        }

        let mut h3 = [0i8; HIDDEN3_SIZE];
        for i in 0..HIDDEN3_SIZE {
            let mut sum = self.b3[i] as i32;
            for j in 0..HIDDEN2_SIZE {
                sum += self.w3[i][j] as i32 * h2[j] as i32;
            }
            h3[i] = isigmoid(sum, HIDDEN2_SIZE as i32);
        }

        let mut h4 = [0i8; HIDDEN4_SIZE];
        for i in 0..HIDDEN4_SIZE {
            let mut sum = self.b4[i] as i32;
            for j in 0..HIDDEN3_SIZE {
                sum += self.w4[i][j] as i32 * h3[j] as i32;
            }
            h4[i] = isigmoid(sum, HIDDEN3_SIZE as i32);
        }

        let mut h5 = [0i8; HIDDEN5_SIZE];
        for i in 0..HIDDEN5_SIZE {
            let mut sum = self.b5[i] as i32;
            for j in 0..HIDDEN4_SIZE {
                sum += self.w5[i][j] as i32 * h4[j] as i32;
            }
            h5[i] = isigmoid(sum, HIDDEN4_SIZE as i32);
        }

        let mut output = [0u8; 4];
        for i in 0..OUTPUT_SIZE {
            let mut sum = self.b6[i] as i32;
            for j in 0..HIDDEN5_SIZE {
                sum += self.w6[i][j] as i32 * h5[j] as i32;
            }
            let out_scale = (HIDDEN5_SIZE * HIDDEN5_SIZE * HIDDEN5_SIZE) as i32;
            let s = isigmoid(sum, out_scale) as i32;
            output[i] = ((s + 127) * 255 / 254) as u8;
        }

        output
    }

    // returns all layer activations as bias-encoded u8 (value + 128)
    // layout: [state(4096), h1(2048), h2(1024), h3(256), h4(64), h5(16), output(4)]
    pub fn forward_all(&self, state: &[i8; STATE_SIZE]) -> Vec<u8> {
        let encode = |v: i8| -> u8 { (v as i16 + 128) as u8 };

        let mut result = Vec::with_capacity(
            STATE_SIZE + HIDDEN1_SIZE + HIDDEN2_SIZE + HIDDEN3_SIZE +
            HIDDEN4_SIZE + HIDDEN5_SIZE + OUTPUT_SIZE
        );

        // input state
        for i in 0..STATE_SIZE { result.push(encode(state[i])); }

        let mut h1 = [0i8; HIDDEN1_SIZE];
        for i in 0..HIDDEN1_SIZE {
            let mut sum = self.b1[i] as i32;
            for j in 0..STATE_SIZE {
                sum += self.w1[i][j] as i32 * state[j] as i32;
            }
            h1[i] = isigmoid(sum, STATE_SIZE as i32);
        }
        for i in 0..HIDDEN1_SIZE { result.push(encode(h1[i])); }

        let mut h2 = [0i8; HIDDEN2_SIZE];
        for i in 0..HIDDEN2_SIZE {
            let mut sum = self.b2[i] as i32;
            for j in 0..HIDDEN1_SIZE {
                sum += self.w2[i][j] as i32 * h1[j] as i32;
            }
            h2[i] = isigmoid(sum, HIDDEN1_SIZE as i32);
        }
        for i in 0..HIDDEN2_SIZE { result.push(encode(h2[i])); }

        let mut h3 = [0i8; HIDDEN3_SIZE];
        for i in 0..HIDDEN3_SIZE {
            let mut sum = self.b3[i] as i32;
            for j in 0..HIDDEN2_SIZE {
                sum += self.w3[i][j] as i32 * h2[j] as i32;
            }
            h3[i] = isigmoid(sum, HIDDEN2_SIZE as i32);
        }
        for i in 0..HIDDEN3_SIZE { result.push(encode(h3[i])); }

        let mut h4 = [0i8; HIDDEN4_SIZE];
        for i in 0..HIDDEN4_SIZE {
            let mut sum = self.b4[i] as i32;
            for j in 0..HIDDEN3_SIZE {
                sum += self.w4[i][j] as i32 * h3[j] as i32;
            }
            h4[i] = isigmoid(sum, HIDDEN3_SIZE as i32);
        }
        for i in 0..HIDDEN4_SIZE { result.push(encode(h4[i])); }

        let mut h5 = [0i8; HIDDEN5_SIZE];
        for i in 0..HIDDEN5_SIZE {
            let mut sum = self.b5[i] as i32;
            for j in 0..HIDDEN4_SIZE {
                sum += self.w5[i][j] as i32 * h4[j] as i32;
            }
            h5[i] = isigmoid(sum, HIDDEN4_SIZE as i32);
        }
        for i in 0..HIDDEN5_SIZE { result.push(encode(h5[i])); }

        for i in 0..OUTPUT_SIZE {
            let mut sum = self.b6[i] as i32;
            for j in 0..HIDDEN5_SIZE {
                sum += self.w6[i][j] as i32 * h5[j] as i32;
            }
            let out_scale = (HIDDEN5_SIZE * HIDDEN5_SIZE * HIDDEN5_SIZE) as i32;
            let s = isigmoid(sum, out_scale) as i32;
            result.push(((s + 127) * 255 / 254) as u8);
        }

        result
    }
}

// f32 shadow copies of i8 weights for gradient accumulation
// stored as flat Vec<f32> (row-major: weights[i * in_size + j])
struct ShadowWeights {
    w1: Vec<f32>, b1: Vec<f32>,
    w2: Vec<f32>, b2: Vec<f32>,
    w3: Vec<f32>, b3: Vec<f32>,
    w4: Vec<f32>, b4: Vec<f32>,
    w5: Vec<f32>, b5: Vec<f32>,
    w6: Vec<f32>, b6: Vec<f32>,
}

impl ShadowWeights {
    fn from_network(net: &Network) -> Self {
        let flatten = |w: &[&[i8]]| -> Vec<f32> {
            w.iter().flat_map(|row| row.iter().map(|&v| v as f32)).collect()
        };
        let bias_f32 = |b: &[i8]| -> Vec<f32> {
            b.iter().map(|&v| v as f32).collect()
        };
        let w1r: Vec<&[i8]> = net.w1.iter().map(|r| r.as_slice()).collect();
        let w2r: Vec<&[i8]> = net.w2.iter().map(|r| r.as_slice()).collect();
        let w3r: Vec<&[i8]> = net.w3.iter().map(|r| r.as_slice()).collect();
        let w4r: Vec<&[i8]> = net.w4.iter().map(|r| r.as_slice()).collect();
        let w5r: Vec<&[i8]> = net.w5.iter().map(|r| r.as_slice()).collect();
        let w6r: Vec<&[i8]> = net.w6.iter().map(|r| r.as_slice()).collect();
        ShadowWeights {
            w1: flatten(&w1r), b1: bias_f32(&net.b1),
            w2: flatten(&w2r), b2: bias_f32(&net.b2),
            w3: flatten(&w3r), b3: bias_f32(&net.b3),
            w4: flatten(&w4r), b4: bias_f32(&net.b4),
            w5: flatten(&w5r), b5: bias_f32(&net.b5),
            w6: flatten(&w6r), b6: bias_f32(&net.b6),
        }
    }

    // quantize f32 shadow weights back into the i8 network
    fn quantize_into(&self, net: &mut Network) {
        let q_weights = |src: &[f32], dst: &mut [i8]| {
            for (d, &s) in dst.iter_mut().zip(src.iter()) {
                *d = s.round().clamp(-128.0, 127.0) as i8;
            }
        };
        // weights
        for (i, row) in net.w1.iter_mut().enumerate() {
            for (j, v) in row.iter_mut().enumerate() {
                *v = self.w1[i * STATE_SIZE + j].round().clamp(-128.0, 127.0) as i8;
            }
        }
        for (i, row) in net.w2.iter_mut().enumerate() {
            for (j, v) in row.iter_mut().enumerate() {
                *v = self.w2[i * HIDDEN1_SIZE + j].round().clamp(-128.0, 127.0) as i8;
            }
        }
        for (i, row) in net.w3.iter_mut().enumerate() {
            for (j, v) in row.iter_mut().enumerate() {
                *v = self.w3[i * HIDDEN2_SIZE + j].round().clamp(-128.0, 127.0) as i8;
            }
        }
        for (i, row) in net.w4.iter_mut().enumerate() {
            for (j, v) in row.iter_mut().enumerate() {
                *v = self.w4[i * HIDDEN3_SIZE + j].round().clamp(-128.0, 127.0) as i8;
            }
        }
        for (i, row) in net.w5.iter_mut().enumerate() {
            for (j, v) in row.iter_mut().enumerate() {
                *v = self.w5[i * HIDDEN4_SIZE + j].round().clamp(-128.0, 127.0) as i8;
            }
        }
        for (i, row) in net.w6.iter_mut().enumerate() {
            for (j, v) in row.iter_mut().enumerate() {
                *v = self.w6[i * HIDDEN5_SIZE + j].round().clamp(-128.0, 127.0) as i8;
            }
        }
        // biases
        q_weights(&self.b1, &mut net.b1);
        q_weights(&self.b2, &mut net.b2);
        q_weights(&self.b3, &mut net.b3);
        q_weights(&self.b4, &mut net.b4);
        q_weights(&self.b5, &mut net.b5);
        q_weights(&self.b6, &mut net.b6);
    }
}

// TODO: less lossy fold — multiple passes, better mixing
pub fn fold_bytes(data: &[u8]) -> [i8; STATE_SIZE] {
    let mut state = [0i32; STATE_SIZE];

    for (pos, &byte) in data.iter().enumerate() {
        let idx = pos % STATE_SIZE;
        state[idx] = state[idx].wrapping_add(byte as i32);
        state[idx] ^= (pos as i32).wrapping_mul(0x9E3779B9_u32 as i32);
    }

    let mut out = [0i8; STATE_SIZE];
    for i in 0..STATE_SIZE {
        out[i] = (state[i] % 128) as i8;
    }
    out
}

#[wasm_bindgen]
pub struct CpuPipeline {
    net: Network,
    shadow: ShadowWeights,
    lr: f32,
    steps: u32,
}

impl CpuPipeline {
    fn train_state(&mut self, state: &[i8; STATE_SIZE], target: [u8; 4]) {
        let lr = self.lr;
        let input0: Vec<f32> = state.iter().map(|&v| v as f32).collect();

        // forward pass using f32 shadow weights, storing intermediates
        let (pre1, act1) = forward_layer(&self.shadow.w1, &self.shadow.b1, &input0,
            HIDDEN1_SIZE, STATE_SIZE, STATE_SIZE as f32);
        let (pre2, act2) = forward_layer(&self.shadow.w2, &self.shadow.b2, &act1,
            HIDDEN2_SIZE, HIDDEN1_SIZE, HIDDEN1_SIZE as f32);
        let (pre3, act3) = forward_layer(&self.shadow.w3, &self.shadow.b3, &act2,
            HIDDEN3_SIZE, HIDDEN2_SIZE, HIDDEN2_SIZE as f32);
        let (pre4, act4) = forward_layer(&self.shadow.w4, &self.shadow.b4, &act3,
            HIDDEN4_SIZE, HIDDEN3_SIZE, HIDDEN3_SIZE as f32);
        let (pre5, act5) = forward_layer(&self.shadow.w5, &self.shadow.b5, &act4,
            HIDDEN5_SIZE, HIDDEN4_SIZE, HIDDEN4_SIZE as f32);

        let out_scale = (HIDDEN5_SIZE * HIDDEN5_SIZE * HIDDEN5_SIZE) as f32;
        let (pre6, act6) = forward_layer(&self.shadow.w6, &self.shadow.b6, &act5,
            OUTPUT_SIZE, HIDDEN5_SIZE, out_scale);

        // output remapping: isigmoid [-127,127] -> [0,255]
        let mut output = [0.0f32; OUTPUT_SIZE];
        for i in 0..OUTPUT_SIZE {
            output[i] = (act6[i] + 127.0) * 255.0 / 254.0;
        }

        // MSE loss gradient, chained through output remapping (255/254)
        let mut d_act6 = vec![0.0f32; OUTPUT_SIZE];
        for i in 0..OUTPUT_SIZE {
            d_act6[i] = 2.0 * (output[i] - target[i] as f32) * (255.0 / 254.0);
        }

        // backward pass: compute gradients + update weights in place
        let d5 = backward_and_update(&d_act6, &pre6, &act5,
            &mut self.shadow.w6, &mut self.shadow.b6,
            OUTPUT_SIZE, HIDDEN5_SIZE, out_scale, lr);
        let d4 = backward_and_update(&d5, &pre5, &act4,
            &mut self.shadow.w5, &mut self.shadow.b5,
            HIDDEN5_SIZE, HIDDEN4_SIZE, HIDDEN4_SIZE as f32, lr);
        let d3 = backward_and_update(&d4, &pre4, &act3,
            &mut self.shadow.w4, &mut self.shadow.b4,
            HIDDEN4_SIZE, HIDDEN3_SIZE, HIDDEN3_SIZE as f32, lr);
        let d2 = backward_and_update(&d3, &pre3, &act2,
            &mut self.shadow.w3, &mut self.shadow.b3,
            HIDDEN3_SIZE, HIDDEN2_SIZE, HIDDEN2_SIZE as f32, lr);
        let d1 = backward_and_update(&d2, &pre2, &act1,
            &mut self.shadow.w2, &mut self.shadow.b2,
            HIDDEN2_SIZE, HIDDEN1_SIZE, HIDDEN1_SIZE as f32, lr);
        let _ = backward_and_update(&d1, &pre1, &input0,
            &mut self.shadow.w1, &mut self.shadow.b1,
            HIDDEN1_SIZE, STATE_SIZE, STATE_SIZE as f32, lr);

        // quantize f32 shadow weights back to i8 for inference
        self.shadow.quantize_into(&mut self.net);
        self.steps += 1;
    }
}

#[wasm_bindgen]
impl CpuPipeline {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let net = Network::new(0xCEEDEE); // Z NUTZ HA, gottem ;)
        let shadow = ShadowWeights::from_network(&net);
        CpuPipeline { net, shadow, lr: 1e-4, steps: 0 }
    }

    pub fn compute(&self, input: &str) -> String {
        self.compute_bytes(input.as_bytes())
    }

    pub fn compute_bytes(&self, input: &[u8]) -> String {
        let state = fold_bytes(input);
        let p = self.net.forward(&state);
        format!("#{:02x}{:02x}{:02x}{:02x}", p[0], p[1], p[2], p[3])
    }

    // one training step: forward + backward + weight update
    // returns the new output hex after update
    pub fn train(&mut self, input: &str, target_hex: &str) -> String {
        self.train_bytes(input.as_bytes(), target_hex)
    }

    pub fn train_bytes(&mut self, input: &[u8], target_hex: &str) -> String {
        let state = fold_bytes(input);
        let target = parse_hex_rgba(target_hex);
        self.train_state(&state, target);
        let p = self.net.forward(&state);
        format!("#{:02x}{:02x}{:02x}{:02x}", p[0], p[1], p[2], p[3])
    }

    pub fn set_learning_rate(&mut self, lr: f32) {
        self.lr = lr;
    }

    pub fn training_steps(&self) -> u32 {
        self.steps
    }

    // returns all layer activations as flat u8 array (bias-encoded: original + 128)
    // layout: [input(4096), h1(2048), h2(1024), h3(256), h4(64), h5(16), output(4)]
    pub fn activations(&self, input: &str) -> Vec<u8> {
        self.activations_bytes(input.as_bytes())
    }

    pub fn activations_bytes(&self, input: &[u8]) -> Vec<u8> {
        let state = fold_bytes(input);
        self.net.forward_all(&state)
    }

    // layer sizes so JS doesn't need to hardcode them
    pub fn layer_sizes(&self) -> Vec<u32> {
        vec![
            STATE_SIZE as u32,
            HIDDEN1_SIZE as u32,
            HIDDEN2_SIZE as u32,
            HIDDEN3_SIZE as u32,
            HIDDEN4_SIZE as u32,
            HIDDEN5_SIZE as u32,
            OUTPUT_SIZE as u32,
        ]
    }
}
