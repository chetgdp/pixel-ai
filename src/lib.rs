use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
mod gpu;

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
    (x as i64 * 256 / (x.abs() as i64 + scale as i64)).clamp(-128, 127) as i8
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
            // shift from [-127,127] to [0,255]
            output[i] = (isigmoid(sum, HIDDEN5_SIZE as i32) as i32 + 128) as u8;
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
            result.push((isigmoid(sum, HIDDEN5_SIZE as i32) as i32 + 128) as u8);
        }

        result
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
pub struct CpuPipeline;

#[wasm_bindgen]
impl CpuPipeline {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        CpuPipeline
    }

    pub fn compute(&self, input: &str) -> String {
        self.compute_bytes(input.as_bytes())
    }

    pub fn compute_bytes(&self, input: &[u8]) -> String {
        let state = fold_bytes(input);
        let net = Network::new(0xCEEDEE); // Z NUTZ HA, gottem ;)
        let p = net.forward(&state);
        format!("#{:02x}{:02x}{:02x}{:02x}", p[0], p[1], p[2], p[3])
    }

    // returns all layer activations as flat u8 array (bias-encoded: original + 128)
    // layout: [input(4096), h1(2048), h2(1024), h3(256), h4(64), h5(16), output(4)]
    pub fn activations(&self, input: &str) -> Vec<u8> {
        self.activations_bytes(input.as_bytes())
    }

    pub fn activations_bytes(&self, input: &[u8]) -> Vec<u8> {
        let state = fold_bytes(input);
        let net = Network::new(0xCEEDEE);
        net.forward_all(&state)
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
