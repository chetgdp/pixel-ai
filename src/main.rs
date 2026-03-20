use std::env;
use std::fs;
use std::io::{self, Read};

// fold to 1mb then division by 4 all the way down
const STATE_SIZE:   usize = 1024;
const HIDDEN1_SIZE: usize = 256;
const HIDDEN2_SIZE: usize = 64;
const HIDDEN3_SIZE: usize = 16;
const OUTPUT_SIZE:  usize = 4; // RGBA

// we use i8 because it matches our u8 output, ez
struct Network {
    w1: [[i8; STATE_SIZE]; HIDDEN1_SIZE],
    b1: [i8; HIDDEN1_SIZE],
    w2: [[i8; HIDDEN1_SIZE]; HIDDEN2_SIZE],
    b2: [i8; HIDDEN2_SIZE],
    w3: [[i8; HIDDEN2_SIZE]; HIDDEN3_SIZE],
    b3: [i8; HIDDEN3_SIZE],
    w4: [[i8; HIDDEN3_SIZE]; OUTPUT_SIZE],
    b4: [i8; OUTPUT_SIZE],
}

// simple neural network
fn isigmoid(x: i32, scale: i32) -> i8 {
    ((x * 127) / (x.abs() + scale)) as i8
}

impl Network {
    // when we create a new network we want to init all the weights and biases at random
    fn new(seed: u64) -> Self {
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

        let mut w4 = [[0i8; HIDDEN3_SIZE]; OUTPUT_SIZE];
        let mut b4 = [0i8; OUTPUT_SIZE];
        for i in 0..OUTPUT_SIZE {
            for j in 0..HIDDEN3_SIZE { w4[i][j] = next(); }
            b4[i] = next();
        }

        Network { w1, b1, w2, b2, w3, b3, w4, b4 }
    }

    fn forward(&self, state: &[i8; STATE_SIZE]) -> [u8; 4] {
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

        let mut output = [0u8; 4];
        for i in 0..OUTPUT_SIZE {
            let mut sum = self.b4[i] as i32;
            for j in 0..HIDDEN3_SIZE {
                sum += self.w4[i][j] as i32 * h3[j] as i32;
            }
            // shift from [-127,127] to [0,255]
            output[i] = (isigmoid(sum, HIDDEN3_SIZE as i32) as i32 + 128) as u8;
        }

        output
    }
}

// TODO: less lossy fold — multiple passes, better mixing
fn fold_bytes(data: &[u8]) -> [i8; STATE_SIZE] {
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

fn main() {
    let data = match env::args().nth(1) {
        Some(arg) => {
            let path = std::path::Path::new(&arg);
            if path.exists() {
                fs::read(path).expect("could not read file")
            } else {
                arg.into_bytes()
            }
        }
        None => {
            let mut buf = Vec::new();
            io::stdin().read_to_end(&mut buf).expect("could not read stdin");
            buf
        }
    };

    if data.is_empty() {
        eprintln!("no input");
        std::process::exit(1);
    }

    let state = fold_bytes(&data);
    let net = Network::new(0xDEAD_BEEF);
    let pixel = net.forward(&state);

    println!(
        "rgba({}, {}, {}, {:.2})",
        pixel[0], pixel[1], pixel[2], pixel[3] as f32 / 255.0
    );
    println!("#{:02x}{:02x}{:02x}{:02x}", pixel[0], pixel[1], pixel[2], pixel[3]);

    // write a 64x64 PPM image
    let size = 64;
    let mut ppm = format!("P6\n{size} {size}\n255\n").into_bytes();
    for _ in 0..size * size {
        ppm.push(pixel[0]);
        ppm.push(pixel[1]);
        ppm.push(pixel[2]);
    }
    fs::write("pixel.ppm", &ppm).expect("could not write pixel.ppm");
    eprintln!("wrote pixel.ppm");
}
