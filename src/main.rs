use std::env;
use std::fs;
use std::io::{self, Read};

use pixel_ai::{fold_bytes, Network};

fn run() {
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
    let net = Network::new(0xCEEDEE);
    let pixel = net.forward(&state);

    println!(
        "rgba({}, {}, {}, {:.2})",
        pixel[0], pixel[1], pixel[2], pixel[3] as f32 / 255.0
    );
    println!("#{:02x}{:02x}{:02x}{:02x}", pixel[0], pixel[1], pixel[2], pixel[3]);

    // write a 1x1 PPM image
    let size = 1;
    let mut ppm = format!("P6\n{size} {size}\n255\n").into_bytes();
    for _ in 0..size * size {
        ppm.push(pixel[0]);
        ppm.push(pixel[1]);
        ppm.push(pixel[2]);
    }
    fs::write("pixel.ppm", &ppm).expect("could not write pixel.ppm");
    eprintln!("wrote pixel.ppm");
}

fn main() {
    // need 16MB stack for the ~10MB network on stack
    std::thread::Builder::new()
        .stack_size(32 * 1024 * 1024)
        .spawn(run)
        .unwrap()
        .join()
        .unwrap();
}
