#version 300 es
// hidden.frag
precision highp float;

// hidden layers 1-5: each pixel computes 4 neurons (RGBA), dot product + isigmoid
uniform sampler2D u_weights;
uniform sampler2D u_biases;
uniform sampler2D u_input;
uniform int u_input_size;

out vec4 fragColor;

void main() {
    int pixel = int(gl_FragCoord.x);

    // convert between u8 and i8 that the weights are supposed to be
    // biases: read all 4 at once from RGBA-packed texture
    vec4 sums = texelFetch(u_biases, ivec2(pixel, 0), 0) * 255.0 - 128.0;
    // unrolled by 4: read one input pixel (4 values) per iteration
    // 5 fetches per iteration for 16 multiply-adds instead of 8 fetches
    for (int j = 0; j < u_input_size / 4; j++) {
        vec4 a = texelFetch(u_input, ivec2(j, 0), 0) * 255.0 - 128.0;
        vec4 w0 = texelFetch(u_weights, ivec2(j * 4 + 0, pixel), 0) * 255.0 - 128.0;
        vec4 w1 = texelFetch(u_weights, ivec2(j * 4 + 1, pixel), 0) * 255.0 - 128.0;
        vec4 w2 = texelFetch(u_weights, ivec2(j * 4 + 2, pixel), 0) * 255.0 - 128.0;
        vec4 w3 = texelFetch(u_weights, ivec2(j * 4 + 3, pixel), 0) * 255.0 - 128.0;
        sums += w0 * a.x + w1 * a.y + w2 * a.z + w3 * a.w;
    }

    // isigmoid: (x * 127) / (|x| + scale), truncated to match Rust's as i8
    vec4 act = trunc((sums * 127.0) / (abs(sums) + float(u_input_size)));
    // bias encode back to [0,1] for RGBA8 storage
    fragColor = (act + 128.0) / 255.0;
}
