#version 300 es
// output.frag
precision highp float;

// output layer: 1x1 pixel, all 4 RGBA outputs computed in one fragment
// the neural network literally outputs one pixel
uniform sampler2D u_weights;
uniform sampler2D u_biases;
uniform sampler2D u_input;

out vec4 fragColor;

void main() {
    // biases and weights are RGBA-packed: 4 output neurons per pixel
    vec4 sums = texelFetch(u_biases, ivec2(0, 0), 0) * 255.0 - 128.0;
    // unrolled by 4: read one input pixel (4 values) per iteration
    for (int j = 0; j < 4; j++) {
        vec4 a = texelFetch(u_input, ivec2(j, 0), 0) * 255.0 - 128.0;
        vec4 w0 = texelFetch(u_weights, ivec2(j * 4 + 0, 0), 0) * 255.0 - 128.0;
        vec4 w1 = texelFetch(u_weights, ivec2(j * 4 + 1, 0), 0) * 255.0 - 128.0;
        vec4 w2 = texelFetch(u_weights, ivec2(j * 4 + 2, 0), 0) * 255.0 - 128.0;
        vec4 w3 = texelFetch(u_weights, ivec2(j * 4 + 3, 0), 0) * 255.0 - 128.0;
        sums += w0 * a.x + w1 * a.y + w2 * a.z + w3 * a.w;
    }
    // isigmoid with scale³ (16³=4096), remap [-127,127] -> [0,255]
    vec4 act = trunc((sums * 127.0) / (abs(sums) + 4096.0));
    fragColor = (act + 127.0) * 255.0 / 254.0 / 255.0;
}
