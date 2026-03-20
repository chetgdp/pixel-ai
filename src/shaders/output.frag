#version 300 es
precision highp float;

// output layer: 1x1 pixel, all 4 RGBA outputs computed in one fragment
// the neural network literally outputs one pixel
uniform sampler2D u_weights;
uniform sampler2D u_biases;
uniform sampler2D u_input;

out vec4 fragColor;

void main() {
    vec4 result;
    for (int i = 0; i < 4; i++) {
        float sum = texelFetch(u_biases, ivec2(i, 0), 0).r * 255.0 - 128.0;
        for (int j = 0; j < 16; j++) {
            float w = texelFetch(u_weights, ivec2(j, i), 0).r * 255.0 - 128.0;
            float a = texelFetch(u_input, ivec2(j, 0), 0).r * 255.0 - 128.0;
            sum += w * a;
        }
        // isigmoid + shift to [0,255]
        float act = trunc((sum * 127.0) / (abs(sum) + 16.0));
        result[i] = (act + 128.0) / 255.0;
    }
    fragColor = result;
}
