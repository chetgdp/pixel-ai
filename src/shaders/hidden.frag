#version 300 es
precision highp float;

// hidden layers 1-3: one fragment per neuron, dot product + isigmoid
uniform sampler2D u_weights;
uniform sampler2D u_biases;
uniform sampler2D u_input;
uniform int u_input_size;

out vec4 fragColor;

void main() {
    int neuron = int(gl_FragCoord.x);

    float sum = texelFetch(u_biases, ivec2(neuron, 0), 0).r * 255.0 - 128.0;
    for (int j = 0; j < u_input_size; j++) {
        float w = texelFetch(u_weights, ivec2(j, neuron), 0).r * 255.0 - 128.0;
        float a = texelFetch(u_input, ivec2(j, 0), 0).r * 255.0 - 128.0;
        sum += w * a;
    }

    // isigmoid: (x * 127) / (|x| + scale), truncated to match Rust's as i8
    float act = trunc((sum * 127.0) / (abs(sum) + float(u_input_size)));
    // bias encode back to [0,1] for RGBA8 storage
    fragColor = vec4((act + 128.0) / 255.0, 0.0, 0.0, 1.0);
}
