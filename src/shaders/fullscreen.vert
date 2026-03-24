#version 300 es
// fullscreen.vert
// what comes in from vertex buffer
// 
layout(location = 0) in vec2 a_pos;
void main() {
    // write to gl_Position which gets reasterized by GPU
    // this is later consumed by gl_FragCoord
    // vec2(apos.x, apos.y), z=0.0, w=1.0
    // so we make a 4d clip space, no depth from z and divide by w is no op
    gl_Position = vec4(a_pos, 0.0, 1.0);
}
