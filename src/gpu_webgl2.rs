/* gpu_webgl2.rs 
a neural network compute pipeline via WebGL2
rather than using compute shaders such as is possible with WebGPU
you make use of webgl texturess and fragment shaders as a hack
this file is mostly boilerplate binding together the shader inputs and outputs

each pixel of a texture holds 4 neurons (RGBA-packed)
the textures are our intermediate layer data

t_in (input texture)
  → hidden shader → th1 (texture, 4 neurons per pixel via RGBA)
    → hidden shader → th2
      → hidden shader → th3
        → hidden shader → th4
          → hidden shader → th5
            → output shader → t_out (1x1 pixel, RGBA = 4 outputs)

where the first input texture is what has been folded. its a 1 by SIZE/4 texture.
same goes for the intermediate data layers
weight textures are width/input by height/(output/4), RGBA-packed by output group
biases are SIZE/4 by 1, RGBA-packed
*/
use wasm_bindgen::prelude::*;
//use wasm_bindgen::JsCast;
// almost everything we do here goes through this rendering context
use web_sys::WebGl2RenderingContext as GL;
use web_sys::{
    WebGlFramebuffer, WebGlProgram, WebGlShader, WebGlTexture,
    WebGlUniformLocation, WebGlVertexArrayObject,
};

// import shader files
const VS: &str = include_str!("shaders/fullscreen.vert");
const FS_HIDDEN: &str = include_str!("shaders/hidden.frag");
const FS_OUTPUT: &str = include_str!("shaders/output.frag");

// turn an array of i8s from the base network into a vector of unsigned bytes
fn bias_encode(data: &[i8]) -> Vec<u8> {
    data.iter().map(|&v| (v as i16 + 128) as u8).collect()
}

// interleave weights by groups of 4 output neurons for RGBA packing
// pixel (x, y) RGBA = w[y*4+0][x], w[y*4+1][x], w[y*4+2][x], w[y*4+3][x]
fn flat_rgba(arr: &[&[i8]]) -> Vec<i8> {
    let out_size = arr.len();
    let in_size = arr[0].len();
    let mut result = Vec::with_capacity(out_size * in_size);
    for group in (0..out_size).step_by(4) {
        for j in 0..in_size {
            result.push(arr[group][j]);
            result.push(arr[group + 1][j]);
            result.push(arr[group + 2][j]);
            result.push(arr[group + 3][j]);
        }
    }
    result
}

// take a kind of shader and a source string and compile it
fn compile(gl: &GL, kind: u32, src: &str) -> Result<WebGlShader, JsValue> {
    let shader = gl.create_shader(kind).ok_or("could not create shader")?;
    gl.shader_source(&shader, src);
    gl.compile_shader(&shader);
    if !gl.get_shader_parameter(&shader, GL::COMPILE_STATUS).as_bool().unwrap_or(false) {
        return Err(gl.get_shader_info_log(&shader).unwrap_or_default().into());
    }
    Ok(shader)
}

// vertex shader + fragment shader
fn link(gl: &GL, vs: &WebGlShader, fs: &WebGlShader) -> Result<WebGlProgram, JsValue> {
    let p = gl.create_program().ok_or("could not create program")?;
    gl.attach_shader(&p, vs);
    gl.attach_shader(&p, fs);
    gl.link_program(&p);
    if !gl.get_program_parameter(&p, GL::LINK_STATUS).as_bool().unwrap_or(false) {
        return Err(gl.get_program_info_log(&p).unwrap_or_default().into());
    }
    Ok(p)
}

// take bias encoded data (i8->u8) and turn it into an RGBA8 texture
// 4 values packed per pixel, so width should be original_size/4
fn rgba8_data_tex(gl: &GL, w: i32, h: i32, data: &[u8]) -> Result<WebGlTexture, JsValue> {
    let t = gl.create_texture().ok_or("failed to create texture")?;
    gl.bind_texture(GL::TEXTURE_2D, Some(&t));
    // what in the magic fuckery is this function name?
    gl.tex_image_2d_with_i32_and_i32_and_i32_and_format_and_type_and_opt_u8_array(
        GL::TEXTURE_2D, 0, GL::RGBA8 as i32, w, h, 0, GL::RGBA, GL::UNSIGNED_BYTE, Some(data),
    )?;
    // important to use nearest filtering to get exact values
    gl.tex_parameteri(GL::TEXTURE_2D, GL::TEXTURE_MIN_FILTER, GL::NEAREST as i32);
    gl.tex_parameteri(GL::TEXTURE_2D, GL::TEXTURE_MAG_FILTER, GL::NEAREST as i32);
    // we clamp so texture does not repeat
    gl.tex_parameteri(GL::TEXTURE_2D, GL::TEXTURE_WRAP_S, GL::CLAMP_TO_EDGE as i32);
    gl.tex_parameteri(GL::TEXTURE_2D, GL::TEXTURE_WRAP_T, GL::CLAMP_TO_EDGE as i32);
    Ok(t)
}

// empty RGBA8 texture for activation render targets
fn rgba8_tex(gl: &GL, w: i32, h: i32) -> Result<WebGlTexture, JsValue> {
    let t = gl.create_texture().ok_or("failed to create texture")?;
    gl.bind_texture(GL::TEXTURE_2D, Some(&t));
    gl.tex_image_2d_with_i32_and_i32_and_i32_and_format_and_type_and_opt_u8_array(
        GL::TEXTURE_2D, 0, GL::RGBA8 as i32, w, h, 0, GL::RGBA, GL::UNSIGNED_BYTE, None,
    )?;
    gl.tex_parameteri(GL::TEXTURE_2D, GL::TEXTURE_MIN_FILTER, GL::NEAREST as i32);
    gl.tex_parameteri(GL::TEXTURE_2D, GL::TEXTURE_MAG_FILTER, GL::NEAREST as i32);
    gl.tex_parameteri(GL::TEXTURE_2D, GL::TEXTURE_WRAP_S, GL::CLAMP_TO_EDGE as i32);
    gl.tex_parameteri(GL::TEXTURE_2D, GL::TEXTURE_WRAP_T, GL::CLAMP_TO_EDGE as i32);
    Ok(t)
}

// make a framebuffer so we can write to an activation texture
fn make_fbo(gl: &GL, tex: &WebGlTexture) -> Result<WebGlFramebuffer, JsValue> {
    let f = gl.create_framebuffer().ok_or("create_framebuffer")?;
    gl.bind_framebuffer(GL::FRAMEBUFFER, Some(&f));
    gl.framebuffer_texture_2d(
        GL::FRAMEBUFFER, GL::COLOR_ATTACHMENT0, GL::TEXTURE_2D, Some(tex), 0,
    );
    Ok(f)
}

#[wasm_bindgen]
pub struct GpuPipeline {
    gl: GL,
    hidden_prog: WebGlProgram,
    output_prog: WebGlProgram,
    vao: WebGlVertexArrayObject,
    // weights
    tw1: WebGlTexture, tb1: WebGlTexture,
    tw2: WebGlTexture, tb2: WebGlTexture,
    tw3: WebGlTexture, tb3: WebGlTexture,
    tw4: WebGlTexture, tb4: WebGlTexture,
    tw5: WebGlTexture, tb5: WebGlTexture,
    tw6: WebGlTexture, tb6: WebGlTexture,
    // input
    t_in: WebGlTexture,
    // activation targets
    th1: WebGlTexture, th2: WebGlTexture, th3: WebGlTexture,
    th4: WebGlTexture, th5: WebGlTexture,
    // fbos
    fh1: WebGlFramebuffer, fh2: WebGlFramebuffer, fh3: WebGlFramebuffer,
    fh4: WebGlFramebuffer, fh5: WebGlFramebuffer, f_out: WebGlFramebuffer,
    // hidden shader uniforms
    h_w: WebGlUniformLocation, h_b: WebGlUniformLocation,
    h_i: WebGlUniformLocation, h_s: WebGlUniformLocation,
    // output shader uniforms
    o_w: WebGlUniformLocation, o_b: WebGlUniformLocation, o_i: WebGlUniformLocation,
}

#[wasm_bindgen]
impl GpuPipeline {
    #[wasm_bindgen(constructor)]
    pub fn new(canvas_id: &str) -> Result<GpuPipeline, JsValue> {
        // could error this part better back to the user
        // we need a display:none canvas?
        let doc = web_sys::window().unwrap().document().unwrap();
        let canvas = doc
            .get_element_by_id(canvas_id)
            .ok_or("canvas not found")?
            .dyn_into::<web_sys::HtmlCanvasElement>()?;
        let gl = canvas
            .get_context("webgl2")?
            .ok_or("webgl2 not supported")?
            .dyn_into::<GL>()?;

        // byte aligned, not padded
        gl.pixel_storei(GL::UNPACK_ALIGNMENT, 1);

        // compile shaders
        // we have two shader programs
        // one that compute the hidden layers 1-5 and one for the output layer
        // they both reuse the simple vertex shader: vs
        let vs = compile(&gl, GL::VERTEX_SHADER, VS)?;
        let hidden_prog = link(&gl, &vs, &compile(&gl, GL::FRAGMENT_SHADER, FS_HIDDEN)?)?;
        let output_prog = link(&gl, &vs, &compile(&gl, GL::FRAGMENT_SHADER, FS_OUTPUT)?)?;

        // init network, flatten weights for texture upload
        let net = crate::Network::new(0xCEEDEE); // Z NUTZ HA, gottem ;)

        let w1_refs: Vec<&[i8]> = net.w1.iter().map(|r| r.as_slice()).collect();
        let w2_refs: Vec<&[i8]> = net.w2.iter().map(|r| r.as_slice()).collect();
        let w3_refs: Vec<&[i8]> = net.w3.iter().map(|r| r.as_slice()).collect();
        let w4_refs: Vec<&[i8]> = net.w4.iter().map(|r| r.as_slice()).collect();
        let w5_refs: Vec<&[i8]> = net.w5.iter().map(|r| r.as_slice()).collect();
        let w6_refs: Vec<&[i8]> = net.w6.iter().map(|r| r.as_slice()).collect();

        use crate::*;

        // weight & bias textures (RGBA8, bias-encoded: value + 128)
        // weights: input_size wide, output_size/4 tall (4 outputs per pixel via RGBA)
        // biases: size/4 wide (4 biases per pixel via RGBA)
        let tw1 = rgba8_data_tex(&gl, STATE_SIZE as i32, (HIDDEN1_SIZE / 4) as i32, &bias_encode(&flat_rgba(&w1_refs)))?;
        let tb1 = rgba8_data_tex(&gl, (HIDDEN1_SIZE / 4) as i32, 1, &bias_encode(&net.b1))?;
        let tw2 = rgba8_data_tex(&gl, HIDDEN1_SIZE as i32, (HIDDEN2_SIZE / 4) as i32, &bias_encode(&flat_rgba(&w2_refs)))?;
        let tb2 = rgba8_data_tex(&gl, (HIDDEN2_SIZE / 4) as i32, 1, &bias_encode(&net.b2))?;
        let tw3 = rgba8_data_tex(&gl, HIDDEN2_SIZE as i32, (HIDDEN3_SIZE / 4) as i32, &bias_encode(&flat_rgba(&w3_refs)))?;
        let tb3 = rgba8_data_tex(&gl, (HIDDEN3_SIZE / 4) as i32, 1, &bias_encode(&net.b3))?;
        let tw4 = rgba8_data_tex(&gl, HIDDEN3_SIZE as i32, (HIDDEN4_SIZE / 4) as i32, &bias_encode(&flat_rgba(&w4_refs)))?;
        let tb4 = rgba8_data_tex(&gl, (HIDDEN4_SIZE / 4) as i32, 1, &bias_encode(&net.b4))?;
        let tw5 = rgba8_data_tex(&gl, HIDDEN4_SIZE as i32, (HIDDEN5_SIZE / 4) as i32, &bias_encode(&flat_rgba(&w5_refs)))?;
        let tb5 = rgba8_data_tex(&gl, (HIDDEN5_SIZE / 4) as i32, 1, &bias_encode(&net.b5))?;
        let tw6 = rgba8_data_tex(&gl, HIDDEN5_SIZE as i32, (OUTPUT_SIZE / 4) as i32, &bias_encode(&flat_rgba(&w6_refs)))?;
        let tb6 = rgba8_data_tex(&gl, (OUTPUT_SIZE / 4) as i32, 1, &bias_encode(&net.b6))?;
        let t_in = rgba8_data_tex(&gl, (STATE_SIZE / 4) as i32, 1, &vec![128u8; STATE_SIZE])?;

        // activation render targets (RGBA8)
        // empty textures for the pipeline to write all 4 channels into
        // SIZE/4 by 1 (4 neurons per pixel)
        let th1 = rgba8_tex(&gl, (HIDDEN1_SIZE / 4) as i32, 1)?;
        let th2 = rgba8_tex(&gl, (HIDDEN2_SIZE / 4) as i32, 1)?;
        let th3 = rgba8_tex(&gl, (HIDDEN3_SIZE / 4) as i32, 1)?;
        let th4 = rgba8_tex(&gl, (HIDDEN4_SIZE / 4) as i32, 1)?;
        let th5 = rgba8_tex(&gl, (HIDDEN5_SIZE / 4) as i32, 1)?;
        let t_out = rgba8_tex(&gl, 1, 1)?;

        // framebuffers
        let fh1 = make_fbo(&gl, &th1)?;
        let fh2 = make_fbo(&gl, &th2)?;
        let fh3 = make_fbo(&gl, &th3)?;
        let fh4 = make_fbo(&gl, &th4)?;
        let fh5 = make_fbo(&gl, &th5)?;
        let f_out = make_fbo(&gl, &t_out)?;
        gl.bind_framebuffer(GL::FRAMEBUFFER, None);

        // fullscreen triangle that goes past viewport
        // we operate on the -1 to 1 clip space while the tri goes to 3.0
        // need this as a trigger for the pipeline to run
        // we use one oversized tris vertices, no need for a quad (2 tris)
        let vao = gl.create_vertex_array().ok_or("failed to create_vertex_array")?;
        gl.bind_vertex_array(Some(&vao));
        let vbo = gl.create_buffer().ok_or("failed to create_buffer")?;
        gl.bind_buffer(GL::ARRAY_BUFFER, Some(&vbo));
        let vertices: [f32; 6] = [-1.0, -1.0, 3.0, -1.0, -1.0, 3.0];
        unsafe {
            let view = js_sys::Float32Array::view(&vertices);
            gl.buffer_data_with_array_buffer_view(GL::ARRAY_BUFFER, &view, GL::STATIC_DRAW);
        }
        gl.enable_vertex_attrib_array(0);
        gl.vertex_attrib_pointer_with_i32(0, 2, GL::FLOAT, false, 0, 0);
        gl.bind_vertex_array(None);

        // uniform locations
        // these are basically variables that the shaders have access to 
        // give them a location so GPU can access its memory
        let loc = |p: &WebGlProgram, n: &str| gl.get_uniform_location(p, n).unwrap();
        let h_w = loc(&hidden_prog, "u_weights");
        let h_b = loc(&hidden_prog, "u_biases");
        let h_i = loc(&hidden_prog, "u_input");
        let h_s = loc(&hidden_prog, "u_input_size");
        let o_w = loc(&output_prog, "u_weights");
        let o_b = loc(&output_prog, "u_biases");
        let o_i = loc(&output_prog, "u_input");

        Ok(GpuPipeline {
            gl, hidden_prog, output_prog, vao,
            tw1, tb1, tw2, tb2, tw3, tb3, tw4, tb4, tw5, tb5, tw6, tb6,
            t_in, th1, th2, th3, th4, th5,
            fh1, fh2, fh3, fh4, fh5, f_out,
            h_w, h_b, h_i, h_s,
            o_w, o_b, o_i,
        })
    }

    pub fn compute_bytes(&self, input: &[u8]) -> Result<String, JsValue> {
        let state = crate::fold_bytes(input);
        self.compute_state(&state)
    }

    // run the forward pass on the GPU, returns hex color string
    pub fn compute(&self, input: &str) -> Result<String, JsValue> {
        // fold still done on cpu, this is still the bottleneck
        let state = crate::fold_bytes(input.as_bytes());
        self.compute_state(&state)
    }

    fn compute_state(&self, state: &[i8; crate::STATE_SIZE]) -> Result<String, JsValue> {
        let gl = &self.gl;
        let encoded = bias_encode(state);

        // upload folded input (RGBA-packed, 4 values per pixel)
        gl.bind_texture(GL::TEXTURE_2D, Some(&self.t_in));
        gl.tex_image_2d_with_i32_and_i32_and_i32_and_format_and_type_and_opt_u8_array(
            GL::TEXTURE_2D, 0, GL::RGBA8 as i32, (crate::STATE_SIZE / 4) as i32, 1, 0,
            GL::RGBA, GL::UNSIGNED_BYTE, Some(&encoded),
        )?;

        gl.bind_vertex_array(Some(&self.vao));
        gl.use_program(Some(&self.hidden_prog));

        let bind = |unit: u32, tex: &WebGlTexture| {
            gl.active_texture(GL::TEXTURE0 + unit);
            gl.bind_texture(GL::TEXTURE_2D, Some(tex));
        };

        // layer 1: 4096 -> 2048
        gl.bind_framebuffer(GL::FRAMEBUFFER, Some(&self.fh1));
        gl.viewport(0, 0, (crate::HIDDEN1_SIZE / 4) as i32, 1);
        bind(0, &self.tw1); bind(1, &self.tb1); bind(2, &self.t_in);
        gl.uniform1i(Some(&self.h_w), 0);
        gl.uniform1i(Some(&self.h_b), 1);
        gl.uniform1i(Some(&self.h_i), 2);
        gl.uniform1i(Some(&self.h_s), crate::STATE_SIZE as i32);
        gl.draw_arrays(GL::TRIANGLES, 0, 3);

        // layer 2: 2048 -> 1024
        gl.bind_framebuffer(GL::FRAMEBUFFER, Some(&self.fh2));
        gl.viewport(0, 0, (crate::HIDDEN2_SIZE / 4) as i32, 1);
        bind(0, &self.tw2); bind(1, &self.tb2); bind(2, &self.th1);
        gl.uniform1i(Some(&self.h_s), crate::HIDDEN1_SIZE as i32);
        gl.draw_arrays(GL::TRIANGLES, 0, 3);

        // layer 3: 1024 -> 256
        gl.bind_framebuffer(GL::FRAMEBUFFER, Some(&self.fh3));
        gl.viewport(0, 0, (crate::HIDDEN3_SIZE / 4) as i32, 1);
        bind(0, &self.tw3); bind(1, &self.tb3); bind(2, &self.th2);
        gl.uniform1i(Some(&self.h_s), crate::HIDDEN2_SIZE as i32);
        gl.draw_arrays(GL::TRIANGLES, 0, 3);

        // layer 4: 256 -> 64
        gl.bind_framebuffer(GL::FRAMEBUFFER, Some(&self.fh4));
        gl.viewport(0, 0, (crate::HIDDEN4_SIZE / 4) as i32, 1);
        bind(0, &self.tw4); bind(1, &self.tb4); bind(2, &self.th3);
        gl.uniform1i(Some(&self.h_s), crate::HIDDEN3_SIZE as i32);
        gl.draw_arrays(GL::TRIANGLES, 0, 3);

        // layer 5: 64 -> 16
        gl.bind_framebuffer(GL::FRAMEBUFFER, Some(&self.fh5));
        gl.viewport(0, 0, (crate::HIDDEN5_SIZE / 4) as i32, 1);
        bind(0, &self.tw5); bind(1, &self.tb5); bind(2, &self.th4);
        gl.uniform1i(Some(&self.h_s), crate::HIDDEN4_SIZE as i32);
        gl.draw_arrays(GL::TRIANGLES, 0, 3);

        // output layer: 16 -> RGBA (one pixel)
        gl.use_program(Some(&self.output_prog));
        gl.bind_framebuffer(GL::FRAMEBUFFER, Some(&self.f_out));
        gl.viewport(0, 0, 1, 1);
        bind(0, &self.tw6); bind(1, &self.tb6); bind(2, &self.th5);
        gl.uniform1i(Some(&self.o_w), 0);
        gl.uniform1i(Some(&self.o_b), 1);
        gl.uniform1i(Some(&self.o_i), 2);
        gl.draw_arrays(GL::TRIANGLES, 0, 3);

        // read the one pixel
        let mut px = [0u8; 4];
        gl.read_pixels_with_opt_u8_array(
            0, 0, 1, 1, GL::RGBA, GL::UNSIGNED_BYTE, Some(&mut px),
        )?;

        gl.bind_vertex_array(None);
        gl.bind_framebuffer(GL::FRAMEBUFFER, None);

        Ok(format!("#{:02x}{:02x}{:02x}{:02x}", px[0], px[1], px[2], px[3]))
    }
}
