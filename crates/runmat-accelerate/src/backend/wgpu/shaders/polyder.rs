pub const POLYDER_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct Params {
    input_len: u32,
    output_len: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> Input: Tensor;
@group(0) @binding(1) var<storage, read_write> Output: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.output_len) {
        return;
    }
    if (params.input_len <= 1u) {
        Output.data[idx] = 0.0;
        return;
    }
    let degree = params.input_len - idx - 1u;
    Output.data[idx] = f64(degree) * Input.data[idx];
}
"#;

pub const POLYDER_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct Params {
    input_len: u32,
    output_len: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> Input: Tensor;
@group(0) @binding(1) var<storage, read_write> Output: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.output_len) {
        return;
    }
    if (params.input_len <= 1u) {
        Output.data[idx] = 0.0;
        return;
    }
    let degree = params.input_len - idx - 1u;
    Output.data[idx] = f32(degree) * Input.data[idx];
}
"#;
