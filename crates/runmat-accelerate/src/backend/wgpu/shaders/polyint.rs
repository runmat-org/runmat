pub const POLYINT_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct Params {
    input_len: u32,
    output_len: u32,
    constant: f64,
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
    if (params.input_len == 0u) {
        if (idx == 0u) {
            Output.data[0u] = params.constant;
        }
        return;
    }
    let last = params.output_len - 1u;
    if (idx == last) {
        Output.data[idx] = params.constant;
        return;
    }
    let power = params.input_len - idx;
    Output.data[idx] = Input.data[idx] / f64(power);
}
"#;

pub const POLYINT_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct Params {
    input_len: u32,
    output_len: u32,
    constant: f32,
    _pad0: f32,
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
    if (params.input_len == 0u) {
        if (idx == 0u) {
            Output.data[0u] = params.constant;
        }
        return;
    }
    let last = params.output_len - 1u;
    if (idx == last) {
        Output.data[idx] = params.constant;
        return;
    }
    let power = params.input_len - idx;
    Output.data[idx] = Input.data[idx] / f32(power);
}
"#;
