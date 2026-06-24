pub const POLYINT_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct Params {
    input_len: u32,
    output_len: u32,
    storage_factor: u32,
    _pad0: u32,
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
    let storage_factor = max(params.storage_factor, 1u);
    let logical_idx = idx / storage_factor;
    let lane = idx - logical_idx * storage_factor;
    if (params.input_len == 0u) {
        if (logical_idx == 0u) {
            if (lane == 0u) {
                Output.data[idx] = params.constant;
            } else {
                Output.data[idx] = 0.0;
            }
        }
        return;
    }
    let last = params.input_len;
    if (logical_idx == last) {
        if (lane == 0u) {
            Output.data[idx] = params.constant;
        } else {
            Output.data[idx] = 0.0;
        }
        return;
    }
    let power = params.input_len - logical_idx;
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
    storage_factor: u32,
    _pad0: u32,
    constant: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
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
    let storage_factor = max(params.storage_factor, 1u);
    let logical_idx = idx / storage_factor;
    let lane = idx - logical_idx * storage_factor;
    if (params.input_len == 0u) {
        if (logical_idx == 0u) {
            if (lane == 0u) {
                Output.data[idx] = params.constant;
            } else {
                Output.data[idx] = 0.0;
            }
        }
        return;
    }
    let last = params.input_len;
    if (logical_idx == last) {
        if (lane == 0u) {
            Output.data[idx] = params.constant;
        } else {
            Output.data[idx] = 0.0;
        }
        return;
    }
    let power = params.input_len - logical_idx;
    Output.data[idx] = Input.data[idx] / f32(power);
}
"#;
