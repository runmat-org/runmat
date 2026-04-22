pub const GRADIENT_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct GradientParams {
    stride_before: u32,
    segment_len: u32,
    block: u32,
    total: u32,
    spacing: f64,
    _pad0: f64,
    _pad1: f64,
    _pad2: f64,
};

@group(0) @binding(0) var<storage, read> Input: Tensor;
@group(0) @binding(1) var<storage, read_write> Output: Tensor;
@group(0) @binding(2) var<uniform> params: GradientParams;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.total) {
        return;
    }
    if (params.segment_len <= 1u) {
        Output.data[idx] = 0.0;
        return;
    }

    let block = params.block;
    let after = idx / block;
    let within = idx % block;
    let before = within % params.stride_before;
    let k = within / params.stride_before;
    let base = after * block;
    let center = base + before + k * params.stride_before;

    if (k == 0u) {
        let right = center + params.stride_before;
        Output.data[idx] = (Input.data[right] - Input.data[center]) / params.spacing;
        return;
    }
    if (k + 1u == params.segment_len) {
        let left = center - params.stride_before;
        Output.data[idx] = (Input.data[center] - Input.data[left]) / params.spacing;
        return;
    }

    let left = center - params.stride_before;
    let right = center + params.stride_before;
    Output.data[idx] = (Input.data[right] - Input.data[left]) / (2.0 * params.spacing);
}
"#;

pub const GRADIENT_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct GradientParams {
    meta0: vec4<u32>,
    meta1: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> Input: Tensor;
@group(0) @binding(1) var<storage, read_write> Output: Tensor;
@group(0) @binding(2) var<uniform> params: GradientParams;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let stride_before = params.meta0.x;
    let segment_len = params.meta0.y;
    let block = params.meta0.z;
    let total = params.meta0.w;
    let spacing = params.meta1.x;

    if (idx >= total) {
        return;
    }
    if (segment_len <= 1u) {
        Output.data[idx] = 0.0;
        return;
    }

    let after = idx / block;
    let within = idx % block;
    let before = within % stride_before;
    let k = within / stride_before;
    let base = after * block;
    let center = base + before + k * stride_before;

    if (k == 0u) {
        let right = center + stride_before;
        Output.data[idx] = (Input.data[right] - Input.data[center]) / spacing;
        return;
    }
    if (k + 1u == segment_len) {
        let left = center - stride_before;
        Output.data[idx] = (Input.data[center] - Input.data[left]) / spacing;
        return;
    }

    let left = center - stride_before;
    let right = center + stride_before;
    Output.data[idx] = (Input.data[right] - Input.data[left]) / (2.0 * spacing);
}
"#;
