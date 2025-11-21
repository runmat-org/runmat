pub const DIFF_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct DiffParams {
    stride_before: u32,
    segments: u32,
    segment_len: u32,
    segment_out: u32,
    block: u32,
    total_out: u32,
    total_in: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> Input: Tensor;
@group(0) @binding(1) var<storage, read_write> Output: Tensor;
@group(0) @binding(2) var<uniform> params: DiffParams;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (params.segment_out == 0u) {
        return;
    }
    let idx = gid.x;
    if (idx >= params.total_out) {
        return;
    }
    let segment_idx = idx / params.segment_out;
    if (segment_idx >= params.segments) {
        return;
    }
    let offset = idx % params.segment_out;
    let before = segment_idx % params.stride_before;
    let after = segment_idx / params.stride_before;
    let base = after * params.block;
    let i0 = base + before + offset * params.stride_before;
    let i1 = i0 + params.stride_before;
    if (i1 >= params.total_in || i0 >= params.total_in) {
        return;
    }
    let a = Input.data[i1];
    let b = Input.data[i0];
    Output.data[idx] = a - b;
}
"#;

pub const DIFF_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct DiffParams {
    stride_before: u32,
    segments: u32,
    segment_len: u32,
    segment_out: u32,
    block: u32,
    total_out: u32,
    total_in: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> Input: Tensor;
@group(0) @binding(1) var<storage, read_write> Output: Tensor;
@group(0) @binding(2) var<uniform> params: DiffParams;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (params.segment_out == 0u) {
        return;
    }
    let idx = gid.x;
    if (idx >= params.total_out) {
        return;
    }
    let segment_idx = idx / params.segment_out;
    if (segment_idx >= params.segments) {
        return;
    }
    let offset = idx % params.segment_out;
    let before = segment_idx % params.stride_before;
    let after = segment_idx / params.stride_before;
    let base = after * params.block;
    let i0 = base + before + offset * params.stride_before;
    let i1 = i0 + params.stride_before;
    if (i1 >= params.total_in || i0 >= params.total_in) {
        return;
    }
    let a = Input.data[i1];
    let b = Input.data[i0];
    Output.data[idx] = a - b;
}
"#;
