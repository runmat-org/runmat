pub const EYE_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct EyeParams {
    rows: u32,
    cols: u32,
    diag_len: u32,
    slices: u32,
    stride_slice: u32,
    diag_total: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read_write> Out: Tensor;
@group(0) @binding(1) var<uniform> params: EyeParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if params.diag_total == 0u {
        return;
    }
    let idx = gid.x;
    if idx >= params.diag_total {
        return;
    }
    let diag = idx % params.diag_len;
    let slice = idx / params.diag_len;
    let base = slice * params.stride_slice + diag + diag * params.rows;
    Out.data[base] = 1.0;
}
"#;

pub const EYE_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct EyeParams {
    rows: u32,
    cols: u32,
    diag_len: u32,
    slices: u32,
    stride_slice: u32,
    diag_total: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read_write> Out: Tensor;
@group(0) @binding(1) var<uniform> params: EyeParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if params.diag_total == 0u {
        return;
    }
    let idx = gid.x;
    if idx >= params.diag_total {
        return;
    }
    let diag = idx % params.diag_len;
    let slice = idx / params.diag_len;
    let base = slice * params.stride_slice + diag + diag * params.rows;
    Out.data[base] = 1.0;
}
"#;
