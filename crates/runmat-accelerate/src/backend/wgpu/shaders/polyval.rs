pub const POLYVAL_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct Params {
    chunk_len: u32,
    coeff_len: u32,
    offset: u32,
    has_mu: u32,
    mu_mean: f64,
    mu_scale: f64,
};

@group(0) @binding(0) var<storage, read> Coeffs: Tensor;
@group(0) @binding(1) var<storage, read> Points: Tensor;
@group(0) @binding(2) var<storage, read_write> Output: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local_idx = gid.x;
    if (local_idx >= params.chunk_len) {
        return;
    }

    let idx = params.offset + local_idx;
    var x = Points.data[idx];
    if (params.has_mu != 0u) {
        x = (x - params.mu_mean) / params.mu_scale;
    }

    var acc = Coeffs.data[0u];
    var k: u32 = 1u;
    loop {
        if (k >= params.coeff_len) {
            break;
        }
        acc = acc * x + Coeffs.data[k];
        k = k + 1u;
    }

    Output.data[idx] = acc;
}
"#;

pub const POLYVAL_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct Params {
    chunk_len: u32,
    coeff_len: u32,
    offset: u32,
    has_mu: u32,
    mu_mean: f32,
    mu_scale: f32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> Coeffs: Tensor;
@group(0) @binding(1) var<storage, read> Points: Tensor;
@group(0) @binding(2) var<storage, read_write> Output: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local_idx = gid.x;
    if (local_idx >= params.chunk_len) {
        return;
    }

    let idx = params.offset + local_idx;
    var x = Points.data[idx];
    if (params.has_mu != 0u) {
        x = (x - params.mu_mean) / params.mu_scale;
    }

    var acc = Coeffs.data[0u];
    var k: u32 = 1u;
    loop {
        if (k >= params.coeff_len) {
            break;
        }
        acc = acc * x + Coeffs.data[k];
        k = k + 1u;
    }

    Output.data[idx] = acc;
}
"#;
