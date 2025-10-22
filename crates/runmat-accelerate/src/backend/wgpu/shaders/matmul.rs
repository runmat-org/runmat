pub const MATMUL_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct Params {
    m: u32,
    n: u32,
    k: u32,
    lda: u32,
    ldb: u32,
    ldc: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> A: Tensor;
@group(0) @binding(1) var<storage, read> B: Tensor;
@group(0) @binding(2) var<storage, read_write> Out: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let col = gid.x;
    let row = gid.y;
    if row >= params.m || col >= params.n {
        return;
    }
    var acc: f64 = 0.0;
    let lda = params.lda;
    let ldb = params.ldb;
    let ldc = params.ldc;
    for (var kk: u32 = 0u; kk < params.k; kk = kk + 1u) {
        let a_idx = row + kk * lda;
        let b_idx = kk + col * ldb;
        acc = acc + A.data[a_idx] * B.data[b_idx];
    }
    let out_idx = row + col * ldc;
    Out.data[out_idx] = acc;
}
"#;

pub const MATMUL_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct Params {
    m: u32,
    n: u32,
    k: u32,
    lda: u32,
    ldb: u32,
    ldc: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> A: Tensor;
@group(0) @binding(1) var<storage, read> B: Tensor;
@group(0) @binding(2) var<storage, read_write> Out: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let col = gid.x;
    let row = gid.y;
    if row >= params.m || col >= params.n {
        return;
    }
    var acc: f32 = 0.0;
    let lda = params.lda;
    let ldb = params.ldb;
    let ldc = params.ldc;
    for (var kk: u32 = 0u; kk < params.k; kk = kk + 1u) {
        let a_idx = row + kk * lda;
        let b_idx = kk + col * ldb;
        acc = acc + A.data[a_idx] * B.data[b_idx];
    }
    let out_idx = row + col * ldc;
    Out.data[out_idx] = acc;
}
"#;


