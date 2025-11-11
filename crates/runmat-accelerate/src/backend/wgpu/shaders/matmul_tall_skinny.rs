pub const MATMUL_TALL_SKINNY_F32: &str = r#"
struct Tensor { data: array<f32>, };
struct Params {
    m: u32, n: u32, k: u32,
    lda: u32, ldb: u32, ldc: u32,
    offset_a: u32, offset_b: u32, offset_out: u32,
    flags: u32,
};
@group(0) @binding(0) var<storage, read> A: Tensor;
@group(0) @binding(1) var<storage, read> B: Tensor;
@group(0) @binding(2) var<storage, read_write> Out: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let row = gid.y;
    let col = gid.x;
    let row_active = row < params.m;
    let col_active = col < params.n;
    let transpose_a = (params.flags & 1u) != 0u;
    let transpose_b = (params.flags & 2u) != 0u;
    var acc: f32 = 0.0;
    var c: f32 = 0.0;
    for (var kk: u32 = 0u; kk < params.k; kk = kk + 1u) {
        var a_idx = params.offset_a + row + kk * params.lda;
        if (transpose_a) {
            a_idx = params.offset_a + kk + row * params.lda;
        }
        var b_idx = params.offset_b + kk + col * params.ldb;
        if (transpose_b) {
            b_idx = params.offset_b + col + kk * params.ldb;
        }
        var product: f32 = 0.0;
        if (row_active && col_active && kk < params.k) {
            product = A.data[a_idx] * B.data[b_idx];
        }
        let y = product - c;
        let t = acc + y;
        c = (t - acc) - y;
        acc = t;
    }

    if (row_active && col_active) {
        let out_idx = params.offset_out + row + col * params.ldc;
        Out.data[out_idx] = acc;
    }
}
"#;

pub const MATMUL_TALL_SKINNY_F64: &str = r#"
struct Tensor { data: array<f64>, };
struct Params {
    m: u32, n: u32, k: u32,
    lda: u32, ldb: u32, ldc: u32,
    offset_a: u32, offset_b: u32, offset_out: u32,
    flags: u32,
};
@group(0) @binding(0) var<storage, read> A: Tensor;
@group(0) @binding(1) var<storage, read> B: Tensor;
@group(0) @binding(2) var<storage, read_write> Out: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let row = gid.y;
    let col = gid.x;
    let row_active = row < params.m;
    let col_active = col < params.n;
    let transpose_a = (params.flags & 1u) != 0u;
    let transpose_b = (params.flags & 2u) != 0u;
    var acc: f64 = 0.0;
    var c: f64 = 0.0;
    for (var kk: u32 = 0u; kk < params.k; kk = kk + 1u) {
        var a_idx = params.offset_a + row + kk * params.lda;
        if (transpose_a) {
            a_idx = params.offset_a + kk + row * params.lda;
        }
        var b_idx = params.offset_b + kk + col * params.ldb;
        if (transpose_b) {
            b_idx = params.offset_b + col + kk * params.ldb;
        }
        var product: f64 = 0.0;
        if (row_active && col_active && kk < params.k) {
            product = A.data[a_idx] * B.data[b_idx];
        }
        let y = product - c;
        let t = acc + y;
        c = (t - acc) - y;
        acc = t;
    }

    if (row_active && col_active) {
        let out_idx = params.offset_out + row + col * params.ldc;
        Out.data[out_idx] = acc;
    }
}
"#;
