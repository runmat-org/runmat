pub const MATMUL_SMALLK_SHADER_F64: &str = r#"
struct Tensor { data: array<f64>, };
struct Params {
    m: u32,
    n: u32,
    k: u32,
    lda: u32,
    ldb: u32,
    ldc: u32,
    offset_a: u32,
    offset_b: u32,
    offset_out: u32,
    _pad: u32,
};

const SMALL_K_MAX: u32 = 8u;

@group(0) @binding(0) var<storage, read> A: Tensor;
@group(0) @binding(1) var<storage, read> B: Tensor;
@group(0) @binding(2) var<storage, read_write> Out: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(@MT@, @MT@, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let tile = @MT@u;
    let global_row = wid.y * tile + lid.y;
    let global_col = wid.x * tile + lid.x;

    if global_row >= params.m || global_col >= params.n {
        return;
    }

    let lda = params.lda;
    let ldb = params.ldb;
    let ldc = params.ldc;
    let base_a = params.offset_a;
    let base_b = params.offset_b;
    let base_out = params.offset_out;

    var acc: f64 = 0.0;
    let k = params.k;
    for (var kk: u32 = 0u; kk < SMALL_K_MAX; kk = kk + 1u) {
        if (kk >= k) {
            break;
        }
        let a_idx = base_a + global_row + kk * lda;
        let b_idx = base_b + kk + global_col * ldb;
        acc = acc + A.data[a_idx] * B.data[b_idx];
    }

    let out_idx = base_out + global_row + global_col * ldc;
    Out.data[out_idx] = acc;
}
"#;

pub const MATMUL_SMALLK_SHADER_F32: &str = r#"
struct Tensor { data: array<f32>, };
struct Params {
    m: u32,
    n: u32,
    k: u32,
    lda: u32,
    ldb: u32,
    ldc: u32,
    offset_a: u32,
    offset_b: u32,
    offset_out: u32,
    _pad: u32,
};

const SMALL_K_MAX: u32 = 8u;

@group(0) @binding(0) var<storage, read> A: Tensor;
@group(0) @binding(1) var<storage, read> B: Tensor;
@group(0) @binding(2) var<storage, read_write> Out: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(@MT@, @MT@, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let tile = @MT@u;
    let global_row = wid.y * tile + lid.y;
    let global_col = wid.x * tile + lid.x;

    if global_row >= params.m || global_col >= params.n {
        return;
    }

    let lda = params.lda;
    let ldb = params.ldb;
    let ldc = params.ldc;
    let base_a = params.offset_a;
    let base_b = params.offset_b;
    let base_out = params.offset_out;

    var acc: f32 = 0.0;
    let k = params.k;
    for (var kk: u32 = 0u; kk < SMALL_K_MAX; kk = kk + 1u) {
        if (kk >= k) {
            break;
        }
        let a_idx = base_a + global_row + kk * lda;
        let b_idx = base_b + kk + global_col * ldb;
        acc = acc + A.data[a_idx] * B.data[b_idx];
    }

    let out_idx = base_out + global_row + global_col * ldc;
    Out.data[out_idx] = acc;
}
"#;

