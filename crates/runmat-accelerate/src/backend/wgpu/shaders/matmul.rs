pub const MATMUL_SHADER_F64: &str = r#"
struct Tensor { data: array<f64>, };
struct Params {
    m: u32, n: u32, k: u32,
    lda: u32, ldb: u32, ldc: u32,
    offset_a: u32, offset_b: u32, offset_out: u32,
    _pad: u32,
};
@group(0) @binding(0) var<storage, read> A: Tensor;
@group(0) @binding(1) var<storage, read> B: Tensor;
@group(0) @binding(2) var<storage, read_write> Out: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> tileA: array<array<f64, @MT@>, @MT@>;
var<workgroup> tileB: array<array<f64, @MT@>, @MT@>;

@compute @workgroup_size(@MT@, @MT@, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let tile = @MT@u;
    let lda = params.lda;
    let ldb = params.ldb;
    let ldc = params.ldc;
    let base_a = params.offset_a;
    let base_b = params.offset_b;
    let base_out = params.offset_out;

    let global_row = wid.y * tile + lid.y;
    let global_col = wid.x * tile + lid.x;
    if (global_row >= params.m || global_col >= params.n) {
        // still participate in barriers to avoid divergence issues
        // but return before final write
    }

    var acc: f64 = 0.0;
    let tiles_k = (params.k + tile - 1u) / tile;
    for (var t: u32 = 0u; t < tiles_k; t = t + 1u) {
        let kk_base = t * tile;
        let a_col = kk_base + lid.x;
        let b_row = kk_base + lid.y;
        var a_in: f64 = 0.0;
        if (global_row < params.m && a_col < params.k) {
            a_in = A.data[base_a + global_row + a_col * lda];
        }
        var b_in: f64 = 0.0;
        if (b_row < params.k && global_col < params.n) {
            b_in = B.data[base_b + b_row + global_col * ldb];
        }
        tileA[lid.y][lid.x] = a_in;
        tileB[lid.y][lid.x] = b_in;
        workgroupBarrier();
        for (var p: u32 = 0u; p < tile; p = p + 1u) {
            let a_val = tileA[lid.y][p];
            let b_val = tileB[p][lid.x];
            acc = acc + a_val * b_val;
        }
        workgroupBarrier();
    }
    if (global_row < params.m && global_col < params.n) {
        Out.data[base_out + global_row + global_col * ldc] = acc;
    }
}
"#;

pub const MATMUL_SHADER_F32: &str = r#"
struct Tensor { data: array<f32>, };
struct Params {
    m: u32, n: u32, k: u32,
    lda: u32, ldb: u32, ldc: u32,
    offset_a: u32, offset_b: u32, offset_out: u32,
    _pad: u32,
};
@group(0) @binding(0) var<storage, read> A: Tensor;
@group(0) @binding(1) var<storage, read> B: Tensor;
@group(0) @binding(2) var<storage, read_write> Out: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> tileA: array<array<f32, @MT@>, @MT@>;
var<workgroup> tileB: array<array<f32, @MT@>, @MT@>;

@compute @workgroup_size(@MT@, @MT@, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let tile = @MT@u;
    let lda = params.lda;
    let ldb = params.ldb;
    let ldc = params.ldc;
    let base_a = params.offset_a;
    let base_b = params.offset_b;
    let base_out = params.offset_out;

    let global_row = wid.y * tile + lid.y;
    let global_col = wid.x * tile + lid.x;
    if (global_row >= params.m || global_col >= params.n) {
        // proceed to barriers; final write is guarded
    }

    var acc: f32 = 0.0;
    let tiles_k = (params.k + tile - 1u) / tile;
    for (var t: u32 = 0u; t < tiles_k; t = t + 1u) {
        let kk_base = t * tile;
        let a_col = kk_base + lid.x;
        let b_row = kk_base + lid.y;
        var a_in: f32 = 0.0;
        if (global_row < params.m && a_col < params.k) {
            a_in = A.data[base_a + global_row + a_col * lda];
        }
        var b_in: f32 = 0.0;
        if (b_row < params.k && global_col < params.n) {
            b_in = B.data[base_b + b_row + global_col * ldb];
        }
        tileA[lid.y][lid.x] = a_in;
        tileB[lid.y][lid.x] = b_in;
        workgroupBarrier();
        for (var p: u32 = 0u; p < tile; p = p + 1u) {
            let a_val = tileA[lid.y][p];
            let b_val = tileB[p][lid.x];
            acc = acc + a_val * b_val;
        }
        workgroupBarrier();
    }
    if (global_row < params.m && global_col < params.n) {
        Out.data[base_out + global_row + global_col * ldc] = acc;
    }
}
"#;

pub const MATMUL_EPILOGUE_SHADER_F64: &str = r#"
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
    offset_a: u32,
    offset_b: u32,
    offset_out: u32,
    _pad: u32,
};

struct EpilogueF64 {
    alpha: f64,
    beta: f64,
    clamp_min: f64,
    clamp_max: f64,
    pow_exponent: f64,
    flags: u32,
    _pad0: u32,
};

@group(0) @binding(0) var<storage, read> A: Tensor;
@group(0) @binding(1) var<storage, read> B: Tensor;
@group(0) @binding(2) var<storage, read_write> Out: Tensor;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read> Row: Tensor;
@group(0) @binding(5) var<storage, read> Col: Tensor;
@group(0) @binding(6) var<uniform> ep: EpilogueF64;

@compute @workgroup_size(@MT@, @MT@, 1)
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
    let base_a = params.offset_a;
    let base_b = params.offset_b;
    let base_out = params.offset_out;
    for (var kk: u32 = 0u; kk < params.k; kk = kk + 1u) {
        let a_idx = base_a + row + kk * lda;
        let b_idx = base_b + kk + col * ldb;
        acc = acc + A.data[a_idx] * B.data[b_idx];
    }
    var val = acc * ep.alpha + ep.beta;

    let flags = ep.flags;
    if ((flags & 1u) != 0u) {
        let s = Row.data[row];
        if ((flags & 4u) != 0u) { val = val / s; } else { val = val * s; }
    }
    if ((flags & 2u) != 0u) {
        let s = Col.data[col];
        if ((flags & 8u) != 0u) { val = val / s; } else { val = val * s; }
    }
    if ((flags & 16u) != 0u) {
        val = max(val, ep.clamp_min);
    }
    if ((flags & 32u) != 0u) {
        val = min(val, ep.clamp_max);
    }
    if ((flags & 64u) != 0u) {
        val = pow(val, ep.pow_exponent);
    }
    let out_idx = base_out + row + col * ldc;
    Out.data[out_idx] = val;
}
"#;

pub const MATMUL_EPILOGUE_SHADER_F32: &str = r#"
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
    offset_a: u32,
    offset_b: u32,
    offset_out: u32,
    _pad: u32,
};

struct EpilogueF32 {
    alpha: f32,
    beta: f32,
    clamp_min: f32,
    clamp_max: f32,
    pow_exponent: f32,
    flags: u32,
};

@group(0) @binding(0) var<storage, read> A: Tensor;
@group(0) @binding(1) var<storage, read> B: Tensor;
@group(0) @binding(2) var<storage, read_write> Out: Tensor;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read> Row: Tensor;
@group(0) @binding(5) var<storage, read> Col: Tensor;
@group(0) @binding(6) var<uniform> ep: EpilogueF32;

@compute @workgroup_size(@MT@, @MT@, 1)
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
    let base_a = params.offset_a;
    let base_b = params.offset_b;
    let base_out = params.offset_out;
    for (var kk: u32 = 0u; kk < params.k; kk = kk + 1u) {
        let a_idx = base_a + row + kk * lda;
        let b_idx = base_b + kk + col * ldb;
        acc = acc + A.data[a_idx] * B.data[b_idx];
    }
    var val = acc * ep.alpha + ep.beta;
    let flags = ep.flags;
    if ((flags & 1u) != 0u) {
        let s = Row.data[row];
        if ((flags & 4u) != 0u) { val = val / s; } else { val = val * s; }
    }
    if ((flags & 2u) != 0u) {
        let s = Col.data[col];
        if ((flags & 8u) != 0u) { val = val / s; } else { val = val * s; }
    }
    if ((flags & 16u) != 0u) {
        val = max(val, ep.clamp_min);
    }
    if ((flags & 32u) != 0u) {
        val = min(val, ep.clamp_max);
    }
    if ((flags & 64u) != 0u) {
        val = pow(val, ep.pow_exponent);
    }
    let out_idx = base_out + row + col * ldc;
    Out.data[out_idx] = val;
}
"#;
