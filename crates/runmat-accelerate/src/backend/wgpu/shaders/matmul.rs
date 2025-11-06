pub const MATMUL_SHADER_F64: &str = r#"
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
    let transpose_a = (params.flags & 1u) != 0u;
    let transpose_b = (params.flags & 2u) != 0u;

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
            if (!transpose_a) {
                a_in = A.data[base_a + global_row + a_col * lda];
            } else {
                a_in = A.data[base_a + a_col + global_row * lda];
            }
        }
        var b_in: f64 = 0.0;
        if (b_row < params.k && global_col < params.n) {
            if (!transpose_b) {
                b_in = B.data[base_b + b_row + global_col * ldb];
            } else {
                b_in = B.data[base_b + global_col + b_row * ldb];
            }
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
    flags: u32,
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
    let transpose_a = (params.flags & 1u) != 0u;
    let transpose_b = (params.flags & 2u) != 0u;

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
            if (!transpose_a) {
                a_in = A.data[base_a + global_row + a_col * lda];
            } else {
                a_in = A.data[base_a + a_col + global_row * lda];
            }
        }
        var b_in: f32 = 0.0;
        if (b_row < params.k && global_col < params.n) {
            if (!transpose_b) {
                b_in = B.data[base_b + b_row + global_col * ldb];
            } else {
                b_in = B.data[base_b + global_col + b_row * ldb];
            }
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

pub const MATMUL_SHADER_VEC4_F32: &str = r#"
struct TensorVec4 {
    data: array<vec4<f32>>,
};

struct TensorScalar {
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
    flags: u32,
};

var<workgroup> tileA: array<array<vec4<f32>, @MT@>, @MT@>;
var<workgroup> tileB: array<array<f32, @MT@>, @MT@>;

@group(0) @binding(0) var<storage, read> A: TensorVec4;
@group(0) @binding(1) var<storage, read> B: TensorScalar;
@group(0) @binding(2) var<storage, read_write> Out: TensorScalar;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(@MT@, @MT@, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let tile = @MT@u;
    let row_block = wid.y * tile + lid.y;
    let row_base = row_block * 4u;
    let col = wid.x * tile + lid.x;

    let row_active = row_base < params.m;
    let col_active = col < params.n;

    let lda_vec = params.lda / 4u;
    let ldc_vec = params.ldc / 4u;
    let offset_a_vec = params.offset_a / 4u;
    let offset_out_vec = params.offset_out / 4u;

    var acc = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    let tiles_k = (params.k + tile - 1u) / tile;

    for (var t: u32 = 0u; t < tiles_k; t = t + 1u) {
        let kk_base = t * tile;
        let a_col = kk_base + lid.x;

        var a_vec = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        if (row_active && a_col < params.k) {
            let idx = offset_a_vec + row_block + a_col * lda_vec;
            a_vec = A.data[idx];
        }
        tileA[lid.y][lid.x] = a_vec;

        let b_row = kk_base + lid.y;
        var b_val: f32 = 0.0;
        if (col_active && b_row < params.k) {
            let idx = params.offset_b + b_row + col * params.ldb;
            b_val = B.data[idx];
        }
        tileB[lid.y][lid.x] = b_val;

        workgroupBarrier();

        for (var p: u32 = 0u; p < tile; p = p + 1u) {
            let a_block = tileA[lid.y][p];
            let b_block = tileB[p][lid.x];
            acc = acc + a_block * b_block;
        }

        workgroupBarrier();
    }

    if (!(row_active && col_active)) {
        return;
    }

    for (var lane: u32 = 0u; lane < 4u; lane = lane + 1u) {
        let row = row_base + lane;
        if (row >= params.m) {
            break;
        }
        let value = acc[lane];
        let out_idx = params.offset_out + row + col * params.ldc;
        Out.data[out_idx] = value;
    }
}
"#;

pub const MATMUL_SHADER_VEC4_F64: &str = MATMUL_SHADER_F64;

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
    flags: u32,
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
    let transpose_a = (params.flags & 1u) != 0u;
    let transpose_b = (params.flags & 2u) != 0u;
    for (var kk: u32 = 0u; kk < params.k; kk = kk + 1u) {
        var a_idx = base_a + row + kk * lda;
        if (transpose_a) {
            a_idx = base_a + kk + row * lda;
        }
        var b_idx = base_b + kk + col * ldb;
        if (transpose_b) {
            b_idx = base_b + col + kk * ldb;
        }
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
    flags: u32,
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
    let transpose_a = (params.flags & 1u) != 0u;
    let transpose_b = (params.flags & 2u) != 0u;    
    for (var kk: u32 = 0u; kk < params.k; kk = kk + 1u) {
        var a_idx = base_a + row + kk * lda;
        if (transpose_a) {
            a_idx = base_a + kk + row * lda;
        }
        var b_idx = base_b + kk + col * ldb;
        if (transpose_b) {
            b_idx = base_b + col + kk * ldb;
        }
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
