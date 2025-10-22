pub const TRANSPOSE_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct Params {
    rows: u32,
    cols: u32,
    len: u32,
    offset: u32,
};

@group(0) @binding(0) var<storage, read> A: Tensor;
@group(0) @binding(1) var<storage, read_write> Out: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

fn isNan(x: f64) -> bool { return x != x; }

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local = gid.x;
    if local >= params.len {
        return;
    }
    let rows = params.rows;
    let cols = params.cols;
    let idx = params.offset + local;
    let row = idx % rows;
    let col = idx / rows;
    let out_rows = cols;
    let tgt_idx = col + row * out_rows;
    Out.data[tgt_idx] = A.data[idx];
}
"#;

pub const TRANSPOSE_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct Params {
    rows: u32,
    cols: u32,
    len: u32,
    offset: u32,
};

@group(0) @binding(0) var<storage, read> A: Tensor;
@group(0) @binding(1) var<storage, read_write> Out: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

fn isNan(x: f32) -> bool { return x != x; }

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local = gid.x;
    if local >= params.len {
        return;
    }
    let rows = params.rows;
    let cols = params.cols;
    let idx = params.offset + local;
    let row = idx % rows;
    let col = idx / rows;
    let out_rows = cols;
    let tgt_idx = col + row * out_rows;
    Out.data[tgt_idx] = A.data[idx];
}
"#;


