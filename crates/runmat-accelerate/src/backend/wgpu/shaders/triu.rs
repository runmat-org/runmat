pub const TRIU_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct TriuParams {
    len: u32,
    start: u32,
    rows: u32,
    cols: u32,
    plane: u32,
    diag_offset: i32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read_write> output: Tensor;
@group(0) @binding(2) var<uniform> params: TriuParams;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.len {
        return;
    }
    let index: u32 = params.start + gid.x;
    let plane = params.plane;
    if plane == 0u {
        output.data[index] = input0.data[index];
        return;
    }
    let rows = params.rows;
    let within = index % plane;
    let row = within % rows;
    let col = within / rows;

    let row_i = i32(row);
    let col_i = i32(col);
    let diff = col_i - row_i;
    if diff < params.diag_offset {
        output.data[index] = 0.0;
    } else {
        output.data[index] = input0.data[index];
    }
}
"#;

pub const TRIU_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct TriuParams {
    len: u32,
    start: u32,
    rows: u32,
    cols: u32,
    plane: u32,
    diag_offset: i32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read_write> output: Tensor;
@group(0) @binding(2) var<uniform> params: TriuParams;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.len {
        return;
    }
    let index: u32 = params.start + gid.x;
    let plane = params.plane;
    if plane == 0u {
        output.data[index] = input0.data[index];
        return;
    }
    let rows = params.rows;
    let within = index % plane;
    let row = within % rows;
    let col = within / rows;

    let row_i = i32(row);
    let col_i = i32(col);
    let diff = col_i - row_i;
    if diff < params.diag_offset {
        output.data[index] = 0.0;
    } else {
        output.data[index] = input0.data[index];
    }
}
"#;
