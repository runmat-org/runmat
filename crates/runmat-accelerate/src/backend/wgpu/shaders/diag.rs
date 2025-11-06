pub const DIAG_FROM_VECTOR_SHADER_F64: &str = r#"
struct Vector {
    data: array<f64>,
};

struct Matrix {
    data: array<f64>,
};

struct DiagVecParams {
    len: u32,
    size: u32,
    offset: i32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> Input: Vector;
@group(0) @binding(1) var<storage, read_write> Output: Matrix;
@group(0) @binding(2) var<uniform> params: DiagVecParams;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.len {
        return;
    }
    var row = idx;
    var col = idx;
    if params.offset >= 0 {
        col = idx + u32(params.offset);
    } else {
        let shift = u32(-params.offset);
        row = idx + shift;
    }
    let base = row + col * params.size;
    Output.data[base] = Input.data[idx];
}
"#;

pub const DIAG_FROM_VECTOR_SHADER_F32: &str = r#"
struct Vector {
    data: array<f32>,
};

struct Matrix {
    data: array<f32>,
};

struct DiagVecParams {
    len: u32,
    size: u32,
    offset: i32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> Input: Vector;
@group(0) @binding(1) var<storage, read_write> Output: Matrix;
@group(0) @binding(2) var<uniform> params: DiagVecParams;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.len {
        return;
    }
    var row = idx;
    var col = idx;
    if params.offset >= 0 {
        col = idx + u32(params.offset);
    } else {
        let shift = u32(-params.offset);
        row = idx + shift;
    }
    let base = row + col * params.size;
    Output.data[base] = Input.data[idx];
}
"#;

pub const DIAG_EXTRACT_SHADER_F64: &str = r#"
struct Matrix {
    data: array<f64>,
};

struct Vector {
    data: array<f64>,
};

struct DiagExtractParams {
    rows: u32,
    cols: u32,
    offset: i32,
    diag_len: u32,
};

@group(0) @binding(0) var<storage, read> Input: Matrix;
@group(0) @binding(1) var<storage, read_write> Output: Vector;
@group(0) @binding(2) var<uniform> params: DiagExtractParams;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.diag_len {
        return;
    }
    var row = idx;
    var col = idx;
    if params.offset >= 0 {
        col = idx + u32(params.offset);
    } else {
        let shift = u32(-params.offset);
        row = idx + shift;
    }
    let base = row + col * params.rows;
    Output.data[idx] = Input.data[base];
}
"#;

pub const DIAG_EXTRACT_SHADER_F32: &str = r#"
struct Matrix {
    data: array<f32>,
};

struct Vector {
    data: array<f32>,
};

struct DiagExtractParams {
    rows: u32,
    cols: u32,
    offset: i32,
    diag_len: u32,
};

@group(0) @binding(0) var<storage, read> Input: Matrix;
@group(0) @binding(1) var<storage, read_write> Output: Vector;
@group(0) @binding(2) var<uniform> params: DiagExtractParams;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.diag_len {
        return;
    }
    var row = idx;
    var col = idx;
    if params.offset >= 0 {
        col = idx + u32(params.offset);
    } else {
        let shift = u32(-params.offset);
        row = idx + shift;
    }
    let base = row + col * params.rows;
    Output.data[idx] = Input.data[base];
}
"#;
