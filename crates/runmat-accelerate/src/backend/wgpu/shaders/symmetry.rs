pub const SYMMETRY_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct SymmetryResult {
    flag: atomic<u32>,
};

struct SymmetryParams {
    rows: u32,
    cols: u32,
    len: u32,
    mode: u32,
    tolerance: f64,
    _pad: f64,
    _pad2: f64,
};

@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read_write> output: SymmetryResult;
@group(0) @binding(2) var<uniform> params: SymmetryParams;

fn mark_false() {
    atomicStore(&output.flag, 0u);
}

fn is_nan_f64(value: f64) -> bool {
    return value != value;
}

fn is_inf_f64(value: f64) -> bool {
    let inf = bitcast<f64>(0x7ff0000000000000u);
    return value == inf || value == -inf;
}

fn finite_f64(value: f64) -> bool {
    return !(is_nan_f64(value) || is_inf_f64(value));
}

fn real_within(value: f64, reference: f64, tol: f64) -> bool {
    if value == reference {
        return true;
    }
    if !(finite_f64(value) && finite_f64(reference)) {
        return false;
    }
    return abs(value - reference) <= tol;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.len || params.rows == 0u {
        return;
    }
    if atomicLoad(&output.flag) == 0u {
        return;
    }
    let rows = params.rows;
    let row = idx % rows;
    let col = idx / rows;
    let value = input0.data[idx];

    if row == col {
        if params.mode == 1u {
            if !real_within(value, 0.0, params.tolerance) {
                mark_false();
            }
        }
        return;
    }
    if col < row {
        return;
    }
    let mate_index = (col * rows) + row;
    if mate_index >= params.len {
        mark_false();
        return;
    }
    let mate = input0.data[mate_index];
    var reference: f64 = mate;
    if params.mode == 1u {
        reference = -mate;
    }
    if value == reference {
        return;
    }
    if !(finite_f64(value) && finite_f64(reference)) {
        mark_false();
        return;
    }
    if abs(value - reference) > params.tolerance {
        mark_false();
    }
}
"#;

pub const SYMMETRY_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct SymmetryResult {
    flag: atomic<u32>,
};

struct SymmetryParams {
    rows: u32,
    cols: u32,
    len: u32,
    mode: u32,
    tolerance: f32,
    _pad: vec3<f32>,
};

@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read_write> output: SymmetryResult;
@group(0) @binding(2) var<uniform> params: SymmetryParams;

fn mark_false() {
    atomicStore(&output.flag, 0u);
}

fn is_nan_f32(value: f32) -> bool {
    return value != value;
}

fn is_inf_f32(value: f32) -> bool {
    let inf = bitcast<f32>(0x7f800000u);
    return value == inf || value == -inf;
}

fn finite_f32(value: f32) -> bool {
    return !(is_nan_f32(value) || is_inf_f32(value));
}

fn real_within(value: f32, reference: f32, tol: f32) -> bool {
    if value == reference {
        return true;
    }
    if !(finite_f32(value) && finite_f32(reference)) {
        return false;
    }
    return abs(value - reference) <= tol;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.len || params.rows == 0u {
        return;
    }
    if atomicLoad(&output.flag) == 0u {
        return;
    }
    let rows = params.rows;
    let row = idx % rows;
    let col = idx / rows;
    let value = input0.data[idx];

    if row == col {
        if params.mode == 1u {
            if !real_within(value, 0.0, params.tolerance) {
                mark_false();
            }
        }
        return;
    }
    if col < row {
        return;
    }
    let mate_index = (col * rows) + row;
    if mate_index >= params.len {
        mark_false();
        return;
    }
    let mate = input0.data[mate_index];
    var reference: f32 = mate;
    if params.mode == 1u {
        reference = -mate;
    }
    if value == reference {
        return;
    }
    if !(finite_f32(value) && finite_f32(reference)) {
        mark_false();
        return;
    }
    if abs(value - reference) > params.tolerance {
        mark_false();
    }
}
"#;
