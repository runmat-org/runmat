pub(crate) const LOGICAL_AND_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read> input1: Tensor;
@group(0) @binding(2) var<storage, read_write> output: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    let lhs = input0.data[idx];
    let rhs = input1.data[idx];
    let lhs_true = !(lhs == 0.0);
    let rhs_true = !(rhs == 0.0);
    let cond = lhs_true && rhs_true;
    output.data[idx] = select(0.0, 1.0, cond);
}
"#;

pub(crate) const LOGICAL_AND_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read> input1: Tensor;
@group(0) @binding(2) var<storage, read_write> output: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    let lhs = input0.data[idx];
    let rhs = input1.data[idx];
    let lhs_true = !(lhs == f64(0.0));
    let rhs_true = !(rhs == f64(0.0));
    let cond = lhs_true && rhs_true;
    output.data[idx] = select(f64(0.0), f64(1.0), cond);
}
"#;

pub(crate) const ELEM_EQ_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read> input1: Tensor;
@group(0) @binding(2) var<storage, read_write> output: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    let lhs = input0.data[idx];
    let rhs = input1.data[idx];
    let cond = lhs == rhs;
    output.data[idx] = select(0.0, 1.0, cond);
}
"#;

pub(crate) const ELEM_EQ_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read> input1: Tensor;
@group(0) @binding(2) var<storage, read_write> output: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    let lhs = input0.data[idx];
    let rhs = input1.data[idx];
    let cond = lhs == rhs;
    output.data[idx] = select(f64(0.0), f64(1.0), cond);
}
"#;

pub(crate) const ELEM_NE_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read> input1: Tensor;
@group(0) @binding(2) var<storage, read_write> output: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    let lhs = input0.data[idx];
    let rhs = input1.data[idx];
    let cond = !(lhs == rhs);
    output.data[idx] = select(0.0, 1.0, cond);
}
"#;

pub(crate) const ELEM_NE_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read> input1: Tensor;
@group(0) @binding(2) var<storage, read_write> output: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    let lhs = input0.data[idx];
    let rhs = input1.data[idx];
    let cond = !(lhs == rhs);
    output.data[idx] = select(f64(0.0), f64(1.0), cond);
}
"#;

pub(crate) const ELEM_LT_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read> input1: Tensor;
@group(0) @binding(2) var<storage, read_write> output: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    let lhs = input0.data[idx];
    let rhs = input1.data[idx];
    let cond = lhs < rhs;
    output.data[idx] = select(0.0, 1.0, cond);
}
"#;

pub(crate) const ELEM_LT_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read> input1: Tensor;
@group(0) @binding(2) var<storage, read_write> output: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    let lhs = input0.data[idx];
    let rhs = input1.data[idx];
    let cond = lhs < rhs;
    output.data[idx] = select(f64(0.0), f64(1.0), cond);
}
"#;

pub(crate) const ELEM_LE_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read> input1: Tensor;
@group(0) @binding(2) var<storage, read_write> output: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    let lhs = input0.data[idx];
    let rhs = input1.data[idx];
    let cond = lhs <= rhs;
    output.data[idx] = select(0.0, 1.0, cond);
}
"#;

pub(crate) const ELEM_LE_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read> input1: Tensor;
@group(0) @binding(2) var<storage, read_write> output: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    let lhs = input0.data[idx];
    let rhs = input1.data[idx];
    let cond = lhs <= rhs;
    output.data[idx] = select(f64(0.0), f64(1.0), cond);
}
"#;

pub(crate) const ELEM_GT_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read> input1: Tensor;
@group(0) @binding(2) var<storage, read_write> output: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    let lhs = input0.data[idx];
    let rhs = input1.data[idx];
    let cond = lhs > rhs;
    output.data[idx] = select(0.0, 1.0, cond);
}
"#;

pub(crate) const ELEM_GT_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read> input1: Tensor;
@group(0) @binding(2) var<storage, read_write> output: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    let lhs = input0.data[idx];
    let rhs = input1.data[idx];
    let cond = lhs > rhs;
    output.data[idx] = select(f64(0.0), f64(1.0), cond);
}
"#;

pub(crate) const ELEM_GE_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read> input1: Tensor;
@group(0) @binding(2) var<storage, read_write> output: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    let lhs = input0.data[idx];
    let rhs = input1.data[idx];
    let cond = lhs >= rhs;
    output.data[idx] = select(0.0, 1.0, cond);
}
"#;

pub(crate) const ELEM_GE_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read> input1: Tensor;
@group(0) @binding(2) var<storage, read_write> output: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    let lhs = input0.data[idx];
    let rhs = input1.data[idx];
    let cond = lhs >= rhs;
    output.data[idx] = select(f64(0.0), f64(1.0), cond);
}
"#;
pub(crate) const LOGICAL_OR_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read> input1: Tensor;
@group(0) @binding(2) var<storage, read_write> output: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    let lhs = input0.data[idx];
    let rhs = input1.data[idx];
    let lhs_true = !(lhs == 0.0);
    let rhs_true = !(rhs == 0.0);
    let cond = lhs_true || rhs_true;
    output.data[idx] = select(0.0, 1.0, cond);
}
"#;

pub(crate) const LOGICAL_OR_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};
struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};
@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read> input1: Tensor;
@group(0) @binding(2) var<storage, read_write> output: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    let lhs = input0.data[idx];
    let rhs = input1.data[idx];
    let lhs_true = !(lhs == f64(0.0));
    let rhs_true = !(rhs == f64(0.0));
    let cond = lhs_true || rhs_true;
    output.data[idx] = select(f64(0.0), f64(1.0), cond);
}
"#;

pub(crate) const LOGICAL_XOR_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read> input1: Tensor;
@group(0) @binding(2) var<storage, read_write> output: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    let lhs = input0.data[idx];
    let rhs = input1.data[idx];
    let lhs_true = !(lhs == 0.0);
    let rhs_true = !(rhs == 0.0);
    let cond = lhs_true != rhs_true;
    output.data[idx] = select(0.0, 1.0, cond);
}
"#;

pub(crate) const LOGICAL_XOR_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read> input1: Tensor;
@group(0) @binding(2) var<storage, read_write> output: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    let lhs = input0.data[idx];
    let rhs = input1.data[idx];
    let lhs_true = !(lhs == f64(0.0));
    let rhs_true = !(rhs == f64(0.0));
    let cond = lhs_true != rhs_true;
    output.data[idx] = select(f64(0.0), f64(1.0), cond);
}
"#;

pub(crate) const LOGICAL_NOT_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read_write> output: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    let value = input0.data[idx];
    let cond = (value != 0.0);
    output.data[idx] = select(1.0, 0.0, cond);
}
"#;

pub(crate) const LOGICAL_NOT_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};
struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read_write> output: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    let value = input0.data[idx];
    let cond = (value != f64(0.0));
    output.data[idx] = select(f64(1.0), f64(0.0), cond);
}
"#;

pub(crate) const LOGICAL_ISNAN_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};
@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read_write> output: Tensor;
@group(0) @binding(2) var<uniform> params: Params;
fn isNan(x: f32) -> bool { return x != x; }

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    let value = input0.data[idx];
    let cond = isNan(value);
    output.data[idx] = select(0.0, 1.0, cond);
}
"#;

pub(crate) const LOGICAL_ISNAN_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read_write> output: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

fn isNan(x: f64) -> bool { return x != x; }

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    let value = input0.data[idx];
    let cond = isNan(value);
    output.data[idx] = select(f64(0.0), f64(1.0), cond);
}
"#;
pub(crate) const LOGICAL_ISINF_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};
struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};
@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read_write> output: Tensor;
@group(0) @binding(2) var<uniform> params: Params;
fn isInf(x: f32) -> bool { return (x == x) && !(abs(x) < 3.4028234663852886e38); }
@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    let value = input0.data[idx];
    let cond = isInf(value);
    output.data[idx] = select(0.0, 1.0, cond);
}
"#;

pub(crate) const LOGICAL_ISINF_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read_write> output: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

fn isInf(x: f64) -> bool { return (x == x) && !(abs(x) < f64(1.7976931348623157e308)); }

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    let value = input0.data[idx];
    let cond = isInf(value);
    output.data[idx] = select(f64(0.0), f64(1.0), cond);
}
"#;

pub(crate) const LOGICAL_ISFINITE_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read_write> output: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

fn isFinite(x: f32) -> bool { return (x == x) && (abs(x) < 3.4028234663852886e38); }

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    let value = input0.data[idx];
    let cond = isFinite(value);
    output.data[idx] = select(0.0, 1.0, cond);
}
"#;

pub(crate) const LOGICAL_ISFINITE_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read_write> output: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

fn isFinite(x: f64) -> bool { return (x == x) && (abs(x) < f64(1.7976931348623157e308)); }

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    let value = input0.data[idx];
    let cond = isFinite(value);
    output.data[idx] = select(f64(0.0), f64(1.0), cond);
}
"#;
