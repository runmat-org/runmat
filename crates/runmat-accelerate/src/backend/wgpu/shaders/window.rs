pub const WINDOW_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct WindowParams {
    len: u32,
    total: u32,
    chunk: u32,
    offset: u32,
    kind: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read_write> Out: Tensor;
@group(0) @binding(1) var<uniform> params: WindowParams;

fn coeff(kind: u32, idx: u32, len: u32) -> f64 {
    if (len == 0u) {
        return 0.0;
    }
    if (len == 1u) {
        return 1.0;
    }
    let phase = 2.0 * 3.141592653589793 * f64(idx) / f64(len - 1u);
    switch kind {
        case 0u: { return 0.5 - 0.5 * cos(phase); }
        case 1u: { return 0.54 - 0.46 * cos(phase); }
        default: { return 0.42 - 0.5 * cos(phase) + 0.08 * cos(2.0 * phase); }
    }
}

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (params.chunk == 0u || params.total == 0u) {
        return;
    }
    let local = gid.x;
    if (local >= params.chunk) {
        return;
    }
    let idx = params.offset + local;
    if (idx >= params.total) {
        return;
    }
    Out.data[idx] = coeff(params.kind, idx, params.len);
}
"#;

pub const WINDOW_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct WindowParams {
    len: u32,
    total: u32,
    chunk: u32,
    offset: u32,
    kind: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read_write> Out: Tensor;
@group(0) @binding(1) var<uniform> params: WindowParams;

fn coeff(kind: u32, idx: u32, len: u32) -> f32 {
    if (len == 0u) {
        return 0.0;
    }
    if (len == 1u) {
        return 1.0;
    }
    let phase = 2.0 * 3.1415927 * f32(idx) / f32(len - 1u);
    switch kind {
        case 0u: { return 0.5 - 0.5 * cos(phase); }
        case 1u: { return 0.54 - 0.46 * cos(phase); }
        default: { return 0.42 - 0.5 * cos(phase) + 0.08 * cos(2.0 * phase); }
    }
}

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (params.chunk == 0u || params.total == 0u) {
        return;
    }
    let local = gid.x;
    if (local >= params.chunk) {
        return;
    }
    let idx = params.offset + local;
    if (idx >= params.total) {
        return;
    }
    Out.data[idx] = coeff(params.kind, idx, params.len);
}
"#;
