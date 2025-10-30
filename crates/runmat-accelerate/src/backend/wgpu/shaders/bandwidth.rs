pub const BANDWIDTH_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct BandwidthOutput {
    lower: atomic<u32>,
    upper: atomic<u32>,
};

struct BandwidthParams {
    rows: u32,
    cols: u32,
    len: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read_write> output: BandwidthOutput;
@group(0) @binding(2) var<uniform> params: BandwidthParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.len || params.rows == 0u {
        return;
    }
    let rows = params.rows;
    let row = idx % rows;
    let col = idx / rows;
    let value = input0.data[idx];
    if value != 0.0 {
        if row >= col {
            let delta = row - col;
            atomicMax(&output.lower, delta);
        } else {
            let delta = col - row;
            atomicMax(&output.upper, delta);
        }
    }
}
"#;

pub const BANDWIDTH_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct BandwidthOutput {
    lower: atomic<u32>,
    upper: atomic<u32>,
};

struct BandwidthParams {
    rows: u32,
    cols: u32,
    len: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> input0: Tensor;
@group(0) @binding(1) var<storage, read_write> output: BandwidthOutput;
@group(0) @binding(2) var<uniform> params: BandwidthParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.len || params.rows == 0u {
        return;
    }
    let rows = params.rows;
    let row = idx % rows;
    let col = idx / rows;
    let value = input0.data[idx];
    if value != 0.0 {
        if row >= col {
            let delta = row - col;
            atomicMax(&output.lower, delta);
        } else {
            let delta = col - row;
            atomicMax(&output.upper, delta);
        }
    }
}
"#;
