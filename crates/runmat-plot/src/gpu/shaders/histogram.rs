pub mod counts {
    pub const F32: &str = r#"override WORKGROUP_SIZE: u32 = {{WORKGROUP_SIZE}}u;

struct HistogramParams {
    min_value: f32,
    inv_bin_width: f32,
    sample_count: u32,
    bin_count: u32,
};

@group(0) @binding(0)
var<storage, read> samples: array<f32>;

@group(0) @binding(1)
var<storage, read_write> counts: array<atomic<u32>>;

@group(0) @binding(2)
var<uniform> params: HistogramParams;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.sample_count) {
        return;
    }

    let value = samples[idx];
    let normalized = (value - params.min_value) * params.inv_bin_width;
    let raw_bin = i32(floor(normalized));
    let clamped = clamp(raw_bin, 0, i32(params.bin_count) - 1);
    let bin_index = u32(clamped);
    atomicAdd(&counts[bin_index], 1u);
}
"#;

    pub const F64: &str = r#"override WORKGROUP_SIZE: u32 = {{WORKGROUP_SIZE}}u;

struct HistogramParams {
    min_value: f32,
    inv_bin_width: f32,
    sample_count: u32,
    bin_count: u32,
};

@group(0) @binding(0)
var<storage, read> samples: array<f64>;

@group(0) @binding(1)
var<storage, read_write> counts: array<atomic<u32>>;

@group(0) @binding(2)
var<uniform> params: HistogramParams;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.sample_count) {
        return;
    }

    let value = f32(samples[idx]);
    let normalized = (value - params.min_value) * params.inv_bin_width;
    let raw_bin = i32(floor(normalized));
    let clamped = clamp(raw_bin, 0, i32(params.bin_count) - 1);
    let bin_index = u32(clamped);
    atomicAdd(&counts[bin_index], 1u);
}
"#;
}

pub const CONVERT: &str = r#"override WORKGROUP_SIZE: u32 = {{WORKGROUP_SIZE}}u;

struct ConvertParams {
    bin_count: u32,
    _pad0: vec3<u32>,
    scale: f32,
    _pad1: vec3<u32>,
};

@group(0) @binding(0)
var<storage, read> counts: array<u32>;

@group(0) @binding(1)
var<storage, read_write> values: array<f32>;

@group(0) @binding(2)
var<uniform> params: ConvertParams;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.bin_count) {
        return;
    }

    values[idx] = f32(counts[idx]) * params.scale;
}
"#;
