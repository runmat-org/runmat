pub mod counts {
    pub const F32_UNIFORM: &str = r#"const WORKGROUP_SIZE: u32 = {{WORKGROUP_SIZE}}u;

struct HistogramParams {
    min_value: f32,
    inv_bin_width: f32,
    sample_count: u32,
    bin_count: u32,
    accumulate_total: u32,
    _pad: vec3<u32>,
};

@group(0) @binding(0)
var<storage, read> samples: array<f32>;

@group(0) @binding(1)
var<storage, read_write> counts: array<atomic<u32>>;

@group(0) @binding(2)
var<storage, read_write> total_weight: atomic<u32>;

@group(0) @binding(3)
var<uniform> params: HistogramParams;

fn atomic_add_f32(target: ptr<storage, atomic<u32>>, value: f32) {
    var old_bits = atomicLoad(target);
    loop {
        let old_value = bitcast<f32>(old_bits);
        let new_bits = bitcast<u32>(old_value + value);
        let exchange = atomicCompareExchangeWeak(target, old_bits, new_bits);
        if (exchange.exchanged) {
            break;
        }
        old_bits = exchange.old_value;
    }
}

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

    atomic_add_f32(&counts[bin_index], 1.0);
    if (params.accumulate_total != 0u) {
        atomic_add_f32(&total_weight, 1.0);
    }
}
"#;

    pub const F32_WEIGHTS_F32: &str = r#"const WORKGROUP_SIZE: u32 = {{WORKGROUP_SIZE}}u;

struct HistogramParams {
    min_value: f32,
    inv_bin_width: f32,
    sample_count: u32,
    bin_count: u32,
    accumulate_total: u32,
    _pad: vec3<u32>,
};

@group(0) @binding(0)
var<storage, read> samples: array<f32>;

@group(0) @binding(1)
var<storage, read> weights: array<f32>;

@group(0) @binding(2)
var<storage, read_write> counts: array<atomic<u32>>;

@group(0) @binding(3)
var<storage, read_write> total_weight: atomic<u32>;

@group(0) @binding(4)
var<uniform> params: HistogramParams;

fn atomic_add_f32(target: ptr<storage, atomic<u32>>, value: f32) {
    var old_bits = atomicLoad(target);
    loop {
        let old_value = bitcast<f32>(old_bits);
        let new_bits = bitcast<u32>(old_value + value);
        let exchange = atomicCompareExchangeWeak(target, old_bits, new_bits);
        if (exchange.exchanged) {
            break;
        }
        old_bits = exchange.old_value;
    }
}

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
    let weight = weights[idx];

    atomic_add_f32(&counts[bin_index], weight);
    if (params.accumulate_total != 0u) {
        atomic_add_f32(&total_weight, weight);
    }
}
"#;

    pub const F32_WEIGHTS_F64: &str = r#"const WORKGROUP_SIZE: u32 = {{WORKGROUP_SIZE}}u;

struct HistogramParams {
    min_value: f32,
    inv_bin_width: f32,
    sample_count: u32,
    bin_count: u32,
    accumulate_total: u32,
    _pad: vec3<u32>,
};

@group(0) @binding(0)
var<storage, read> samples: array<f32>;

@group(0) @binding(1)
var<storage, read> weights: array<f64>;

@group(0) @binding(2)
var<storage, read_write> counts: array<atomic<u32>>;

@group(0) @binding(3)
var<storage, read_write> total_weight: atomic<u32>;

@group(0) @binding(4)
var<uniform> params: HistogramParams;

fn atomic_add_f32(target: ptr<storage, atomic<u32>>, value: f32) {
    var old_bits = atomicLoad(target);
    loop {
        let old_value = bitcast<f32>(old_bits);
        let new_bits = bitcast<u32>(old_value + value);
        let exchange = atomicCompareExchangeWeak(target, old_bits, new_bits);
        if (exchange.exchanged) {
            break;
        }
        old_bits = exchange.old_value;
    }
}

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
    let weight = f32(weights[idx]);

    atomic_add_f32(&counts[bin_index], weight);
    if (params.accumulate_total != 0u) {
        atomic_add_f32(&total_weight, weight);
    }
}
"#;

    pub const F64_UNIFORM: &str = r#"const WORKGROUP_SIZE: u32 = {{WORKGROUP_SIZE}}u;

struct HistogramParams {
    min_value: f32,
    inv_bin_width: f32,
    sample_count: u32,
    bin_count: u32,
    accumulate_total: u32,
    _pad: vec3<u32>,
};

@group(0) @binding(0)
var<storage, read> samples: array<f64>;

@group(0) @binding(1)
var<storage, read_write> counts: array<atomic<u32>>;

@group(0) @binding(2)
var<storage, read_write> total_weight: atomic<u32>;

@group(0) @binding(3)
var<uniform> params: HistogramParams;

fn atomic_add_f32(target: ptr<storage, atomic<u32>>, value: f32) {
    var old_bits = atomicLoad(target);
    loop {
        let old_value = bitcast<f32>(old_bits);
        let new_bits = bitcast<u32>(old_value + value);
        let exchange = atomicCompareExchangeWeak(target, old_bits, new_bits);
        if (exchange.exchanged) {
            break;
        }
        old_bits = exchange.old_value;
    }
}

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

    atomic_add_f32(&counts[bin_index], 1.0);
    if (params.accumulate_total != 0u) {
        atomic_add_f32(&total_weight, 1.0);
    }
}
"#;

    pub const F64_WEIGHTS_F32: &str = r#"const WORKGROUP_SIZE: u32 = {{WORKGROUP_SIZE}}u;

struct HistogramParams {
    min_value: f32,
    inv_bin_width: f32,
    sample_count: u32,
    bin_count: u32,
    accumulate_total: u32,
    _pad: vec3<u32>,
};

@group(0) @binding(0)
var<storage, read> samples: array<f64>;

@group(0) @binding(1)
var<storage, read> weights: array<f32>;

@group(0) @binding(2)
var<storage, read_write> counts: array<atomic<u32>>;

@group(0) @binding(3)
var<storage, read_write> total_weight: atomic<u32>;

@group(0) @binding(4)
var<uniform> params: HistogramParams;

fn atomic_add_f32(target: ptr<storage, atomic<u32>>, value: f32) {
    var old_bits = atomicLoad(target);
    loop {
        let old_value = bitcast<f32>(old_bits);
        let new_bits = bitcast<u32>(old_value + value);
        let exchange = atomicCompareExchangeWeak(target, old_bits, new_bits);
        if (exchange.exchanged) {
            break;
        }
        old_bits = exchange.old_value;
    }
}

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
    let weight = weights[idx];

    atomic_add_f32(&counts[bin_index], weight);
    if (params.accumulate_total != 0u) {
        atomic_add_f32(&total_weight, weight);
    }
}
"#;

    pub const F64_WEIGHTS_F64: &str = r#"const WORKGROUP_SIZE: u32 = {{WORKGROUP_SIZE}}u;

struct HistogramParams {
    min_value: f32,
    inv_bin_width: f32,
    sample_count: u32,
    bin_count: u32,
    accumulate_total: u32,
    _pad: vec3<u32>,
};

@group(0) @binding(0)
var<storage, read> samples: array<f64>;

@group(0) @binding(1)
var<storage, read> weights: array<f64>;

@group(0) @binding(2)
var<storage, read_write> counts: array<atomic<u32>>;

@group(0) @binding(3)
var<storage, read_write> total_weight: atomic<u32>;

@group(0) @binding(4)
var<uniform> params: HistogramParams;

fn atomic_add_f32(target: ptr<storage, atomic<u32>>, value: f32) {
    var old_bits = atomicLoad(target);
    loop {
        let old_value = bitcast<f32>(old_bits);
        let new_bits = bitcast<u32>(old_value + value);
        let exchange = atomicCompareExchangeWeak(target, old_bits, new_bits);
        if (exchange.exchanged) {
            break;
        }
        old_bits = exchange.old_value;
    }
}

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
    let weight = f32(weights[idx]);

    atomic_add_f32(&counts[bin_index], weight);
    if (params.accumulate_total != 0u) {
        atomic_add_f32(&total_weight, weight);
    }
}
"#;
}

pub mod convert {
    pub const TEMPLATE: &str = r#"const WORKGROUP_SIZE: u32 = {{WORKGROUP_SIZE}}u;

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

    let count_value = bitcast<f32>(counts[idx]);
    values[idx] = count_value * params.scale;
}
"#;
}
