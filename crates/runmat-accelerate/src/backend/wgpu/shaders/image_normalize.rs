pub const IMAGE_NORMALIZE_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct Params {
    batches: u32,
    height: u32,
    width: u32,
    plane: u32,
    stride_h: u32,
    stride_w: u32,
    flags: u32,
    _pad0: u32,
    epsilon: f64,
    gain: f64,
    bias: f64,
    gamma: f64,
};

@group(0) @binding(0) var<storage, read> input_tensor: Tensor;
@group(0) @binding(1) var<storage, read_write> output_tensor: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> partial_mean: array<f64, @WG@>;
var<workgroup> partial_m2: array<f64, @WG@>;
var<workgroup> partial_count: array<u32, @WG@>;
var<workgroup> shared_mean: f64;
var<workgroup> shared_inv_sigma: f64;

fn has_flag(mask: u32) -> bool {
    return (params.flags & mask) != 0u;
}

fn index_for(batch: u32, idx: u32) -> u32 {
    let h = idx % params.height;
    let w = idx / params.height;
    return batch + h * params.stride_h + w * params.stride_w;
}

@compute @workgroup_size(@WG@)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let batch = wid.x;
    if batch >= params.batches {
        return;
    }
    let lane = lid.x;
    let plane = params.plane;
    if plane == 0u {
        return;
    }

    var mean = 0.0f64;
    var m2 = 0.0f64;
    var count: u32 = 0u;

    var idx = lane;
    loop {
        if idx >= plane {
            break;
        }
        let offset = index_for(batch, idx);
        let value = input_tensor.data[offset];
        let new_count = count + 1u;
        let delta = value - mean;
        mean = mean + delta / f64(new_count);
        let delta2 = value - mean;
        m2 = m2 + delta * delta2;
        count = new_count;
        idx = idx + @WG@u;
    }

    partial_mean[lane] = mean;
    partial_m2[lane] = m2;
    partial_count[lane] = count;
    workgroupBarrier();

    if lane == 0u {
        var agg_mean = 0.0f64;
        var agg_m2 = 0.0f64;
        var agg_count: u32 = 0u;
        var i = 0u;
        loop {
            if i >= @WG@u {
                break;
            }
            let c = partial_count[i];
            if c != 0u {
                let pm = partial_mean[i];
                let pm2 = partial_m2[i];
                if agg_count == 0u {
                    agg_mean = pm;
                    agg_m2 = pm2;
                    agg_count = c;
                } else {
                    let delta = pm - agg_mean;
                    let new_count = agg_count + c;
                    let nf = f64(new_count);
                    let cf = f64(c);
                    agg_mean = agg_mean + delta * (cf / nf);
                    agg_m2 = agg_m2 + pm2 + delta * delta * f64(agg_count) * cf / nf;
                    agg_count = new_count;
                }
            }
            i = i + 1u;
        }

        shared_mean = agg_mean;
        var variance = 0.0f64;
        if agg_count > 0u {
            variance = agg_m2 / f64(agg_count);
        }
        let sigma = sqrt(variance + params.epsilon);
        var inv_sigma = 0.0f64;
        if sigma > 0.0f64 {
            inv_sigma = 1.0f64 / sigma;
        }
        shared_inv_sigma = inv_sigma;
    }

    workgroupBarrier();

    let mean_val = shared_mean;
    let inv_sigma = shared_inv_sigma;
    let apply_gain = has_flag(1u);
    let apply_bias = has_flag(2u);
    let apply_gamma = has_flag(4u);

    var write_idx = lane;
    loop {
        if write_idx >= plane {
            break;
        }
        let offset = index_for(batch, write_idx);
        let value = input_tensor.data[offset];
        var out_value = (value - mean_val) * inv_sigma;
        if apply_gain {
            out_value = out_value * params.gain;
        }
        if apply_bias {
            out_value = out_value + params.bias;
        }
        out_value = max(out_value, 0.0f64);
        if apply_gamma {
            out_value = pow(out_value, params.gamma);
        }
        output_tensor.data[offset] = out_value;
        write_idx = write_idx + @WG@u;
    }
}
"#;

pub const IMAGE_NORMALIZE_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct Params {
    batches: u32,
    height: u32,
    width: u32,
    plane: u32,
    stride_h: u32,
    stride_w: u32,
    flags: u32,
    _pad0: u32,
    epsilon: f32,
    gain: f32,
    bias: f32,
    gamma: f32,
};

@group(0) @binding(0) var<storage, read> input_tensor: Tensor;
@group(0) @binding(1) var<storage, read_write> output_tensor: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> partial_mean: array<f32, @WG@>;
var<workgroup> partial_m2: array<f32, @WG@>;
var<workgroup> partial_count: array<u32, @WG@>;
var<workgroup> shared_mean: f32;
var<workgroup> shared_inv_sigma: f32;

fn has_flag(mask: u32) -> bool {
    return (params.flags & mask) != 0u;
}

fn index_for(batch: u32, idx: u32) -> u32 {
    let h = idx % params.height;
    let w = idx / params.height;
    return batch + h * params.stride_h + w * params.stride_w;
}

@compute @workgroup_size(@WG@)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let batch = wid.x;
    if batch >= params.batches {
        return;
    }
    let lane = lid.x;
    let plane = params.plane;
    if plane == 0u {
        return;
    }

    var mean = 0.0f;
    var m2 = 0.0f;
    var count: u32 = 0u;

    var idx = lane;
    loop {
        if idx >= plane {
            break;
        }
        let offset = index_for(batch, idx);
        let value = input_tensor.data[offset];
        let new_count = count + 1u;
        let delta = value - mean;
        mean = mean + delta / f32(new_count);
        let delta2 = value - mean;
        m2 = m2 + delta * delta2;
        count = new_count;
        idx = idx + @WG@u;
    }

    partial_mean[lane] = mean;
    partial_m2[lane] = m2;
    partial_count[lane] = count;
    workgroupBarrier();

    if lane == 0u {
        var agg_mean = 0.0f;
        var agg_m2 = 0.0f;
        var agg_count: u32 = 0u;
        var i = 0u;
        loop {
            if i >= @WG@u {
                break;
            }
            let c = partial_count[i];
            if c != 0u {
                let pm = partial_mean[i];
                let pm2 = partial_m2[i];
                if agg_count == 0u {
                    agg_mean = pm;
                    agg_m2 = pm2;
                    agg_count = c;
                } else {
                    let delta = pm - agg_mean;
                    let new_count = agg_count + c;
                    let nf = f32(new_count);
                    let cf = f32(c);
                    agg_mean = agg_mean + delta * (cf / nf);
                    agg_m2 = agg_m2 + pm2 + delta * delta * f32(agg_count) * cf / nf;
                    agg_count = new_count;
                }
            }
            i = i + 1u;
        }

        shared_mean = agg_mean;
        var variance = 0.0f;
        if agg_count > 0u {
            variance = agg_m2 / f32(agg_count);
        }
        let sigma = sqrt(variance + params.epsilon);
        var inv_sigma = 0.0f;
        if sigma > 0.0f {
            inv_sigma = 1.0f / sigma;
        }
        shared_inv_sigma = inv_sigma;
    }

    workgroupBarrier();

    let mean_val = shared_mean;
    let inv_sigma = shared_inv_sigma;
    let apply_gain = has_flag(1u);
    let apply_bias = has_flag(2u);
    let apply_gamma = has_flag(4u);

    var write_idx = lane;
    loop {
        if write_idx >= plane {
            break;
        }
        let offset = index_for(batch, write_idx);
        let value = input_tensor.data[offset];
        var out_value = (value - mean_val) * inv_sigma;
        if apply_gain {
            out_value = out_value * params.gain;
        }
        if apply_bias {
            out_value = out_value + params.bias;
        }
        out_value = max(out_value, 0.0f);
        if apply_gamma {
            out_value = pow(out_value, params.gamma);
        }
        output_tensor.data[offset] = out_value;
        write_idx = write_idx + @WG@u;
    }
}
"#;
