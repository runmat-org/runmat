pub const IMAGE_NORMALIZE_SHADER_F64: &str = r#"
const BATCH_TILE: u32 = @BT@u;
const VALUES_PER_THREAD: u32 = @VP@u;
const LANE_COUNT: u32 = @WG@u;
const SPATIAL_TILE: u32 = @ST@u;
const LANE_STRIDE: u32 = LANE_COUNT * SPATIAL_TILE;
const BATCH_VEC_WIDTH: u32 = @BV@u;
const BATCH_GROUPS: u32 = (BATCH_TILE + BATCH_VEC_WIDTH - 1u) / BATCH_VEC_WIDTH;
const PARTIAL_STRIDE: u32 = LANE_STRIDE * BATCH_GROUPS;

struct TensorScalar {
    data: array<f64>,
};

struct TensorVec {
    data: array<vec4<f64>>,
};

struct Params {
    batch_count: u32,
    height: u32,
    width: u32,
    plane: u32,
    stride_h: u32,
    stride_w: u32,
    flags: u32,
    batch_stride: u32,
    batch_offset: u32,
    _pad0: u32,
    epsilon: f64,
    gain: f64,
    bias: f64,
    gamma: f64,
};

@group(0) @binding(0) var<storage, read> input_tensor: TensorScalar;
@group(0) @binding(1) var<storage, read_write> output_tensor: TensorScalar;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> input_tensor_vec: TensorVec;
@group(0) @binding(4) var<storage, read_write> output_tensor_vec: TensorVec;

var<workgroup> partial_sum: array<vec4<f64>, PARTIAL_STRIDE>;
var<workgroup> partial_sum_sq: array<vec4<f64>, PARTIAL_STRIDE>;
var<workgroup> shared_mean: array<vec4<f64>, BATCH_GROUPS>;
var<workgroup> shared_inv_sigma: array<vec4<f64>, BATCH_GROUPS>;

fn has_flag(mask: u32) -> bool {
    return (params.flags & mask) != 0u;
}

fn plane_offset(idx: u32) -> u32 {
    return idx * params.batch_stride;
}

fn can_use_vec_path(base: u32, count: u32) -> bool {
    return count == BATCH_VEC_WIDTH && (base & (BATCH_VEC_WIDTH - 1u)) == 0u;
}

fn group_count(active_batches: u32) -> u32 {
    return (active_batches + BATCH_VEC_WIDTH - 1u) / BATCH_VEC_WIDTH;
}

fn group_component_count(active_batches: u32, group: u32) -> u32 {
    let consumed = group * BATCH_VEC_WIDTH;
    if consumed >= active_batches {
        return 0u;
    }
    let remaining = active_batches - consumed;
    return min(BATCH_VEC_WIDTH, remaining);
}

fn batch_group_mask(active_batches: u32, group: u32) -> vec4<f64> {
    let count = group_component_count(active_batches, group);
    if count == BATCH_VEC_WIDTH {
        return vec4<f64>(1.0);
    }
    var mask = vec4<f64>(0.0);
    for (var i = 0u; i < count; i = i + 1u) {
        mask[i] = 1.0;
    }
    return mask;
}

fn load_batch_group(base: u32, count: u32) -> vec4<f64> {
    if count == 0u {
        return vec4<f64>(0.0);
    }
    if can_use_vec_path(base, count) {
        let vec_index = base >> 2u;
        return input_tensor_vec.data[vec_index];
    }
    var value = vec4<f64>(0.0);
    for (var i = 0u; i < count; i = i + 1u) {
        value[i] = input_tensor.data[base + i];
    }
    return value;
}

fn store_batch_group(base: u32, count: u32, value: vec4<f64>) {
    if count == 0u {
        return;
    }
    if can_use_vec_path(base, count) {
        let vec_index = base >> 2u;
        output_tensor_vec.data[vec_index] = value;
        return;
    }
    for (var i = 0u; i < count; i = i + 1u) {
        output_tensor.data[base + i] = value[i];
    }
}

@compute @workgroup_size(@WG@, @ST@, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let local_base_batch = wid.x * BATCH_TILE;
    if local_base_batch >= params.batch_count {
        return;
    }
    let active_batches = min(BATCH_TILE, params.batch_count - local_base_batch);
    let base_batch = params.batch_offset + local_base_batch;
    let plane = params.plane;
    if plane == 0u {
        return;
    }

    let lane = lid.x;
    let spatial_lane = lid.y;
    let lane_index = spatial_lane * LANE_COUNT + lane;
    let lane_stride = LANE_STRIDE * VALUES_PER_THREAD;
    let lane_base = lane_index * BATCH_GROUPS;
    for (var g = 0u; g < BATCH_GROUPS; g = g + 1u) {
        partial_sum[lane_base + g] = vec4<f64>(0.0);
        partial_sum_sq[lane_base + g] = vec4<f64>(0.0);
    }
    workgroupBarrier();
    var idx = spatial_lane * (LANE_COUNT * VALUES_PER_THREAD) + lane * VALUES_PER_THREAD;
    let total_groups = group_count(active_batches);
    var sum: array<vec4<f64>, BATCH_GROUPS>;
    var sum_sq: array<vec4<f64>, BATCH_GROUPS>;
    for (var g = 0u; g < BATCH_GROUPS; g = g + 1u) {
        sum[g] = vec4<f64>(0.0);
        sum_sq[g] = vec4<f64>(0.0);
    }
    loop {
        if idx >= plane {
            break;
        }
        var inner = 0u;
        loop {
            if inner >= VALUES_PER_THREAD {
                break;
            }
            let sample_idx = idx + inner;
            if sample_idx >= plane {
                break;
            }
            let sample_base = plane_offset(sample_idx) + base_batch;
            var group = 0u;
            loop {
                if group >= total_groups {
                    break;
                }
                let comp = group_component_count(active_batches, group);
                let offset = sample_base + group * BATCH_VEC_WIDTH;
                let value = load_batch_group(offset, comp);
                sum[group] = sum[group] + value;
                sum_sq[group] = sum_sq[group] + value * value;
                group = group + 1u;
            }
            inner = inner + 1u;
        }
        idx = idx + lane_stride;
    }

    for (var g = 0u; g < BATCH_GROUPS; g = g + 1u) {
        partial_sum[lane_base + g] = sum[g];
        partial_sum_sq[lane_base + g] = sum_sq[g];
    }
    workgroupBarrier();

    var stride = LANE_STRIDE / 2u;
    loop {
        if stride == 0u {
            break;
        }
        if lane_index < stride {
            let dst = lane_index * BATCH_GROUPS;
            let src = (lane_index + stride) * BATCH_GROUPS;
            for (var g = 0u; g < BATCH_GROUPS; g = g + 1u) {
                partial_sum[dst + g] = partial_sum[dst + g] + partial_sum[src + g];
                partial_sum_sq[dst + g] =
                    partial_sum_sq[dst + g] + partial_sum_sq[src + g];
            }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if lane_index == 0u {
        let inv_plane = 1.0f64 / f64(plane);
        for (var g = 0u; g < BATCH_GROUPS; g = g + 1u) {
            let mask = batch_group_mask(active_batches, g);
            let total_sum = partial_sum[g];
            let total_sq = partial_sum_sq[g];
            let mean = total_sum * inv_plane;
            var variance = total_sq * inv_plane - mean * mean;
            variance = max(variance, vec4<f64>(0.0));
            let sigma = sqrt(variance + vec4<f64>(params.epsilon));
            var inv_sigma_vec = vec4<f64>(0.0);
            inv_sigma_vec =
                select(inv_sigma_vec, vec4<f64>(1.0) / sigma, sigma > vec4<f64>(0.0));
            shared_mean[g] = mean * mask;
            shared_inv_sigma[g] = inv_sigma_vec * mask;
        }
    }

    workgroupBarrier();

    let apply_gain = has_flag(1u);
    let apply_bias = has_flag(2u);
    let apply_gamma = has_flag(4u);

    var write_idx = spatial_lane * (LANE_COUNT * VALUES_PER_THREAD) + lane * VALUES_PER_THREAD;
    loop {
        if write_idx >= plane {
            break;
        }
        var inner = 0u;
        loop {
            if inner >= VALUES_PER_THREAD {
                break;
            }
            let sample_idx = write_idx + inner;
            if sample_idx >= plane {
                break;
            }
            let sample_base = plane_offset(sample_idx) + base_batch;
            var group = 0u;
            loop {
                if group >= total_groups {
                    break;
                }
                let comp = group_component_count(active_batches, group);
                if comp == 0u {
                    break;
                }
                let offset = sample_base + group * BATCH_VEC_WIDTH;
                var out_value = load_batch_group(offset, comp);
                var normalized = (out_value - shared_mean[group]) * shared_inv_sigma[group];
                for (var c = 0u; c < comp; c = c + 1u) {
                    var component = normalized[c];
                    if apply_gain {
                        component = component * params.gain;
                    }
                    if apply_bias {
                        component = component + params.bias;
                    }
                    component = max(component, 0.0f64);
                    if apply_gamma {
                        component = pow(component, params.gamma);
                    }
                    normalized[c] = component;
                }
                store_batch_group(offset, comp, normalized);
                group = group + 1u;
            }
            inner = inner + 1u;
        }
        write_idx = write_idx + lane_stride;
    }
}
"#;

pub const IMAGE_NORMALIZE_SHADER_F32: &str = r#"
const BATCH_TILE: u32 = @BT@u;
const VALUES_PER_THREAD: u32 = @VP@u;
const LANE_COUNT: u32 = @WG@u;
const SPATIAL_TILE: u32 = @ST@u;
const LANE_STRIDE: u32 = LANE_COUNT * SPATIAL_TILE;
const BATCH_VEC_WIDTH: u32 = @BV@u;
const BATCH_GROUPS: u32 = (BATCH_TILE + BATCH_VEC_WIDTH - 1u) / BATCH_VEC_WIDTH;
const PARTIAL_STRIDE: u32 = LANE_STRIDE * BATCH_GROUPS;

struct TensorScalar {
    data: array<f32>,
};

struct TensorVec {
    data: array<vec4<f32>>,
};

struct Params {
    batch_count: u32,
    height: u32,
    width: u32,
    plane: u32,
    stride_h: u32,
    stride_w: u32,
    flags: u32,
    batch_stride: u32,
    batch_offset: u32,
    _pad0: u32,
    epsilon: f32,
    gain: f32,
    bias: f32,
    gamma: f32,
};

@group(0) @binding(0) var<storage, read> input_tensor: TensorScalar;
@group(0) @binding(1) var<storage, read_write> output_tensor: TensorScalar;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> input_tensor_vec: TensorVec;
@group(0) @binding(4) var<storage, read_write> output_tensor_vec: TensorVec;

var<workgroup> partial_sum: array<vec4<f32>, PARTIAL_STRIDE>;
var<workgroup> partial_sum_sq: array<vec4<f32>, PARTIAL_STRIDE>;
var<workgroup> shared_mean: array<vec4<f32>, BATCH_GROUPS>;
var<workgroup> shared_inv_sigma: array<vec4<f32>, BATCH_GROUPS>;

fn has_flag(mask: u32) -> bool {
    return (params.flags & mask) != 0u;
}

fn plane_offset(idx: u32) -> u32 {
    return idx * params.batch_stride;
}

fn can_use_vec_path(base: u32, count: u32) -> bool {
    return count == BATCH_VEC_WIDTH && (base & (BATCH_VEC_WIDTH - 1u)) == 0u;
}

fn group_count(active_batches: u32) -> u32 {
    return (active_batches + BATCH_VEC_WIDTH - 1u) / BATCH_VEC_WIDTH;
}

fn group_component_count(active_batches: u32, group: u32) -> u32 {
    let consumed = group * BATCH_VEC_WIDTH;
    if consumed >= active_batches {
        return 0u;
    }
    let remaining = active_batches - consumed;
    return min(BATCH_VEC_WIDTH, remaining);
}

fn batch_group_mask(active_batches: u32, group: u32) -> vec4<f32> {
    let count = group_component_count(active_batches, group);
    if count == BATCH_VEC_WIDTH {
        return vec4<f32>(1.0);
    }
    var mask = vec4<f32>(0.0);
    for (var i = 0u; i < count; i = i + 1u) {
        mask[i] = 1.0;
    }
    return mask;
}

fn load_batch_group(base: u32, count: u32) -> vec4<f32> {
    if count == 0u {
        return vec4<f32>(0.0);
    }
    if can_use_vec_path(base, count) {
        let vec_index = base >> 2u;
        return input_tensor_vec.data[vec_index];
    }
    var value = vec4<f32>(0.0);
    for (var i = 0u; i < count; i = i + 1u) {
        value[i] = input_tensor.data[base + i];
    }
    return value;
}

fn store_batch_group(base: u32, count: u32, value: vec4<f32>) {
    if count == 0u {
        return;
    }
    if can_use_vec_path(base, count) {
        let vec_index = base >> 2u;
        output_tensor_vec.data[vec_index] = value;
        return;
    }
    for (var i = 0u; i < count; i = i + 1u) {
        output_tensor.data[base + i] = value[i];
    }
}

@compute @workgroup_size(@WG@, @ST@, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let local_base_batch = wid.x * BATCH_TILE;
    if local_base_batch >= params.batch_count {
        return;
    }
    let active_batches = min(BATCH_TILE, params.batch_count - local_base_batch);
    let base_batch = params.batch_offset + local_base_batch;
    let plane = params.plane;
    if plane == 0u {
        return;
    }

    let lane = lid.x;
    let spatial_lane = lid.y;
    let lane_index = spatial_lane * LANE_COUNT + lane;
    let lane_stride = LANE_STRIDE * VALUES_PER_THREAD;
    let lane_base = lane_index * BATCH_GROUPS;
    for (var g = 0u; g < BATCH_GROUPS; g = g + 1u) {
        partial_sum[lane_base + g] = vec4<f32>(0.0);
        partial_sum_sq[lane_base + g] = vec4<f32>(0.0);
    }
    workgroupBarrier();
    var idx = spatial_lane * (LANE_COUNT * VALUES_PER_THREAD) + lane * VALUES_PER_THREAD;
    let total_groups = group_count(active_batches);
    var sum: array<vec4<f32>, BATCH_GROUPS>;
    var sum_comp: array<vec4<f32>, BATCH_GROUPS>;
    var sum_sq: array<vec4<f32>, BATCH_GROUPS>;
    var sum_sq_comp: array<vec4<f32>, BATCH_GROUPS>;
    for (var g = 0u; g < BATCH_GROUPS; g = g + 1u) {
        sum[g] = vec4<f32>(0.0);
        sum_comp[g] = vec4<f32>(0.0);
        sum_sq[g] = vec4<f32>(0.0);
        sum_sq_comp[g] = vec4<f32>(0.0);
    }
    loop {
        if idx >= plane {
            break;
        }
        var inner = 0u;
        loop {
            if inner >= VALUES_PER_THREAD {
                break;
            }
            let sample_idx = idx + inner;
            if sample_idx >= plane {
                break;
            }
            let sample_base = plane_offset(sample_idx) + base_batch;
            var group = 0u;
            loop {
                if group >= total_groups {
                    break;
                }
                let comp = group_component_count(active_batches, group);
                let offset = sample_base + group * BATCH_VEC_WIDTH;
                let value = load_batch_group(offset, comp);
                let y = value - sum_comp[group];
                let t = sum[group] + y;
                sum_comp[group] = (t - sum[group]) - y;
                sum[group] = t;
                let sq_val = value * value;
                let y_sq = sq_val - sum_sq_comp[group];
                let t_sq = sum_sq[group] + y_sq;
                sum_sq_comp[group] = (t_sq - sum_sq[group]) - y_sq;
                sum_sq[group] = t_sq;
                group = group + 1u;
            }
            inner = inner + 1u;
        }
        idx = idx + lane_stride;
    }

    for (var g = 0u; g < BATCH_GROUPS; g = g + 1u) {
        partial_sum[lane_base + g] = sum[g];
        partial_sum_sq[lane_base + g] = sum_sq[g];
    }
    workgroupBarrier();

    var stride = LANE_STRIDE / 2u;
    loop {
        if stride == 0u {
            break;
        }
        if lane_index < stride {
            let dst = lane_index * BATCH_GROUPS;
            let src = (lane_index + stride) * BATCH_GROUPS;
            for (var g = 0u; g < BATCH_GROUPS; g = g + 1u) {
                partial_sum[dst + g] = partial_sum[dst + g] + partial_sum[src + g];
                partial_sum_sq[dst + g] =
                    partial_sum_sq[dst + g] + partial_sum_sq[src + g];
            }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if lane_index == 0u {
        let inv_plane = 1.0f / f32(plane);
        for (var g = 0u; g < BATCH_GROUPS; g = g + 1u) {
            let mask = batch_group_mask(active_batches, g);
            let total_sum = partial_sum[g];
            let total_sq = partial_sum_sq[g];
            let mean = total_sum * inv_plane;
            var variance = total_sq * inv_plane - mean * mean;
            variance = max(variance, vec4<f32>(0.0));
            let sigma = sqrt(variance + vec4<f32>(params.epsilon));
            var inv_sigma_vec = vec4<f32>(0.0);
            inv_sigma_vec =
                select(inv_sigma_vec, vec4<f32>(1.0) / sigma, sigma > vec4<f32>(0.0));
            shared_mean[g] = mean * mask;
            shared_inv_sigma[g] = inv_sigma_vec * mask;
        }
    }

    workgroupBarrier();

    let apply_gain = has_flag(1u);
    let apply_bias = has_flag(2u);
    let apply_gamma = has_flag(4u);

    var write_idx = spatial_lane * (LANE_COUNT * VALUES_PER_THREAD) + lane * VALUES_PER_THREAD;
    loop {
        if write_idx >= plane {
            break;
        }
        var inner = 0u;
        loop {
            if inner >= VALUES_PER_THREAD {
                break;
            }
            let sample_idx = write_idx + inner;
            if sample_idx >= plane {
                break;
            }
            let sample_base = plane_offset(sample_idx) + base_batch;
            var group = 0u;
            loop {
                if group >= total_groups {
                    break;
                }
                let comp = group_component_count(active_batches, group);
                if comp == 0u {
                    break;
                }
                let offset = sample_base + group * BATCH_VEC_WIDTH;
                var out_value = load_batch_group(offset, comp);
                var normalized = (out_value - shared_mean[group]) * shared_inv_sigma[group];
                if apply_gain {
                    normalized = normalized * vec4<f32>(params.gain);
                }
                if apply_bias {
                    normalized = normalized + vec4<f32>(params.bias);
                }
                normalized = max(normalized, vec4<f32>(0.0));
                if apply_gamma {
                    normalized = pow(normalized, vec4<f32>(params.gamma));
                }
                store_batch_group(offset, comp, normalized);
                group = group + 1u;
            }
            inner = inner + 1u;
        }
        write_idx = write_idx + lane_stride;
    }
}
"#;
