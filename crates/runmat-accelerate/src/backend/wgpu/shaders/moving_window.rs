pub const MOVING_WINDOW_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct MovingWindowParams {
    meta0: vec4<u32>,
    meta1: vec4<u32>,
    meta2: vec4<u32>,
    fill_value: f64,
    _pad1: f64,
}

@group(0) @binding(0) var<storage, read> Input: Tensor;
@group(0) @binding(1) var<storage, read_write> Output: Tensor;
@group(0) @binding(2) var<uniform> params: MovingWindowParams;

fn isNan(value: f64) -> bool {
    let bits = bitcast<u64>(value);
    return (bits & 0x7ff0000000000000u) == 0x7ff0000000000000u &&
        (bits & 0x000fffffffffffffu) != 0u;
}

fn nanValue() -> f64 {
    var bits: u64 = 0x7ff8000000000000u;
    return bitcast<f64>(bits);
}

fn kthValue(
    target: u32,
    in_start: i32,
    in_end: i32,
    before_idx: u32,
    after_idx: u32,
    fill_count: u32,
    fill: f64,
) -> f64 {
    var selected = 0.0;
    var found = false;

    if (in_start <= in_end) {
        var cand_pos = u32(in_start);
        loop {
            if (cand_pos > u32(in_end)) {
                break;
            }
            let cand_idx = before_idx + cand_pos * params.meta0.w + after_idx * params.meta0.w * params.meta0.y;
            let candidate = Input.data[cand_idx];
            if (!isNan(candidate)) {
                var less = 0u;
                var equal = 0u;
                var pos = u32(in_start);
                loop {
                    if (pos > u32(in_end)) {
                        break;
                    }
                    let in_idx = before_idx + pos * params.meta0.w + after_idx * params.meta0.w * params.meta0.y;
                    let value = Input.data[in_idx];
                    if (!isNan(value)) {
                        if (value < candidate) {
                            less = less + 1u;
                        } else if (value == candidate) {
                            equal = equal + 1u;
                        }
                    }
                    pos = pos + 1u;
                }
                if (fill_count > 0u) {
                    if (fill < candidate) {
                        less = less + fill_count;
                    } else if (fill == candidate) {
                        equal = equal + fill_count;
                    }
                }
                if (less <= target && target < less + equal) {
                    if (!found || candidate < selected) {
                        selected = candidate;
                        found = true;
                    }
                }
            }
            cand_pos = cand_pos + 1u;
        }
    }

    if (fill_count > 0u) {
        let candidate = fill;
        var less = 0u;
        var equal = fill_count;
        if (in_start <= in_end) {
            var pos = u32(in_start);
            loop {
                if (pos > u32(in_end)) {
                    break;
                }
                let in_idx = before_idx + pos * params.meta0.w + after_idx * params.meta0.w * params.meta0.y;
                let value = Input.data[in_idx];
                if (!isNan(value)) {
                    if (value < candidate) {
                        less = less + 1u;
                    } else if (value == candidate) {
                        equal = equal + 1u;
                    }
                }
                pos = pos + 1u;
            }
        }
        if (less <= target && target < less + equal) {
            if (!found || candidate < selected) {
                selected = candidate;
                found = true;
            }
        }
    }

    if (found) {
        return selected;
    }
    return nanValue();
}

fn medianValue(
    total_count: u32,
    in_start: i32,
    in_end: i32,
    before_idx: u32,
    after_idx: u32,
    fill_count: u32,
    fill: f64,
) -> f64 {
    if (total_count == 0u) {
        return nanValue();
    }
    let lower_rank = (total_count - 1u) / 2u;
    let upper_rank = total_count / 2u;
    let lower = kthValue(lower_rank, in_start, in_end, before_idx, after_idx, fill_count, fill);
    if (lower_rank == upper_rank) {
        return lower;
    }
    let upper = kthValue(upper_rank, in_start, in_end, before_idx, after_idx, fill_count, fill);
    return (lower + upper) * 0.5;
}

fn pushValue(
    value: f64,
    count: ptr<function, u32>,
    sum: ptr<function, f64>,
    prod: ptr<function, f64>,
    min_value: ptr<function, f64>,
    max_value: ptr<function, f64>,
    mean: ptr<function, f64>,
    m2: ptr<function, f64>,
    has_value: ptr<function, bool>,
) {
    (*count) = (*count) + 1u;
    (*sum) = (*sum) + value;
    (*prod) = (*prod) * value;
    if ((*has_value)) {
        (*min_value) = min((*min_value), value);
        (*max_value) = max((*max_value), value);
    } else {
        (*min_value) = value;
        (*max_value) = value;
        (*has_value) = true;
    }
    let delta = value - (*mean);
    (*mean) = (*mean) + delta / f64((*count));
    let delta2 = value - (*mean);
    (*m2) = (*m2) + delta * delta2;
}

fn pushRepeated(
    value: f64,
    repeats: u32,
    count: ptr<function, u32>,
    sum: ptr<function, f64>,
    prod: ptr<function, f64>,
    min_value: ptr<function, f64>,
    max_value: ptr<function, f64>,
    mean: ptr<function, f64>,
    m2: ptr<function, f64>,
    has_value: ptr<function, bool>,
) {
    if (repeats == 0u) {
        return;
    }
    (*sum) = (*sum) + value * f64(repeats);
    (*prod) = (*prod) * pow(value, f64(repeats));
    if ((*has_value)) {
        (*min_value) = min((*min_value), value);
        (*max_value) = max((*max_value), value);
    } else {
        (*min_value) = value;
        (*max_value) = value;
        (*has_value) = true;
    }
    if ((*count) == 0u) {
        (*count) = repeats;
        (*mean) = value;
        (*m2) = 0.0;
        return;
    }
    let old_count = (*count);
    let new_count = old_count + repeats;
    let delta = value - (*mean);
    (*mean) = (*mean) + delta * f64(repeats) / f64(new_count);
    (*m2) = (*m2) + delta * delta * f64(old_count) * f64(repeats) / f64(new_count);
    (*count) = new_count;
}

fn finishValue(
    count: u32,
    sum: f64,
    prod: f64,
    min_value: f64,
    max_value: f64,
    m2: f64,
) -> f64 {
    if (count == 0u) {
        if (params.meta2.x == 0u) {
            return 0.0;
        }
        if (params.meta2.x == 2u) {
            return 1.0;
        }
        return nanValue();
    }
    if (params.meta2.x == 0u) {
        return sum;
    }
    if (params.meta2.x == 1u) {
        return sum / f64(count);
    }
    if (params.meta2.x == 2u) {
        return prod;
    }
    if (params.meta2.x == 3u) {
        return min_value;
    }
    if (params.meta2.x == 4u) {
        return max_value;
    }
    var denom = count;
    if (params.meta2.y == 0u && count > 1u) {
        denom = count - 1u;
    }
    let variance = m2 / f64(denom);
    if (params.meta2.x == 5u) {
        return sqrt(variance);
    }
    return variance;
}

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.meta0.x) {
        return;
    }
    let before_idx = idx % params.meta0.w;
    let tmp = idx / params.meta0.w;
    let out_pos = tmp % params.meta0.z;
    let after_idx = tmp / params.meta0.z;
    var center = out_pos;
    if (params.meta1.x == 1u) {
        center = center + params.meta1.z;
    }
    let start = i32(center) - i32(params.meta1.z);
    let end = i32(center) + i32(params.meta1.w);
    let last_axis = i32(params.meta0.y) - 1;
    let in_start = max(start, 0);
    let in_end = min(end, last_axis);

    var fill_count = 0u;
    if (params.meta1.x == 2u) {
        if (start < 0) {
            fill_count = fill_count + u32(-start);
        }
        if (end >= i32(params.meta0.y)) {
            fill_count = fill_count + u32(end - i32(params.meta0.y) + 1);
        }
    }

    var count = 0u;
    var sum = 0.0;
    var prod = 1.0;
    var min_value = 0.0;
    var max_value = 0.0;
    var mean = 0.0;
    var m2 = 0.0;
    var has_value = false;
    var saw_nan = false;
    var median_fill_count = 0u;

    if (in_start <= in_end) {
        var pos = u32(in_start);
        loop {
            if (pos > u32(in_end)) {
                break;
            }
            let in_idx = before_idx + pos * params.meta0.w + after_idx * params.meta0.w * params.meta0.y;
            let value = Input.data[in_idx];
            if (isNan(value)) {
                if (params.meta1.y == 0u) {
                    saw_nan = true;
                    break;
                }
            } else {
                pushValue(value, &count, &sum, &prod, &min_value, &max_value, &mean, &m2, &has_value);
            }
            pos = pos + 1u;
        }
    }

    if (!saw_nan && fill_count > 0u) {
        let fill = params.fill_value;
        if (isNan(fill)) {
            if (params.meta1.y == 0u) {
                saw_nan = true;
            }
        } else if (params.meta2.x == 7u) {
            median_fill_count = fill_count;
        } else {
            pushRepeated(fill, fill_count, &count, &sum, &prod, &min_value, &max_value, &mean, &m2, &has_value);
        }
    }

    if (saw_nan) {
        Output.data[idx] = nanValue();
    } else if (params.meta2.x == 7u) {
        Output.data[idx] = medianValue(count + median_fill_count, in_start, in_end, before_idx, after_idx, median_fill_count, params.fill_value);
    } else {
        Output.data[idx] = finishValue(count, sum, prod, min_value, max_value, m2);
    }
}
"#;

pub const MOVING_WINDOW_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct MovingWindowParams {
    meta0: vec4<u32>,
    meta1: vec4<u32>,
    meta2: vec4<u32>,
    meta3: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> Input: Tensor;
@group(0) @binding(1) var<storage, read_write> Output: Tensor;
@group(0) @binding(2) var<uniform> params: MovingWindowParams;

fn isNan(value: f32) -> bool {
    let bits = bitcast<u32>(value);
    return (bits & 0x7f800000u) == 0x7f800000u &&
        (bits & 0x007fffffu) != 0u;
}

fn nanValue() -> f32 {
    var bits: u32 = 0x7fc00000u;
    return bitcast<f32>(bits);
}

fn kthValue(
    target: u32,
    in_start: i32,
    in_end: i32,
    before_idx: u32,
    after_idx: u32,
    fill_count: u32,
    fill: f32,
) -> f32 {
    var selected = 0.0f;
    var found = false;

    if (in_start <= in_end) {
        var cand_pos = u32(in_start);
        loop {
            if (cand_pos > u32(in_end)) {
                break;
            }
            let cand_idx = before_idx + cand_pos * params.meta0.w + after_idx * params.meta0.w * params.meta0.y;
            let candidate = Input.data[cand_idx];
            if (!isNan(candidate)) {
                var less = 0u;
                var equal = 0u;
                var pos = u32(in_start);
                loop {
                    if (pos > u32(in_end)) {
                        break;
                    }
                    let in_idx = before_idx + pos * params.meta0.w + after_idx * params.meta0.w * params.meta0.y;
                    let value = Input.data[in_idx];
                    if (!isNan(value)) {
                        if (value < candidate) {
                            less = less + 1u;
                        } else if (value == candidate) {
                            equal = equal + 1u;
                        }
                    }
                    pos = pos + 1u;
                }
                if (fill_count > 0u) {
                    if (fill < candidate) {
                        less = less + fill_count;
                    } else if (fill == candidate) {
                        equal = equal + fill_count;
                    }
                }
                if (less <= target && target < less + equal) {
                    if (!found || candidate < selected) {
                        selected = candidate;
                        found = true;
                    }
                }
            }
            cand_pos = cand_pos + 1u;
        }
    }

    if (fill_count > 0u) {
        let candidate = fill;
        var less = 0u;
        var equal = fill_count;
        if (in_start <= in_end) {
            var pos = u32(in_start);
            loop {
                if (pos > u32(in_end)) {
                    break;
                }
                let in_idx = before_idx + pos * params.meta0.w + after_idx * params.meta0.w * params.meta0.y;
                let value = Input.data[in_idx];
                if (!isNan(value)) {
                    if (value < candidate) {
                        less = less + 1u;
                    } else if (value == candidate) {
                        equal = equal + 1u;
                    }
                }
                pos = pos + 1u;
            }
        }
        if (less <= target && target < less + equal) {
            if (!found || candidate < selected) {
                selected = candidate;
                found = true;
            }
        }
    }

    if (found) {
        return selected;
    }
    return nanValue();
}

fn medianValue(
    total_count: u32,
    in_start: i32,
    in_end: i32,
    before_idx: u32,
    after_idx: u32,
    fill_count: u32,
    fill: f32,
) -> f32 {
    if (total_count == 0u) {
        return nanValue();
    }
    let lower_rank = (total_count - 1u) / 2u;
    let upper_rank = total_count / 2u;
    let lower = kthValue(lower_rank, in_start, in_end, before_idx, after_idx, fill_count, fill);
    if (lower_rank == upper_rank) {
        return lower;
    }
    let upper = kthValue(upper_rank, in_start, in_end, before_idx, after_idx, fill_count, fill);
    return (lower + upper) * 0.5f;
}

fn pushValue(
    value: f32,
    count: ptr<function, u32>,
    sum: ptr<function, f32>,
    prod: ptr<function, f32>,
    min_value: ptr<function, f32>,
    max_value: ptr<function, f32>,
    mean: ptr<function, f32>,
    m2: ptr<function, f32>,
    has_value: ptr<function, bool>,
) {
    (*count) = (*count) + 1u;
    (*sum) = (*sum) + value;
    (*prod) = (*prod) * value;
    if ((*has_value)) {
        (*min_value) = min((*min_value), value);
        (*max_value) = max((*max_value), value);
    } else {
        (*min_value) = value;
        (*max_value) = value;
        (*has_value) = true;
    }
    let delta = value - (*mean);
    (*mean) = (*mean) + delta / f32((*count));
    let delta2 = value - (*mean);
    (*m2) = (*m2) + delta * delta2;
}

fn pushRepeated(
    value: f32,
    repeats: u32,
    count: ptr<function, u32>,
    sum: ptr<function, f32>,
    prod: ptr<function, f32>,
    min_value: ptr<function, f32>,
    max_value: ptr<function, f32>,
    mean: ptr<function, f32>,
    m2: ptr<function, f32>,
    has_value: ptr<function, bool>,
) {
    if (repeats == 0u) {
        return;
    }
    (*sum) = (*sum) + value * f32(repeats);
    (*prod) = (*prod) * pow(value, f32(repeats));
    if ((*has_value)) {
        (*min_value) = min((*min_value), value);
        (*max_value) = max((*max_value), value);
    } else {
        (*min_value) = value;
        (*max_value) = value;
        (*has_value) = true;
    }
    if ((*count) == 0u) {
        (*count) = repeats;
        (*mean) = value;
        (*m2) = 0.0f;
        return;
    }
    let old_count = (*count);
    let new_count = old_count + repeats;
    let delta = value - (*mean);
    (*mean) = (*mean) + delta * f32(repeats) / f32(new_count);
    (*m2) = (*m2) + delta * delta * f32(old_count) * f32(repeats) / f32(new_count);
    (*count) = new_count;
}

fn finishValue(
    count: u32,
    sum: f32,
    prod: f32,
    min_value: f32,
    max_value: f32,
    m2: f32,
) -> f32 {
    if (count == 0u) {
        if (params.meta2.x == 0u) {
            return 0.0f;
        }
        if (params.meta2.x == 2u) {
            return 1.0f;
        }
        return nanValue();
    }
    if (params.meta2.x == 0u) {
        return sum;
    }
    if (params.meta2.x == 1u) {
        return sum / f32(count);
    }
    if (params.meta2.x == 2u) {
        return prod;
    }
    if (params.meta2.x == 3u) {
        return min_value;
    }
    if (params.meta2.x == 4u) {
        return max_value;
    }
    var denom = count;
    if (params.meta2.y == 0u && count > 1u) {
        denom = count - 1u;
    }
    let variance = m2 / f32(denom);
    if (params.meta2.x == 5u) {
        return sqrt(variance);
    }
    return variance;
}

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.meta0.x) {
        return;
    }
    let before_idx = idx % params.meta0.w;
    let tmp = idx / params.meta0.w;
    let out_pos = tmp % params.meta0.z;
    let after_idx = tmp / params.meta0.z;
    var center = out_pos;
    if (params.meta1.x == 1u) {
        center = center + params.meta1.z;
    }
    let start = i32(center) - i32(params.meta1.z);
    let end = i32(center) + i32(params.meta1.w);
    let last_axis = i32(params.meta0.y) - 1;
    let in_start = max(start, 0);
    let in_end = min(end, last_axis);

    var fill_count = 0u;
    if (params.meta1.x == 2u) {
        if (start < 0) {
            fill_count = fill_count + u32(-start);
        }
        if (end >= i32(params.meta0.y)) {
            fill_count = fill_count + u32(end - i32(params.meta0.y) + 1);
        }
    }

    var count = 0u;
    var sum = 0.0f;
    var prod = 1.0f;
    var min_value = 0.0f;
    var max_value = 0.0f;
    var mean = 0.0f;
    var m2 = 0.0f;
    var has_value = false;
    var saw_nan = false;
    var median_fill_count = 0u;

    if (in_start <= in_end) {
        var pos = u32(in_start);
        loop {
            if (pos > u32(in_end)) {
                break;
            }
            let in_idx = before_idx + pos * params.meta0.w + after_idx * params.meta0.w * params.meta0.y;
            let value = Input.data[in_idx];
            if (isNan(value)) {
                if (params.meta1.y == 0u) {
                    saw_nan = true;
                    break;
                }
            } else {
                pushValue(value, &count, &sum, &prod, &min_value, &max_value, &mean, &m2, &has_value);
            }
            pos = pos + 1u;
        }
    }

    if (!saw_nan && fill_count > 0u) {
        let fill = params.meta3.x;
        if (isNan(fill)) {
            if (params.meta1.y == 0u) {
                saw_nan = true;
            }
        } else if (params.meta2.x == 7u) {
            median_fill_count = fill_count;
        } else {
            pushRepeated(fill, fill_count, &count, &sum, &prod, &min_value, &max_value, &mean, &m2, &has_value);
        }
    }

    if (saw_nan) {
        Output.data[idx] = nanValue();
    } else if (params.meta2.x == 7u) {
        Output.data[idx] = medianValue(count + median_fill_count, in_start, in_end, before_idx, after_idx, median_fill_count, params.meta3.x);
    } else {
        Output.data[idx] = finishValue(count, sum, prod, min_value, max_value, m2);
    }
}
"#;
