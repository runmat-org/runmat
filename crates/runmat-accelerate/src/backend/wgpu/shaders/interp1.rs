pub const INTERP1_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct Params {
    sample_len: u32,
    query_len: u32,
    series_count: u32,
    output_len: u32,
    method: u32,
    extrapolation: u32,
    _pad0: u32,
    _pad1: u32,
    extrapolation_value: f64,
    _pad2: f64,
};

@group(0) @binding(0) var<storage, read> X: Tensor;
@group(0) @binding(1) var<storage, read> Y: Tensor;
@group(0) @binding(2) var<storage, read> Xq: Tensor;
@group(0) @binding(3) var<storage, read_write> Output: Tensor;
@group(0) @binding(4) var<uniform> params: Params;

fn nan_f64() -> f64 {
    var bits: u64 = 0x7ff8000000000000u;
    return bitcast<f64>(bits);
}

fn out_of_range_f64() -> f64 {
    if params.extrapolation == 2u {
        return params.extrapolation_value;
    }
    return nan_f64();
}

fn interval_index_f64(xq: f64) -> u32 {
    if xq < X.data[0u] {
        if params.extrapolation == 1u {
            return 0u;
        }
        return 0xffffffffu;
    }
    let last = params.sample_len - 1u;
    if xq > X.data[last] {
        if params.extrapolation == 1u {
            return last - 1u;
        }
        return 0xffffffffu;
    }
    if xq == X.data[last] {
        return last - 1u;
    }

    var lo: u32 = 0u;
    var hi: u32 = last;
    loop {
        if lo >= hi {
            break;
        }
        let mid = (lo + hi) / 2u;
        if X.data[mid] <= xq {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    if lo == 0u || lo >= params.sample_len {
        return 0xffffffffu;
    }
    return lo - 1u;
}

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.output_len {
        return;
    }
    let query_index = idx % params.query_len;
    let series = idx / params.query_len;
    let query = Xq.data[query_index];
    let y_base = series * params.sample_len;

    if query != query {
        Output.data[idx] = nan_f64();
        return;
    }

    if params.method == 1u {
        if query < X.data[0u] {
            Output.data[idx] = select(out_of_range_f64(), Y.data[y_base], params.extrapolation == 1u);
            return;
        }
        let last = params.sample_len - 1u;
        if query > X.data[last] {
            Output.data[idx] = select(out_of_range_f64(), Y.data[y_base + last], params.extrapolation == 1u);
            return;
        }

        var lo: u32 = 0u;
        var hi: u32 = last;
        loop {
            if lo >= hi {
                break;
            }
            let mid = (lo + hi) / 2u;
            if X.data[mid] < query {
                lo = mid + 1u;
            } else {
                hi = mid;
            }
        }
        if X.data[lo] == query {
            Output.data[idx] = Y.data[y_base + lo];
            return;
        }
        let right = min(lo, last);
        var left: u32 = 0u;
        if right > 0u {
            left = right - 1u;
        }
        if abs(query - X.data[left]) <= abs(X.data[right] - query) {
            Output.data[idx] = Y.data[y_base + left];
        } else {
            Output.data[idx] = Y.data[y_base + right];
        }
        return;
    }

    let piece = interval_index_f64(query);
    if piece == 0xffffffffu {
        Output.data[idx] = out_of_range_f64();
        return;
    }
    let x0 = X.data[piece];
    let x1 = X.data[piece + 1u];
    let y0 = Y.data[y_base + piece];
    let y1 = Y.data[y_base + piece + 1u];
    let t = (query - x0) / (x1 - x0);
    Output.data[idx] = y0 + t * (y1 - y0);
}
"#;

pub const INTERP1_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct Params {
    sample_len: u32,
    query_len: u32,
    series_count: u32,
    output_len: u32,
    method: u32,
    extrapolation: u32,
    _pad0: u32,
    _pad1: u32,
    extrapolation_value: f32,
    _pad2: f32,
    _pad3: f32,
    _pad4: f32,
};

@group(0) @binding(0) var<storage, read> X: Tensor;
@group(0) @binding(1) var<storage, read> Y: Tensor;
@group(0) @binding(2) var<storage, read> Xq: Tensor;
@group(0) @binding(3) var<storage, read_write> Output: Tensor;
@group(0) @binding(4) var<uniform> params: Params;

fn nan_f32() -> f32 {
    var bits: u32 = 0x7fc00000u;
    return bitcast<f32>(bits);
}

fn out_of_range_f32() -> f32 {
    if params.extrapolation == 2u {
        return params.extrapolation_value;
    }
    return nan_f32();
}

fn interval_index_f32(xq: f32) -> u32 {
    if xq < X.data[0u] {
        if params.extrapolation == 1u {
            return 0u;
        }
        return 0xffffffffu;
    }
    let last = params.sample_len - 1u;
    if xq > X.data[last] {
        if params.extrapolation == 1u {
            return last - 1u;
        }
        return 0xffffffffu;
    }
    if xq == X.data[last] {
        return last - 1u;
    }

    var lo: u32 = 0u;
    var hi: u32 = last;
    loop {
        if lo >= hi {
            break;
        }
        let mid = (lo + hi) / 2u;
        if X.data[mid] <= xq {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    if lo == 0u || lo >= params.sample_len {
        return 0xffffffffu;
    }
    return lo - 1u;
}

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.output_len {
        return;
    }
    let query_index = idx % params.query_len;
    let series = idx / params.query_len;
    let query = Xq.data[query_index];
    let y_base = series * params.sample_len;

    if query != query {
        Output.data[idx] = nan_f32();
        return;
    }

    if params.method == 1u {
        if query < X.data[0u] {
            Output.data[idx] = select(out_of_range_f32(), Y.data[y_base], params.extrapolation == 1u);
            return;
        }
        let last = params.sample_len - 1u;
        if query > X.data[last] {
            Output.data[idx] = select(out_of_range_f32(), Y.data[y_base + last], params.extrapolation == 1u);
            return;
        }

        var lo: u32 = 0u;
        var hi: u32 = last;
        loop {
            if lo >= hi {
                break;
            }
            let mid = (lo + hi) / 2u;
            if X.data[mid] < query {
                lo = mid + 1u;
            } else {
                hi = mid;
            }
        }
        if X.data[lo] == query {
            Output.data[idx] = Y.data[y_base + lo];
            return;
        }
        let right = min(lo, last);
        var left: u32 = 0u;
        if right > 0u {
            left = right - 1u;
        }
        if abs(query - X.data[left]) <= abs(X.data[right] - query) {
            Output.data[idx] = Y.data[y_base + left];
        } else {
            Output.data[idx] = Y.data[y_base + right];
        }
        return;
    }

    let piece = interval_index_f32(query);
    if piece == 0xffffffffu {
        Output.data[idx] = out_of_range_f32();
        return;
    }
    let x0 = X.data[piece];
    let x1 = X.data[piece + 1u];
    let y0 = Y.data[y_base + piece];
    let y1 = Y.data[y_base + piece + 1u];
    let t = (query - x0) / (x1 - x0);
    Output.data[idx] = y0 + t * (y1 - y0);
}
"#;
