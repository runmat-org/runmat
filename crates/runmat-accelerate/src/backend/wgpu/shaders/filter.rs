pub const FILTER_SHADER_F64: &str = r#"
const MAX_RANK: u32 = 8u;

struct PackedValue {
    value: u32,
    _pad: vec3<u32>,
};

alias PackedArray = array<PackedValue, MAX_RANK>;

struct Tensor {
    data: array<f64>,
};

struct Params {
    dim_len: u32,
    leading: u32,
    trailing: u32,
    order: u32,
    state_len: u32,
    signal_len: u32,
    channel_count: u32,
    zi_present: u32,
    dim_idx: u32,
    rank: u32,
    state_rank: u32,
    lane_factor: u32,
    signal_shape: PackedArray,
    state_shape: PackedArray,
};

@group(0) @binding(0) var<storage, read> Signal: Tensor;
@group(0) @binding(1) var<storage, read> Num: Tensor;
@group(0) @binding(2) var<storage, read> Den: Tensor;
@group(0) @binding(3) var<storage, read> Zi: Tensor;
@group(0) @binding(4) var<storage, read_write> Output: Tensor;
@group(0) @binding(5) var<storage, read_write> States: Tensor;
@group(0) @binding(6) var<storage, read_write> FinalState: Tensor;
@group(0) @binding(7) var<uniform> params: Params;

fn state_linear_index(state_index: u32, channel: u32) -> u32 {
    return channel * params.state_len + state_index;
}
@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let channel = gid.x;
    if channel >= params.channel_count {
        return;
    }
    if params.leading == 0u {
        return;
    }

    let state_len = params.state_len;
    let order = params.order;
    let dim_len = params.dim_len;
    let leading = params.leading;
    let base_before = channel % leading;
    let trailing_index: u32 = select(channel / leading, 0u, leading == 0u);
    let base = trailing_index * dim_len * leading;

    let zi_len = arrayLength(&Zi.data);
    let states_len = arrayLength(&States.data);
    let final_len = arrayLength(&FinalState.data);

    if state_len == 0u {
        let gain = Num.data[0];
        var step: u32 = 0u;
        let stride = leading;
        let mut_index = base + base_before;
        loop {
            if step >= dim_len {
                break;
            }
            let idx = mut_index + step * stride;
            if idx >= params.signal_len {
                break;
            }
            let raw_idx = idx * params.lane_factor;
            var lane: u32 = 0u;
            loop {
                if lane >= params.lane_factor {
                    break;
                }
                let x_n = Signal.data[raw_idx + lane];
                Output.data[raw_idx + lane] = gain * x_n;
                lane = lane + 1u;
            }
            step = step + 1u;
        }
        if final_len > 0u {
            var i: u32 = 0u;
            loop {
                if i >= final_len {
                    break;
                }
                FinalState.data[i] = 0.0;
                i = i + 1u;
            }
        }
        return;
    }

    let state_base = channel * state_len * params.lane_factor;

    if params.zi_present != 0u && zi_len > 0u {
        var s: u32 = 0u;
        loop {
            if s >= state_len {
                break;
            }
            let offset = state_linear_index(s, channel);
            let raw_offset = offset * params.lane_factor;
            let raw_state = state_base + s * params.lane_factor;
            var lane: u32 = 0u;
            loop {
                if lane >= params.lane_factor {
                    break;
                }
                var value: f64 = 0.0;
                if raw_offset + lane < zi_len {
                    value = Zi.data[raw_offset + lane];
                }
                if raw_state + lane < states_len {
                    States.data[raw_state + lane] = value;
                }
                lane = lane + 1u;
            }
            s = s + 1u;
        }
    } else {
        var s: u32 = 0u;
        loop {
            if s >= state_len {
                break;
            }
            let raw_state = state_base + s * params.lane_factor;
            var lane: u32 = 0u;
            loop {
                if lane >= params.lane_factor {
                    break;
                }
                if raw_state + lane < states_len {
                    States.data[raw_state + lane] = 0.0;
                }
                lane = lane + 1u;
            }
            s = s + 1u;
        }
    }

    let stride = leading;
    let mut_index = base + base_before;
    let b0 = Num.data[0];

    var step: u32 = 0u;
    loop {
        if step >= dim_len {
            break;
        }
        let idx = mut_index + step * stride;
        if idx >= params.signal_len {
            break;
        }
        let raw_idx = idx * params.lane_factor;
        var lane: u32 = 0u;
        loop {
            if lane >= params.lane_factor {
                break;
            }
            let x_n = Signal.data[raw_idx + lane];
            var y = b0 * x_n;
            if state_base + lane < states_len {
                y = y + States.data[state_base + lane];
            }
            Output.data[raw_idx + lane] = y;

            var i: u32 = 1u;
            loop {
                if i >= order {
                    break;
                }
                let state_i = state_base + i * params.lane_factor + lane;
                var next_state: f64 = 0.0;
                if i < state_len && state_i < states_len {
                    next_state = States.data[state_i];
                }
                let new_state = Num.data[i] * x_n + next_state - Den.data[i] * y;
                let prev_state = state_base + (i - 1u) * params.lane_factor + lane;
                if prev_state < states_len {
                    States.data[prev_state] = new_state;
                }
                i = i + 1u;
            }
            lane = lane + 1u;
        }

        step = step + 1u;
    }

    if final_len > 0u {
        var s: u32 = 0u;
        loop {
            if s >= state_len {
                break;
            }
            let offset = state_linear_index(s, channel);
            let raw_offset = offset * params.lane_factor;
            let raw_state = state_base + s * params.lane_factor;
            var lane: u32 = 0u;
            loop {
                if lane >= params.lane_factor {
                    break;
                }
                if raw_offset + lane < final_len && raw_state + lane < states_len {
                    FinalState.data[raw_offset + lane] = States.data[raw_state + lane];
                }
                lane = lane + 1u;
            }
            s = s + 1u;
        }
    }
}
"#;

pub const FILTER_SHADER_F32: &str = r#"
const MAX_RANK: u32 = 8u;

struct PackedValue {
    value: u32,
    _pad: vec3<u32>,
};

alias PackedArray = array<PackedValue, MAX_RANK>;

struct Tensor {
    data: array<f32>,
};

struct Params {
    dim_len: u32,
    leading: u32,
    trailing: u32,
    order: u32,
    state_len: u32,
    signal_len: u32,
    channel_count: u32,
    zi_present: u32,
    dim_idx: u32,
    rank: u32,
    state_rank: u32,
    lane_factor: u32,
    signal_shape: PackedArray,
    state_shape: PackedArray,
};

@group(0) @binding(0) var<storage, read> Signal: Tensor;
@group(0) @binding(1) var<storage, read> Num: Tensor;
@group(0) @binding(2) var<storage, read> Den: Tensor;
@group(0) @binding(3) var<storage, read> Zi: Tensor;
@group(0) @binding(4) var<storage, read_write> Output: Tensor;
@group(0) @binding(5) var<storage, read_write> States: Tensor;
@group(0) @binding(6) var<storage, read_write> FinalState: Tensor;
@group(0) @binding(7) var<uniform> params: Params;

fn state_linear_index(state_index: u32, channel: u32) -> u32 {
    return channel * params.state_len + state_index;
}
@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let channel = gid.x;
    if channel >= params.channel_count {
        return;
    }
    if params.leading == 0u {
        return;
    }

    let state_len = params.state_len;
    let order = params.order;
    let dim_len = params.dim_len;
    let leading = params.leading;
    let base_before = channel % leading;
    let trailing_index: u32 = select(channel / leading, 0u, leading == 0u);
    let base = trailing_index * dim_len * leading;

    let zi_len = arrayLength(&Zi.data);
    let states_len = arrayLength(&States.data);
    let final_len = arrayLength(&FinalState.data);

    if state_len == 0u {
        let gain = Num.data[0];
        var step: u32 = 0u;
        let stride = leading;
        let mut_index = base + base_before;
        loop {
            if step >= dim_len {
                break;
            }
            let idx = mut_index + step * stride;
            if idx >= params.signal_len {
                break;
            }
            let raw_idx = idx * params.lane_factor;
            var lane: u32 = 0u;
            loop {
                if lane >= params.lane_factor {
                    break;
                }
                let x_n = Signal.data[raw_idx + lane];
                Output.data[raw_idx + lane] = gain * x_n;
                lane = lane + 1u;
            }
            step = step + 1u;
        }
        if final_len > 0u {
            var i: u32 = 0u;
            loop {
                if i >= final_len {
                    break;
                }
                FinalState.data[i] = 0.0;
                i = i + 1u;
            }
        }
        return;
    }

    let state_base = channel * state_len * params.lane_factor;

    if params.zi_present != 0u && zi_len > 0u {
        var s: u32 = 0u;
        loop {
            if s >= state_len {
                break;
            }
            let offset = state_linear_index(s, channel);
            let raw_offset = offset * params.lane_factor;
            let raw_state = state_base + s * params.lane_factor;
            var lane: u32 = 0u;
            loop {
                if lane >= params.lane_factor {
                    break;
                }
                var value: f32 = 0.0;
                if raw_offset + lane < zi_len {
                    value = Zi.data[raw_offset + lane];
                }
                if raw_state + lane < states_len {
                    States.data[raw_state + lane] = value;
                }
                lane = lane + 1u;
            }
            s = s + 1u;
        }
    } else {
        var s: u32 = 0u;
        loop {
            if s >= state_len {
                break;
            }
            let raw_state = state_base + s * params.lane_factor;
            var lane: u32 = 0u;
            loop {
                if lane >= params.lane_factor {
                    break;
                }
                if raw_state + lane < states_len {
                    States.data[raw_state + lane] = 0.0;
                }
                lane = lane + 1u;
            }
            s = s + 1u;
        }
    }

    let stride = leading;
    let mut_index = base + base_before;
    let b0 = Num.data[0];

    var step: u32 = 0u;
    loop {
        if step >= dim_len {
            break;
        }
        let idx = mut_index + step * stride;
        if idx >= params.signal_len {
            break;
        }
        let raw_idx = idx * params.lane_factor;
        var lane: u32 = 0u;
        loop {
            if lane >= params.lane_factor {
                break;
            }
            let x_n = Signal.data[raw_idx + lane];
            var y = b0 * x_n;
            if state_base + lane < states_len {
                y = y + States.data[state_base + lane];
            }
            Output.data[raw_idx + lane] = y;

            var i: u32 = 1u;
            loop {
                if i >= order {
                    break;
                }
                let state_i = state_base + i * params.lane_factor + lane;
                var next_state: f32 = 0.0;
                if i < state_len && state_i < states_len {
                    next_state = States.data[state_i];
                }
                let new_state = Num.data[i] * x_n + next_state - Den.data[i] * y;
                let prev_state = state_base + (i - 1u) * params.lane_factor + lane;
                if prev_state < states_len {
                    States.data[prev_state] = new_state;
                }
                i = i + 1u;
            }
            lane = lane + 1u;
        }

        step = step + 1u;
    }

    if final_len > 0u {
        var s: u32 = 0u;
        loop {
            if s >= state_len {
                break;
            }
            let offset = state_linear_index(s, channel);
            let raw_offset = offset * params.lane_factor;
            let raw_state = state_base + s * params.lane_factor;
            var lane: u32 = 0u;
            loop {
                if lane >= params.lane_factor {
                    break;
                }
                if raw_offset + lane < final_len && raw_state + lane < states_len {
                    FinalState.data[raw_offset + lane] = States.data[raw_state + lane];
                }
                lane = lane + 1u;
            }
            s = s + 1u;
        }
    }
}
"#;
