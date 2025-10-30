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
    _pad: u32,
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

fn compute_before_coords(channel: u32, before: ptr<function, array<u32, MAX_RANK>>) -> u32 {
    if params.dim_idx == 0u {
        return 0u;
    }
    var idx: u32 = channel % params.leading;
    var axis: u32 = 0u;
    loop {
        if axis >= params.dim_idx {
            break;
        }
        let size = params.signal_shape[axis].value;
        if size == 0u {
            (*before)[axis] = 0u;
        } else {
            (*before)[axis] = idx % size;
            idx = idx / size;
        }
        axis = axis + 1u;
    }
    return channel % params.leading;
}

fn compute_after_coords(
    channel: u32,
    after: ptr<function, array<u32, MAX_RANK>>,
) -> u32 {
    if params.rank <= params.dim_idx + 1u {
        return 0u;
    }
    let leading = max(params.leading, 1u);
    var idx: u32 = channel / leading;
    var axis: u32 = params.dim_idx + 1u;
    var offset: u32 = 0u;
    loop {
        if axis >= params.rank {
            break;
        }
        let size = params.signal_shape[axis].value;
        if size == 0u {
            (*after)[offset] = 0u;
        } else {
            (*after)[offset] = idx % size;
            idx = idx / size;
        }
        offset = offset + 1u;
        axis = axis + 1u;
    }
    return offset;
}

fn state_linear_index(
    state_index: u32,
    before: ptr<function, array<u32, MAX_RANK>>,
    after: ptr<function, array<u32, MAX_RANK>>,
    after_len: u32,
) -> u32 {
    var offset: u32 = 0u;
    var stride: u32 = 1u;
    var axis: u32 = 0u;
    var after_axis: u32 = 0u;
    loop {
        if axis >= params.state_rank {
            break;
        }
        let shape = params.state_shape[axis].value;
        var coord: u32 = 0u;
        if axis < params.dim_idx {
            coord = (*before)[axis];
        } else if axis == params.dim_idx {
            coord = state_index;
        } else {
            if after_axis < after_len {
                coord = (*after)[after_axis];
            } else {
                coord = 0u;
            }
            after_axis = after_axis + 1u;
        }
        offset = offset + coord * stride;
        stride = stride * max(shape, 1u);
        axis = axis + 1u;
    }
    return offset;
}

@compute @workgroup_size(256)
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
            let x_n = Signal.data[idx];
            Output.data[idx] = gain * x_n;
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

    var before_coords: array<u32, MAX_RANK>;
    var after_coords: array<u32, MAX_RANK>;
    _ = compute_before_coords(channel, &before_coords);
    let after_len = compute_after_coords(channel, &after_coords);

    let state_base = channel * state_len;

    if params.zi_present != 0u && zi_len > 0u {
        var s: u32 = 0u;
        loop {
            if s >= state_len {
                break;
            }
            let offset = state_linear_index(s, &before_coords, &after_coords, after_len);
            var value: f64 = 0.0;
            if offset < zi_len {
                value = Zi.data[offset];
            }
            if state_base + s < states_len {
                States.data[state_base + s] = value;
            }
            s = s + 1u;
        }
    } else {
        var s: u32 = 0u;
        loop {
            if s >= state_len {
                break;
            }
            if state_base + s < states_len {
                States.data[state_base + s] = 0.0;
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
        let x_n = Signal.data[idx];
        var y = b0 * x_n;
        if state_base < states_len {
            y = y + States.data[state_base];
        }
        Output.data[idx] = y;

        var i: u32 = 1u;
        loop {
            if i >= order {
                break;
            }
            var next_state: f64 = 0.0;
            if i < state_len && state_base + i < states_len {
                next_state = States.data[state_base + i];
            }
            let new_state = Num.data[i] * x_n + next_state - Den.data[i] * y;
            if state_base + i - 1u < states_len {
                States.data[state_base + i - 1u] = new_state;
            }
            i = i + 1u;
        }

        step = step + 1u;
    }

    if final_len > 0u {
        var s: u32 = 0u;
        loop {
            if s >= state_len {
                break;
            }
            let offset = state_linear_index(s, &before_coords, &after_coords, after_len);
            if offset < final_len && state_base + s < states_len {
                FinalState.data[offset] = States.data[state_base + s];
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
    _pad: u32,
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

fn compute_before_coords(channel: u32, before: ptr<function, array<u32, MAX_RANK>>) -> u32 {
    if params.dim_idx == 0u {
        return 0u;
    }
    var idx: u32 = channel % params.leading;
    var axis: u32 = 0u;
    loop {
        if axis >= params.dim_idx {
            break;
        }
        let size = params.signal_shape[axis].value;
        if size == 0u {
            (*before)[axis] = 0u;
        } else {
            (*before)[axis] = idx % size;
            idx = idx / size;
        }
        axis = axis + 1u;
    }
    return channel % params.leading;
}

fn compute_after_coords(
    channel: u32,
    after: ptr<function, array<u32, MAX_RANK>>,
) -> u32 {
    if params.rank <= params.dim_idx + 1u {
        return 0u;
    }
    let leading = max(params.leading, 1u);
    var idx: u32 = channel / leading;
    var axis: u32 = params.dim_idx + 1u;
    var offset: u32 = 0u;
    loop {
        if axis >= params.rank {
            break;
        }
        let size = params.signal_shape[axis].value;
        if size == 0u {
            (*after)[offset] = 0u;
        } else {
            (*after)[offset] = idx % size;
            idx = idx / size;
        }
        offset = offset + 1u;
        axis = axis + 1u;
    }
    return offset;
}

fn state_linear_index(
    state_index: u32,
    before: ptr<function, array<u32, MAX_RANK>>,
    after: ptr<function, array<u32, MAX_RANK>>,
    after_len: u32,
) -> u32 {
    var offset: u32 = 0u;
    var stride: u32 = 1u;
    var axis: u32 = 0u;
    var after_axis: u32 = 0u;
    loop {
        if axis >= params.state_rank {
            break;
        }
        let shape = params.state_shape[axis].value;
        var coord: u32 = 0u;
        if axis < params.dim_idx {
            coord = (*before)[axis];
        } else if axis == params.dim_idx {
            coord = state_index;
        } else {
            if after_axis < after_len {
                coord = (*after)[after_axis];
            } else {
                coord = 0u;
            }
            after_axis = after_axis + 1u;
        }
        offset = offset + coord * stride;
        stride = stride * max(shape, 1u);
        axis = axis + 1u;
    }
    return offset;
}

@compute @workgroup_size(256)
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
            let x_n = Signal.data[idx];
            Output.data[idx] = gain * x_n;
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

    var before_coords: array<u32, MAX_RANK>;
    var after_coords: array<u32, MAX_RANK>;
    _ = compute_before_coords(channel, &before_coords);
    let after_len = compute_after_coords(channel, &after_coords);

    let state_base = channel * state_len;

    if params.zi_present != 0u && zi_len > 0u {
        var s: u32 = 0u;
        loop {
            if s >= state_len {
                break;
            }
            let offset = state_linear_index(s, &before_coords, &after_coords, after_len);
            var value: f32 = 0.0;
            if offset < zi_len {
                value = Zi.data[offset];
            }
            if state_base + s < states_len {
                States.data[state_base + s] = value;
            }
            s = s + 1u;
        }
    } else {
        var s: u32 = 0u;
        loop {
            if s >= state_len {
                break;
            }
            if state_base + s < states_len {
                States.data[state_base + s] = 0.0;
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
        let x_n = Signal.data[idx];
        var y = b0 * x_n;
        if state_base < states_len {
            y = y + States.data[state_base];
        }
        Output.data[idx] = y;

        var i: u32 = 1u;
        loop {
            if i >= order {
                break;
            }
            var next_state: f32 = 0.0;
            if i < state_len && state_base + i < states_len {
                next_state = States.data[state_base + i];
            }
            let new_state = Num.data[i] * x_n + next_state - Den.data[i] * y;
            if state_base + i - 1u < states_len {
                States.data[state_base + i - 1u] = new_state;
            }
            i = i + 1u;
        }

        step = step + 1u;
    }

    if final_len > 0u {
        var s: u32 = 0u;
        loop {
            if s >= state_len {
                break;
            }
            let offset = state_linear_index(s, &before_coords, &after_coords, after_len);
            if offset < final_len && state_base + s < states_len {
                FinalState.data[offset] = States.data[state_base + s];
            }
            s = s + 1u;
        }
    }
}
"#;
