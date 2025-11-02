pub const IMFILTER_SHADER_F64: &str = r#"
const MAX_RANK: u32 = 512u;

struct Tensor {
    data: array<f64>,
};

struct OffsetsBuffer {
    data: array<i32>,
};

struct KernelValues {
    data: array<f64>,
};

struct PackedValueU32 { value: u32, _pad0: u32, _pad1: u32, _pad2: u32 }
struct PackedValueI32 { value: i32, _pad0: i32, _pad1: i32, _pad2: i32 }
alias PackedArrayU32 = array<PackedValueU32, MAX_RANK>;
alias PackedArrayI32 = array<PackedValueI32, MAX_RANK>;

struct Params {
    len: u32,
    offset: u32,
    rank: u32,
    padding: u32,
    kernel_points: u32,
    image_len: u32,
    _pad0: u32,
    _pad1: u32,
    constant_value: f64,
    _pad_const: f64,
    image_shape: PackedArrayU32,
    image_strides: PackedArrayU32,
    output_shape: PackedArrayU32,
    base_offset: PackedArrayI32,
};

fn clamp_index(coord: i32, len: i32) -> u32 {
    if coord < 0 {
        return 0u;
    }
    if coord >= len {
        return u32(len - 1);
    }
    return u32(coord);
}

fn wrap_index(coord_in: i32, len: i32) -> u32 {
    if len <= 0 {
        return 0u;
    }
    var coord = coord_in % len;
    if coord < 0 {
        coord = coord + len;
    }
    return u32(coord);
}

fn reflect_index(coord_in: i32, len: i32) -> u32 {
    if len <= 0 {
        return 0u;
    }
    if len == 1 {
        return 0u;
    }
    let period = 2 * len - 2;
    var value = coord_in % period;
    if value < 0 {
        value = value + period;
    }
    if value >= len {
        value = period - value;
    }
    return u32(value);
}

@group(0) @binding(0) var<storage, read> Image: Tensor;
@group(0) @binding(1) var<storage, read> KernelOffsets: OffsetsBuffer;
@group(0) @binding(2) var<storage, read> KernelWeights: KernelValues;
@group(0) @binding(3) var<storage, read_write> Output: Tensor;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local_idx = gid.x;
    if local_idx >= params.len {
        return;
    }
    let global_idx = params.offset + local_idx;

    var coords: array<i32, MAX_RANK>;
    var tmp = global_idx;
    var dim: u32 = 0u;
    loop {
        if dim >= params.rank {
            break;
        }
        let size = params.output_shape[dim].value;
        if size == 0u {
            coords[dim] = 0;
        } else {
            coords[dim] = i32(tmp % size);
            tmp = tmp / size;
        }
        dim = dim + 1u;
    }

    var acc: f64 = 0.0;
    let stride = params.rank;
    var kp: u32 = 0u;
    loop {
        if kp >= params.kernel_points {
            break;
        }
        var need_load = true;
        var sample = params.constant_value;
        var linear: u32 = 0u;
        dim = 0u;
        loop {
            if dim >= params.rank {
                break;
            }
            let len = i32(params.image_shape[dim].value);
            let stride_val = params.image_strides[dim].value;
            var coord = coords[dim] + params.base_offset[dim].value;
            let offset_index = kp * stride + dim;
            coord = coord + KernelOffsets.data[offset_index];
            if len <= 0 {
                need_load = false;
                sample = params.constant_value;
                break;
            }
            if coord >= 0 && coord < len {
                linear = linear + u32(coord) * stride_val;
            } else {
                switch params.padding {
                    case 0u: {
                        need_load = false;
                        sample = params.constant_value;
                    }
                    case 1u: {
                        let clamped = clamp_index(coord, len);
                        linear = linear + clamped * stride_val;
                    }
                    case 2u: {
                        let reflected = reflect_index(coord, len);
                        linear = linear + reflected * stride_val;
                    }
                    case 3u: {
                        let wrapped = wrap_index(coord, len);
                        linear = linear + wrapped * stride_val;
                    }
                    default: {
                        need_load = false;
                        sample = params.constant_value;
                    }
                }
                if !need_load {
                    break;
                }
            }
            dim = dim + 1u;
        }
        if need_load {
            if params.image_len == 0u {
                sample = params.constant_value;
            } else {
                sample = Image.data[linear];
            }
        }
        acc = acc + KernelWeights.data[kp] * sample;
        kp = kp + 1u;
    }

    Output.data[global_idx] = acc;
}
"#;

pub const IMFILTER_SHADER_F32: &str = r#"
const MAX_RANK: u32 = 512u;

struct Tensor {
    data: array<f32>,
};

struct OffsetsBuffer {
    data: array<i32>,
};

struct KernelValues {
    data: array<f32>,
};

struct PackedValueU32 { value: u32, _pad0: u32, _pad1: u32, _pad2: u32 }
struct PackedValueI32 { value: i32, _pad0: i32, _pad1: i32, _pad2: i32 }
alias PackedArrayU32 = array<PackedValueU32, MAX_RANK>;
alias PackedArrayI32 = array<PackedValueI32, MAX_RANK>;

struct Params {
    len: u32,
    offset: u32,
    rank: u32,
    padding: u32,
    kernel_points: u32,
    image_len: u32,
    _pad0: u32,
    _pad1: u32,
    constant_value: f32,
    _pad_const0: f32,
    _pad_const1: f32,
    _pad_const2: f32,
    image_shape: PackedArrayU32,
    image_strides: PackedArrayU32,
    output_shape: PackedArrayU32,
    base_offset: PackedArrayI32,
};

fn clamp_index(coord: i32, len: i32) -> u32 {
    if coord < 0 {
        return 0u;
    }
    if coord >= len {
        return u32(len - 1);
    }
    return u32(coord);
}

fn wrap_index(coord_in: i32, len: i32) -> u32 {
    if len <= 0 {
        return 0u;
    }
    var coord = coord_in % len;
    if coord < 0 {
        coord = coord + len;
    }
    return u32(coord);
}

fn reflect_index(coord_in: i32, len: i32) -> u32 {
    if len <= 0 {
        return 0u;
    }
    if len == 1 {
        return 0u;
    }
    let period = 2 * len - 2;
    var value = coord_in % period;
    if value < 0 {
        value = value + period;
    }
    if value >= len {
        value = period - value;
    }
    return u32(value);
}

@group(0) @binding(0) var<storage, read> Image: Tensor;
@group(0) @binding(1) var<storage, read> KernelOffsets: OffsetsBuffer;
@group(0) @binding(2) var<storage, read> KernelWeights: KernelValues;
@group(0) @binding(3) var<storage, read_write> Output: Tensor;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local_idx = gid.x;
    if local_idx >= params.len {
        return;
    }
    let global_idx = params.offset + local_idx;

    var coords: array<i32, MAX_RANK>;
    var tmp = global_idx;
    var dim: u32 = 0u;
    loop {
        if dim >= params.rank {
            break;
        }
        let size = params.output_shape[dim].value;
        if size == 0u {
            coords[dim] = 0;
        } else {
            coords[dim] = i32(tmp % size);
            tmp = tmp / size;
        }
        dim = dim + 1u;
    }

    var acc: f32 = 0.0;
    let stride = params.rank;
    var kp: u32 = 0u;
    loop {
        if kp >= params.kernel_points {
            break;
        }
        var need_load = true;
        var sample = params.constant_value;
        var linear: u32 = 0u;
        dim = 0u;
        loop {
            if dim >= params.rank {
                break;
            }
            let len = i32(params.image_shape[dim].value);
            let stride_val = params.image_strides[dim].value;
            var coord = coords[dim] + params.base_offset[dim].value;
            let offset_index = kp * stride + dim;
            coord = coord + KernelOffsets.data[offset_index];
            if len <= 0 {
                need_load = false;
                sample = params.constant_value;
                break;
            }
            if coord >= 0 && coord < len {
                linear = linear + u32(coord) * stride_val;
            } else {
                switch params.padding {
                    case 0u: {
                        need_load = false;
                        sample = params.constant_value;
                    }
                    case 1u: {
                        let clamped = clamp_index(coord, len);
                        linear = linear + clamped * stride_val;
                    }
                    case 2u: {
                        let reflected = reflect_index(coord, len);
                        linear = linear + reflected * stride_val;
                    }
                    case 3u: {
                        let wrapped = wrap_index(coord, len);
                        linear = linear + wrapped * stride_val;
                    }
                    default: {
                        need_load = false;
                        sample = params.constant_value;
                    }
                }
                if !need_load {
                    break;
                }
            }
            dim = dim + 1u;
        }
        if need_load {
            if params.image_len == 0u {
                sample = params.constant_value;
            } else {
                sample = Image.data[linear];
            }
        }
        acc = acc + KernelWeights.data[kp] * sample;
        kp = kp + 1u;
    }

    Output.data[global_idx] = acc;
}
"#;
