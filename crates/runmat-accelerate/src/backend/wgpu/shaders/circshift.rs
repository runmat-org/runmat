pub const CIRCSHIFT_SHADER_F64: &str = r#"
const MAX_RANK: u32 = 128u;

struct PackedValue {
    value: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

alias PackedArray = array<PackedValue, MAX_RANK>;

struct Tensor {
    data: array<f64>,
};

struct Params {
    len: u32,
    offset: u32,
    rank: u32,
    _pad: u32,
    shape: PackedArray,
    strides: PackedArray,
    shifts: PackedArray,
};

@group(0) @binding(0) var<storage, read> Input: Tensor;
@group(0) @binding(1) var<storage, read_write> Output: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local_index = gid.x;
    if local_index >= params.len {
        return;
    }
    let global_index = params.offset + local_index;

    var coords: array<u32, MAX_RANK>;
    var tmp = global_index;
    var dim: u32 = 0u;
    loop {
        if dim >= params.rank {
            break;
        }
        let size = params.shape[dim].value;
        if size == 0u {
            coords[dim] = 0u;
        } else {
            coords[dim] = tmp % size;
            tmp = tmp / size;
        }
        dim = dim + 1u;
    }

    var src_index: u32 = 0u;
    dim = 0u;
    loop {
        if dim >= params.rank {
            break;
        }
        let size = params.shape[dim].value;
        let stride = params.strides[dim].value;
        var coord = coords[dim];
        if size > 1u {
            let shift = params.shifts[dim].value;
            if shift != 0u {
                let wrap = shift % size;
                coord = (coord + size - wrap) % size;
            }
        }
        src_index = src_index + coord * stride;
        dim = dim + 1u;
    }

    Output.data[global_index] = Input.data[src_index];
}
"#;

pub const CIRCSHIFT_SHADER_F32: &str = r#"
const MAX_RANK: u32 = 128u;

struct PackedValue {
    value: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

alias PackedArray = array<PackedValue, MAX_RANK>;

struct Tensor {
    data: array<f32>,
};

struct Params {
    len: u32,
    offset: u32,
    rank: u32,
    _pad: u32,
    shape: PackedArray,
    strides: PackedArray,
    shifts: PackedArray,
};

@group(0) @binding(0) var<storage, read> Input: Tensor;
@group(0) @binding(1) var<storage, read_write> Output: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local_index = gid.x;
    if local_index >= params.len {
        return;
    }
    let global_index = params.offset + local_index;

    var coords: array<u32, MAX_RANK>;
    var tmp = global_index;
    var dim: u32 = 0u;
    loop {
        if dim >= params.rank {
            break;
        }
        let size = params.shape[dim].value;
        if size == 0u {
            coords[dim] = 0u;
        } else {
            coords[dim] = tmp % size;
            tmp = tmp / size;
        }
        dim = dim + 1u;
    }

    var src_index: u32 = 0u;
    dim = 0u;
    loop {
        if dim >= params.rank {
            break;
        }
        let size = params.shape[dim].value;
        let stride = params.strides[dim].value;
        var coord = coords[dim];
        if size > 1u {
            let shift = params.shifts[dim].value;
            if shift != 0u {
                let wrap = shift % size;
                coord = (coord + size - wrap) % size;
            }
        }
        src_index = src_index + coord * stride;
        dim = dim + 1u;
    }

    Output.data[global_index] = Input.data[src_index];
}
"#;
