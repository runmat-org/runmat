pub const REPMAT_SHADER_F64: &str = r#"
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
    base_shape: PackedArray,
    new_shape: PackedArray,
    base_strides: PackedArray,
};

@group(0) @binding(0) var<storage, read> Input: Tensor;
@group(0) @binding(1) var<storage, read_write> Output: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.len {
        return;
    }
    let global_index = params.offset + idx;

    var remaining = global_index;
    var src_index: u32 = 0u;
    var dim: u32 = 0u;

    loop {
        if dim >= params.rank {
            break;
        }
        let size = params.new_shape[dim].value;
        var coord: u32 = 0u;
        if size != 0u {
            coord = remaining % size;
            remaining = remaining / size;
        }
        let base = params.base_shape[dim].value;
        var orig_coord: u32 = 0u;
        if base != 0u {
            orig_coord = coord % base;
        }
        let stride = params.base_strides[dim].value;
        src_index = src_index + orig_coord * stride;
        dim = dim + 1u;
    }

    Output.data[global_index] = Input.data[src_index];
}
"#;

pub const REPMAT_SHADER_F32: &str = r#"
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
    base_shape: PackedArray,
    new_shape: PackedArray,
    base_strides: PackedArray,
};

@group(0) @binding(0) var<storage, read> Input: Tensor;
@group(0) @binding(1) var<storage, read_write> Output: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.len {
        return;
    }
    let global_index = params.offset + idx;

    var remaining = global_index;
    var src_index: u32 = 0u;
    var dim: u32 = 0u;

    loop {
        if dim >= params.rank {
            break;
        }
        let size = params.new_shape[dim].value;
        var coord: u32 = 0u;
        if size != 0u {
            coord = remaining % size;
            remaining = remaining / size;
        }
        let base = params.base_shape[dim].value;
        var orig_coord: u32 = 0u;
        if base != 0u {
            orig_coord = coord % base;
        }
        let stride = params.base_strides[dim].value;
        src_index = src_index + orig_coord * stride;
        dim = dim + 1u;
    }

    Output.data[global_index] = Input.data[src_index];
}
"#;
