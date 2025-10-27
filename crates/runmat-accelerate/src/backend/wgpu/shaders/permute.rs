pub const PERMUTE_SHADER_F64: &str = r#"
const MAX_RANK: u32 = 8u;

struct Tensor {
    data: array<f64>,
};

struct Params {
    len: u32,
    offset: u32,
    rank: u32,
    _pad: u32,
    src_shape: array<vec4<u32>, MAX_RANK>,
    dst_shape: array<vec4<u32>, MAX_RANK>,
    order: array<vec4<u32>, MAX_RANK>,
    src_strides: array<vec4<u32>, MAX_RANK>,
};

@group(0) @binding(0) var<storage, read> Input: Tensor;
@group(0) @binding(1) var<storage, read_write> Output: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
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
        let size = params.dst_shape[dim].x;
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
        let src_dim = params.order[dim].x;
        let stride = params.src_strides[src_dim].x;
        let coord = coords[dim];
        src_index = src_index + coord * stride;
        dim = dim + 1u;
    }

    Output.data[global_index] = Input.data[src_index];
}
"#;

pub const PERMUTE_SHADER_F32: &str = r#"
const MAX_RANK: u32 = 8u;

struct Tensor {
    data: array<f32>,
};

struct Params {
    len: u32,
    offset: u32,
    rank: u32,
    _pad: u32,
    src_shape: array<vec4<u32>, MAX_RANK>,
    dst_shape: array<vec4<u32>, MAX_RANK>,
    order: array<vec4<u32>, MAX_RANK>,
    src_strides: array<vec4<u32>, MAX_RANK>,
};

@group(0) @binding(0) var<storage, read> Input: Tensor;
@group(0) @binding(1) var<storage, read_write> Output: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
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
        let size = params.dst_shape[dim].x;
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
        let src_dim = params.order[dim].x;
        let stride = params.src_strides[src_dim].x;
        let coord = coords[dim];
        src_index = src_index + coord * stride;
        dim = dim + 1u;
    }

    Output.data[global_index] = Input.data[src_index];
}
"#;
