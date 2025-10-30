pub const KRON_SHADER_F64: &str = r#"
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
    len: u32,
    offset: u32,
    rank: u32,
    _pad: u32,
    shape_a: PackedArray,
    shape_b: PackedArray,
    shape_out: PackedArray,
    stride_a: PackedArray,
    stride_b: PackedArray,
};

@group(0) @binding(0) var<storage, read> A: Tensor;
@group(0) @binding(1) var<storage, read> B: Tensor;
@group(0) @binding(2) var<storage, read_write> Out: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.len {
        return;
    }

    let global_index = params.offset + idx;
    var remaining = global_index;
    var index_a: u32 = 0u;
    var index_b: u32 = 0u;
    var dim: u32 = 0u;

    loop {
        if dim >= params.rank {
            break;
        }
        let out_dim = params.shape_out[dim].value;
        var coord_out: u32 = 0u;
        if out_dim != 0u {
            coord_out = remaining % out_dim;
            remaining = remaining / out_dim;
        }

        let dim_b = params.shape_b[dim].value;
        var coord_b: u32 = 0u;
        var coord_a: u32 = 0u;
        if dim_b != 0u {
            coord_b = coord_out % dim_b;
            coord_a = coord_out / dim_b;
        }

        let stride_a = params.stride_a[dim].value;
        let stride_b = params.stride_b[dim].value;
        index_a = index_a + coord_a * stride_a;
        index_b = index_b + coord_b * stride_b;

        dim = dim + 1u;
    }

    Out.data[global_index] = A.data[index_a] * B.data[index_b];
}
"#;

pub const KRON_SHADER_F32: &str = r#"
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
    len: u32,
    offset: u32,
    rank: u32,
    _pad: u32,
    shape_a: PackedArray,
    shape_b: PackedArray,
    shape_out: PackedArray,
    stride_a: PackedArray,
    stride_b: PackedArray,
};

@group(0) @binding(0) var<storage, read> A: Tensor;
@group(0) @binding(1) var<storage, read> B: Tensor;
@group(0) @binding(2) var<storage, read_write> Out: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.len {
        return;
    }

    let global_index = params.offset + idx;
    var remaining = global_index;
    var index_a: u32 = 0u;
    var index_b: u32 = 0u;
    var dim: u32 = 0u;

    loop {
        if dim >= params.rank {
            break;
        }
        let out_dim = params.shape_out[dim].value;
        var coord_out: u32 = 0u;
        if out_dim != 0u {
            coord_out = remaining % out_dim;
            remaining = remaining / out_dim;
        }

        let dim_b = params.shape_b[dim].value;
        var coord_b: u32 = 0u;
        var coord_a: u32 = 0u;
        if dim_b != 0u {
            coord_b = coord_out % dim_b;
            coord_a = coord_out / dim_b;
        }

        let stride_a = params.stride_a[dim].value;
        let stride_b = params.stride_b[dim].value;
        index_a = index_a + coord_a * stride_a;
        index_b = index_b + coord_b * stride_b;

        dim = dim + 1u;
    }

    Out.data[global_index] = A.data[index_a] * B.data[index_b];
}
"#;
