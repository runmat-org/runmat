pub const IMAGE_NORMALIZE_STUB_SHADER_F64: &str = r#"
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

@compute @workgroup_size(1, 1, 1)
fn main() {
    // Minimal bootstrap shader for DX12. Real pipelines are compiled lazily
    // with adapter-aware workgroup sizes once tuning is known.
    if params.batch_count == 0u {
        return;
    }
}
"#;

pub const IMAGE_NORMALIZE_STUB_SHADER_F32: &str = r#"
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

@compute @workgroup_size(1, 1, 1)
fn main() {
    if params.batch_count == 0u {
        return;
    }
}
"#;
