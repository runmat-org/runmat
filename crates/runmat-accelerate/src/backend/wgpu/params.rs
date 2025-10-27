use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct LenOpParams {
    pub len: u32,
    pub op: u32,
    pub offset: u32,
    pub total: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ScalarParamsF64 {
    pub len: u32,
    pub op: u32,
    pub offset: u32,
    pub total: u32,
    pub scalar: f64,
    pub _pad_scalar: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ScalarParamsF32 {
    pub len: u32,
    pub op: u32,
    pub offset: u32,
    pub total: u32,
    pub scalar: f32,
    pub _pad_scalar: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct FusionParams {
    pub len: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct PackedU32(pub [u32; 4]);

impl PackedU32 {
    pub fn from_scalar(value: u32) -> Self {
        Self([value, 0, 0, 0])
    }
}

impl Default for PackedU32 {
    fn default() -> Self {
        Self([0; 4])
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct PackedI32(pub [i32; 4]);

impl PackedI32 {
    pub fn from_scalar(value: i32) -> Self {
        Self([value, 0, 0, 0])
    }
}

impl Default for PackedI32 {
    fn default() -> Self {
        Self([0; 4])
    }
}

pub const PERMUTE_MAX_RANK: usize = 8;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct PermuteParams {
    pub len: u32,
    pub offset: u32,
    pub rank: u32,
    pub _pad: u32,
    pub src_shape: [PackedU32; PERMUTE_MAX_RANK],
    pub dst_shape: [PackedU32; PERMUTE_MAX_RANK],
    pub order: [PackedU32; PERMUTE_MAX_RANK],
    pub src_strides: [PackedU32; PERMUTE_MAX_RANK],
}

pub const FLIP_MAX_RANK: usize = PERMUTE_MAX_RANK;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct FlipParams {
    pub len: u32,
    pub offset: u32,
    pub rank: u32,
    pub _pad: u32,
    pub shape: [PackedU32; FLIP_MAX_RANK],
    pub strides: [PackedU32; FLIP_MAX_RANK],
    pub flags: [PackedU32; FLIP_MAX_RANK],
}

pub const CIRCSHIFT_MAX_RANK: usize = PERMUTE_MAX_RANK;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct CircshiftParams {
    pub len: u32,
    pub offset: u32,
    pub rank: u32,
    pub _pad: u32,
    pub shape: [PackedU32; CIRCSHIFT_MAX_RANK],
    pub strides: [PackedU32; CIRCSHIFT_MAX_RANK],
    pub shifts: [PackedU32; CIRCSHIFT_MAX_RANK],
}

pub const IMFILTER_MAX_RANK: usize = 8;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ImfilterParamsF64 {
    pub len: u32,
    pub offset: u32,
    pub rank: u32,
    pub padding: u32,
    pub kernel_points: u32,
    pub image_len: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub constant_value: f64,
    pub _pad_const: f64,
    pub image_shape: [PackedU32; IMFILTER_MAX_RANK],
    pub image_strides: [PackedU32; IMFILTER_MAX_RANK],
    pub output_shape: [PackedU32; IMFILTER_MAX_RANK],
    pub base_offset: [PackedI32; IMFILTER_MAX_RANK],
    pub _pad_tail: PackedU32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ImfilterParamsF32 {
    pub len: u32,
    pub offset: u32,
    pub rank: u32,
    pub padding: u32,
    pub kernel_points: u32,
    pub image_len: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub constant_value: f32,
    pub _pad_const: [f32; 3],
    pub image_shape: [PackedU32; IMFILTER_MAX_RANK],
    pub image_strides: [PackedU32; IMFILTER_MAX_RANK],
    pub output_shape: [PackedU32; IMFILTER_MAX_RANK],
    pub base_offset: [PackedI32; IMFILTER_MAX_RANK],
    pub _pad_tail: PackedU32,
}

pub const REPMAT_MAX_RANK: usize = PERMUTE_MAX_RANK;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct RepmatParams {
    pub len: u32,
    pub offset: u32,
    pub rank: u32,
    pub _pad: u32,
    pub base_shape: [PackedU32; REPMAT_MAX_RANK],
    pub new_shape: [PackedU32; REPMAT_MAX_RANK],
    pub base_strides: [PackedU32; REPMAT_MAX_RANK],
}

pub const KRON_MAX_RANK: usize = PERMUTE_MAX_RANK;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct KronParams {
    pub len: u32,
    pub offset: u32,
    pub rank: u32,
    pub _pad: u32,
    pub shape_a: [PackedU32; KRON_MAX_RANK],
    pub shape_b: [PackedU32; KRON_MAX_RANK],
    pub shape_out: [PackedU32; KRON_MAX_RANK],
    pub stride_a: [PackedU32; KRON_MAX_RANK],
    pub stride_b: [PackedU32; KRON_MAX_RANK],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct TransposeParams {
    pub rows: u32,
    pub cols: u32,
    pub len: u32,
    pub offset: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct MatmulParams {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub lda: u32,
    pub ldb: u32,
    pub ldc: u32,
    pub _pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ReduceGlobalParams {
    pub len: u32,
    pub op: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ReduceDimParams {
    pub rows: u32,
    pub cols: u32,
    pub dim: u32,
    pub op: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct EyeParams {
    pub rows: u32,
    pub cols: u32,
    pub diag_len: u32,
    pub slices: u32,
    pub stride_slice: u32,
    pub diag_total: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct FillParamsF64 {
    pub value: f64,
    pub len: u32,
    pub _pad: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct FillParamsF32 {
    pub value: f32,
    pub len: u32,
    pub _pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct FspecialParamsF64 {
    pub rows: u32,
    pub cols: u32,
    pub kind: u32,
    pub len: u32,
    pub sigma: f64,
    pub alpha: f64,
    pub norm: f64,
    pub center_x: f64,
    pub center_y: f64,
    pub extra0: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct FspecialParamsF32 {
    pub rows: u32,
    pub cols: u32,
    pub kind: u32,
    pub len: u32,
    pub sigma: f32,
    pub alpha: f32,
    pub norm: f32,
    pub _pad0: f32,
    pub center_x: f32,
    pub center_y: f32,
    pub _pad1: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct DiagFromVectorParams {
    pub len: u32,
    pub size: u32,
    pub offset: i32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct DiagExtractParams {
    pub rows: u32,
    pub cols: u32,
    pub offset: i32,
    pub diag_len: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct LinspaceParamsF64 {
    pub start: f64,
    pub step: f64,
    pub stop: f64,
    pub total: u32,
    pub chunk: u32,
    pub offset: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct LinspaceParamsF32 {
    pub start: f32,
    pub step: f32,
    pub stop: f32,
    pub _pad0: f32,
    pub total: u32,
    pub chunk: u32,
    pub offset: u32,
    pub _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct RandomIntParamsF64 {
    pub lower: f64,
    pub upper: f64,
    pub span: f64,
    pub span_minus_one: f64,
    pub offset: u32,
    pub chunk: u32,
    pub seed: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct RandomIntParamsF32 {
    pub lower: f32,
    pub upper: f32,
    pub span: f32,
    pub span_minus_one: f32,
    pub offset: u32,
    pub chunk: u32,
    pub seed: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct RandomScalarParams {
    pub offset: u32,
    pub chunk: u32,
    pub seed: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct RandPermParams {
    pub n: u32,
    pub k: u32,
    pub seed: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct FindParams {
    pub len: u32,
    pub limit: u32,
    pub rows: u32,
    pub direction: u32,
    pub include_values: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct TrilParams {
    pub len: u32,
    pub start: u32,
    pub rows: u32,
    pub cols: u32,
    pub plane: u32,
    pub diag_offset: i32,
    pub _pad0: u32,
    pub _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct TriuParams {
    pub len: u32,
    pub start: u32,
    pub rows: u32,
    pub cols: u32,
    pub plane: u32,
    pub diag_offset: i32,
    pub _pad0: u32,
    pub _pad1: u32,
}
