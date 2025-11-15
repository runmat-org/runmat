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
    pub _pad_tail: f64,
    pub _pad_tail2: f64,
    pub _pad_tail3: f64,
    pub _pad_tail4: f64,
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
    pub _pad_tail: [f32; 4],
    pub _pad_tail2: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Conv1dParams {
    pub signal_len: u32,
    pub kernel_len: u32,
    pub output_len: u32,
    pub start_offset: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct FusionParams {
    pub len: u32,
    pub offset: u32,
    pub _pad1: u32,
    pub _pad2: u32,
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

pub const PERMUTE_MAX_RANK: usize = 128;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub struct AlignedU32 {
    pub value: u32,
    pub _pad: [u32; 3],
}

impl AlignedU32 {
    pub const fn new(value: u32) -> Self {
        Self {
            value,
            _pad: [0; 3],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct PermuteParams {
    pub len: u32,
    pub offset: u32,
    pub rank: u32,
    pub _pad: u32,
    pub src_shape: [AlignedU32; PERMUTE_MAX_RANK],
    pub dst_shape: [AlignedU32; PERMUTE_MAX_RANK],
    pub order: [AlignedU32; PERMUTE_MAX_RANK],
    pub src_strides: [AlignedU32; PERMUTE_MAX_RANK],
}

pub const FLIP_MAX_RANK: usize = PERMUTE_MAX_RANK;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct FlipParams {
    pub len: u32,
    pub offset: u32,
    pub rank: u32,
    pub _pad: u32,
    pub shape: [AlignedU32; FLIP_MAX_RANK],
    pub strides: [AlignedU32; FLIP_MAX_RANK],
    pub flags: [AlignedU32; FLIP_MAX_RANK],
}

pub const CIRCSHIFT_MAX_RANK: usize = PERMUTE_MAX_RANK;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct CircshiftParams {
    pub len: u32,
    pub offset: u32,
    pub rank: u32,
    pub _pad: u32,
    pub shape: [AlignedU32; CIRCSHIFT_MAX_RANK],
    pub strides: [AlignedU32; CIRCSHIFT_MAX_RANK],
    pub shifts: [AlignedU32; CIRCSHIFT_MAX_RANK],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct DiffParams {
    pub stride_before: u32,
    pub segments: u32,
    pub segment_len: u32,
    pub segment_out: u32,
    pub block: u32,
    pub total_out: u32,
    pub total_in: u32,
    pub _pad: u32,
}

pub const FILTER_MAX_RANK: usize = PERMUTE_MAX_RANK;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct FilterParams {
    pub dim_len: u32,
    pub leading: u32,
    pub trailing: u32,
    pub order: u32,
    pub state_len: u32,
    pub signal_len: u32,
    pub channel_count: u32,
    pub zi_present: u32,
    pub dim_idx: u32,
    pub rank: u32,
    pub state_rank: u32,
    pub _pad: u32,
    pub signal_shape: [AlignedU32; FILTER_MAX_RANK],
    pub state_shape: [AlignedU32; FILTER_MAX_RANK],
}

pub const IMFILTER_MAX_RANK: usize = 512;

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
    pub image_shape: [AlignedU32; IMFILTER_MAX_RANK],
    pub image_strides: [AlignedU32; IMFILTER_MAX_RANK],
    pub output_shape: [AlignedU32; IMFILTER_MAX_RANK],
    pub base_offset: [PackedI32; IMFILTER_MAX_RANK],
    pub _pad_tail: AlignedU32,
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
    pub image_shape: [AlignedU32; IMFILTER_MAX_RANK],
    pub image_strides: [AlignedU32; IMFILTER_MAX_RANK],
    pub output_shape: [AlignedU32; IMFILTER_MAX_RANK],
    pub base_offset: [PackedI32; IMFILTER_MAX_RANK],
    pub _pad_tail: AlignedU32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct PolyvalParamsF64 {
    pub len: u32,
    pub coeff_len: u32,
    pub offset: u32,
    pub has_mu: u32,
    pub mu_mean: f64,
    pub mu_scale: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct PolyvalParamsF32 {
    pub len: u32,
    pub coeff_len: u32,
    pub offset: u32,
    pub has_mu: u32,
    pub mu_mean: f32,
    pub mu_scale: f32,
    pub _pad0: u32,
    pub _pad1: u32,
}

pub const REPMAT_MAX_RANK: usize = PERMUTE_MAX_RANK;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct RepmatParams {
    pub len: u32,
    pub offset: u32,
    pub rank: u32,
    pub _pad: u32,
    pub base_shape: [AlignedU32; REPMAT_MAX_RANK],
    pub new_shape: [AlignedU32; REPMAT_MAX_RANK],
    pub base_strides: [AlignedU32; REPMAT_MAX_RANK],
}

pub const BCAST_MAX_RANK: usize = PERMUTE_MAX_RANK;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BinaryBroadcastParams {
    pub len: u32,
    pub offset: u32,
    pub rank: u32,
    pub op: u32,
    pub out_shape: [AlignedU32; BCAST_MAX_RANK],
    pub a_shape: [AlignedU32; BCAST_MAX_RANK],
    pub b_shape: [AlignedU32; BCAST_MAX_RANK],
    pub a_strides: [AlignedU32; BCAST_MAX_RANK],
    pub b_strides: [AlignedU32; BCAST_MAX_RANK],
}

pub const KRON_MAX_RANK: usize = PERMUTE_MAX_RANK;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct KronParams {
    pub len: u32,
    pub offset: u32,
    pub rank: u32,
    pub _pad: u32,
    pub shape_a: [AlignedU32; KRON_MAX_RANK],
    pub shape_b: [AlignedU32; KRON_MAX_RANK],
    pub shape_out: [AlignedU32; KRON_MAX_RANK],
    pub stride_a: [AlignedU32; KRON_MAX_RANK],
    pub stride_b: [AlignedU32; KRON_MAX_RANK],
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
pub struct BandwidthParams {
    pub rows: u32,
    pub cols: u32,
    pub len: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SymmetryParamsF64 {
    pub rows: u32,
    pub cols: u32,
    pub len: u32,
    pub mode: u32,
    pub tolerance: f64,
    pub _pad: f64,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SymmetryParamsF32 {
    pub rows: u32,
    pub cols: u32,
    pub len: u32,
    pub mode: u32,
    pub tolerance: f32,
    pub _pad: [f32; 3],
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
    pub offset_a: u32,
    pub offset_b: u32,
    pub offset_out: u32,
    pub flags: u32,
}

pub const MATMUL_FLAG_TRANSPOSE_A: u32 = 1 << 0;
pub const MATMUL_FLAG_TRANSPOSE_B: u32 = 1 << 1;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct MatmulEpilogueParamsF64 {
    pub alpha: f64,
    pub beta: f64,
    pub clamp_min: f64,
    pub clamp_max: f64,
    pub pow_exponent: f64,
    pub flags: u32,
    pub diag_offset: u32,
    pub diag_stride: u32,
    pub diag_rows: u32,
    pub _pad: u32,
    pub _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct MatmulEpilogueParamsF32 {
    pub alpha: f32,
    pub beta: f32,
    pub clamp_min: f32,
    pub clamp_max: f32,
    pub pow_exponent: f32,
    pub flags: u32,
    pub diag_offset: u32,
    pub diag_stride: u32,
    pub diag_rows: u32,
    pub _pad: u32,
}

pub const MATMUL_EPILOGUE_FLAG_ROW_SCALE: u32 = 1 << 0;
pub const MATMUL_EPILOGUE_FLAG_COL_SCALE: u32 = 1 << 1;
pub const MATMUL_EPILOGUE_FLAG_ROW_DIV: u32 = 1 << 2;
pub const MATMUL_EPILOGUE_FLAG_COL_DIV: u32 = 1 << 3;
pub const MATMUL_EPILOGUE_FLAG_CLAMP_MIN: u32 = 1 << 4;
pub const MATMUL_EPILOGUE_FLAG_CLAMP_MAX: u32 = 1 << 5;
pub const MATMUL_EPILOGUE_FLAG_POW: u32 = 1 << 6;
pub const MATMUL_EPILOGUE_FLAG_DIAG_WRITE: u32 = 1 << 7;

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct CenteredGramParamsF64 {
    pub rows: u32,
    pub cols: u32,
    pub lda: u32,
    pub ldc: u32,
    pub offset_matrix: u32,
    pub offset_means: u32,
    pub offset_out: u32,
    pub _pad0: u32,
    pub denom: [f64; 2],
    pub _pad1: [f64; 2],
    pub _pad2: [f64; 2],
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct CenteredGramParamsF32 {
    pub rows: u32,
    pub cols: u32,
    pub lda: u32,
    pub ldc: u32,
    pub offset_matrix: u32,
    pub offset_means: u32,
    pub offset_out: u32,
    pub _pad0: u32,
    pub denom: [f32; 4],
    pub _pad1: [f32; 4],
    pub _pad2: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ImageNormalizeUniforms {
    pub batches: u32,
    pub height: u32,
    pub width: u32,
    pub plane: u32,
    pub stride_h: u32,
    pub stride_w: u32,
    pub flags: u32,
    pub _pad0: u32,
    pub epsilon: f32,
    pub gain: f32,
    pub bias: f32,
    pub gamma: f32,
}

pub const IMAGE_NORMALIZE_FLAG_GAIN: u32 = 1 << 0;
pub const IMAGE_NORMALIZE_FLAG_BIAS: u32 = 1 << 1;
pub const IMAGE_NORMALIZE_FLAG_GAMMA: u32 = 1 << 2;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SyrkParams {
    pub rows_total: u32,
    pub cols: u32,
    pub lda: u32,
    pub ldc: u32,
    pub row_offset: u32,
    pub chunk_rows: u32,
    pub flags: u32,
    pub offset_out: u32,
}

pub const SYRK_FLAG_ACCUMULATE: u32 = 1 << 0;
pub const SYRK_FLAG_FILL_BOTH: u32 = 1 << 1;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ReduceGlobalParams {
    pub len: u32,
    pub op: u32,
    pub offset: u32,
    pub total: u32,
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
pub struct CumsumParams {
    pub segment_len: u32,
    pub segments: u32,
    pub stride_before: u32,
    pub block: u32,
    pub flags: u32,
    pub total_len: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct CumprodParams {
    pub segment_len: u32,
    pub segments: u32,
    pub stride_before: u32,
    pub block: u32,
    pub flags: u32,
    pub total_len: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct CumminParams {
    pub segment_len: u32,
    pub segments: u32,
    pub stride_before: u32,
    pub block: u32,
    pub flags: u32,
    pub total_len: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

pub type CummaxParams = CumminParams;

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
    pub key0: u32,
    pub key1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct StochasticEvolutionParamsF32 {
    pub offset: u32,
    pub chunk: u32,
    pub len: u32,
    pub steps: u32,
    pub key0: u32,
    pub key1: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub drift: f32,
    pub scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct StochasticEvolutionParamsF64 {
    pub offset: u32,
    pub chunk: u32,
    pub len: u32,
    pub steps: u32,
    pub key0: u32,
    pub key1: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub drift: f64,
    pub scale: f64,
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

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ReduceNdParams {
    pub rank: u32,
    pub kept_count: u32,
    pub reduce_count: u32,
    pub _pad: u32,
    pub rows: u32,
    pub cols: u32,
    pub _pad2: [u32; 2],
    pub kept_sizes: [AlignedU32; BCAST_MAX_RANK],
    pub reduce_sizes: [AlignedU32; BCAST_MAX_RANK],
    pub kept_strides: [AlignedU32; BCAST_MAX_RANK],
    pub reduce_strides: [AlignedU32; BCAST_MAX_RANK],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct QrPowerIterParams {
    pub cols: u32,
    pub stride: u32,
    pub _pad0: [u32; 2],
}
