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


