use crate::{GpuTensorLayout, ProviderPrecision};
use serde::Serialize;

/// Side-effect-free negative admission for the double-accumulating sum hook.
/// `Unknown` retains the hook's existing effect-aware execution contract;
/// it is not a claim that the reduction will succeed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderDoubleSumPreflight {
    Unknown,
    Unsupported,
}

/// Frozen dimensions/limits supplied by the provider owner, not ambient knobs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ReductionRecipeConfig {
    pub workgroup_size: u32,
    pub chunk_elements: u32,
    pub max_workgroup_size: u32,
    pub max_workgroup_storage_bytes: u64,
    pub max_dispatch_dimension: u32,
    pub max_storage_binding_bytes: u64,
    pub max_rank: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ReductionPartialLayout {
    pub precision: ProviderPrecision,
    pub alignment: u32,
    pub stride: u32,
    pub count_offset: u32,
    pub flags_offset: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ReductionDispatch {
    pub workgroups: [u32; 3],
    pub logical_groups: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ReductionGeometry {
    pub reduce_elements: u32,
    pub slices: u32,
    pub chunks: u32,
    pub partial_elements: u32,
    pub single: ReductionDispatch,
    pub pass1: ReductionDispatch,
    pub pass2: ReductionDispatch,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ReductionRecipeWork {
    /// Operation-owned producer prior, not measured time or a FLOP count.
    pub producer_scalar_work_units: u64,
    pub output_payload_bytes: u64,
    pub output_requested_bytes: u64,
    pub partial_payload_bytes: u64,
    pub partial_requested_bytes: u64,
    pub workgroup_bytes: u64,
    pub single_dispatches: u32,
    pub two_pass_dispatches: u32,
    pub uniform_bytes: u64,
}

/// Prepared binding after upstream upload/cast/materialization. No graph ID,
/// host value or original source allocation is smuggled into this ABI.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ReductionInputBinding {
    pub precision: ProviderPrecision,
    pub shape: Vec<usize>,
    pub layout: GpuTensorLayout,
    /// Existing producer-layout authority supplies these, in iteration order.
    pub broadcast_shape: Vec<u32>,
    pub strides: Vec<u32>,
    pub payload_bytes: u64,
}

pub(super) fn width(precision: ProviderPrecision) -> u64 {
    match precision {
        ProviderPrecision::F32 => 4,
        ProviderPrecision::F64 => 8,
    }
}
