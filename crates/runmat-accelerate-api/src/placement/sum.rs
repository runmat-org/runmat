//! Resolved numerical sum controls supplied by the owning language runtime.
//! Provider implementations derive their physical recipe from these controls
//! and the input representation; this is not an option parser or scheduler.
use runmat_types::{NumericClass, NumericDomain, ReductionNaN, SumInvocation, SumOutputClass};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderSumWorkload {
    #[serde(default, skip_serializing_if = "SumInvocation::is_sum")]
    pub invocation: SumInvocation,
    /// Zero-based in-bounds axes in execution order. An empty list is a
    /// publication-only conversion, not a reduction dispatch.
    pub axes: Vec<u64>,
    pub missing: ReductionNaN,
    pub output: SumOutputClass,
    pub input_class: NumericClass,
    pub input_domain: NumericDomain,
    pub accumulation_class: NumericClass,
    pub publication_class: NumericClass,
}

/// Physical limits used by the provider's dimensional sum kernel. Absence
/// means its launch recipe is not available for automatic cost admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProviderSumGeometry {
    pub max_buffer_bytes: u64,
    pub max_binding_bytes: u64,
    /// Maximum output elements per N-D shader dispatch, from the selected
    /// workgroup geometry and the adapter's dispatch limit.
    pub nd_chunk_elements: u64,
    /// Elements per floating conversion dispatch for this provider instance.
    pub cast_chunk_elements: u64,
    /// Elements per integer-word to F64 sum conversion dispatch. This uses
    /// the integer shader's workgroup, which may differ from floating casts.
    pub integer_cast_chunk_elements: u64,
    /// Maximum output elements admitted by one 2-D dimensional reduction
    /// dispatch, using its fixed reduction workgroup and the adapter limit.
    pub dimension_2_max_output_elements: u64,
    /// Input elements consumed by one global reduction dispatch.
    pub global_chunk_elements: u64,
    /// Input elements reduced to one partial result in the global kernel.
    pub global_group_elements: u64,
}
