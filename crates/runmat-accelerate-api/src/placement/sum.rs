//! Resolved numerical sum controls supplied by the owning language runtime.
//! Provider implementations derive their physical recipe from these controls
//! and the input representation; this is not an option parser or scheduler.
use runmat_types::{NumericClass, NumericDomain, ReductionNaN, SumOutputClass};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderSumWorkload {
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
}
