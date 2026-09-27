//! Provider-owned preparation facts, distinct from numerical feasibility.
use runmat_execution::{CandidatePreparationState, Digest};
use serde::{Deserialize, Serialize};

/// Immutable launch selection owned by a provider instance, never an ambient
/// process setting that another device can replace during an invocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProviderElementwiseGeometry {
    pub workgroup_size: u32,
}

#[derive(Clone, Copy, Debug)]
pub enum ProviderPreparationProgram<'a> {
    Elementwise { shader: &'a str },
}

/// A point-in-time cache observation. None means unavailable/contended, not
/// warm. The identity includes the exact selected recipes and geometry.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProviderPreparationSnapshot {
    pub identity: Digest,
    pub state: Option<CandidatePreparationState>,
    pub cold_pipelines: u32,
    pub warm_pipelines: u32,
}
