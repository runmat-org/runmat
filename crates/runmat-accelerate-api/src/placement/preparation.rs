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
    Elementwise {
        shader: &'a str,
    },
    /// Exact stage sources and selected physical strategy. Cache readiness
    /// is observed from the pipeline keys used by submission.
    TypedReduction {
        selection: &'a crate::reduction::ReductionSelection<'a>,
    },
    /// Provider-owned cast recipe; controls/adaptation are not part of this route.
    FloatingCast {
        target: crate::ProviderPrecision,
    },
    /// Ordinary sum after the runtime has resolved its typed controls.
    /// The associated workload carries the axes, classes and missing policy.
    Sum,
}

/// Resolved physical route, distinct from an operation's numerical capability.
/// A direct hook caller may declare this only when no output-template adaptation
/// follows the hook. Missing evidence does not mean an identity/zero-work route.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ProviderPreparationRoute {
    FloatingCast { target: crate::ProviderPrecision },
    TypedReductionInputs { identity: Digest },
}

impl ProviderPreparationProgram<'_> {
    pub fn route(self) -> Option<ProviderPreparationRoute> {
        match self {
            Self::FloatingCast { target } => {
                Some(ProviderPreparationRoute::FloatingCast { target })
            }
            Self::Sum => None,
            // Elementwise already carries its complete expression workload.
            Self::Elementwise { .. } => None,
            Self::TypedReduction { .. } => None,
        }
    }
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
