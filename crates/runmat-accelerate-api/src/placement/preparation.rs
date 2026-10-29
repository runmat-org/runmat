//! Provider-owned preparation facts, distinct from numerical feasibility.
use crate::{
    ProviderLayout, ProviderRepresentation, ProviderResidency, ProviderUploadPhysicalEvidence,
};
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
    PowerPair {
        recipe: &'a crate::power_service::PowerPairCompound,
    },
    PowerScalar {
        projection: &'a crate::power_service::PowerTypedScalarProjection,
    },
    RealUnary {
        recipe: &'a crate::ProviderRealUnaryCompound,
    },
    Elementwise {
        shader: &'a str,
    },
    /// Exact stage sources and selected physical strategy. Cache readiness
    /// is observed from the pipeline keys used by submission.
    TypedReduction {
        selection: &'a crate::reduction::ReductionSelection<'a>,
        host_metadata: Option<&'a dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
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
    ClassPreservingUpload { identity: Digest },
}

/// Pure exact class-preserving input upload. This is a placement fact, not a
/// future handle or execution permit. The consuming owner must recheck the
/// same provider evidence and source before invoking the raw upload hook.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProviderClassPreservingUpload {
    source: ProviderRepresentation,
    prepared: ProviderRepresentation,
    physical: ProviderUploadPhysicalEvidence,
    identity: Digest,
}

impl ProviderClassPreservingUpload {
    pub fn new_checked(
        device_id: u32,
        source: ProviderRepresentation,
        prepared: ProviderRepresentation,
        physical: ProviderUploadPhysicalEvidence,
    ) -> Option<Self> {
        if source.residency != ProviderResidency::Host
            || prepared.residency != ProviderResidency::Device
            || source.element_type != prepared.element_type
            || source.storage != prepared.storage
            || source.layout != ProviderLayout::ColumnMajorContiguous
            || prepared.layout != ProviderLayout::ColumnMajorContiguous
            || source.shape != prepared.shape
            || prepared.checked_byte_len()?.max(4) > physical.output_requested_bytes
            || physical.completion_queue_slots == 0
        {
            return None;
        }
        let identity = Digest::sha256_serialized(&(
            "class-preserving-input-upload.v1",
            device_id,
            &source,
            &prepared,
            physical,
        ))
        .ok()?;
        Some(Self {
            source,
            prepared,
            physical,
            identity,
        })
    }

    pub fn source(&self) -> &ProviderRepresentation {
        &self.source
    }

    pub fn prepared(&self) -> &ProviderRepresentation {
        &self.prepared
    }

    pub fn physical(&self) -> ProviderUploadPhysicalEvidence {
        self.physical
    }

    pub fn route(&self) -> ProviderPreparationRoute {
        ProviderPreparationRoute::ClassPreservingUpload {
            identity: self.identity,
        }
    }
}

impl ProviderPreparationProgram<'_> {
    pub fn route(self) -> Option<ProviderPreparationRoute> {
        match self {
            Self::RealUnary { .. } | Self::PowerPair { .. } | Self::PowerScalar { .. } => None,
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
