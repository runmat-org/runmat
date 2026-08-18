use runmat_geometry_core::PersistentEntityId;
use runmat_meshing_core::{MeshingCancellationSignal, StableDigest};
use runmat_meshing_size::metric::{MetricFieldRequest, MetricTensor3};
use sha2::{Digest, Sha256};

use super::{
    validate_delaunay_volume_quality, DelaunayTetrahedronQuality, DelaunayVolumeNode,
    DelaunayVolumeProvenance, DelaunayVolumeQuality, DelaunayVolumeQualityError,
    DelaunayVolumeQualityErrorKind, DelaunayVolumeQualityOptions, DelaunayVolumeTopology,
};

mod candidate;
mod insertion;
mod iteration;
mod validation;
mod work;

use candidate::construct_candidate;
pub use insertion::{
    insert_delaunay_volume_refinement_candidate, validate_delaunay_volume_refinement_step,
};
pub use iteration::{refine_delaunay_volume, validate_delaunay_volume_refinement};
pub use validation::validate_delaunay_volume_refinement_candidate;
use work::CandidateWork;

const CANDIDATE_IDENTITY_DOMAIN: &[u8] = b"runmat/meshing/cdt/volume-refinement-candidate/1\0";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DelaunayRefinementCandidateKind {
    MetricCircumcenter,
    InteriorCentroid,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DelaunayVolumeRefinementCandidate {
    pub node: DelaunayVolumeNode,
    pub kind: DelaunayRefinementCandidateKind,
    pub source_node_identities: [StableDigest; 4],
    pub region_id: PersistentEntityId,
    pub incident_metric_entity_ids: Vec<PersistentEntityId>,
    pub resolved_metric: MetricTensor3,
    pub source_violation_ratio: f64,
}

#[derive(Clone, Copy)]
pub struct DelaunayVolumeRefinementInput<'a> {
    pub topology: &'a DelaunayVolumeTopology,
    pub metric_request: &'a MetricFieldRequest,
    pub provenance: &'a DelaunayVolumeProvenance,
    pub quality: &'a DelaunayVolumeQuality,
    pub quality_options: DelaunayVolumeQualityOptions,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DelaunayVolumeRefinementCandidateOptions {
    pub maximum_candidate_evaluations: u64,
    pub cancellation_check_interval: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DelaunayVolumeRefinementStepOptions {
    pub candidate: DelaunayVolumeRefinementCandidateOptions,
    pub insertion: super::DelaunayInsertionOptions,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DelaunayVolumeRefinementStep {
    pub topology: DelaunayVolumeTopology,
    pub quality: DelaunayVolumeQuality,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DelaunayVolumeRefinementStepErrorKind {
    InvalidOptions,
    InvalidInput,
    InvalidCandidate,
    InvalidTopology,
    InvalidProvenance,
    InvalidQuality,
    ResourceLimit,
    Cancelled,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DelaunayVolumeRefinementStepError {
    pub kind: DelaunayVolumeRefinementStepErrorKind,
    pub reason: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DelaunayVolumeRefinementOptions {
    pub step: DelaunayVolumeRefinementStepOptions,
    pub maximum_insertions: u64,
}

impl Default for DelaunayVolumeRefinementOptions {
    fn default() -> Self {
        Self {
            step: DelaunayVolumeRefinementStepOptions::default(),
            maximum_insertions: 10_000_000,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DelaunayVolumeRefinement {
    pub topology: DelaunayVolumeTopology,
    pub quality: DelaunayVolumeQuality,
    /// Canonically sorted inventory of nodes inserted by this refinement run.
    pub inserted_node_identities: Vec<StableDigest>,
}

impl std::fmt::Display for DelaunayVolumeRefinementStepError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "3D Delaunay volume refinement step {:?}: {}",
            self.kind, self.reason
        )
    }
}

impl std::error::Error for DelaunayVolumeRefinementStepError {}

impl Default for DelaunayVolumeRefinementCandidateOptions {
    fn default() -> Self {
        Self {
            maximum_candidate_evaluations: 2,
            cancellation_check_interval: 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DelaunayVolumeRefinementCandidateErrorKind {
    InvalidOptions,
    InvalidTopology,
    InvalidQuality,
    InvalidCandidate,
    NumericalFailure,
    ResourceLimit,
    Cancelled,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DelaunayVolumeRefinementCandidateError {
    pub kind: DelaunayVolumeRefinementCandidateErrorKind,
    pub reason: String,
}

impl std::fmt::Display for DelaunayVolumeRefinementCandidateError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "3D Delaunay refinement candidate {:?}: {}",
            self.kind, self.reason
        )
    }
}

impl std::error::Error for DelaunayVolumeRefinementCandidateError {}

pub fn select_delaunay_volume_refinement_candidate(
    input: DelaunayVolumeRefinementInput<'_>,
    options: DelaunayVolumeRefinementCandidateOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Option<DelaunayVolumeRefinementCandidate>, DelaunayVolumeRefinementCandidateError> {
    validate_options(options)?;
    validate_delaunay_volume_quality(
        input.topology,
        input.metric_request,
        input.provenance,
        input.quality,
        input.quality_options,
        cancellation,
    )
    .map_err(quality_error)?;
    let Some(source_identity) = input.quality.worst_refinement_tetrahedron else {
        return Ok(None);
    };
    let source_index = input
        .quality
        .tetrahedra
        .iter()
        .position(|tetrahedron| tetrahedron.node_identities == source_identity)
        .ok_or_else(|| {
            error(
                DelaunayVolumeRefinementCandidateErrorKind::InvalidQuality,
                "worst refinement identity is absent from quality evidence",
            )
        })?;
    let mut work = CandidateWork::new(options, cancellation);
    let candidate = construct_candidate(
        input.topology,
        &input.quality.tetrahedra[source_index],
        &mut work,
    )?;
    let selected = Some(candidate);
    validate_delaunay_volume_refinement_candidate(input, &selected, options, cancellation)?;
    Ok(selected)
}

/// Coordinates are deterministic derived evidence but not identity input: excluding their final
/// floating-point rounding keeps the logical node stable across supported IEEE-754 hosts. The
/// domain version must change if the construction algorithm changes.
fn candidate_identity(
    source_node_identities: [StableDigest; 4],
    kind: DelaunayRefinementCandidateKind,
    metric: MetricTensor3,
) -> StableDigest {
    let mut source_node_identities = source_node_identities;
    source_node_identities.sort_unstable();
    let mut hasher = Sha256::new();
    hasher.update(CANDIDATE_IDENTITY_DOMAIN);
    for identity in source_node_identities {
        hasher.update(identity.bytes());
    }
    hasher.update([match kind {
        DelaunayRefinementCandidateKind::MetricCircumcenter => 1,
        DelaunayRefinementCandidateKind::InteriorCentroid => 2,
    }]);
    for value in [
        metric.xx, metric.yy, metric.zz, metric.xy, metric.xz, metric.yz,
    ] {
        hasher.update(value.to_bits().to_be_bytes());
    }
    StableDigest::from_bytes(hasher.finalize().into())
}

fn validate_options(
    options: DelaunayVolumeRefinementCandidateOptions,
) -> Result<(), DelaunayVolumeRefinementCandidateError> {
    if options.maximum_candidate_evaluations == 0 || options.cancellation_check_interval == 0 {
        return Err(error(
            DelaunayVolumeRefinementCandidateErrorKind::InvalidOptions,
            "candidate evaluation limit and cancellation interval must be nonzero",
        ));
    }
    Ok(())
}

fn quality_error(failure: DelaunayVolumeQualityError) -> DelaunayVolumeRefinementCandidateError {
    let kind = match failure.kind {
        DelaunayVolumeQualityErrorKind::InvalidOptions => {
            DelaunayVolumeRefinementCandidateErrorKind::InvalidOptions
        }
        DelaunayVolumeQualityErrorKind::InvalidTopology => {
            DelaunayVolumeRefinementCandidateErrorKind::InvalidTopology
        }
        DelaunayVolumeQualityErrorKind::ResourceLimit => {
            DelaunayVolumeRefinementCandidateErrorKind::ResourceLimit
        }
        DelaunayVolumeQualityErrorKind::Cancelled => {
            DelaunayVolumeRefinementCandidateErrorKind::Cancelled
        }
        DelaunayVolumeQualityErrorKind::InvalidMetric
        | DelaunayVolumeQualityErrorKind::InvalidMetricContext
        | DelaunayVolumeQualityErrorKind::InvalidQuality
        | DelaunayVolumeQualityErrorKind::NumericalFailure => {
            DelaunayVolumeRefinementCandidateErrorKind::InvalidQuality
        }
    };
    error(kind, failure.to_string())
}

fn error(
    kind: DelaunayVolumeRefinementCandidateErrorKind,
    reason: impl Into<String>,
) -> DelaunayVolumeRefinementCandidateError {
    DelaunayVolumeRefinementCandidateError {
        kind,
        reason: reason.into(),
    }
}

#[cfg(test)]
#[path = "volume_refinement/tests.rs"]
mod tests;
