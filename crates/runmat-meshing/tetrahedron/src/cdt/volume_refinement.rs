use runmat_geometry_core::PersistentEntityId;
use runmat_meshing_core::{MeshingCancellationSignal, StableDigest};
use runmat_meshing_size::metric::{MetricFieldRequest, MetricTensor3};
use sha2::{Digest, Sha256};

use super::{
    validate_delaunay_volume_quality, DelaunayTetrahedronQuality, DelaunayVolumeNode,
    DelaunayVolumeProvenance, DelaunayVolumeQuality, DelaunayVolumeQualityError,
    DelaunayVolumeQualityErrorKind, DelaunayVolumeQualityOptions, DelaunayVolumeTopology,
};

mod admission;
mod batch;
mod candidate;
pub(in crate::cdt) mod context;
mod insertion;
pub(in crate::cdt) mod iteration;
mod metric_circumcenter;
mod validation;
mod work;
pub(in crate::cdt) mod working;

use candidate::{construct_candidate, CandidateQueries};
pub use insertion::{
    insert_delaunay_volume_refinement_candidate, validate_delaunay_volume_refinement_step,
};
pub use iteration::{refine_delaunay_volume, validate_delaunay_volume_refinement};
pub use validation::validate_delaunay_volume_refinement_candidate;
use validation::validate_delaunay_volume_refinement_candidate_from_admitted_input;
use work::CandidateWork;

const CANDIDATE_IDENTITY_DOMAIN: &[u8] = b"runmat/meshing/cdt/volume-refinement-candidate/4\0";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DelaunayRefinementCandidateKind {
    MetricCircumcenter,
    MetricOffcenter,
    InteriorPoint,
    /// Near-midpoint of the longest size-violating edge with an interior endpoint.
    MetricEdgeMidpoint,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DelaunayVolumeRefinementCandidate {
    pub node: DelaunayVolumeNode,
    pub kind: DelaunayRefinementCandidateKind,
    /// Oriented node identities of the tetrahedron from which the point was
    /// constructed. This is the quality source for ordinary candidates and a
    /// same-region neighbor for the numerical-sliver fallback.
    pub construction_node_identities: [StableDigest; 4],
    /// Index within the construction tetrahedron's canonical point family.
    pub construction_point_index: u16,
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
    /// Sorted identities of construction nodes that refinement must retain unchanged.
    pub fixed_node_identities: &'a [StableDigest],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DelaunayVolumeRefinementCandidateOptions {
    pub maximum_candidate_evaluations: u64,
    pub cancellation_check_interval: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
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
    Accounting(runmat_simulation_usage::LedgerError),
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
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
    /// Deterministic insertion order used to validate exact node lineage.
    pub insertions: Vec<DelaunayVolumeNode>,
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
            maximum_candidate_evaluations: 1_000_000,
            cancellation_check_interval: 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DelaunayVolumeRefinementCandidateErrorKind {
    Accounting(runmat_simulation_usage::LedgerError),
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
    select_delaunay_volume_refinement_candidate_excluding(
        input,
        options,
        &std::collections::BTreeSet::new(),
        cancellation,
    )
}

fn select_delaunay_volume_refinement_candidate_excluding(
    input: DelaunayVolumeRefinementInput<'_>,
    options: DelaunayVolumeRefinementCandidateOptions,
    excluded_node_identities: &std::collections::BTreeSet<StableDigest>,
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
    if input.quality.worst_refinement_tetrahedron.is_none() {
        return Ok(None);
    }
    let queries = CandidateQueries::new(input.topology, cancellation)?;
    select_delaunay_volume_refinement_candidate_excluding_from_admitted_input(
        input,
        options,
        excluded_node_identities,
        cancellation,
        &queries,
    )
}

fn select_delaunay_volume_refinement_candidate_excluding_from_admitted_input(
    input: DelaunayVolumeRefinementInput<'_>,
    options: DelaunayVolumeRefinementCandidateOptions,
    excluded_node_identities: &std::collections::BTreeSet<StableDigest>,
    cancellation: &dyn MeshingCancellationSignal,
    queries: &CandidateQueries,
) -> Result<Option<DelaunayVolumeRefinementCandidate>, DelaunayVolumeRefinementCandidateError> {
    if input.quality.worst_refinement_tetrahedron.is_none() {
        return Ok(None);
    }
    let mut sources = input
        .quality
        .tetrahedra
        .iter()
        .filter(|quality| quality.requires_refinement())
        .collect::<Vec<_>>();
    sources.sort_by(|left, right| {
        super::volume_quality::compare_violation_priority(
            right.refinement_violation_ratio,
            super::volume_quality::tetrahedron_geometry_key(input.topology, right.node_identities),
            right.node_identities,
            left.refinement_violation_ratio,
            super::volume_quality::tetrahedron_geometry_key(input.topology, left.node_identities),
            left.node_identities,
        )
    });
    let mut work = CandidateWork::new(options, cancellation);
    let mut selected = None;
    let mut unavailable_source_count = 0usize;
    let mut unavailable_reasons = Vec::new();
    for source in sources {
        match construct_candidate(
            input.topology,
            queries,
            source,
            input.quality_options,
            input.fixed_node_identities,
            excluded_node_identities,
            &mut work,
        ) {
            Ok(candidate) => {
                selected = Some(candidate);
                break;
            }
            Err(failure)
                if matches!(
                    failure.kind,
                    DelaunayVolumeRefinementCandidateErrorKind::NumericalFailure
                        | DelaunayVolumeRefinementCandidateErrorKind::InvalidCandidate
                ) =>
            {
                unavailable_source_count += 1;
                if unavailable_reasons.len() < 4 {
                    unavailable_reasons.push(failure.reason);
                }
            }
            Err(failure) => return Err(failure),
        }
    }
    if selected.is_none() && unavailable_source_count != 0 {
        return Err(error(
            DelaunayVolumeRefinementCandidateErrorKind::NumericalFailure,
            format!(
                "all {unavailable_source_count} violating tetrahedra exhausted their admissible refinement constructions; representative failures: {}",
                unavailable_reasons.join(" | ")
            ),
        ));
    }
    validate_delaunay_volume_refinement_candidate_from_admitted_input(
        input,
        &selected,
        options,
        cancellation,
    )?;
    Ok(selected)
}

fn candidate_identity(
    source_node_identities: [StableDigest; 4],
    kind: DelaunayRefinementCandidateKind,
    construction_point_index: u16,
) -> StableDigest {
    let mut source_node_identities = source_node_identities;
    source_node_identities.sort_unstable();
    let mut hasher = Sha256::new();
    hasher.update(CANDIDATE_IDENTITY_DOMAIN);
    for identity in source_node_identities {
        hasher.update(identity.bytes());
    }
    hasher.update([match kind {
        DelaunayRefinementCandidateKind::MetricCircumcenter => 0,
        DelaunayRefinementCandidateKind::MetricOffcenter => 1,
        DelaunayRefinementCandidateKind::InteriorPoint => 2,
        DelaunayRefinementCandidateKind::MetricEdgeMidpoint => 3,
    }]);
    hasher.update(construction_point_index.to_be_bytes());
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
        DelaunayVolumeQualityErrorKind::Accounting(cause) => {
            DelaunayVolumeRefinementCandidateErrorKind::Accounting(cause)
        }
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
