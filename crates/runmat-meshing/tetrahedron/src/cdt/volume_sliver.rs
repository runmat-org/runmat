use runmat_meshing_core::{MeshingCancellationSignal, StableDigest};
use serde::{Deserialize, Serialize};

use super::{
    validate_delaunay_volume_provenance, validate_delaunay_volume_quality,
    DelaunayInsertionErrorKind, DelaunayInsertionOptions, DelaunayTetrahedronQuality,
    DelaunayVolumeNode, DelaunayVolumeProvenanceErrorKind, DelaunayVolumeQuality,
    DelaunayVolumeQualityError, DelaunayVolumeQualityErrorKind, DelaunayVolumeRefinementInput,
    DelaunayVolumeTopology,
};

mod admission;
mod frontier;
mod insertion;
mod offcenter;
pub(in crate::cdt) mod reconnection;
#[cfg(test)]
mod reference;
mod relocation;
#[cfg(test)]
mod retained_state;
mod retriangulate;
mod validation;
pub(in crate::cdt) mod work;
mod working;
pub(in crate::cdt) use working::treatment::run_in_state;

pub(super) const CAVITY_SMOOTHING_STEPS: [f64; 4] =
    [1.0 / 1_024.0, 1.0 / 128.0, 1.0 / 16.0, 1.0 / 2.0];
pub(super) const MAXIMUM_CAVITY_SMOOTHING_ITERATIONS: usize = 100;
pub(super) const MAXIMUM_CAVITY_SEARCH_DIRECTIONS: usize = 20;
pub(super) const MAXIMUM_INSERTION_LOOKAHEAD_DEPTH: u8 = 1;

#[cfg(test)]
use admission::worst_quality_source;
use admission::QualityObjective;
#[cfg(test)]
use insertion::insertion_candidates;
pub(in crate::cdt) use insertion::insertion_identity_is_valid;
#[cfg(test)]
use reconnection::reconnection_candidates;
#[cfg(test)]
use relocation::{relocation_candidates, RelocationSearchScope};
pub use validation::validate_delaunay_volume_sliver_treatment;
#[cfg(test)]
use work::OptimizationWork;

#[derive(Clone, Copy)]
pub(super) struct SliverSearchPolicy {
    options: DelaunayVolumeSliverOptions,
    refinement: admission::CandidateRefinementPolicy,
    objective: QualityObjective,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DelaunayVolumeSliverOptions {
    pub maximum_passes: u64,
    pub maximum_candidate_evaluations_per_pass: u64,
    pub maximum_compound_reconnections_per_insertion: u64,
    pub cancellation_check_interval: u64,
    pub insertion: DelaunayInsertionOptions,
}

impl Default for DelaunayVolumeSliverOptions {
    fn default() -> Self {
        Self {
            maximum_passes: 1_000_000,
            // One source has at most 16 immediate reconnections, a bounded set of
            // collapse / deletion / relocation choices, and 390 edge-star insertion
            // choices (six edges, each with a midpoint, up to 60 exact face-direction
            // probes, and four smoothing scales). A width-two, one-level insertion
            // lookahead and bounded reconnection beam share this same hard limit.
            maximum_candidate_evaluations_per_pass: 2_048,
            maximum_compound_reconnections_per_insertion: 4,
            cancellation_check_interval: 32,
            insertion: DelaunayInsertionOptions::default(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DelaunayVolumeSliverRelocation {
    pub source_node_identity: StableDigest,
    pub replacement_node: DelaunayVolumeNode,
    pub source_tetrahedron_node_identities: [StableDigest; 4],
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DelaunayVolumeSliverDeletion {
    pub source_node_identity: StableDigest,
    pub source_tetrahedron_node_identities: [StableDigest; 4],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DelaunayVolumeReconnectionKind {
    TwoToThreeFace,
    ThreeToTwoEdge,
    EdgeStar,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DelaunayVolumeSliverReconnection {
    pub kind: DelaunayVolumeReconnectionKind,
    pub source_tetrahedron_node_identities: [StableDigest; 4],
    #[serde(deserialize_with = "runmat_canonical_codec::deserialize_exact_vec")]
    pub removed_tetrahedron_node_identities: Vec<[StableDigest; 4]>,
    #[serde(deserialize_with = "runmat_canonical_codec::deserialize_exact_vec")]
    pub created_tetrahedron_node_identities: Vec<[StableDigest; 4]>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DelaunayVolumeSliverInsertion {
    pub inserted_node: DelaunayVolumeNode,
    pub source_tetrahedron_node_identities: [StableDigest; 4],
    pub source_edge_node_identities: [StableDigest; 2],
    pub construction_point_index: u16,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    content = "mutation",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum DelaunayVolumeSliverMutation {
    Relocated(DelaunayVolumeSliverRelocation),
    Deleted(DelaunayVolumeSliverDeletion),
    Reconnected(DelaunayVolumeSliverReconnection),
    Inserted(DelaunayVolumeSliverInsertion),
}

#[derive(Clone, Debug, PartialEq)]
pub struct DelaunayVolumeSliverTreatment {
    pub topology: DelaunayVolumeTopology,
    pub quality: DelaunayVolumeQuality,
    /// Deterministic order of relocations and redundant interior-Steiner-node removals.
    pub mutations: Vec<DelaunayVolumeSliverMutation>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DelaunayVolumeSliverErrorKind {
    Accounting(runmat_simulation_usage::LedgerError),
    InvalidOptions,
    InvalidInput,
    InvalidTopology,
    InvalidProvenance,
    InvalidQuality,
    NoAdmissibleOptimization,
    ResourceLimit,
    Cancelled,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DelaunayVolumeSliverError {
    pub kind: DelaunayVolumeSliverErrorKind,
    pub reason: String,
}

impl std::fmt::Display for DelaunayVolumeSliverError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "3D Delaunay volume sliver treatment {:?}: {}",
            self.kind, self.reason
        )
    }
}

impl std::error::Error for DelaunayVolumeSliverError {}

pub fn treat_delaunay_volume_slivers(
    input: DelaunayVolumeRefinementInput<'_>,
    options: DelaunayVolumeSliverOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<DelaunayVolumeSliverTreatment, DelaunayVolumeSliverError> {
    validate_options(options)?;
    validate_input(input, options, cancellation)?;
    let treatment = run_treatment(
        input.topology.clone(),
        input.quality.clone(),
        input.into(),
        options,
        cancellation,
        QualityConvergenceMode::ShapeOnly,
    )?;
    validate_delaunay_volume_sliver_treatment(input, &treatment, options, cancellation)?;
    Ok(treatment)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum QualityConvergenceMode {
    ShapeOnly,
    Coupled,
}

impl SliverSearchPolicy {
    fn current_objective(
        self,
        shape: Option<&DelaunayTetrahedronQuality>,
        refinement: Option<&DelaunayTetrahedronQuality>,
    ) -> QualityObjective {
        if matches!(
            self.refinement,
            admission::CandidateRefinementPolicy::AllowForCoupledConvergence
        ) {
            QualityConvergenceMode::Coupled.objective(shape, refinement)
        } else {
            self.objective
        }
    }
}

impl QualityConvergenceMode {
    fn refinement_policy(self) -> admission::CandidateRefinementPolicy {
        match self {
            Self::ShapeOnly => admission::CandidateRefinementPolicy::Preserve,
            Self::Coupled => admission::CandidateRefinementPolicy::AllowForCoupledConvergence,
        }
    }

    fn objective(
        self,
        shape: Option<&DelaunayTetrahedronQuality>,
        refinement: Option<&DelaunayTetrahedronQuality>,
    ) -> QualityObjective {
        if matches!(self, Self::Coupled)
            && refinement.map_or(0., |cell| cell.refinement_violation_ratio)
                > shape.map_or(0., |cell| cell.optimization_violation_ratio)
        {
            QualityObjective::Refinement
        } else {
            QualityObjective::Shape
        }
    }
}

pub(super) fn run_treatment(
    topology: DelaunayVolumeTopology,
    quality: DelaunayVolumeQuality,
    context: super::volume_refinement::context::RefinementContext<'_>,
    options: DelaunayVolumeSliverOptions,
    cancellation: &dyn MeshingCancellationSignal,
    mode: QualityConvergenceMode,
) -> Result<DelaunayVolumeSliverTreatment, DelaunayVolumeSliverError> {
    let prepared = super::volume_quality::working::PreparedLocalQuality::prepare(
        context.metric_request,
        context.provenance,
        context.quality_options,
    )
    .map_err(quality_error)?;
    working::treatment::run(
        topology,
        quality,
        context,
        &prepared,
        options,
        cancellation,
        mode,
    )
}

pub(super) fn validate_options(
    options: DelaunayVolumeSliverOptions,
) -> Result<(), DelaunayVolumeSliverError> {
    if options.maximum_passes == 0
        || options.maximum_candidate_evaluations_per_pass == 0
        || options.maximum_compound_reconnections_per_insertion == 0
        || options.cancellation_check_interval == 0
    {
        return Err(error(
            DelaunayVolumeSliverErrorKind::InvalidOptions,
            "pass, candidate-evaluation, compound-reconnection, and cancellation limits must be nonzero",
        ));
    }
    super::insertion::validate_options(options.insertion).map_err(|failure| {
        error(
            DelaunayVolumeSliverErrorKind::InvalidOptions,
            failure.to_string(),
        )
    })?;
    Ok(())
}

pub(super) fn validate_input(
    input: DelaunayVolumeRefinementInput<'_>,
    options: DelaunayVolumeSliverOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayVolumeSliverError> {
    if input
        .fixed_node_identities
        .windows(2)
        .any(|pair| pair[0] >= pair[1])
        || input.fixed_node_identities.iter().any(|identity| {
            input
                .topology
                .nodes
                .binary_search_by_key(identity, |node| node.identity)
                .is_err()
        })
    {
        return Err(error(
            DelaunayVolumeSliverErrorKind::InvalidInput,
            "fixed refinement-node identities must be sorted, unique, and present in topology",
        ));
    }
    let protected_faces = input
        .provenance
        .facets
        .iter()
        .map(|facet| facet.node_identities)
        .collect::<Vec<_>>();
    super::insertion::validate_conforming_volume_topology(
        input.topology,
        &protected_faces,
        options.insertion,
        cancellation,
    )
    .map_err(|failure| {
        let kind = match failure.kind {
            DelaunayInsertionErrorKind::InvalidOptions => {
                DelaunayVolumeSliverErrorKind::InvalidOptions
            }
            DelaunayInsertionErrorKind::ResourceLimit => {
                DelaunayVolumeSliverErrorKind::ResourceLimit
            }
            DelaunayInsertionErrorKind::Cancelled => DelaunayVolumeSliverErrorKind::Cancelled,
            _ => DelaunayVolumeSliverErrorKind::InvalidTopology,
        };
        error(kind, failure.to_string())
    })?;
    validate_delaunay_volume_provenance(
        input.topology,
        input.provenance,
        input.quality_options.provenance,
        cancellation,
    )
    .map_err(|failure| {
        let kind = match failure.kind {
            DelaunayVolumeProvenanceErrorKind::Accounting(cause) => {
                DelaunayVolumeSliverErrorKind::Accounting(cause)
            }
            DelaunayVolumeProvenanceErrorKind::InvalidOptions => {
                DelaunayVolumeSliverErrorKind::InvalidOptions
            }
            DelaunayVolumeProvenanceErrorKind::InvalidTopology => {
                DelaunayVolumeSliverErrorKind::InvalidTopology
            }
            DelaunayVolumeProvenanceErrorKind::InvalidProvenance => {
                DelaunayVolumeSliverErrorKind::InvalidProvenance
            }
            DelaunayVolumeProvenanceErrorKind::ResourceLimit => {
                DelaunayVolumeSliverErrorKind::ResourceLimit
            }
            DelaunayVolumeProvenanceErrorKind::Cancelled => {
                DelaunayVolumeSliverErrorKind::Cancelled
            }
        };
        error(kind, failure.to_string())
    })?;
    validate_delaunay_volume_quality(
        input.topology,
        input.metric_request,
        input.provenance,
        input.quality,
        input.quality_options,
        cancellation,
    )
    .map_err(quality_error)
}

pub(super) fn checkpoint(
    step: u64,
    options: DelaunayVolumeSliverOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayVolumeSliverError> {
    if step.is_multiple_of(options.cancellation_check_interval) && cancellation.is_cancelled() {
        return Err(error(DelaunayVolumeSliverErrorKind::Cancelled, "cancelled"));
    }
    Ok(())
}

pub(super) fn quality_error(failure: DelaunayVolumeQualityError) -> DelaunayVolumeSliverError {
    let kind = match failure.kind {
        DelaunayVolumeQualityErrorKind::Accounting(cause) => {
            DelaunayVolumeSliverErrorKind::Accounting(cause)
        }
        DelaunayVolumeQualityErrorKind::InvalidOptions => {
            DelaunayVolumeSliverErrorKind::InvalidOptions
        }
        DelaunayVolumeQualityErrorKind::InvalidTopology => {
            DelaunayVolumeSliverErrorKind::InvalidTopology
        }
        DelaunayVolumeQualityErrorKind::InvalidMetric
        | DelaunayVolumeQualityErrorKind::InvalidMetricContext => {
            DelaunayVolumeSliverErrorKind::InvalidProvenance
        }
        DelaunayVolumeQualityErrorKind::InvalidQuality
        | DelaunayVolumeQualityErrorKind::NumericalFailure => {
            DelaunayVolumeSliverErrorKind::InvalidQuality
        }
        DelaunayVolumeQualityErrorKind::ResourceLimit => {
            DelaunayVolumeSliverErrorKind::ResourceLimit
        }
        DelaunayVolumeQualityErrorKind::Cancelled => DelaunayVolumeSliverErrorKind::Cancelled,
    };
    error(kind, failure.to_string())
}

pub(super) fn relocation_identity_is_valid(relocation: &DelaunayVolumeSliverRelocation) -> bool {
    relocation::relocation_identity_is_valid(relocation)
}

pub(super) fn error(
    kind: DelaunayVolumeSliverErrorKind,
    reason: impl Into<String>,
) -> DelaunayVolumeSliverError {
    DelaunayVolumeSliverError {
        kind,
        reason: reason.into(),
    }
}

fn stable_simplex(mut identities: [StableDigest; 4]) -> String {
    identities.sort_unstable();
    identities
        .into_iter()
        .map(|identity| format!("{:02x?}", &identity.bytes()[..4]))
        .collect::<Vec<_>>()
        .join(":")
}

#[cfg(test)]
#[path = "volume_sliver/tests.rs"]
mod tests;
