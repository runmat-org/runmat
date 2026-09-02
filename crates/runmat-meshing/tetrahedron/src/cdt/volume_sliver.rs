use runmat_meshing_core::{MeshingCancellationSignal, StableDigest};
use serde::{Deserialize, Serialize};

use super::{
    validate_delaunay_volume_provenance, validate_delaunay_volume_quality,
    DelaunayInsertionErrorKind, DelaunayInsertionOptions, DelaunayVolumeNode,
    DelaunayVolumeProvenanceErrorKind, DelaunayVolumeQuality, DelaunayVolumeQualityError,
    DelaunayVolumeQualityErrorKind, DelaunayVolumeRefinementInput, DelaunayVolumeTopology,
};

mod admission;
mod insertion;
pub(in crate::cdt) mod reconnection;
mod relocation;
mod retriangulate;
mod validation;
pub(in crate::cdt) mod work;

pub(super) const CAVITY_SMOOTHING_STEPS: [f64; 4] =
    [1.0 / 1_024.0, 1.0 / 128.0, 1.0 / 16.0, 1.0 / 2.0];
pub(super) const MAXIMUM_CAVITY_SMOOTHING_ITERATIONS: usize = 100;
pub(super) const MAXIMUM_CAVITY_SEARCH_DIRECTIONS: usize = 20;
pub(super) const MAXIMUM_INSERTION_LOOKAHEAD_DEPTH: u8 = 1;

use admission::{quality_spectrum, worst_quality_source, QualityObjective};
use insertion::insertion_candidates;
pub(in crate::cdt) use insertion::insertion_identity_is_valid;
use reconnection::reconnection_candidates;
use relocation::{relocation_candidates, RelocationSearchScope};
pub use validation::validate_delaunay_volume_sliver_treatment;
use work::OptimizationWork;

#[derive(Clone, Copy)]
pub(super) struct SliverSearchPolicy {
    options: DelaunayVolumeSliverOptions,
    refinement: admission::CandidateRefinementPolicy,
    objective: QualityObjective,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
    pub removed_tetrahedron_node_identities: Vec<[StableDigest; 4]>,
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
        input,
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
    StalledRefinement,
}

impl QualityConvergenceMode {
    fn refinement_policy(self) -> admission::CandidateRefinementPolicy {
        match self {
            Self::ShapeOnly => admission::CandidateRefinementPolicy::Preserve,
            Self::Coupled | Self::StalledRefinement => {
                admission::CandidateRefinementPolicy::AllowForCoupledConvergence
            }
        }
    }

    fn objective(self) -> QualityObjective {
        match self {
            Self::ShapeOnly | Self::Coupled => QualityObjective::Shape,
            Self::StalledRefinement => QualityObjective::Refinement,
        }
    }
}

pub(super) fn run_treatment(
    input: DelaunayVolumeRefinementInput<'_>,
    options: DelaunayVolumeSliverOptions,
    cancellation: &dyn MeshingCancellationSignal,
    mode: QualityConvergenceMode,
) -> Result<DelaunayVolumeSliverTreatment, DelaunayVolumeSliverError> {
    let span = tracing::info_span!(
        "meshing.tetrahedron.volume_optimization",
        tetrahedron_count = input.topology.tetrahedra.len(),
    );
    let _guard = span.enter();
    let mut topology = input.topology.clone();
    let mut quality = input.quality.clone();
    let mut mutations = Vec::new();
    for pass in 0..options.maximum_passes {
        checkpoint(pass, options, cancellation)?;
        if matches!(
            mode,
            QualityConvergenceMode::Coupled | QualityConvergenceMode::StalledRefinement
        ) && !mutations.is_empty()
            && quality.worst_refinement_tetrahedron.is_some()
        {
            tracing::info!(
                pass,
                mutation_count = mutations.len(),
                "volume optimization yielded to coupled refinement"
            );
            return Ok(DelaunayVolumeSliverTreatment {
                topology,
                quality,
                mutations,
            });
        }
        let objective = mode.objective();
        let Some(source) = worst_quality_source(&quality, &topology, objective) else {
            tracing::info!(
                pass,
                mutation_count = mutations.len(),
                "volume optimization completed"
            );
            return Ok(DelaunayVolumeSliverTreatment {
                topology,
                quality,
                mutations,
            });
        };
        tracing::info!(
            pass,
            optimization_violation_ratio = source.optimization_violation_ratio,
            boundary_node_count = source.boundary_node_count,
            protected_facet_count = source.protected_facet_count,
            metric_radius_edge_ratio = source.metric_radius_edge_ratio,
            metric_scaled_jacobian = source.metric_scaled_jacobian,
            "volume optimization pass started"
        );
        let current_spectrum = quality_spectrum(&quality, objective);
        let current = DelaunayVolumeRefinementInput {
            topology: &topology,
            metric_request: input.metric_request,
            provenance: input.provenance,
            quality: &quality,
            quality_options: input.quality_options,
            fixed_node_identities: input.fixed_node_identities,
        };
        let mut work = OptimizationWork::new(
            options.maximum_candidate_evaluations_per_pass,
            options.cancellation_check_interval,
            cancellation,
        );
        let reconnection = reconnection_candidates(
            current,
            source,
            &current_spectrum,
            cancellation,
            &mut work,
            SliverSearchPolicy {
                options,
                refinement: mode.refinement_policy(),
                objective,
            },
            reconnection::ReconnectionMode {
                allow_lookahead: true,
                maximum_candidates: None,
            },
        )?;
        if let Some(candidate) = reconnection.candidate {
            topology = candidate.topology;
            quality = candidate.quality;
            mutations.extend(candidate.mutations);
            tracing::info!(pass, "volume optimization admitted a local reconnection");
            continue;
        }
        let search = relocation_candidates(
            current,
            source,
            &current_spectrum,
            cancellation,
            &mut work,
            SliverSearchPolicy {
                options,
                refinement: mode.refinement_policy(),
                objective,
            },
            RelocationSearchScope::Complete,
        )?;
        if let Some(candidate) = search.candidate {
            topology = candidate.topology;
            quality = candidate.quality;
            mutations.push(candidate.mutation);
            continue;
        }
        let insertion = insertion_candidates(
            current,
            source,
            &current_spectrum,
            cancellation,
            &mut work,
            SliverSearchPolicy {
                options,
                refinement: mode.refinement_policy(),
                objective,
            },
            MAXIMUM_INSERTION_LOOKAHEAD_DEPTH,
        )?;
        let Some(candidate) = insertion.candidate else {
            let source_nodes = source.node_identities.map(|identity| {
                topology
                    .nodes
                    .binary_search_by_key(&identity, |node| node.identity)
                    .ok()
                    .map(|index| topology.nodes[index].coordinates_m)
            });
            return Err(error(
                DelaunayVolumeSliverErrorKind::NoAdmissibleOptimization,
                format!(
                    "no legal reconnection, interior relocation, or protected-boundary insertion improves volume defect {} at {:?} with {} boundary nodes and {} protected facets, optimization ratio {}, scaled Jacobian {}, and target {}; reconnection rejections {:?}; relocation rejections {:?}; insertion rejections {:?}; first insertion rejection details {:?}",
                    stable_simplex(source.node_identities),
                    source_nodes,
                    source.boundary_node_count,
                    source.protected_facet_count,
                    source.optimization_violation_ratio,
                    source.metric_scaled_jacobian,
                    input.quality_options.minimum_metric_scaled_jacobian,
                    reconnection.rejection_counts,
                    search.rejection_counts,
                    insertion.rejection_counts,
                    insertion.first_rejection_details,
                ),
            ));
        };
        topology = candidate.topology;
        quality = candidate.quality;
        mutations.extend(candidate.mutations);
        tracing::info!(
            pass,
            "volume optimization admitted a local Steiner insertion"
        );
    }
    let remaining = worst_sliver(
        &quality,
        &topology,
        input.quality_options.minimum_metric_scaled_jacobian,
    )
    .map(|tetrahedron| tetrahedron.metric_scaled_jacobian)
    .unwrap_or(input.quality_options.minimum_metric_scaled_jacobian);
    Err(error(
        DelaunayVolumeSliverErrorKind::ResourceLimit,
        format!(
            "sliver treatment exhausted {} passes with minimum metric scaled Jacobian {remaining}",
            options.maximum_passes
        ),
    ))
}

fn worst_sliver<'a>(
    quality: &'a DelaunayVolumeQuality,
    topology: &DelaunayVolumeTopology,
    _minimum: f64,
) -> Option<&'a super::DelaunayTetrahedronQuality> {
    quality
        .tetrahedra
        .iter()
        .filter(|tetrahedron| tetrahedron.requires_optimization())
        .max_by(|left, right| {
            super::volume_quality::compare_violation_priority(
                left.optimization_violation_ratio,
                super::volume_quality::tetrahedron_geometry_key(topology, left.node_identities),
                left.node_identities,
                right.optimization_violation_ratio,
                super::volume_quality::tetrahedron_geometry_key(topology, right.node_identities),
                right.node_identities,
            )
        })
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
