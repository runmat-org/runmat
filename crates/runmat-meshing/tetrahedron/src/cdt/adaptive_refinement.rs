//! Deterministic local h-refinement for solver-selected tetrahedra.
//!
//! The solver owns estimator choice and marking. This module admits those marks against a checked
//! quality snapshot, imposes a canonical mutation order, and delegates every topology change to
//! the constrained Delaunay insertion authority. A later insertion may subsume a still-pending
//! marked cell; that outcome is recorded explicitly rather than reconstructed heuristically.

use std::collections::BTreeSet;

use runmat_meshing_core::{MeshingCancellationSignal, StableDigest};
use sha2::{Digest, Sha256};

use super::{
    evaluate_delaunay_volume_quality,
    insertion::{
        insert_delaunay_volume_node_with_barriers, validate_constrained_delaunay_volume_topology,
    },
    validate_delaunay_volume_provenance, validate_delaunay_volume_quality, DelaunayInsertionError,
    DelaunayInsertionErrorKind, DelaunayInsertionOptions, DelaunayVolumeNode,
    DelaunayVolumeQuality, DelaunayVolumeQualityError, DelaunayVolumeQualityErrorKind,
    DelaunayVolumeRefinementInput, DelaunayVolumeTopology,
};

const ADAPTIVE_NODE_IDENTITY_DOMAIN: &[u8] = b"runmat/meshing/cdt/adaptive-refinement-node/1\0";

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DelaunayAdaptiveRefinementMark {
    /// Oriented stable node identities from the admitted tetrahedron-quality record.
    pub node_identities: [StableDigest; 4],
    /// Finite positive solver indicator used only for deterministic priority ordering.
    pub indicator_value: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DelaunayAdaptiveRefinementOptions {
    pub insertion: DelaunayInsertionOptions,
    pub maximum_marks: u64,
    pub maximum_insertions: u64,
    pub cancellation_check_interval: u64,
}

impl Default for DelaunayAdaptiveRefinementOptions {
    fn default() -> Self {
        Self {
            insertion: DelaunayInsertionOptions::default(),
            maximum_marks: 10_000_000,
            maximum_insertions: 10_000_000,
            cancellation_check_interval: 1_024,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DelaunayAdaptiveRefinementDecision {
    Inserted {
        mark: DelaunayAdaptiveRefinementMark,
        node: DelaunayVolumeNode,
    },
    CoveredByPriorInsertion {
        mark: DelaunayAdaptiveRefinementMark,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct DelaunayAdaptiveRefinementResult {
    pub topology: DelaunayVolumeTopology,
    pub quality: DelaunayVolumeQuality,
    pub decisions: Vec<DelaunayAdaptiveRefinementDecision>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DelaunayAdaptiveRefinementErrorKind {
    InvalidOptions,
    InvalidInput,
    InvalidMarks,
    InvalidResult,
    ResourceLimit,
    Cancelled,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DelaunayAdaptiveRefinementError {
    pub kind: DelaunayAdaptiveRefinementErrorKind,
    pub reason: String,
}

impl std::fmt::Display for DelaunayAdaptiveRefinementError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "marked Delaunay adaptation {:?}: {}",
            self.kind, self.reason
        )
    }
}

impl std::error::Error for DelaunayAdaptiveRefinementError {}

pub fn refine_marked_delaunay_volume(
    input: DelaunayVolumeRefinementInput<'_>,
    marks: &[DelaunayAdaptiveRefinementMark],
    options: DelaunayAdaptiveRefinementOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<DelaunayAdaptiveRefinementResult, DelaunayAdaptiveRefinementError> {
    let canonical_marks = validate_input(input, marks, options, cancellation)?;
    let result = apply_marks(input, &canonical_marks, options, cancellation)?;
    validate_marked_delaunay_volume_refinement(input, marks, &result, options, cancellation)?;
    Ok(result)
}

pub fn validate_marked_delaunay_volume_refinement(
    input: DelaunayVolumeRefinementInput<'_>,
    marks: &[DelaunayAdaptiveRefinementMark],
    result: &DelaunayAdaptiveRefinementResult,
    options: DelaunayAdaptiveRefinementOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayAdaptiveRefinementError> {
    let canonical_marks = validate_input(input, marks, options, cancellation)?;
    if result.decisions.len() != canonical_marks.len()
        || result
            .decisions
            .iter()
            .zip(&canonical_marks)
            .any(|(decision, mark)| decision_mark(*decision) != *mark)
    {
        return Err(error(
            DelaunayAdaptiveRefinementErrorKind::InvalidResult,
            "adaptive decisions do not match the canonical marked-cell order",
        ));
    }
    let replay = apply_marks(input, &canonical_marks, options, cancellation)?;
    if replay != *result {
        return Err(error(
            DelaunayAdaptiveRefinementErrorKind::InvalidResult,
            "adaptive topology, quality, or ordered mutation lineage does not match replay",
        ));
    }
    Ok(())
}

fn validate_input(
    input: DelaunayVolumeRefinementInput<'_>,
    marks: &[DelaunayAdaptiveRefinementMark],
    options: DelaunayAdaptiveRefinementOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Vec<DelaunayAdaptiveRefinementMark>, DelaunayAdaptiveRefinementError> {
    if options.maximum_marks == 0
        || options.maximum_insertions == 0
        || options.cancellation_check_interval == 0
    {
        return Err(error(
            DelaunayAdaptiveRefinementErrorKind::InvalidOptions,
            "adaptive mark, insertion, and cancellation limits must be nonzero",
        ));
    }
    if marks.len() as u64 > options.maximum_marks {
        return Err(resource(format!(
            "adaptive mark inventory {} exceeds its hard limit {}",
            marks.len(),
            options.maximum_marks
        )));
    }
    validate_delaunay_volume_quality(
        input.topology,
        input.metric_request,
        input.provenance,
        input.quality,
        input.quality_options,
        cancellation,
    )
    .map_err(|failure| quality_error(failure, DelaunayAdaptiveRefinementErrorKind::InvalidInput))?;
    let admitted = input
        .quality
        .tetrahedra
        .iter()
        .map(|quality| quality.node_identities)
        .collect::<BTreeSet<_>>();
    let mut unique = BTreeSet::new();
    for (index, mark) in marks.iter().enumerate() {
        checkpoint(index as u64, options, cancellation)?;
        if mark.node_identities.contains(&StableDigest::ZERO)
            || !mark.indicator_value.is_finite()
            || mark.indicator_value <= 0.0
            || !admitted.contains(&mark.node_identities)
            || !unique.insert(mark.node_identities)
        {
            return Err(error(
                DelaunayAdaptiveRefinementErrorKind::InvalidMarks,
                "adaptive marks must be unique admitted cells with finite positive indicators",
            ));
        }
    }
    let mut canonical = marks.to_vec();
    canonical.sort_by(|left, right| {
        right
            .indicator_value
            .total_cmp(&left.indicator_value)
            .then_with(|| left.node_identities.cmp(&right.node_identities))
    });
    Ok(canonical)
}

fn apply_marks(
    input: DelaunayVolumeRefinementInput<'_>,
    marks: &[DelaunayAdaptiveRefinementMark],
    options: DelaunayAdaptiveRefinementOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<DelaunayAdaptiveRefinementResult, DelaunayAdaptiveRefinementError> {
    let protected_faces = input
        .provenance
        .facets
        .iter()
        .map(|facet| facet.node_identities)
        .collect::<Vec<_>>();
    let mut topology = input.topology.clone();
    let mut decisions = Vec::with_capacity(marks.len());
    let mut insertion_count = 0_u64;
    for (index, mark) in marks.iter().copied().enumerate() {
        checkpoint(index as u64, options, cancellation)?;
        let Some(tetrahedron) = topology.tetrahedra.iter().find(|tetrahedron| {
            tetrahedron
                .vertex_indices
                .map(|vertex| topology.nodes[vertex as usize].identity)
                == mark.node_identities
        }) else {
            decisions.push(DelaunayAdaptiveRefinementDecision::CoveredByPriorInsertion { mark });
            continue;
        };
        if insertion_count >= options.maximum_insertions {
            return Err(resource(format!(
                "adaptive refinement requires more than {} insertions",
                options.maximum_insertions
            )));
        }
        let node = adaptive_node(&topology, tetrahedron.vertex_indices, mark);
        topology = insert_delaunay_volume_node_with_barriers(
            topology,
            node,
            &protected_faces,
            options.insertion,
            cancellation,
        )
        .map_err(insertion_error)?;
        insertion_count += 1;
        decisions.push(DelaunayAdaptiveRefinementDecision::Inserted { mark, node });
    }
    validate_constrained_delaunay_volume_topology(
        &topology,
        &protected_faces,
        options.insertion,
        cancellation,
    )
    .map_err(insertion_error)?;
    validate_delaunay_volume_provenance(
        &topology,
        input.provenance,
        input.quality_options.provenance,
        cancellation,
    )
    .map_err(|failure| {
        error(
            DelaunayAdaptiveRefinementErrorKind::InvalidResult,
            failure.to_string(),
        )
    })?;
    let quality = evaluate_delaunay_volume_quality(
        &topology,
        input.metric_request,
        input.provenance,
        input.quality_options,
        cancellation,
    )
    .map_err(|failure| {
        quality_error(failure, DelaunayAdaptiveRefinementErrorKind::InvalidResult)
    })?;
    Ok(DelaunayAdaptiveRefinementResult {
        topology,
        quality,
        decisions,
    })
}

fn adaptive_node(
    topology: &DelaunayVolumeTopology,
    vertices: [u32; 4],
    mark: DelaunayAdaptiveRefinementMark,
) -> DelaunayVolumeNode {
    let coordinates_m = std::array::from_fn(|axis| {
        vertices
            .iter()
            .map(|vertex| topology.nodes[*vertex as usize].coordinates_m[axis] * 0.25)
            .sum()
    });
    // Indicator magnitude is solver evidence, not geometric identity. The same marked cell must
    // produce the same node even when estimator scaling changes between otherwise identical runs.
    let mut identities = mark.node_identities;
    identities.sort_unstable();
    let mut hasher = Sha256::new();
    hasher.update(ADAPTIVE_NODE_IDENTITY_DOMAIN);
    for identity in identities {
        hasher.update(identity.bytes());
    }
    DelaunayVolumeNode {
        identity: StableDigest::from_bytes(hasher.finalize().into()),
        coordinates_m,
    }
}

fn decision_mark(decision: DelaunayAdaptiveRefinementDecision) -> DelaunayAdaptiveRefinementMark {
    match decision {
        DelaunayAdaptiveRefinementDecision::Inserted { mark, .. }
        | DelaunayAdaptiveRefinementDecision::CoveredByPriorInsertion { mark } => mark,
    }
}

fn checkpoint(
    work: u64,
    options: DelaunayAdaptiveRefinementOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayAdaptiveRefinementError> {
    if work.is_multiple_of(options.cancellation_check_interval) && cancellation.is_cancelled() {
        return Err(error(
            DelaunayAdaptiveRefinementErrorKind::Cancelled,
            "cancelled",
        ));
    }
    Ok(())
}

fn insertion_error(failure: DelaunayInsertionError) -> DelaunayAdaptiveRefinementError {
    let kind = match failure.kind {
        DelaunayInsertionErrorKind::ResourceLimit => {
            DelaunayAdaptiveRefinementErrorKind::ResourceLimit
        }
        DelaunayInsertionErrorKind::Cancelled => DelaunayAdaptiveRefinementErrorKind::Cancelled,
        DelaunayInsertionErrorKind::InvalidOptions => {
            DelaunayAdaptiveRefinementErrorKind::InvalidOptions
        }
        DelaunayInsertionErrorKind::InvalidTopology
        | DelaunayInsertionErrorKind::InvalidNode
        | DelaunayInsertionErrorKind::PointOutsideTopology => {
            DelaunayAdaptiveRefinementErrorKind::InvalidResult
        }
    };
    error(kind, failure.to_string())
}

fn quality_error(
    failure: DelaunayVolumeQualityError,
    invalid_kind: DelaunayAdaptiveRefinementErrorKind,
) -> DelaunayAdaptiveRefinementError {
    let kind = match failure.kind {
        DelaunayVolumeQualityErrorKind::InvalidOptions => {
            DelaunayAdaptiveRefinementErrorKind::InvalidOptions
        }
        DelaunayVolumeQualityErrorKind::ResourceLimit => {
            DelaunayAdaptiveRefinementErrorKind::ResourceLimit
        }
        DelaunayVolumeQualityErrorKind::Cancelled => DelaunayAdaptiveRefinementErrorKind::Cancelled,
        DelaunayVolumeQualityErrorKind::InvalidTopology
        | DelaunayVolumeQualityErrorKind::InvalidMetric
        | DelaunayVolumeQualityErrorKind::InvalidMetricContext
        | DelaunayVolumeQualityErrorKind::InvalidQuality
        | DelaunayVolumeQualityErrorKind::NumericalFailure => invalid_kind,
    };
    error(kind, failure.to_string())
}

fn resource(reason: impl Into<String>) -> DelaunayAdaptiveRefinementError {
    error(DelaunayAdaptiveRefinementErrorKind::ResourceLimit, reason)
}

fn error(
    kind: DelaunayAdaptiveRefinementErrorKind,
    reason: impl Into<String>,
) -> DelaunayAdaptiveRefinementError {
    DelaunayAdaptiveRefinementError {
        kind,
        reason: reason.into(),
    }
}
