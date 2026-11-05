use std::cmp::Ordering;
use std::collections::BTreeSet;

use runmat_geometry_core::PersistentEntityId;
use runmat_meshing_core::{MeshingCancellationSignal, StableDigest};
use runmat_meshing_size::metric::{
    MetricContractError, MetricFieldRequest, MetricSourceKind, MetricTensor3,
};
use serde::{Deserialize, Serialize};

use super::{
    topology::validate_canonical_volume_topology,
    volume_provenance::validate_delaunay_volume_provenance, DelaunayTopologyOptions,
    DelaunayVolumeProvenance, DelaunayVolumeProvenanceError, DelaunayVolumeProvenanceErrorKind,
    DelaunayVolumeProvenanceOptions, DelaunayVolumeTopology,
};

pub(super) mod context;
use context::{MetricContextInventory, MetricContexts};
pub(in crate::cdt) mod boundary_compatibility;
mod evaluation;
mod jacobian;
#[cfg(test)]
pub(in crate::cdt) mod joint;
pub(in crate::cdt) use jacobian::{evaluate_tetrahedron_jacobian, PreparedJacobian};
mod metrics;
mod mutation;
pub(in crate::cdt) mod node_size;
mod stream;
mod summary;
use stream::visit_from_metric_contexts;
mod handoff;
pub(in crate::cdt) use handoff::verify_complete_handoff;
pub(in crate::cdt) mod working;
pub(super) use mutation::evaluate_quality_after_mutation;
mod validation;

use context::DelaunayVolumeMetricContext;
use evaluation::evaluate_tetrahedron;
pub(in crate::cdt) use evaluation::{evaluate_tetrahedron_geometry, violation_ratios};
pub use validation::validate_delaunay_volume_quality;

/// Below this shape condition, a floating-point circumradius is not stable
/// enough to serve as quantitative evidence. The cell is already rejected by
/// the much stronger scaled-Jacobian quality gate, so both construction and
/// independent validation report a deterministic worst-case radius until the
/// optimizer repairs it.
const CIRCUMRADIUS_CONDITION_FLOOR: f64 = 1.0e-8;

/// Quality work ordering is part of deterministic topology construction, but
/// the low mantissa bits of independently equivalent metric evaluations are
/// not. Keep the quality gate on the full value and discard only those bits
/// when choosing which violating cell to process first.
const QUALITY_PRIORITY_DISCARDED_MANTISSA_BITS: u32 = 24;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::cdt) enum QualityObjective {
    Shape,
    Refinement,
}

pub(super) fn compare_violation_priority(
    left_ratio: f64,
    left_geometry: [[u64; 3]; 4],
    left_identity: [StableDigest; 4],
    right_ratio: f64,
    right_geometry: [[u64; 3]; 4],
    right_identity: [StableDigest; 4],
) -> Ordering {
    violation_priority_key(left_ratio)
        .cmp(&violation_priority_key(right_ratio))
        .then_with(|| right_geometry.cmp(&left_geometry))
        .then_with(|| right_identity.cmp(&left_identity))
}

pub(super) fn tetrahedron_geometry_key(
    topology: &DelaunayVolumeTopology,
    identities: [StableDigest; 4],
) -> [[u64; 3]; 4] {
    let mut key = identities.map(|identity| {
        topology
            .nodes
            .binary_search_by_key(&identity, |node| node.identity)
            .map(|index| {
                topology.nodes[index]
                    .coordinates_m
                    .canonical_value()
                    .map(ordered_float_key)
            })
            .unwrap_or([u64::MAX; 3])
    });
    key.sort_unstable();
    key
}

pub(super) fn ordered_float_key(value: f64) -> u64 {
    let bits = if value == 0.0 { 0.0_f64 } else { value }.to_bits();
    if bits >> 63 == 0 {
        bits | (1_u64 << 63)
    } else {
        !bits
    }
}

pub(super) fn violation_priority_key(ratio: f64) -> u64 {
    debug_assert!(ratio.is_finite() && ratio >= 0.0);
    let rounding = 1_u64 << (QUALITY_PRIORITY_DISCARDED_MANTISSA_BITS - 1);
    ratio.to_bits().saturating_add(rounding) >> QUALITY_PRIORITY_DISCARDED_MANTISSA_BITS
}

/// Physical bounds use the complete computed value. Scheduling buckets must
/// never make an above-bound value admissible.
pub(super) fn violates_quality_bound(ratio: f64) -> bool {
    !ratio.is_finite() || ratio > 1.0
}

pub(super) fn compare_spectra(left: &[f64], right: &[f64]) -> Ordering {
    left.iter()
        .zip(right)
        .map(|(left, right)| violation_priority_key(*left).cmp(&violation_priority_key(*right)))
        .find(|ordering| *ordering != Ordering::Equal)
        .unwrap_or_else(|| left.len().cmp(&right.len()))
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DelaunayVolumeQualityOptions {
    pub maximum_nodes: u64,
    pub maximum_tetrahedra: u64,
    pub maximum_metric_edge_length: f64,
    pub maximum_radius_edge_ratio: f64,
    pub minimum_metric_scaled_jacobian: f64,
    pub cancellation_check_interval: u64,
    pub provenance: DelaunayVolumeProvenanceOptions,
}

impl Default for DelaunayVolumeQualityOptions {
    fn default() -> Self {
        Self {
            maximum_nodes: 1_000_000_000,
            maximum_tetrahedra: 2_000_000_000,
            maximum_metric_edge_length: 1.0,
            maximum_radius_edge_ratio: 2.0,
            minimum_metric_scaled_jacobian: 0.1,
            cancellation_check_interval: 1_024,
            provenance: DelaunayVolumeProvenanceOptions::default(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DelaunayTetrahedronQuality {
    pub node_identities: [StableDigest; 4],
    pub region_id: PersistentEntityId,
    pub incident_metric_entity_ids: Vec<PersistentEntityId>,
    pub resolved_metric: MetricTensor3,
    pub active_metric_sources: Vec<MetricSourceKind>,
    pub applied_metric_contribution_count: u32,
    pub clipped_metric_contribution_count: u32,
    pub rejected_metric_contribution_count: u32,
    pub minimum_metric_edge_length: f64,
    pub maximum_metric_edge_length: f64,
    pub metric_circumradius: f64,
    pub metric_radius_edge_ratio: f64,
    pub metric_scaled_jacobian: f64,
    /// Number of vertices constrained by the published surface closure.
    pub boundary_node_count: u8,
    /// Number of this tetrahedron's faces present in protected facet provenance.
    pub protected_facet_count: u8,
    /// Size plus volumetrically repairable radius-edge defect assigned to point insertion.
    pub refinement_violation_ratio: f64,
    /// Scaled-Jacobian shape defect assigned to topology optimization.
    pub optimization_violation_ratio: f64,
}

impl DelaunayTetrahedronQuality {
    pub fn requires_refinement(&self) -> bool {
        violates_quality_bound(self.refinement_violation_ratio)
    }

    pub fn requires_optimization(&self) -> bool {
        violates_quality_bound(self.optimization_violation_ratio)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DelaunayVolumeQuality {
    pub tetrahedra: Vec<DelaunayTetrahedronQuality>,
    pub worst_refinement_tetrahedron: Option<[StableDigest; 4]>,
    pub worst_optimization_tetrahedron: Option<[StableDigest; 4]>,
    pub maximum_metric_edge_length: f64,
    pub maximum_radius_edge_ratio: f64,
    pub minimum_metric_scaled_jacobian: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DelaunayVolumeQualityErrorKind {
    Accounting(runmat_simulation_usage::LedgerError),
    InvalidOptions,
    InvalidTopology,
    InvalidMetric,
    InvalidMetricContext,
    InvalidQuality,
    NumericalFailure,
    ResourceLimit,
    Cancelled,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DelaunayVolumeQualityError {
    pub kind: DelaunayVolumeQualityErrorKind,
    pub tetrahedron_index: Option<u32>,
    pub reason: String,
}

impl std::fmt::Display for DelaunayVolumeQualityError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "3D Delaunay volume quality {:?} at tetrahedron {:?}: {}",
            self.kind, self.tetrahedron_index, self.reason
        )
    }
}

impl std::error::Error for DelaunayVolumeQualityError {}

pub fn evaluate_delaunay_volume_quality(
    topology: &DelaunayVolumeTopology,
    metric_request: &MetricFieldRequest,
    provenance: &DelaunayVolumeProvenance,
    options: DelaunayVolumeQualityOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<DelaunayVolumeQuality, DelaunayVolumeQualityError> {
    let metric_contexts =
        validate_inputs(topology, metric_request, provenance, options, cancellation)?;
    let quality = evaluate_from_metric_contexts(
        topology,
        metric_request,
        options,
        metric_contexts,
        &boundary_node_identities(provenance),
        &protected_facet_identities(provenance),
        provenance,
        cancellation,
    )?;
    validate_delaunay_volume_quality(
        topology,
        metric_request,
        provenance,
        &quality,
        options,
        cancellation,
    )?;
    Ok(quality)
}

/// Evaluates a topology produced by a checked internal mutation. The caller must
/// already have admitted the metric request and persistent boundary provenance;
/// the completed artifact is independently validated at the public boundary.
pub(super) fn evaluate_delaunay_volume_quality_from_admitted_input(
    topology: &DelaunayVolumeTopology,
    metric_request: &MetricFieldRequest,
    provenance: &DelaunayVolumeProvenance,
    options: DelaunayVolumeQualityOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<DelaunayVolumeQuality, DelaunayVolumeQualityError> {
    if topology.nodes.len() as u64 > options.maximum_nodes
        || topology.tetrahedra.len() as u64 > options.maximum_tetrahedra
    {
        return Err(error(
            DelaunayVolumeQualityErrorKind::ResourceLimit,
            None,
            "volume topology exceeds the quality evaluation inventory limit",
        ));
    }
    let metric_contexts =
        MetricContexts::from_admitted(topology, provenance, metric_request, options, cancellation)?;
    evaluate_from_metric_contexts(
        topology,
        metric_request,
        options,
        metric_contexts,
        &boundary_node_identities(provenance),
        &protected_facet_identities(provenance),
        provenance,
        cancellation,
    )
}

fn evaluate_from_metric_contexts(
    topology: &DelaunayVolumeTopology,
    metric_request: &MetricFieldRequest,
    options: DelaunayVolumeQualityOptions,
    metric_contexts: MetricContexts<'_>,
    boundary_nodes: &BTreeSet<StableDigest>,
    protected_facets: &ProtectedFacets,
    provenance: &DelaunayVolumeProvenance,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<DelaunayVolumeQuality, DelaunayVolumeQualityError> {
    let mut tetrahedra = Vec::with_capacity(topology.tetrahedra.len());
    visit_from_metric_contexts(
        topology,
        metric_request,
        options,
        metric_contexts,
        stream::BoundaryProtection {
            nodes: boundary_nodes,
            facets: protected_facets,
            provenance,
        },
        cancellation,
        |_, quality| {
            tetrahedra.push(quality);
            Ok(())
        },
    )?;
    summary::summarize(topology, tetrahedra, options, cancellation)
}

fn boundary_node_identities(provenance: &DelaunayVolumeProvenance) -> BTreeSet<StableDigest> {
    provenance
        .facets
        .iter()
        .flat_map(|facet| facet.node_identities)
        .collect()
}

/// The published surface the volume stage must preserve: its triangles and, derived from the same
/// triangles, its edges. An edge is protected because it belongs to the surface, whether or not a
/// given tetrahedron also carries one of its triangles.
pub(in crate::cdt) struct ProtectedFacets {
    pub(in crate::cdt) facets: BTreeSet<[StableDigest; 3]>,
    pub(in crate::cdt) edges: BTreeSet<[StableDigest; 2]>,
}

fn protected_facet_identities(provenance: &DelaunayVolumeProvenance) -> ProtectedFacets {
    let facets = provenance
        .facets
        .iter()
        .map(|facet| facet.node_identities)
        .collect::<BTreeSet<_>>();
    let edges = facets
        .iter()
        .flat_map(|&[a, b, c]| {
            [[a, b], [a, c], [b, c]].map(|mut edge| {
                edge.sort_unstable();
                edge
            })
        })
        .collect();
    ProtectedFacets { facets, edges }
}

/// Vertex pairs of a tetrahedron's six edges, in the bit order of [`protected_edge_mask`].
pub(in crate::cdt) const TETRAHEDRON_EDGES: [(usize, usize); 6] =
    [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

/// Bit `k` is set when edge `TETRAHEDRON_EDGES[k]` is an edge of the published surface mesh,
/// which the volume stage never splits. A tetrahedron can hold a surface edge without holding
/// either surface triangle on it; that edge is still the surface stage's, not the volume's.
fn protected_edge_mask(tetrahedron: [StableDigest; 4], protected: &ProtectedFacets) -> u8 {
    let mut mask = 0u8;
    for (bit, (left, right)) in TETRAHEDRON_EDGES.iter().enumerate() {
        let mut edge = [tetrahedron[*left], tetrahedron[*right]];
        edge.sort_unstable();
        if protected.edges.contains(&edge) {
            mask |= 1 << bit;
        }
    }
    mask
}

fn protected_facet_count(tetrahedron: [StableDigest; 4], protected: &ProtectedFacets) -> u8 {
    let protected_facets = &protected.facets;
    (0..4)
        .filter(|opposite| {
            let mut facet = [StableDigest::ZERO; 3];
            let mut cursor = 0;
            for (index, identity) in tetrahedron.iter().copied().enumerate() {
                if index != *opposite {
                    facet[cursor] = identity;
                    cursor += 1;
                }
            }
            facet.sort_unstable();
            protected_facets.contains(&facet)
        })
        .count() as u8
}

fn validate_inputs<'a>(
    topology: &'a DelaunayVolumeTopology,
    metric_request: &MetricFieldRequest,
    provenance: &'a DelaunayVolumeProvenance,
    options: DelaunayVolumeQualityOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<MetricContexts<'a>, DelaunayVolumeQualityError> {
    validate_options(options)?;
    metric_request.validate().map_err(metric_error)?;
    if topology.nodes.len() as u64 > options.maximum_nodes
        || topology.tetrahedra.len() as u64 > options.maximum_tetrahedra
    {
        return Err(error(
            DelaunayVolumeQualityErrorKind::ResourceLimit,
            None,
            "volume topology exceeds the quality evaluation inventory limit",
        ));
    }
    if topology
        .tetrahedra
        .iter()
        .any(|tetrahedron| tetrahedron.region_id.is_none())
    {
        return Err(error(
            DelaunayVolumeQualityErrorKind::InvalidTopology,
            None,
            "every evaluated tetrahedron must have one assigned region",
        ));
    }
    validate_canonical_volume_topology(
        topology,
        DelaunayTopologyOptions {
            maximum_nodes: options.maximum_nodes,
            maximum_tetrahedra: options.maximum_tetrahedra,
            cancellation_check_interval: options.cancellation_check_interval,
        },
        cancellation,
    )
    .map_err(|failure| {
        let kind = match failure.kind {
            super::DelaunayTopologyErrorKind::Cancelled => {
                DelaunayVolumeQualityErrorKind::Cancelled
            }
            super::DelaunayTopologyErrorKind::ResourceLimit => {
                DelaunayVolumeQualityErrorKind::ResourceLimit
            }
            _ => DelaunayVolumeQualityErrorKind::InvalidTopology,
        };
        error(kind, None, failure.to_string())
    })?;
    validate_delaunay_volume_provenance(topology, provenance, options.provenance, cancellation)
        .map_err(provenance_error)?;
    MetricContexts::from_admitted(topology, provenance, metric_request, options, cancellation)
}

fn provenance_error(failure: DelaunayVolumeProvenanceError) -> DelaunayVolumeQualityError {
    let kind = match failure.kind {
        DelaunayVolumeProvenanceErrorKind::Accounting(cause) => {
            DelaunayVolumeQualityErrorKind::Accounting(cause)
        }
        DelaunayVolumeProvenanceErrorKind::InvalidOptions => {
            DelaunayVolumeQualityErrorKind::InvalidOptions
        }
        DelaunayVolumeProvenanceErrorKind::InvalidTopology => {
            DelaunayVolumeQualityErrorKind::InvalidTopology
        }
        DelaunayVolumeProvenanceErrorKind::InvalidProvenance => {
            DelaunayVolumeQualityErrorKind::InvalidMetricContext
        }
        DelaunayVolumeProvenanceErrorKind::ResourceLimit => {
            DelaunayVolumeQualityErrorKind::ResourceLimit
        }
        DelaunayVolumeProvenanceErrorKind::Cancelled => DelaunayVolumeQualityErrorKind::Cancelled,
    };
    error(kind, None, failure.to_string())
}

fn validate_options(
    options: DelaunayVolumeQualityOptions,
) -> Result<(), DelaunayVolumeQualityError> {
    if options.maximum_nodes == 0
        || options.maximum_tetrahedra == 0
        || options.cancellation_check_interval == 0
        || !options.maximum_metric_edge_length.is_finite()
        || options.maximum_metric_edge_length <= 0.0
        || !options.maximum_radius_edge_ratio.is_finite()
        || options.maximum_radius_edge_ratio <= 0.0
        || !options.minimum_metric_scaled_jacobian.is_finite()
        || options.minimum_metric_scaled_jacobian <= 0.0
        || options.minimum_metric_scaled_jacobian > 1.0
    {
        return Err(error(
            DelaunayVolumeQualityErrorKind::InvalidOptions,
            None,
            "inventory limits and cancellation interval must be nonzero; upper quality bounds must be finite and positive; minimum scaled Jacobian must be finite and in (0, 1]",
        ));
    }
    Ok(())
}

fn checkpoint(
    step: u64,
    options: DelaunayVolumeQualityOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayVolumeQualityError> {
    if step.is_multiple_of(options.cancellation_check_interval) && cancellation.is_cancelled() {
        return Err(error(
            DelaunayVolumeQualityErrorKind::Cancelled,
            None,
            "cancelled",
        ));
    }
    Ok(())
}

fn metric_error(failure: MetricContractError) -> DelaunayVolumeQualityError {
    error(
        DelaunayVolumeQualityErrorKind::InvalidMetric,
        None,
        failure.to_string(),
    )
}

fn error(
    kind: DelaunayVolumeQualityErrorKind,
    tetrahedron_index: Option<usize>,
    reason: impl Into<String>,
) -> DelaunayVolumeQualityError {
    DelaunayVolumeQualityError {
        kind,
        tetrahedron_index: tetrahedron_index.and_then(|index| u32::try_from(index).ok()),
        reason: reason.into(),
    }
}

#[cfg(test)]
#[path = "volume_quality/tests.rs"]
mod tests;
