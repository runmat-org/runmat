use std::collections::BTreeSet;

use runmat_geometry_core::PersistentEntityId;
use runmat_meshing_core::{MeshingCancellationSignal, StableDigest};
use runmat_meshing_size::metric::{
    MetricContractError, MetricFieldRequest, MetricSourceKind, MetricTensor3, ResolvedMetricField,
};

use super::{
    topology::build_delaunay_volume_topology_with_regions, DelaunayTopologyOptions,
    DelaunayVolumeTopology,
};

mod context;
mod evaluation;
mod validation;

use context::validate_metric_contexts;
pub use context::DelaunayVolumeMetricContext;
use evaluation::evaluate_tetrahedron;
pub use validation::validate_delaunay_volume_quality;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DelaunayVolumeQualityOptions {
    pub maximum_nodes: u64,
    pub maximum_tetrahedra: u64,
    pub maximum_metric_edge_length: f64,
    pub maximum_radius_edge_ratio: f64,
    pub cancellation_check_interval: u64,
}

impl Default for DelaunayVolumeQualityOptions {
    fn default() -> Self {
        Self {
            maximum_nodes: 1_000_000_000,
            maximum_tetrahedra: 2_000_000_000,
            maximum_metric_edge_length: 1.0,
            maximum_radius_edge_ratio: 2.0,
            cancellation_check_interval: 1_024,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
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
    pub refinement_violation_ratio: f64,
}

impl DelaunayTetrahedronQuality {
    pub fn requires_refinement(&self) -> bool {
        self.refinement_violation_ratio > 1.0
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DelaunayVolumeQuality {
    pub tetrahedra: Vec<DelaunayTetrahedronQuality>,
    pub worst_refinement_tetrahedron: Option<[StableDigest; 4]>,
    pub maximum_metric_edge_length: f64,
    pub maximum_radius_edge_ratio: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DelaunayVolumeQualityErrorKind {
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
    metric_contexts: &[DelaunayVolumeMetricContext],
    options: DelaunayVolumeQualityOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<DelaunayVolumeQuality, DelaunayVolumeQualityError> {
    validate_inputs(
        topology,
        metric_request,
        metric_contexts,
        options,
        cancellation,
    )?;
    let field = ResolvedMetricField::new(metric_request).map_err(metric_error)?;
    let mut tetrahedra = Vec::with_capacity(topology.tetrahedra.len());
    let mut worst = None::<(f64, [StableDigest; 4])>;
    let mut maximum_metric_edge_length = 0.0_f64;
    let mut maximum_radius_edge_ratio = 0.0_f64;

    for (index, (tetrahedron, context)) in
        topology.tetrahedra.iter().zip(metric_contexts).enumerate()
    {
        checkpoint(index as u64, options, cancellation)?;
        let region_id = tetrahedron.region_id.clone().ok_or_else(|| {
            error(
                DelaunayVolumeQualityErrorKind::InvalidTopology,
                Some(index),
                "every evaluated tetrahedron must have one assigned region",
            )
        })?;
        let incident_entities = context
            .incident_entity_ids
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();
        let metric = field.resolve(&incident_entities).map_err(metric_error)?;
        let quality = evaluate_tetrahedron(
            topology,
            index,
            region_id,
            context.incident_entity_ids.clone(),
            metric,
            options,
        )?;
        maximum_metric_edge_length =
            maximum_metric_edge_length.max(quality.maximum_metric_edge_length);
        maximum_radius_edge_ratio = maximum_radius_edge_ratio.max(quality.metric_radius_edge_ratio);
        if quality.requires_refinement() {
            let candidate = (quality.refinement_violation_ratio, quality.node_identities);
            if worst.as_ref().is_none_or(|current| {
                candidate.0.total_cmp(&current.0).is_gt()
                    || candidate.0.total_cmp(&current.0).is_eq() && candidate.1 < current.1
            }) {
                worst = Some(candidate);
            }
        }
        tetrahedra.push(quality);
    }

    let quality = DelaunayVolumeQuality {
        tetrahedra,
        worst_refinement_tetrahedron: worst.map(|(_, identity)| identity),
        maximum_metric_edge_length,
        maximum_radius_edge_ratio,
    };
    validate_delaunay_volume_quality(
        topology,
        metric_request,
        metric_contexts,
        &quality,
        options,
        cancellation,
    )?;
    Ok(quality)
}

fn validate_inputs(
    topology: &DelaunayVolumeTopology,
    metric_request: &MetricFieldRequest,
    metric_contexts: &[DelaunayVolumeMetricContext],
    options: DelaunayVolumeQualityOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayVolumeQualityError> {
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
    validate_metric_contexts(topology, metric_contexts)?;
    let rebuilt = build_delaunay_volume_topology_with_regions(
        topology.nodes.clone(),
        topology
            .tetrahedra
            .iter()
            .map(|tetrahedron| (tetrahedron.vertex_indices, tetrahedron.region_id.clone()))
            .collect(),
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
    if rebuilt != *topology {
        return Err(error(
            DelaunayVolumeQualityErrorKind::InvalidTopology,
            None,
            "volume topology is not in canonical checked form",
        ));
    }
    Ok(())
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
    {
        return Err(error(
            DelaunayVolumeQualityErrorKind::InvalidOptions,
            None,
            "inventory limits and cancellation interval must be nonzero and quality bounds must be finite and positive",
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
