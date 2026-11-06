use std::collections::BTreeMap;

use runmat_geometry_core::{ExactBRepTopology, GeometryEvaluationControl, PersistentEntityId};
use runmat_meshing_core::MetricFieldRequest;
use runmat_meshing_size::grading::grade_metric_evaluations;
use runmat_meshing_size::incidence::TopologyMetricIncidence;
use runmat_meshing_size::metric::{
    ExactFaceMetricResolutionError, ResolvedMetricEvaluation, ResolvedMetricField,
};

use super::{
    CurveMetricEvaluation, CurveMetricField, CurveMetricQuery, SharedCurveError,
    SharedCurveErrorKind,
};

/// Exact-topology projection of the canonical resolved metric request for curve evaluation.
///
/// Applicable contributions are intersected by the shared sizing owner in canonical order.
/// Repeated physical tensors retain provenance without multiplying sizing density.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedCurveMetricField {
    by_edge: BTreeMap<PersistentEntityId, EdgeMetricRecord>,
    /// Uniform face sizing; each face's entity-local sources are graded at the query point.
    face_metrics: Vec<ResolvedMetricEvaluation>,
    face_local: Vec<Vec<runmat_meshing_size::metric::LocalMetricSource>>,
    resolver: ResolvedMetricField,
}

#[derive(Debug, Clone, PartialEq)]
struct EdgeMetricRecord {
    /// Contributions that apply uniformly along the edge, graded over edge adjacency.
    resolved: ResolvedMetricEvaluation,
    /// Edge-local sources of this edge and the edges meeting it at a vertex, graded by distance
    /// from their exact supports at each query point.
    local: Vec<runmat_meshing_size::metric::LocalMetricSource>,
    faces: Vec<usize>,
}

impl ResolvedCurveMetricField {
    pub fn new(
        topology: &ExactBRepTopology,
        request: &MetricFieldRequest,
    ) -> Result<Self, SharedCurveError> {
        Self::prepare(topology, request, None)
    }
    pub fn new_with_control(
        topology: &ExactBRepTopology,
        request: &MetricFieldRequest,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<Self, SharedCurveError> {
        Self::prepare(topology, request, Some(control))
    }
    fn prepare(
        topology: &ExactBRepTopology,
        request: &MetricFieldRequest,
        control: Option<&dyn GeometryEvaluationControl>,
    ) -> Result<Self, SharedCurveError> {
        let resolver = ResolvedMetricField::new(request).map_err(|error| {
            SharedCurveError::invalid_request("resolved curve metric", error.to_string())
        })?;
        let incidence = TopologyMetricIncidence::new(topology);
        let face_metrics = resolver
            .resolve_exact_faces(topology, &incidence, control)
            .map_err(|error| match error {
                ExactFaceMetricResolutionError::Metric(error) => {
                    SharedCurveError::invalid_request("boundary face metric", error.to_string())
                }
                ExactFaceMetricResolutionError::Control(error) => SharedCurveError::new(
                    SharedCurveErrorKind::GeometryEvaluation(error.kind),
                    "boundary face metric",
                    error.reason,
                ),
            })?;
        // An edge-local source (curvature, feature, proximity, contact) with an exact support is
        // graded by distance from that support, as on faces. Grading it over edge adjacency would
        // impose its size on the whole of each neighbor and, hop by hop, on edges far from the
        // feature: a gap at one end of a long edge would refine the corners at its other end.
        let adjacency = TopologyMetricIncidence::edge_adjacency(topology);
        let mut by_edge = BTreeMap::new();
        let mut local_by_edge = BTreeMap::new();
        for edge in &topology.edges {
            let incident = incidence.incident_edge_entities(edge);
            let intersection_error = |error: runmat_meshing_size::metric::MetricContractError| {
                SharedCurveError::invalid_request(
                    "resolved curve metric intersection",
                    error.to_string(),
                )
                .for_edge(&edge.id)
            };
            let resolved = resolver
                .resolve_constant(&incident)
                .map_err(intersection_error)?;
            let mut local_entities = adjacency.get(&edge.id).cloned().unwrap_or_default();
            local_entities.insert(edge.id.clone());
            let local = resolver
                .local_sources(&local_entities)
                .map_err(intersection_error)?;
            by_edge.insert(edge.id.clone(), resolved);
            local_by_edge.insert(edge.id.clone(), local);
        }
        grade_metric_evaluations(request.maximum_grading_ratio, &adjacency, &mut by_edge)
        .map_err(|error| {
            SharedCurveError::invalid_request("resolved curve metric grading", error.to_string())
        })?;
        let mut records = BTreeMap::new();
        for edge in &topology.edges {
            if let Some(control) = control {
                control.checkpoint().map_err(|error| {
                    SharedCurveError::new(
                        SharedCurveErrorKind::GeometryEvaluation(error.kind),
                        "curve metric preparation",
                        error.reason,
                    )
                    .for_edge(&edge.id)
                })?;
            }
            let ids = incidence.edge_face_ids(&edge.id);
            let scratch_bytes = (ids.len() as u64)
                .checked_mul(std::mem::size_of::<&PersistentEntityId>() as u64)
                .ok_or_else(|| {
                    SharedCurveError::invalid_request("edge face references", "storage overflow")
                })?;
            let _scratch = control
                .map(|control| {
                    runmat_geometry_core::GeometryAllocationReservation::new(control, scratch_bytes)
                })
                .transpose()
                .map_err(control_error)?;
            let mut face_ids = incidence.edge_face_ids(&edge.id).iter().collect::<Vec<_>>();
            face_ids.sort_unstable();
            face_ids.dedup();
            let mut incident = incidence.incident_edge_entities(edge);
            if let Some(control) = control {
                let bytes = (face_ids.len() as u64)
                    .checked_mul(std::mem::size_of::<usize>() as u64)
                    .ok_or_else(|| {
                        SharedCurveError::invalid_request("edge face indices", "storage overflow")
                    })?;
                control
                    .consume_allocation_bytes(bytes)
                    .map_err(control_error)?;
            }
            let mut faces = Vec::with_capacity(face_ids.len());
            for id in face_ids {
                let face = topology
                    .faces
                    .binary_search_by(|face| face.id.cmp(id))
                    .map(|index| &topology.faces[index])
                    .map_err(|_| {
                        SharedCurveError::invalid_request(
                            "curve face incidence",
                            "references an absent exact face",
                        )
                        .for_edge(&edge.id)
                    })?;
                incident.extend(incidence.incident_face_entities(face));
                faces.push(
                    topology
                        .faces
                        .binary_search_by(|face| face.id.cmp(id))
                        .expect("admitted exact face"),
                );
            }
            let mut resolved = by_edge.remove(&edge.id).expect("resolved every exact edge");
            let provenance = resolver.resolve(&incident).map_err(|error| {
                SharedCurveError::invalid_request("curve boundary provenance", error.to_string())
                    .for_edge(&edge.id)
            })?;
            resolved.active_sources = provenance.active_sources;
            resolved.applied_contribution_count = provenance.applied_contribution_count;
            let local = local_by_edge
                .remove(&edge.id)
                .expect("resolved every exact edge");
            records.insert(
                edge.id.clone(),
                EdgeMetricRecord {
                    resolved,
                    local,
                    faces,
                },
            );
        }
        if let Some(control) = control {
            let slots = face_metrics
                .len()
                .checked_mul(std::mem::size_of::<ResolvedMetricEvaluation>())
                .and_then(|n| u64::try_from(n).ok())
                .ok_or_else(|| {
                    SharedCurveError::invalid_request("face metric table", "storage overflow")
                })?;
            control
                .consume_allocation_bytes(slots)
                .map_err(control_error)?;
        }
        let mut face_local = Vec::with_capacity(face_metrics.len());
        let mut face_uniform = Vec::with_capacity(face_metrics.len());
        for sizing in face_metrics.into_values() {
            if let Some(control) = control {
                let bytes = (sizing.local.capacity() as u64)
                    .checked_mul(
                        std::mem::size_of::<runmat_meshing_size::metric::LocalMetricSource>()
                            as u64,
                    )
                    .ok_or_else(|| {
                        SharedCurveError::invalid_request("face local metric", "storage overflow")
                    })?;
                control
                    .consume_allocation_bytes(bytes)
                    .map_err(control_error)?;
            }
            face_uniform.push(sizing.uniform);
            face_local.push(sizing.local);
        }
        Ok(Self {
            by_edge: records,
            face_metrics: face_uniform,
            face_local,
            resolver,
        })
    }
}

impl CurveMetricField for ResolvedCurveMetricField {
    fn evaluate(
        &self,
        query: CurveMetricQuery<'_>,
    ) -> Result<CurveMetricEvaluation, SharedCurveError> {
        if query.point_m.iter().any(|value| !value.is_finite())
            || query.unit_tangent.iter().any(|value| !value.is_finite())
        {
            return Err(SharedCurveError::new(
                SharedCurveErrorKind::MetricEvaluation,
                "resolved curve metric query",
                "point and tangent must be finite",
            )
            .for_edge(query.edge_id));
        }
        let record = self.by_edge.get(query.edge_id).ok_or_else(|| {
            SharedCurveError::new(
                SharedCurveErrorKind::MetricEvaluation,
                "resolved curve metric query",
                "edge is absent from the admitted exact topology",
            )
            .for_edge(query.edge_id)
        })?;
        let resolved = &record.resolved;
        let failure = |step: &'static str, detail: String| {
            move |error: runmat_meshing_size::metric::MetricContractError| {
                SharedCurveError::new(
                    SharedCurveErrorKind::MetricEvaluation,
                    "curve boundary trace",
                    format!("{step}: {error} ({detail})"),
                )
                .for_edge(query.edge_id)
            }
        };
        let edge_metric = self
            .resolver
            .evaluate_graded(resolved.metric, &record.local, query.point_m)
            .map_err(failure(
                "graded edge metric",
                format!("uniform {:?} at {:?}", resolved.metric, query.point_m),
            ))?;
        let mut density = edge_metric
            .directional_density(query.unit_tangent)
            .map_err(failure("edge metric", format!("{edge_metric:?}")))?;
        for face in &record.faces {
            let face_metric = self
                .resolver
                .evaluate_graded(
                    self.face_metrics[*face].metric,
                    &self.face_local[*face],
                    query.point_m,
                )
                .map_err(failure(
                    "graded face metric",
                    format!(
                        "uniform {:?} at {:?}",
                        self.face_metrics[*face].metric, query.point_m
                    ),
                ))?;
            density = density.max(
                face_metric
                    .directional_density(query.unit_tangent)
                    .map_err(failure(
                        "face directional density",
                        format!("{face_metric:?}"),
                    ))?,
            );
        }
        let metric = edge_metric
            .with_directional_density(query.unit_tangent, density)
            .map_err(failure(
                "tightened edge metric",
                format!(
                    "base {edge_metric:?}, tangent {:?}, density {density:e}",
                    query.unit_tangent
                ),
            ))?;
        Ok(CurveMetricEvaluation {
            metric,
            active_sources: resolved.active_sources.clone(),
            applied_contribution_count: resolved.applied_contribution_count,
            clipped_contribution_count: resolved
                .clipped_contribution_count
                .saturating_add(u32::from(metric != resolved.metric)),
            rejected_contribution_count: resolved.rejected_contribution_count,
        })
    }
}

fn control_error(error: runmat_geometry_core::GeometryEvaluationError) -> SharedCurveError {
    SharedCurveError::new(
        SharedCurveErrorKind::GeometryEvaluation(error.kind),
        "curve metric preparation",
        error.reason,
    )
}

#[cfg(test)]
mod tests;
