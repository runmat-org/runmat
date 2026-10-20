//! Call-scoped metric preparation with the original incidence and grading rules.
use super::{invalid_request, FaceMetricRecord, ResolvedFaceMetricField};
use crate::{ExactFaceMetricError, ExactFaceMetricErrorKind};
use runmat_geometry_core::{ExactBRepTopology, GeometryEvaluationControl};
use runmat_meshing_core::MetricFieldRequest;
use runmat_meshing_size::{
    incidence::TopologyMetricIncidence,
    metric::{ExactFaceMetricResolutionError, ResolvedMetricField},
};
use std::{collections::BTreeMap, sync::OnceLock};

impl ResolvedFaceMetricField {
    pub fn new(
        topology: &ExactBRepTopology,
        request: &MetricFieldRequest,
    ) -> Result<Self, ExactFaceMetricError> {
        Self::prepare(topology, request, None)
    }

    /// Prepares the same immutable field while observing the execution control.
    pub fn new_with_control(
        topology: &ExactBRepTopology,
        request: &MetricFieldRequest,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<Self, ExactFaceMetricError> {
        Self::prepare(topology, request, Some(control))
    }

    fn prepare(
        topology: &ExactBRepTopology,
        request: &MetricFieldRequest,
        control: Option<&dyn GeometryEvaluationControl>,
    ) -> Result<Self, ExactFaceMetricError> {
        checkpoint(control)?;
        request
            .validate()
            .map_err(|error| invalid_request(error.to_string()))?;
        let resolver = ResolvedMetricField::new(request)
            .map_err(|error| invalid_request(error.to_string()))?;
        let incidence = TopologyMetricIncidence::new(topology);
        let mut resolved_by_face = resolver
            .resolve_exact_faces(topology, &incidence, control)
            .map_err(|error| {
                let kind = match &error {
                    ExactFaceMetricResolutionError::Metric(_) => {
                        ExactFaceMetricErrorKind::InvalidRequest
                    }
                    ExactFaceMetricResolutionError::Control(error) => {
                        ExactFaceMetricErrorKind::GeometryEvaluation(error.kind)
                    }
                };
                ExactFaceMetricError::new(kind, None, error.to_string())
            })?;

        let mut by_face = BTreeMap::new();
        for face in &topology.faces {
            checkpoint(control)?;
            let world_transform = topology.world_transform_for(&face.id).map_err(|error| {
                ExactFaceMetricError::new(
                    ExactFaceMetricErrorKind::InvalidRequest,
                    Some(&face.id),
                    error.to_string(),
                )
            })?;
            let sizing = resolved_by_face
                .remove(&face.id)
                .expect("resolved every exact face");
            let resolved = sizing.uniform;
            let local = sizing.local;
            let incident = incidence.incident_face_entities(face);
            // Provenance names every source that can act on the face, graded or uniform.
            let provenance_sources = resolver
                .resolve(&incident)
                .map_err(|error| invalid_request(error.to_string()))?;
            tracing::info!(
                event_code = "meshing.surface_metric.face",
                face = ?face.id,
                uniform_minimum_length_m = resolved.metric.conservative_minimum_length_m().unwrap_or(f64::NAN),
                local_sources = local.len(),
                active_sources = ?provenance_sources.active_sources,
                applied = provenance_sources.applied_contribution_count,
                "resolved face metric"
            );
            if let Some(control) = control {
                let bytes = (local.capacity() as u64)
                    .checked_mul(
                        std::mem::size_of::<runmat_meshing_size::metric::LocalMetricSource>()
                            as u64,
                    )
                    .ok_or_else(|| invalid_request("local metric source storage overflow"))?;
                control.consume_allocation_bytes(bytes).map_err(|error| {
                    ExactFaceMetricError::new(
                        ExactFaceMetricErrorKind::GeometryEvaluation(error.kind),
                        Some(&face.id),
                        error.reason,
                    )
                })?;
            }
            let provenance_bytes = super::super::ExactFaceMetricProvenance::storage_bytes(
                &face.id,
                &provenance_sources.active_sources,
            )
            .ok_or_else(|| invalid_request("face provenance storage overflow"))?;
            if let Some(control) = control {
                control
                    .consume_allocation_bytes(provenance_bytes)
                    .map_err(|error| {
                        ExactFaceMetricError::new(
                            ExactFaceMetricErrorKind::GeometryEvaluation(error.kind),
                            Some(&face.id),
                            error.reason,
                        )
                    })?;
            }
            let provenance = std::sync::Arc::new(super::super::ExactFaceMetricProvenance {
                source_face_id: face.id.clone(),
                active_sources: provenance_sources.active_sources,
                applied_contribution_count: provenance_sources.applied_contribution_count,
                clipped_contribution_count: resolved.clipped_contribution_count,
                rejected_contribution_count: resolved.rejected_contribution_count,
            });
            let boundary_metrics = super::boundary::prepare(face, &incidence, &resolver, control)?;
            by_face.insert(
                face.id.clone(),
                FaceMetricRecord {
                    surface_evaluator_id: face.surface_evaluator_id.clone(),
                    world_transform,
                    resolved,
                    local,
                    boundary_metrics,
                    provenance,
                    evaluator_parameter_domain: OnceLock::new(),
                },
            );
        }
        checkpoint(control)?;
        Ok(Self {
            by_face,
            resolver: std::sync::Arc::new(resolver),
        })
    }
}

#[track_caller]
fn checkpoint(control: Option<&dyn GeometryEvaluationControl>) -> Result<(), ExactFaceMetricError> {
    if let Some(control) = control {
        control.checkpoint().map_err(|error| {
            ExactFaceMetricError::new(
                ExactFaceMetricErrorKind::GeometryEvaluation(error.kind),
                None,
                error.reason,
            )
        })?;
    }
    Ok(())
}
