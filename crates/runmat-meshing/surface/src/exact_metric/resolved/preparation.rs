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
            let resolved = resolved_by_face
                .remove(&face.id)
                .expect("resolved every exact face");
            let provenance_bytes = super::super::ExactFaceMetricProvenance::storage_bytes(
                &face.id,
                &resolved.active_sources,
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
                active_sources: resolved.active_sources.clone(),
                applied_contribution_count: resolved.applied_contribution_count,
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
                    boundary_metrics,
                    provenance,
                    evaluator_parameter_domain: OnceLock::new(),
                },
            );
        }
        checkpoint(control)?;
        Ok(Self { by_face })
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
