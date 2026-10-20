//! Immutable independent sizing reference, prepared per validation invocation.
//! It is reconstructed from the inputs, never taken from producer evidence.
use super::{ExactFaceMetricError, ExactFaceMetricErrorKind, ExactFaceMetricEvaluation};
use crate::ExactFaceChartParameterization;
use runmat_geometry_core::{
    ExactBRepTopology, ExactSurfaceEvaluator, GeometryEvaluationControl, PersistentEntityId,
};
use runmat_meshing_core::{MetricContributionScope, MetricFieldRequest};
use runmat_meshing_size::{
    grading::grade_metric_evaluations,
    incidence::TopologyMetricIncidence,
    metric::{ResolvedMetricEvaluation, ResolvedMetricField},
};
use std::collections::BTreeMap;

pub(super) struct FaceMetricReference<'a> {
    pub face: &'a runmat_geometry_core::ExactFace,
    pub world_transform: runmat_geometry_core::GeometryTransform,
    /// Graded uniform part of the face sizing.
    pub metric: ResolvedMetricEvaluation,
    /// Entity-local sources graded from `metric` at each evaluated point.
    pub local: Vec<runmat_meshing_size::metric::LocalMetricSource>,
    /// Every source that can act on the face, uniform or graded.
    pub provenance: ResolvedMetricEvaluation,
}
pub(crate) struct PreparedFaceMetricValidation<'a> {
    reference: BTreeMap<PersistentEntityId, FaceMetricReference<'a>>,
    resolver: ResolvedMetricField,
}
impl<'a> PreparedFaceMetricValidation<'a> {
    pub(crate) fn new(
        topology: &'a ExactBRepTopology,
        request: &MetricFieldRequest,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<Self, ExactFaceMetricError> {
        let (resolver, mut resolved) = resolve_face_metrics(topology, request, control)?;
        let incidence = TopologyMetricIncidence::new(topology);
        let transforms =
            runmat_geometry_core::model::PreparedOccurrenceTransforms::new(topology, control)
                .map_err(control_error)?;
        let mut reference = BTreeMap::new();
        for face in &topology.faces {
            checkpoint(control)?;
            let world_transform = transforms
                .world_transform_for(&face.id)
                .map_err(control_error)?;
            let metric = resolved.remove(&face.id).ok_or_else(|| {
                ExactFaceMetricError::new(
                    ExactFaceMetricErrorKind::InvalidRequest,
                    Some(&face.id),
                    "independent reference missing face",
                )
            })?;
            let incident = incidence.incident_face_entities(face);
            let request_error = |error: runmat_meshing_size::metric::MetricContractError| {
                ExactFaceMetricError::new(
                    ExactFaceMetricErrorKind::InvalidRequest,
                    Some(&face.id),
                    error.to_string(),
                )
            };
            let local = resolver.local_sources(&incident).map_err(request_error)?;
            let provenance = resolver.resolve(&incident).map_err(request_error)?;
            reference.insert(
                face.id.clone(),
                FaceMetricReference {
                    face,
                    world_transform,
                    metric,
                    local,
                    provenance,
                },
            );
        }
        Ok(Self {
            reference,
            resolver,
        })
    }

    pub(crate) fn validate(
        &self,
        evaluation: &ExactFaceMetricEvaluation,
        parameterization: &ExactFaceChartParameterization,
        evaluator: &(impl ExactSurfaceEvaluator + ?Sized),
        control: &dyn GeometryEvaluationControl,
    ) -> Result<(), ExactFaceMetricError> {
        super::validate::validate_prepared_face_metric_evaluation(
            evaluation,
            &self.reference,
            &self.resolver,
            parameterization,
            evaluator,
            control,
        )
    }
}
fn resolve_face_metrics(
    topology: &ExactBRepTopology,
    request: &MetricFieldRequest,
    control: &dyn GeometryEvaluationControl,
) -> Result<
    (
        ResolvedMetricField,
        BTreeMap<PersistentEntityId, runmat_meshing_size::metric::ResolvedMetricEvaluation>,
    ),
    ExactFaceMetricError,
> {
    checkpoint(control)?;
    request.validate().map_err(|error| {
        ExactFaceMetricError::new(
            ExactFaceMetricErrorKind::InvalidRequest,
            None,
            error.to_string(),
        )
    })?;
    let resolver = ResolvedMetricField::new(request).map_err(|error| {
        ExactFaceMetricError::new(
            ExactFaceMetricErrorKind::InvalidRequest,
            None,
            error.to_string(),
        )
    })?;
    let incidence = TopologyMetricIncidence::new(topology);
    for contribution in &request.contributions {
        checkpoint(control)?;
        let entity = match &contribution.scope {
            MetricContributionScope::Region { region_id } => region_id,
            MetricContributionScope::Entity { entity_id } => entity_id,
        };
        if !incidence.knows(entity) {
            return Err(ExactFaceMetricError::new(
                ExactFaceMetricErrorKind::InvalidRequest,
                None,
                format!("metric contribution references unknown exact entity {entity:?}"),
            ));
        }
    }
    let mut by_face = topology
        .faces
        .iter()
        .map(|face| {
            checkpoint(control)?;
            resolver
                .resolve_constant(&incidence.incident_face_entities(face))
                .map(|metric| (face.id.clone(), metric))
                .map_err(|error| {
                    ExactFaceMetricError::new(
                        ExactFaceMetricErrorKind::InvalidEvaluation,
                        Some(&face.id),
                        error.to_string(),
                    )
                })
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    grade_metric_evaluations(
        request.maximum_grading_ratio,
        &TopologyMetricIncidence::face_adjacency(topology),
        &mut by_face,
    )
    .map_err(|error| {
        ExactFaceMetricError::new(
            ExactFaceMetricErrorKind::InvalidRequest,
            None,
            error.to_string(),
        )
    })?;
    Ok((resolver, by_face))
}

fn checkpoint(control: &dyn GeometryEvaluationControl) -> Result<(), ExactFaceMetricError> {
    control.checkpoint().map_err(|error| {
        ExactFaceMetricError::new(
            ExactFaceMetricErrorKind::GeometryEvaluation(error.kind),
            None,
            error.reason,
        )
    })
}

fn control_error(error: runmat_geometry_core::GeometryEvaluationError) -> ExactFaceMetricError {
    ExactFaceMetricError::new(
        ExactFaceMetricErrorKind::GeometryEvaluation(error.kind),
        None,
        error.reason,
    )
}
