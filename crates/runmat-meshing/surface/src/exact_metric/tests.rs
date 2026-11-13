mod ownership;
use runmat_geometry_core::{
    GeometryEvaluationControl, GeometryEvaluationError, GeometryModel, GeometryTransform,
    PersistentEntityId, PersistentEntityKind, PortableExactEvaluator,
};
use runmat_meshing_core::{
    MetricCombinationRule, MetricContribution, MetricContributionScope, MetricFieldRequest,
    MetricSourceKind, MetricTensor3,
};

use super::*;

#[test]
fn exact_plane_pulls_the_resolved_world_metric_into_uv() {
    let (document, topology, registry) = runmat_geometry_fixtures::exact_circle();
    let GeometryModel::ExactBRep { model } = &document.model else {
        panic!("fixture must be exact")
    };
    let evaluator = PortableExactEvaluator::new(&registry, &topology, model).unwrap();
    let request = request(&topology.faces[0].id);
    let field = ResolvedFaceMetricField::new(&topology, &request).unwrap();

    let evaluation = field
        .evaluate(&topology.faces[0].id, [0.25, -0.5], &evaluator, &Control)
        .unwrap();

    assert_eq!(evaluation.uv, evaluation.evaluator_uv);
    assert_eq!(evaluation.point_m, [0.25, -0.5, 0.0]);
    assert_eq!(evaluation.derivative_u_m, [1.0, 0.0, 0.0]);
    assert_eq!(evaluation.derivative_v_m, [0.0, 1.0, 0.0]);
    assert_eq!(
        evaluation.physical_metric,
        ParametricMetricTensor {
            uu: 1.0,
            uv: 0.0,
            vv: 1.0,
        }
    );
    assert_eq!(
        evaluation.sizing_metric,
        ParametricMetricTensor {
            uu: 3.0,
            uv: 1.0,
            vv: 8.0,
        }
    );
    assert_eq!(
        evaluation.provenance.active_sources,
        vec![MetricSourceKind::Global, MetricSourceKind::Face]
    );
    assert_eq!(evaluation.provenance.applied_contribution_count, 1);
    assert_eq!(
        evaluation
            .sizing_metric
            .squared_length([0.5, 0.25])
            .unwrap(),
        1.5
    );
    validate_exact_face_metric_evaluation(&evaluation, &topology, &request, &evaluator, &Control)
        .unwrap();
}

#[test]
fn independent_validation_rejects_geometry_metric_and_provenance_tampering() {
    let (document, topology, registry) = runmat_geometry_fixtures::exact_circle();
    let GeometryModel::ExactBRep { model } = &document.model else {
        panic!("fixture must be exact")
    };
    let evaluator = PortableExactEvaluator::new(&registry, &topology, model).unwrap();
    let request = request(&topology.faces[0].id);
    let field = ResolvedFaceMetricField::new(&topology, &request).unwrap();
    let evaluation = field
        .evaluate(&topology.faces[0].id, [0.25, -0.5], &evaluator, &Control)
        .unwrap();

    let mut altered_geometry = evaluation.clone();
    altered_geometry.point_m[0] += 1.0;
    assert_invalid(&altered_geometry, &topology, &request, &evaluator);

    let mut altered_metric = evaluation.clone();
    altered_metric.sizing_metric.uv = 0.0;
    assert_invalid(&altered_metric, &topology, &request, &evaluator);

    let mut altered_evaluator_uv = evaluation.clone();
    altered_evaluator_uv.evaluator_uv[0] += 1.0;
    assert_invalid(&altered_evaluator_uv, &topology, &request, &evaluator);

    let mut altered_provenance = evaluation;
    std::sync::Arc::make_mut(&mut altered_provenance.provenance).applied_contribution_count = 0;
    assert_invalid(&altered_provenance, &topology, &request, &evaluator);
}

#[test]
fn occurrence_transform_is_applied_before_world_metric_pullback() {
    let (document, mut topology, registry) = runmat_geometry_fixtures::exact_circle();
    topology.instances[0].transform = GeometryTransform([
        2.0, 0.0, 0.0, 1.0, 0.0, 3.0, 0.0, -2.0, 0.0, 0.0, 1.0, 4.0, 0.0, 0.0, 0.0, 1.0,
    ]);
    let GeometryModel::ExactBRep { model } = &document.model else {
        panic!("fixture must be exact")
    };
    let evaluator = PortableExactEvaluator::new(&registry, &topology, model).unwrap();
    let request = request(&topology.faces[0].id);
    let field = ResolvedFaceMetricField::new(&topology, &request).unwrap();

    let evaluation = field
        .evaluate(&topology.faces[0].id, [0.25, -0.5], &evaluator, &Control)
        .unwrap();

    assert_eq!(evaluation.point_m, [1.5, -3.5, 4.0]);
    assert_eq!(evaluation.derivative_u_m, [2.0, 0.0, 0.0]);
    assert_eq!(evaluation.derivative_v_m, [0.0, 3.0, 0.0]);
    assert_eq!(
        evaluation.physical_metric,
        ParametricMetricTensor {
            uu: 4.0,
            uv: 0.0,
            vv: 9.0,
        }
    );
    assert_eq!(
        evaluation.sizing_metric,
        ParametricMetricTensor {
            uu: 12.0,
            uv: 6.0,
            vv: 72.0,
        }
    );
    validate_exact_face_metric_evaluation(&evaluation, &topology, &request, &evaluator, &Control)
        .unwrap();
}

#[test]
fn resolved_face_metric_rejects_unknown_scopes_and_queries() {
    let (_, topology, _) = runmat_geometry_fixtures::exact_circle();
    let unknown = id(PersistentEntityKind::Face, "unknown");
    let error = ResolvedFaceMetricField::new(&topology, &request(&unknown)).unwrap_err();
    assert_eq!(error.kind, ExactFaceMetricErrorKind::InvalidRequest);

    let (document, topology, registry) = runmat_geometry_fixtures::exact_circle();
    let GeometryModel::ExactBRep { model } = &document.model else {
        panic!("fixture must be exact")
    };
    let evaluator = PortableExactEvaluator::new(&registry, &topology, model).unwrap();
    let field = ResolvedFaceMetricField::new(&topology, &request(&topology.faces[0].id)).unwrap();
    let error = field
        .evaluate(&unknown, [0.0, 0.0], &evaluator, &Control)
        .unwrap_err();
    assert_eq!(error.kind, ExactFaceMetricErrorKind::UnknownFace);
}

fn request(face_id: &PersistentEntityId) -> MetricFieldRequest {
    MetricFieldRequest {
        entity_supports: Vec::new(),
        located_sources: Vec::new(),
        combination: MetricCombinationRule::MostRestrictiveIntersection,
        global_metric: MetricTensor3::isotropic_length_m(1.0).unwrap(),
        maximum_grading_ratio: 2.0,
        contributions: vec![MetricContribution {
            source: MetricSourceKind::Face,
            scope: MetricContributionScope::Entity {
                entity_id: face_id.clone(),
            },
            metric: MetricTensor3 {
                xx: 3.0,
                yy: 8.0,
                zz: 15.0,
                xy: 1.0,
                xz: 0.0,
                yz: 0.0,
            },
        }],
        cells_across_gap: None,
    }
}

#[test]
fn controlled_preparation_preserves_field_and_rejects_mid_inventory_cancellation() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct CancelAtFace(AtomicUsize);
    impl GeometryEvaluationControl for CancelAtFace {
        fn checkpoint(&self) -> Result<(), GeometryEvaluationError> {
            if self.0.fetch_add(1, Ordering::SeqCst) == 2 {
                return Err(GeometryEvaluationError::new(
                    runmat_geometry_core::GeometryEvaluationErrorKind::Cancelled,
                    "cancel during face resolution",
                ));
            }
            Ok(())
        }
        fn consume_iterations(&self, _: u64) -> Result<(), GeometryEvaluationError> {
            Ok(())
        }
        fn consume_search_work(&self, _: u64) -> Result<(), GeometryEvaluationError> {
            Ok(())
        }
        fn consume_allocation_bytes(&self, _: u64) -> Result<(), GeometryEvaluationError> {
            Ok(())
        }
    }
    let (_, topology, _) = runmat_geometry_fixtures::exact_circle();
    let request = request(&topology.faces[0].id);
    assert_eq!(
        ResolvedFaceMetricField::new(&topology, &request).unwrap(),
        ResolvedFaceMetricField::new_with_control(&topology, &request, &Control).unwrap(),
    );
    let control = CancelAtFace(AtomicUsize::new(0));
    let error =
        ResolvedFaceMetricField::new_with_control(&topology, &request, &control).unwrap_err();
    assert_eq!(
        error.kind,
        ExactFaceMetricErrorKind::GeometryEvaluation(
            runmat_geometry_core::GeometryEvaluationErrorKind::Cancelled,
        )
    );
    assert_eq!(control.0.load(Ordering::SeqCst), 3);
}

fn assert_invalid(
    evaluation: &ExactFaceMetricEvaluation,
    topology: &runmat_geometry_core::ExactBRepTopology,
    request: &MetricFieldRequest,
    evaluator: &PortableExactEvaluator<'_>,
) {
    let error =
        validate_exact_face_metric_evaluation(evaluation, topology, request, evaluator, &Control)
            .unwrap_err();
    assert_eq!(error.kind, ExactFaceMetricErrorKind::InvalidEvaluation);
}

struct Control;

impl GeometryEvaluationControl for Control {
    fn checkpoint(&self) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }

    fn consume_iterations(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }

    fn consume_search_work(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }

    fn consume_allocation_bytes(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }
}

fn id(kind: PersistentEntityKind, name: &str) -> PersistentEntityId {
    PersistentEntityId::new(kind, name.into(), vec!["root".into()])
}
