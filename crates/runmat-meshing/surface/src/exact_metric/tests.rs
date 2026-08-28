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
            uu: 4.0,
            uv: 1.0,
            vv: 9.0,
        }
    );
    assert_eq!(
        evaluation.active_sources,
        vec![MetricSourceKind::Global, MetricSourceKind::Face]
    );
    assert_eq!(evaluation.applied_contribution_count, 1);
    assert_eq!(
        evaluation
            .sizing_metric
            .squared_length([0.5, 0.25])
            .unwrap(),
        1.8125
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
    altered_provenance.applied_contribution_count = 0;
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
            uu: 16.0,
            uv: 6.0,
            vv: 81.0,
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
    }
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
    PersistentEntityId {
        kind,
        source_topology_id: name.into(),
        assembly_path: vec!["root".into()],
    }
}
