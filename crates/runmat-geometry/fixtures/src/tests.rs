use super::*;

struct UnlimitedEvaluation;

impl GeometryEvaluationControl for UnlimitedEvaluation {
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

#[test]
fn exact_circle_is_a_complete_authoritative_fixture() {
    let (document, topology, evaluators) = exact_circle();
    document.validate().unwrap();
    let GeometryModel::ExactBRep { model } = &document.model else {
        panic!("exact-circle fixture must remain exact")
    };
    topology.validate_against(model).unwrap();
    evaluators.validate_against(&topology, model).unwrap();
}

#[test]
fn exact_singular_edge_is_a_complete_authoritative_fixture() {
    let (document, topology, evaluators) = exact_singular_edge();
    document.validate().unwrap();
    let GeometryModel::ExactBRep { model } = &document.model else {
        panic!("singular-edge fixture must remain exact")
    };
    topology.validate_against(model).unwrap();
    evaluators.validate_against(&topology, model).unwrap();
    let evaluator = PortableExactEvaluator::new(&evaluators, &topology, model).unwrap();
    evaluator
        .validate_incidence_consistency(1.0e-8, &UnlimitedEvaluation)
        .unwrap();
    let curve_id = &topology.edges[0].curve_evaluator_id;
    let domain = ExactCurveEvaluator::parameter_range(&evaluator, curve_id).unwrap();
    let control = UnlimitedEvaluation;
    let derivatives =
        ExactCurveEvaluator::derivatives(&evaluator, curve_id, 1.25, &control).unwrap();
    assert_eq!(derivatives.point_m, [0.0, 0.0, 1.0]);
    assert_eq!(derivatives.first_m, [0.0; 3]);
    assert_eq!(derivatives.second_m, [0.0; 3]);
    assert_eq!(
        evaluator
            .arc_length_m(curve_id, domain, 1.0e-12, &control)
            .unwrap(),
        0.0
    );
    let projection = evaluator
        .inverse_project(curve_id, [0.0, 3.0, 1.0], 1.0e-12, &control)
        .unwrap();
    assert_eq!(projection.parameter, domain.start);
    assert_eq!(projection.point_m, [0.0, 0.0, 1.0]);
    assert_eq!(projection.distance_m, 3.0);
    assert_eq!(
        evaluator
            .unit_tangent(curve_id, 1.25, &control)
            .unwrap_err()
            .kind,
        GeometryEvaluationErrorKind::InvalidResult
    );
    assert_eq!(
        evaluator
            .curvature_1_per_m(curve_id, 1.25, &control)
            .unwrap_err()
            .kind,
        GeometryEvaluationErrorKind::InvalidResult
    );
}

#[test]
fn exact_spherical_octant_is_a_complete_authoritative_fixture() {
    let (document, topology, evaluators) = exact_spherical_octant();
    document.validate().unwrap();
    let GeometryModel::ExactBRep { model } = &document.model else {
        panic!("spherical-octant fixture must remain exact")
    };
    topology.validate_against(model).unwrap();
    topology.validate_solid_shell_boundaries().unwrap();
    evaluators.validate_against(&topology, model).unwrap();
    let evaluator = PortableExactEvaluator::new(&evaluators, &topology, model).unwrap();
    evaluator
        .validate_incidence_consistency(1.0e-8, &UnlimitedEvaluation)
        .unwrap();
    assert!(topology.bodies[0].is_sheet_body);
    assert!(topology.faces[0].has_singularity);
}
