use super::super::super::exact_topology_tests::{model, topology};
use super::super::tests::registry;
use super::super::*;
use super::test_support::BudgetControl;
use super::*;

#[test]
fn exact_edge_use_and_vertex_incidence_pass_adaptive_consistency() {
    let registry = registry();
    let topology = topology();
    let evaluator = PortableExactEvaluator::new(&registry, &topology, &model()).unwrap();
    evaluator
        .validate_incidence_consistency(1.0e-6, &BudgetControl::generous())
        .unwrap();
}

#[test]
fn inconsistent_curve_pcurve_surface_and_vertex_fail_closed() {
    let mut mismatched_pcurve = registry();
    let ExactPcurveImplementation::Portable {
        definition: ExactPcurveDefinition::Circle { radius_uv, .. },
    } = &mut mismatched_pcurve.pcurves[0].implementation
    else {
        panic!("fixture pcurve must be portable circle")
    };
    *radius_uv = 0.9;
    let admitted_topology = topology();
    let evaluator =
        PortableExactEvaluator::new(&mismatched_pcurve, &admitted_topology, &model()).unwrap();
    let error = evaluator
        .validate_incidence_consistency(1.0e-6, &BudgetControl::generous())
        .unwrap_err();
    assert_eq!(
        error.kind,
        GeometryEvaluationErrorKind::InconsistentGeometry
    );

    let registry = registry();
    let mut bad_vertex = topology();
    bad_vertex.vertices[0].point_m = [0.0; 3];
    let evaluator = PortableExactEvaluator::new(&registry, &bad_vertex, &model()).unwrap();
    let error = evaluator
        .validate_incidence_consistency(1.0e-6, &BudgetControl::generous())
        .unwrap_err();
    assert_eq!(
        error.kind,
        GeometryEvaluationErrorKind::InconsistentGeometry
    );
}

#[test]
fn consistency_requires_the_owning_kernel_and_obeys_budgets() {
    let mut kernel_registry = registry();
    kernel_registry.curves[0].implementation = ExactCurveImplementation::Kernel {
        reference: KernelEvaluatorRef {
            entity_token: "edge:1".into(),
            representation_digest: [8; 32],
        },
    };
    let topology = topology();
    let evaluator = PortableExactEvaluator::new(&kernel_registry, &topology, &model()).unwrap();
    let error = evaluator
        .validate_incidence_consistency(1.0e-6, &BudgetControl::generous())
        .unwrap_err();
    assert_eq!(error.kind, GeometryEvaluationErrorKind::KernelUnavailable);

    let registry = registry();
    let evaluator = PortableExactEvaluator::new(&registry, &topology, &model()).unwrap();
    let error = evaluator
        .validate_incidence_consistency(1.0e-6, &BudgetControl::with_limits(u64::MAX, 0, u64::MAX))
        .unwrap_err();
    assert_eq!(error.kind, GeometryEvaluationErrorKind::BudgetExceeded);
    let error = evaluator
        .validate_incidence_consistency(0.0, &BudgetControl::generous())
        .unwrap_err();
    assert_eq!(error.kind, GeometryEvaluationErrorKind::InvalidResult);
}
