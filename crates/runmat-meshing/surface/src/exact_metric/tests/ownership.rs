use super::*;
// witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Thousands of point evaluations share one immutable face identity and
//   source-provenance allocation while retaining the same numerical values. Copy-on-write tampering
//   rejects independently and leaves other evaluations intact. Sharing is not serialized proof
//   authority.
#[test]
fn face_provenance_is_shared_without_weakening_independent_tamper_rejection() {
    let (document, topology, registry) = runmat_geometry_fixtures::exact_circle();
    let GeometryModel::ExactBRep { model } = &document.model else {
        panic!("exact fixture")
    };
    let evaluator = PortableExactEvaluator::new(&registry, &topology, model).unwrap();
    let request = request(&topology.faces[0].id);
    let field = ResolvedFaceMetricField::new(&topology, &request).unwrap();
    let first = field
        .evaluate(&topology.faces[0].id, [0.1, 0.2], &evaluator, &Control)
        .unwrap();
    let second = field
        .evaluate(&topology.faces[0].id, [0.2, 0.3], &evaluator, &Control)
        .unwrap();
    assert!(std::sync::Arc::ptr_eq(
        &first.provenance,
        &second.provenance
    ));
    for _ in 0..4096 {
        let point = field
            .evaluate(&topology.faces[0].id, [0.1, 0.2], &evaluator, &Control)
            .unwrap();
        assert!(std::sync::Arc::ptr_eq(&first.provenance, &point.provenance));
        assert_eq!(first, point);
    }
    let mut altered = first.clone();
    std::sync::Arc::make_mut(&mut altered.provenance).applied_contribution_count += 1;
    assert!(validate_exact_face_metric_evaluation(
        &altered, &topology, &request, &evaluator, &Control
    )
    .is_err());
    validate_exact_face_metric_evaluation(&first, &topology, &request, &evaluator, &Control)
        .unwrap();
    validate_exact_face_metric_evaluation(&second, &topology, &request, &evaluator, &Control)
        .unwrap();
}
