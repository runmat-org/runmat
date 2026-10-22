use super::*;

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Off-edge size-only refinement preserves a sizing-valid exact curve, including
//   reversed uses. Geometric-error candidates, on-edge points and oversized protected edges still
//   require shared splits. Final face quality and complete CAD conformity require independent
//   checks.
#[test]
fn sized_curve_allows_off_edge_size_refinement_but_retains_required_shared_splits() {
    let (document, topology, registry) = runmat_geometry_fixtures::exact_circle();
    let GeometryModel::ExactBRep { model } = &document.model else {
        panic!("exact fixture")
    };
    let evaluator = PortableExactEvaluator::new(&registry, &topology, model).unwrap();
    let face = topology.faces[0].id.clone();
    let mut pslg = one_segment_pslg(face.clone());
    pslg.vertices[0].uv = [-0.25, 0.0];
    pslg.vertices[1].uv = [0.25, 0.0];
    let mut point = candidate(face, [0.0, 0.1]);
    let classify = |point: &ExactFaceRefinementCandidate, pslg: &ExactFacePslg| {
        classify_exact_face_refinement_candidate(
            point,
            pslg,
            &topology,
            &metric_request(),
            &evaluator,
            &Control,
        )
        .unwrap()
    };
    assert_eq!(
        classify(&point, &pslg),
        ExactFaceCandidateDisposition::Insert
    );
    pslg.segments[0].vertex_indices.reverse();
    assert_eq!(
        classify(&point, &pslg),
        ExactFaceCandidateDisposition::Insert
    );
    for reason in [
        ExactFaceRefinementReason::ChordalDeviation,
        ExactFaceRefinementReason::NormalDeviation,
    ] {
        point.reason = reason;
        assert!(matches!(
            classify(&point, &pslg),
            ExactFaceCandidateDisposition::SplitProtectedSegment(_)
        ));
    }
    point.reason = ExactFaceRefinementReason::MetricEdgeLength;
    point.uv = [0.0, 0.0];
    assert!(matches!(
        classify(&point, &pslg),
        ExactFaceCandidateDisposition::SplitProtectedSegment(_)
    ));
    point.uv = [0.0, 0.1];
    pslg.vertices[0].uv = [-1.0, 0.0];
    pslg.vertices[1].uv = [1.0, 0.0];
    assert!(matches!(
        classify(&point, &pslg),
        ExactFaceCandidateDisposition::SplitProtectedSegment(_)
    ));
}
