use super::*;

// witness: FEA-SHARED-RESOURCES-001/governed_path_equivalence @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: A size-refinement midpoint close to a nonincident vertex is replaced by a bounded
//   separated point in the same triangle before boundary classification. An infeasible spacing
//   request remains rejected; original final quality checks still apply.
#[test]
fn size_refinement_separates_the_midpoint_from_a_nonincident_node() {
    let mut geometry = geometry([2.0, 1.0, 1.0], 0.5, 1.0, 0.0, 0.0);
    let mut pslg = triangle_pslg(&geometry);
    let midpoint = [1.0, 0.0];
    let targets = SurfaceQualityTargets {
        minimum_metric_edge_length: 0.1,
        ..quality()
    };
    pslg.vertices.push(ExactFacePslgVertex {
        node_id: node(99),
        seam_image: None,
        uv: [1.0, 0.01],
    });
    let extra_evaluation = evaluation(&geometry.source_face_id, [1.0, 0.01]);
    geometry.vertices.push(ExactFaceGeometryVertex {
        pslg_vertex_index: 3,
        point_m: extra_evaluation.point_m,
        evaluation: extra_evaluation,
        unit_normal: [0.0, 0.0, 1.0],
    });
    let candidate = select_exact_face_refinement_candidate(
        &geometry,
        &pslg,
        &empty_collars(&geometry),
        targets,
    )
    .unwrap()
    .unwrap();
    assert_eq!(
        candidate.reason,
        ExactFaceRefinementReason::MetricEdgeLength
    );
    assert_ne!(candidate.uv, midpoint);
    let metric = geometry.triangles[0].deviation_samples[3].sizing_metric;
    assert!(pslg
        .vertices
        .iter()
        .all(
            |vertex| super::super::select::metric_distance(candidate.uv, vertex.uv, metric)
                >= targets.minimum_metric_edge_length
        ));
    let selected = super::super::select::independent_candidate_batch(
        vec![candidate],
        &geometry,
        &pslg,
        targets.minimum_metric_edge_length,
        None,
    )
    .unwrap();
    assert_eq!(selected.len(), 1);
    assert!(super::super::select::separate_refinement_candidate(
        midpoint, 0, &geometry, &pslg, 100.0, None,
    )
    .unwrap()
    .is_none());
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Candidate separation observes cancellation inside the original node inventory,
//   retains only one best candidate, and propagates a typed failure before mutation. This component
//   check does not measure arbitrary CAD throughput.
#[test]
fn separated_candidate_observes_cancellation_inside_the_node_inventory() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct CancelAfterEight(AtomicUsize);
    impl GeometryEvaluationControl for CancelAfterEight {
        fn checkpoint(&self) -> Result<(), GeometryEvaluationError> {
            if self.0.fetch_add(1, Ordering::Relaxed) >= 8 {
                Err(GeometryEvaluationError::new(
                    runmat_geometry_core::GeometryEvaluationErrorKind::Cancelled,
                    "cancel inside candidate node walk",
                ))
            } else {
                Ok(())
            }
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
    let geometry = geometry([2.0, 1.0, 1.0], 0.5, 1.0, 0.0, 0.0);
    let mut pslg = triangle_pslg(&geometry);
    for index in 0..32u8 {
        pslg.vertices.push(ExactFacePslgVertex {
            node_id: node(100 + index),
            seam_image: None,
            uv: [100.0 + index as f64, 100.0],
        });
    }
    let control = CancelAfterEight(AtomicUsize::new(0));
    let failure = super::super::select::separate_refinement_candidate(
        [1.0, 0.0],
        0,
        &geometry,
        &pslg,
        0.1,
        Some(&control),
    )
    .unwrap_err();
    assert_eq!(
        failure.kind,
        ExactFaceRefinementErrorKind::Metric(crate::ExactFaceMetricErrorKind::GeometryEvaluation(
            runmat_geometry_core::GeometryEvaluationErrorKind::Cancelled
        ))
    );
    assert_eq!(control.0.load(Ordering::Relaxed), 9);
}
