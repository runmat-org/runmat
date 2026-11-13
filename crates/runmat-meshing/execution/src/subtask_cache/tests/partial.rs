//! A real face-specific size restriction forces refinement while other faces converge.
use super::*;
pub(super) fn verify(
    geometry: &crate::PreparedExactGeometryObjects,
    evaluator: &dyn crate::ExactMeshingGeometryEvaluation,
    curves: &runmat_meshing_curve::SharedCurveMesh,
    request: &runmat_meshing_core::MeshingRequest,
    control: &MeshingGeometryEvaluationControl<'_>,
) {
    use runmat_meshing_core::{MetricContribution, MetricContributionScope, MetricSourceKind};
    use runmat_meshing_surface::ExactFacePartitionOutcome;
    let mut request = request.clone();
    // Isolate one face's refinement. The normal grading ratio deliberately
    // propagates its fine size to every adjacent tetrahedron face, which would
    // leave no accepted face to exercise partial-batch retention.
    request.metric.maximum_grading_ratio = 1000.0;
    request.metric.contributions.push(MetricContribution {
        source: MetricSourceKind::Face,
        scope: MetricContributionScope::Entity {
            entity_id: geometry.topology.faces[0].id.clone(),
        },
        metric: MetricTensor3::isotropic_length_m(0.1).unwrap(),
    });
    let store = Arc::new(Store::default());
    let partition = face_partition_descriptors(&geometry.topology, 4)
        .unwrap()
        .remove(0);
    let context = || ExactFacePartitionContext {
        topology: &geometry.topology,
        curves,
        metric_request: &request.metric,
        quality: request.quality.surface,
        evaluator,
        geometry_control: control,
        cancellation: &NeverCancelled,
    };
    let mut first = SurfaceFaceReuse::new(
        store.clone(),
        geometry,
        &request,
        &request.metric,
        curves,
        control,
    )
    .unwrap()
    .unwrap();
    let outcome = mesh_exact_face_partition_with_reuse(
        partition.clone(),
        context(),
        ExactFacePartitionOptions::default(),
        &mut first,
    )
    .unwrap();
    assert!(matches!(
        outcome.outcome,
        ExactFacePartitionOutcome::RequiresCurveSplits { .. }
    ));
    assert_eq!(
        first.retained(),
        3,
        "accepted faces survive another face's split request"
    );
    drop(first);
    let mut cold =
        SurfaceFaceReuse::new(store, geometry, &request, &request.metric, curves, control)
            .unwrap()
            .unwrap();
    let replay = mesh_exact_face_partition_with_reuse(
        partition,
        context(),
        ExactFacePartitionOptions::default(),
        &mut cold,
    )
    .unwrap();
    assert_eq!(replay, outcome);
    assert_eq!(cold.hits(), 3);
    assert_eq!(cold.retained(), 0);
}
