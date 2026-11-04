mod partial;
mod storage;
use super::*;
use crate::{
    ExactMeshingEvaluatorProvider, MeshingGeometryEvaluationControl,
    PortableMeshingEvaluatorProvider,
};
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_meshing_core::{MeshingRequestSettings, MetricTensor3, NeverCancelled};
use runmat_meshing_curve::{
    discretize_shared_curves, CurveResolutionPolicy, SharedCurveDiscretizationOptions,
    UniformCurveMetric,
};
use runmat_meshing_surface::{
    build_exact_surface_boundary, face_partition_descriptors, mesh_exact_face_partition_with_reuse,
    ExactFacePartitionContext, ExactFacePartitionOptions,
};
use std::{
    collections::BTreeMap,
    sync::{atomic::AtomicU64, Mutex},
    time::Instant,
};
#[derive(Debug, Default)]
pub(crate) struct Store(pub(crate) Mutex<BTreeMap<StableDigest, (Vec<u8>, u64)>>);
impl MeshingSubtaskCacheStore for Store {
    fn volume_implementation_identity(&self) -> Option<StableDigest> {
        Some(StableDigest::from_bytes([7; 32]))
    }
    fn implementation_identity(&self) -> StableDigest {
        StableDigest::from_bytes([6; 32])
    }
    fn maximum_entry_bytes(&self) -> u64 {
        1024 * 1024
    }
    fn read<'a>(
        &self,
        key: StableDigest,
        control: &'a dyn GeometryEvaluationControl,
    ) -> Result<Option<MeshingSubtaskCacheEntry<'a>>, MeshingSubtaskCacheError> {
        let records = self.0.lock().unwrap();
        let Some((bytes, workspace)) = records.get(&key) else {
            return Ok(None);
        };
        let storage = runmat_geometry_core::GeometryAllocationReservation::new(
            control,
            bytes.len() as u64 + workspace,
        )
        .map_err(MeshingSubtaskCacheError::Control)?;
        Ok(Some(MeshingSubtaskCacheEntry {
            bytes: bytes.clone(),
            storage,
            materialization_resident_bytes: *workspace,
        }))
    }
    fn write(
        &self,
        key: StableDigest,
        bytes: &[u8],
        workspace: u64,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<bool, MeshingSubtaskCacheError> {
        control
            .checkpoint()
            .map_err(MeshingSubtaskCacheError::Control)?;
        self.0
            .lock()
            .unwrap()
            .insert(key, (bytes.to_vec(), workspace));
        Ok(true)
    }
}
// witness: FEA-SHARED-RESOURCES-001/shared_immutable_inventories @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Fresh partition layouts reuse the same complete original faces and match uncached
//   canonical results. Original boundary coordinate and UV changes invalidate identity; unrelated
//   coordinates preserve it. This component witness does not establish native cache throughput or
//   whole-study acceptance.
// witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Fresh and cold restored faces preserve the original accepted values while streamed
//   rich-face storage releases after successful or rejected consumption. Producer dictionary and
//   decoder workspace remain charged through their actual use. This component witness does not
//   close native publication, whole-join memory, or complete-study scalability.
#[test]
fn original_faces_reuse_across_partition_layouts_with_complete_boundary_identity() {
    let ceilings = runmat_meshing_core::MeshingExecutionCeilings::PORTABLE_DEFAULT;
    let (document, topology, registry) = runmat_geometry_fixtures::exact_tetrahedron();
    let geometry = crate::prepare_exact_geometry_objects(
        document,
        topology,
        registry,
        None,
        None,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let settings = MeshingRequestSettings::default();
    let now = Instant::now();
    let control = MeshingGeometryEvaluationControl::new(
        &NeverCancelled,
        now,
        Arc::new(crate::checkpoint_clock::CheckpointClock::new(now)),
        Arc::new(AtomicU64::new(0)),
        Arc::new(AtomicU64::new(0)),
        &runmat_meshing_core::MeshingResourceBudget::new(settings.limits, ceilings),
        &settings.cancellation,
    );
    let evaluator = PortableMeshingEvaluatorProvider
        .evaluator(&geometry, &control)
        .unwrap();
    let curves = discretize_shared_curves(
        &geometry.topology,
        evaluator.as_ref(),
        evaluator.as_ref(),
        evaluator.as_ref(),
        &UniformCurveMetric::from_target_size_m(0.5).unwrap(),
        &control,
        SharedCurveDiscretizationOptions {
            resolution: CurveResolutionPolicy {
                maximum_chordal_deviation_m: runmat_types::physical::Length::new(0.01),
                maximum_tangent_change_rad: 0.2,
                minimum_metric_edge_length: 0.01,
                maximum_metric_edge_length: 1.0,
            },
            maximum_nodes_per_edge: 1024,
            maximum_subdivision_depth: 20,
            geometry_absolute_error_m: runmat_types::physical::Length::new(1e-10),
            pcurve_absolute_error: 1e-10,
            arc_length_absolute_error_m: runmat_types::physical::Length::new(1e-10),
        },
    )
    .unwrap();
    let mut request = crate::serial_tests::request();
    request.metric.global_metric = MetricTensor3::isotropic_length_m(100.0).unwrap();
    request.quality.surface.minimum_metric_edge_length = 0.001;
    request.quality.surface.minimum_metric_angle_degrees = 0.01;
    request.quality.surface.maximum_physical_aspect_ratio = 1e9;
    request.quality.surface.maximum_chordal_deviation_m = runmat_types::physical::Length::new(1e9);
    request.quality.surface.maximum_normal_deviation_degrees = 180.0;
    let store = Arc::new(Store::default());
    let mut first = SurfaceFaceReuse::new(
        store.clone(),
        &geometry,
        &request,
        &request.metric,
        &curves,
        &control,
    )
    .unwrap()
    .unwrap();
    let context = || ExactFacePartitionContext {
        topology: &geometry.topology,
        curves: &curves,
        metric_request: &request.metric,
        quality: request.quality.surface,
        evaluator: evaluator.as_ref(),
        geometry_control: &control,
        cancellation: &NeverCancelled,
    };
    for partition in face_partition_descriptors(&geometry.topology, 1).unwrap() {
        mesh_exact_face_partition_with_reuse(
            partition,
            context(),
            ExactFacePartitionOptions::default(),
            &mut storage::RepeatedRetain {
                reuse: &mut first,
                control: &control,
            },
        )
        .unwrap();
    }
    assert_eq!(first.retained(), 32);
    assert_eq!(first.hits(), 0);
    let mut second = SurfaceFaceReuse::new(
        store.clone(),
        &geometry,
        &request,
        &request.metric,
        &curves,
        &control,
    )
    .unwrap()
    .unwrap();
    let partition = face_partition_descriptors(&geometry.topology, 4)
        .unwrap()
        .remove(0);
    let actual = mesh_exact_face_partition_with_reuse(
        partition.clone(),
        context(),
        ExactFacePartitionOptions::default(),
        &mut storage::RepeatedRetain {
            reuse: &mut second,
            control: &control,
        },
    )
    .unwrap();
    let expected = runmat_meshing_surface::mesh_exact_face_partition(
        partition,
        context(),
        ExactFacePartitionOptions::default(),
    )
    .unwrap();
    assert_eq!(actual, expected);
    assert_eq!(second.hits(), 4);
    assert_eq!(second.retained(), 0);
    // A streamed batch retains only the current rich face, even on cold cache
    // restoration. Successful and failed visitors release their original charges.
    let before_stream = control.usage().live_allocation_bytes;
    let runmat_meshing_surface::ExactFacePartitionOutcome::Converged {
        faces: expected_faces,
    } = &expected.outcome
    else {
        panic!("original accepted faces");
    };
    for fail_after in [None, Some(2)] {
        let mut visited = 0;
        let streamed = runmat_meshing_surface::visit_exact_face_partition_with_reuse(
            expected.partition.clone(),
            context(),
            ExactFacePartitionOptions::default(),
            &mut second,
            &mut |accepted| {
                assert_eq!(accepted.mesh(), &expected_faces[visited]);
                visited += 1;
                if fail_after == Some(visited) {
                    return Err(ExactFacePartitionError {
                        kind: ExactFacePartitionErrorKind::Reuse,
                        reason: "visitor deliberately rejected publication".into(),
                    });
                }
                Ok(())
            },
        );
        if let Some(limit) = fail_after {
            assert!(streamed.is_err());
            assert_eq!(visited, limit);
        } else {
            assert_eq!(streamed.unwrap().accepted_face_count, expected_faces.len());
        }
        assert_eq!(
            control.usage().live_allocation_bytes,
            before_stream,
            "streamed rich meshes and all decode preparation must release on completion or failure"
        );
    }
    let completion = runmat_meshing_surface::visit_exact_face_partition(
        expected.partition.clone(),
        context(),
        ExactFacePartitionOptions::default(),
        &mut |_| Ok(()),
    )
    .unwrap();
    assert_eq!(completion.accepted_face_count, expected_faces.len());
    assert_eq!(
        control.usage().live_allocation_bytes,
        before_stream,
        "fresh production visitor releases rich point inventories too"
    );
    let boundaries = build_exact_surface_boundary(&geometry.topology, &curves).unwrap();
    let face = &boundaries.faces[0];
    let coordinates = ExactBoundaryNodeCoordinates::prepare(&curves, &control).unwrap();
    let scope = identity::scope(
        store.implementation_identity(),
        &geometry.root.digest,
        &request,
        &request.metric,
        &control,
    )
    .unwrap();
    let original = identity::face(scope, face, &coordinates, &control).unwrap();
    let mut uv = face.clone();
    uv.outer_loop.segments[0].node_uv[0][0] += 1e-7;
    assert_ne!(
        original,
        identity::face(scope, &uv, &coordinates, &control).unwrap()
    );
    let mut changed = curves.clone();
    let id = face.outer_loop.segments[0].node_ids[0];
    for edge in &mut changed.edges {
        for node in &mut edge.nodes {
            if node.node_id == id {
                {
                    let quantity_change =
                        f64::from_bits(node.coordinates_m.canonical_value()[0].to_bits() + 1);
                    let quantity_index = 0;
                    let quantity_slot = &mut node.coordinates_m;
                    let mut quantity_coordinates = quantity_slot.canonical_value();
                    quantity_coordinates[quantity_index] = quantity_change;
                    *quantity_slot = runmat_types::physical::Quantity::<
                        runmat_types::physical::kind::Position,
                        _,
                    >::new(quantity_coordinates);
                };
            }
        }
    }
    let changed_coordinates = ExactBoundaryNodeCoordinates::prepare(&changed, &control).unwrap();
    assert_ne!(
        original,
        identity::face(scope, face, &changed_coordinates, &control).unwrap()
    );
    let local_nodes = face
        .outer_loop
        .segments
        .iter()
        .flat_map(|segment| segment.node_ids)
        .collect::<std::collections::BTreeSet<_>>();
    let mut unrelated = curves.clone();
    let other = unrelated
        .edges
        .iter_mut()
        .flat_map(|edge| &mut edge.nodes)
        .find(|node| !local_nodes.contains(&node.node_id))
        .unwrap();
    let other_id = other.node_id;
    let mut point = other.coordinates_m.canonical_value();
    point[0] += 1e-8;
    for edge in &mut unrelated.edges {
        for node in &mut edge.nodes {
            if node.node_id == other_id {
                node.coordinates_m = runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new(point);
            }
        }
    }
    let unrelated_coordinates =
        ExactBoundaryNodeCoordinates::prepare(&unrelated, &control).unwrap();
    assert_eq!(
        original,
        identity::face(scope, face, &unrelated_coordinates, &control).unwrap(),
        "an unrelated face's coordinate change must not invalidate this accepted face"
    );
    let mut different = request.clone();
    different.deterministic_seed += 1;
    assert_ne!(
        scope,
        identity::scope(
            store.implementation_identity(),
            &geometry.root.digest,
            &different,
            &request.metric,
            &control
        )
        .unwrap()
    );
    partial::verify(&geometry, evaluator.as_ref(), &curves, &request, &control);
}
