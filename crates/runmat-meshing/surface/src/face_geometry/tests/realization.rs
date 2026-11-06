use super::*;
use crate::{
    ExactBoundaryNodeCoordinates, ExactFaceChartParameterization, ExactFaceRefinedMesh,
    ExactFaceRefinedTopology, ExactFaceRefinementContext,
};

// witness: FEA-SHARED-RESOURCES-001/governed_path_equivalence @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Original shared curve coordinates define realized boundary nodes while exact surface
//   samples remain unchanged. Independent admission measures their discrepancy against the original
//   chordal envelope; fixed coordinates outside that envelope fail before futile interior
//   insertion. This kernel witness does not establish native CAD or whole-study qualification.
#[test]
fn boundary_realization_preserves_original_surface_samples_and_geometric_acceptance() {
    for transform in [
        None,
        Some(GeometryTransform([
            2.0, 0.0, 0.0, 1.0, 0.0, 3.0, 0.0, -2.0, 0.0, 0.0, 1.0, 4.0, 0.0, 0.0, 0.0, 1.0,
        ])),
    ] {
        let fixture = fixture(transform);
        let evaluator = evaluator(&fixture);
        let mut coordinates = fixture.curves.clone();
        let offset = 1e-4;
        for edge in &mut coordinates.edges {
            for node in &mut edge.nodes {
                {
                    let quantity_change = offset;
                    let quantity_index = 2;
                    let quantity_slot = &mut node.coordinates_m;
                    let mut quantity_coordinates = quantity_slot.canonical_value();
                    quantity_coordinates[quantity_index] += quantity_change;
                    *quantity_slot = runmat_types::physical::Quantity::<
                        runmat_types::physical::kind::Position,
                        _,
                    >::new(quantity_coordinates);
                };
            }
        }
        let authority = ExactBoundaryNodeCoordinates::prepare(&coordinates, &Control).unwrap();
        let request = request();
        let parameterization = ExactFaceChartParameterization::EvaluatorParameters;
        let context =
            ExactFaceGeometryContext::new(&fixture.topology, &request, &evaluator, &Control)
                .with_boundary_coordinates(&authority);
        let geometry = evaluate_exact_face_geometry_in_parameterization(
            &fixture.trimmed,
            &fixture.pslg,
            &parameterization,
            context,
        )
        .unwrap();
        assert!(geometry
            .vertices
            .iter()
            .all(|vertex| vertex.point_m[2] == vertex.evaluation.point_m[2] + offset));
        assert!((geometry.maximum_chordal_deviation_m.canonical_value() - offset).abs() < 1e-14);
        validate_exact_face_geometry_in_parameterization(
            &geometry,
            &fixture.trimmed,
            &fixture.pslg,
            &parameterization,
            context,
        )
        .unwrap();
        let mut altered = geometry.clone();
        altered.vertices[0].point_m[2] += offset;
        assert!(validate_exact_face_geometry_in_parameterization(
            &altered,
            &fixture.trimmed,
            &fixture.pslg,
            &parameterization,
            context
        )
        .is_err());
        let quality = runmat_meshing_core::SurfaceQualityTargets {
            minimum_metric_edge_length: 0.01,
            minimum_metric_angle_degrees: 0.1,
            maximum_physical_aspect_ratio: 1000.0,
            maximum_chordal_deviation_m: runmat_types::physical::Length::new(offset * 2.0),
            maximum_normal_deviation_degrees: 180.0,
        };
        let collars =
            crate::derive_exact_face_feature_collars(&fixture.pslg, &geometry, quality).unwrap();
        let mesh = ExactFaceRefinedMesh {
            topology: ExactFaceRefinedTopology {
                pslg: fixture.pslg.clone(),
                constrained: fixture.constrained.clone(),
                trimmed: fixture.trimmed.clone(),
            },
            geometry,
            feature_collars: collars,
            interior_insertion_count: 0,
        };
        let context = ExactFaceRefinementContext::new(
            &fixture.topology,
            &request,
            &evaluator,
            &Control,
            &NeverCancelled,
        )
        .with_boundary_coordinates(&authority);
        let report =
            crate::accept_exact_face_mesh(&mesh, context, quality, Default::default()).unwrap();
        assert!((report.maximum_chordal_deviation_m.canonical_value() - offset).abs() < 1e-14);
        let strict = runmat_meshing_core::SurfaceQualityTargets {
            maximum_chordal_deviation_m: runmat_types::physical::Length::new(offset * 0.5),
            ..quality
        };
        let policy = crate::ExactFaceRefinementPolicy {
            quality: strict,
            acceptance: Default::default(),
            delaunay: Default::default(),
            refinement: crate::ExactFaceRefinementOptions {
                maximum_interior_insertions: 1,
            },
        };
        let boundary =
            crate::build_exact_surface_boundary(&fixture.topology, &fixture.curves).unwrap();
        let face = boundary
            .faces
            .iter()
            .find(|face| face.source_face_id == mesh.topology.pslg.source_face_id)
            .unwrap();
        assert_eq!(
            crate::refine_exact_face_until_blocked(face, &mesh.topology, context, policy)
                .unwrap_err()
                .kind,
            crate::ExactFaceRefinementErrorKind::InvalidQuality
        );
        assert_eq!(
            crate::accept_exact_face_mesh(&mesh, context, strict, Default::default())
                .unwrap_err()
                .kind,
            crate::ExactFaceAcceptanceErrorKind::UnsatisfiedQuality
        );
    }
}

// witness: FEA-SHARED-RESOURCES-001/shared_immutable_inventories @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Conflicting coordinates for one original shared identity, cancelled preparation and
//   denied allocation reject. Borrowed coordinate authority does not clone curve inventories; this
//   is component admission evidence, not measured whole-worker memory.
#[test]
fn boundary_coordinate_index_rejects_conflicting_shared_identities_and_denied_preparation() {
    let fixture = fixture(None);
    let mut curves = fixture.curves.clone();
    let mut extra = curves.edges[0].clone();
    {
        let quantity_change = 1.0;
        let quantity_index = 2;
        let quantity_slot = &mut extra.nodes[0].coordinates_m;
        let mut quantity_coordinates = quantity_slot.canonical_value();
        quantity_coordinates[quantity_index] += quantity_change;
        *quantity_slot = runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Position,
            _,
        >::new(quantity_coordinates);
    };
    curves.edges.push(extra);
    assert!(ExactBoundaryNodeCoordinates::prepare(&curves, &Control).is_err());
    assert!(ExactBoundaryNodeCoordinates::prepare(&fixture.curves, &CancelledControl).is_err());
    struct Denied;
    impl GeometryEvaluationControl for Denied {
        fn checkpoint(&self) -> Result<(), GeometryEvaluationError> {
            Ok(())
        }
        fn consume_iterations(&self, _: u64) -> Result<(), GeometryEvaluationError> {
            Ok(())
        }
        fn consume_search_work(&self, _: u64) -> Result<(), GeometryEvaluationError> {
            Ok(())
        }
        fn consume_allocation_bytes(&self, _: u64) -> Result<(), GeometryEvaluationError> {
            Err(GeometryEvaluationError::new(
                GeometryEvaluationErrorKind::AllocationBudgetExceeded,
                "denied",
            ))
        }
    }
    assert!(matches!(
        ExactBoundaryNodeCoordinates::prepare(&fixture.curves, &Denied),
        Err(GeometryEvaluationError {
            kind: GeometryEvaluationErrorKind::AllocationBudgetExceeded,
            ..
        })
    ));
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: A multi-run borrowed boundary index preserves every original point, charges pointer
//   inventory and bounded cooperative sorting, and rejects denied search work. This finite fixture
//   does not prove arbitrary native model throughput.
#[test]
fn large_borrowed_boundary_index_preserves_every_point_and_bounds_sort_preparation() {
    use std::sync::atomic::{AtomicU64, Ordering};
    let fixture = fixture(None);
    let evaluator = evaluator(&fixture);
    let curves = discretize_shared_curves(
        &fixture.topology,
        &evaluator,
        &evaluator,
        &UniformCurveMetric::from_target_size_m(0.02).unwrap(),
        &Control,
        curve_options(),
    )
    .unwrap();
    let count = curves
        .edges
        .iter()
        .map(|edge| edge.nodes.len())
        .sum::<usize>();
    assert!(count > 256);
    struct Budget {
        search: AtomicU64,
        storage: AtomicU64,
        limit: u64,
    }
    impl GeometryEvaluationControl for Budget {
        fn checkpoint(&self) -> Result<(), GeometryEvaluationError> {
            Ok(())
        }
        fn consume_iterations(&self, _: u64) -> Result<(), GeometryEvaluationError> {
            Ok(())
        }
        fn consume_allocation_bytes(&self, bytes: u64) -> Result<(), GeometryEvaluationError> {
            self.storage.fetch_add(bytes, Ordering::Relaxed);
            Ok(())
        }
        fn consume_search_work(&self, work: u64) -> Result<(), GeometryEvaluationError> {
            if self
                .search
                .fetch_add(work, Ordering::Relaxed)
                .saturating_add(work)
                > self.limit
            {
                Err(GeometryEvaluationError::new(
                    GeometryEvaluationErrorKind::SearchWorkBudgetExceeded,
                    "sort work denied",
                ))
            } else {
                Ok(())
            }
        }
    }
    let control = Budget {
        search: AtomicU64::new(0),
        storage: AtomicU64::new(0),
        limit: u64::MAX,
    };
    let authority = ExactBoundaryNodeCoordinates::prepare(&curves, &control).unwrap();
    assert_eq!(
        control.storage.load(Ordering::Relaxed),
        (count * 2 * std::mem::size_of::<&runmat_meshing_curve::SharedCurveNode>()) as u64
    );
    for edge in &curves.edges {
        for node in &edge.nodes {
            assert_eq!(
                authority.point(node.node_id, &control).unwrap(),
                Some(node.coordinates_m.canonical_value())
            );
        }
    }
    let denied = Budget {
        search: AtomicU64::new(0),
        storage: AtomicU64::new(0),
        limit: count as u64,
    };
    assert!(matches!(
        ExactBoundaryNodeCoordinates::prepare(&curves, &denied),
        Err(GeometryEvaluationError {
            kind: GeometryEvaluationErrorKind::SearchWorkBudgetExceeded,
            ..
        })
    ));
}
