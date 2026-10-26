//! Broad-phase completeness and work bounds, not acceptance of synthetic CAD.
use super::*;
use runmat_geometry_core::{GeometryEvaluationError, GeometryEvaluationErrorKind};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Default)]
struct Control {
    work: AtomicU64,
}
impl GeometryEvaluationControl for Control {
    fn checkpoint(&self) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }
    fn consume_search_work(&self, count: u64) -> Result<(), GeometryEvaluationError> {
        self.work.fetch_add(count, Ordering::Relaxed);
        Ok(())
    }
    fn consume_iterations(&self, _: u64) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }
    fn consume_allocation_bytes(&self, _: u64) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Face/chart-local broad-phase queries preserve exhaustive barycentric locations on
//   boundaries and near them across scale and translation. Original narrow-phase and
//   parameter-image admission remain authoritative; this is finite component equivalence evidence.
#[test]
fn local_candidates_preserve_exhaustive_boundary_and_near_boundary_locations() {
    let (_, original) = crate::cdt::constraints::tests::tetrahedron();
    for (scale, shift) in [(1.0, 0.0), (1.0e-3, 0.0), (1.0e3, 0.0), (1.0, 1.0e9)] {
        let mut surface = original.clone();
        for node in &mut surface.nodes {
            node.point_m = node.point_m.map(|coordinate| coordinate * scale + shift);
        }
        let nodes = surface
            .nodes
            .iter()
            .map(|node| (node.node_id, node))
            .collect::<BTreeMap<_, _>>();
        let control = Control::default();
        let tolerance = 1.0e-8 * scale;
        let lookup = SurfaceLookup::prepare(
            &surface,
            tolerance,
            &NodeSources::new(&surface, &control).unwrap(),
            &control,
        )
        .unwrap();
        for triangle in &surface.triangles {
            let points = triangle.node_ids.map(|id| nodes[&id].point_m);
            let owner = SurfaceUse {
                source_face_id: &triangle.source_face_id,
                chart_id: triangle.chart_id,
            };
            for first in 0..=4 {
                for second in 0..=(4 - first) {
                    let weights = [first, second, 4 - first - second].map(|v| v as f64 / 4.0);
                    let point: [f64; 3] = std::array::from_fn(|axis| {
                        (0..3)
                            .map(|corner| points[corner][axis] * weights[corner])
                            .sum()
                    });
                    for perturbation in [0.0, tolerance * 0.5, -tolerance * 0.5, tolerance * 4.0] {
                        let query = [point[0] + perturbation, point[1], point[2]];
                        let accepted = |index: usize| {
                            let triangle = &surface.triangles[index];
                            &triangle.source_face_id == owner.source_face_id && triangle.chart_id == owner.chart_id
                            && crate::cdt::solver_topology::order_elevation::geometry::triangle_coordinates(
                                triangle.node_ids.map(|id| nodes[&id].point_m), query, tolerance).is_some()
                        };
                        let expected = (0..surface.triangles.len())
                            .filter(|index| accepted(*index))
                            .collect::<std::collections::BTreeSet<_>>();
                        let observed = lookup
                            .candidates(&owner, query, &control)
                            .unwrap()
                            .into_iter()
                            .filter(|index| accepted(*index))
                            .collect::<std::collections::BTreeSet<_>>();
                        assert_eq!(observed, expected);
                    }
                }
            }
        }
    }
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: A separated spatial inventory checks bounded local query work instead of
//   complete-surface scans and observes cancellation inside tree traversal. Synthetic spatial
//   copies are not accepted CAD, native throughput, or whole-mesh qualification.
#[test]
fn localized_queries_do_not_scan_the_complete_surface_inventory_and_cancel() {
    let (_, original) = crate::cdt::constraints::tests::tetrahedron();
    let mut surface = original.clone();
    surface.nodes.clear();
    surface.triangles.clear();
    // A spatial inventory of separated triangle copies exercises the index only;
    // it is never submitted as a valid exact-geometry closure.
    for copy in 0..4096u64 {
        let mut ids = BTreeMap::new();
        for (index, original_node) in original.nodes.iter().enumerate() {
            let mut node = original_node.clone();
            let mut identity = [0u8; 32];
            identity[..8].copy_from_slice(
                &(copy * original.nodes.len() as u64 + index as u64 + 1).to_le_bytes(),
            );
            node.node_id = StableDigest::from_bytes(identity);
            node.point_m[0] += copy as f64 * 10.0;
            ids.insert(original_node.node_id, node.node_id);
            surface.nodes.push(node);
        }
        for original_triangle in &original.triangles {
            let mut triangle = original_triangle.clone();
            triangle.node_ids = triangle.node_ids.map(|id| ids[&id]);
            surface.triangles.push(triangle);
        }
    }
    let nodes = surface
        .nodes
        .iter()
        .map(|node| (node.node_id, node))
        .collect::<BTreeMap<_, _>>();
    let control = Control::default();
    let lookup = SurfaceLookup::prepare(
        &surface,
        1.0e-8,
        &NodeSources::new(&surface, &control).unwrap(),
        &control,
    )
    .unwrap();
    let selected_copy = 4095;
    let triangle = &surface.triangles[selected_copy * original.triangles.len()];
    let owner = SurfaceUse {
        source_face_id: &triangle.source_face_id,
        chart_id: triangle.chart_id,
    };
    let point = std::array::from_fn(|axis| {
        triangle
            .node_ids
            .iter()
            .map(|id| nodes[id].point_m[axis])
            .sum::<f64>()
            / 3.0
    });
    control.work.store(0, Ordering::Relaxed);
    let found = lookup.candidates(&owner, point, &control).unwrap();
    assert_eq!(found, [selected_copy * original.triangles.len()]);
    assert!(control.work.load(Ordering::Relaxed) < 200);
    struct CancelAfter(AtomicU64);
    impl GeometryEvaluationControl for CancelAfter {
        fn checkpoint(&self) -> Result<(), GeometryEvaluationError> {
            Ok(())
        }
        fn consume_search_work(&self, count: u64) -> Result<(), GeometryEvaluationError> {
            if self.0.fetch_add(count, Ordering::Relaxed) >= 10 {
                Err(GeometryEvaluationError::new(
                    GeometryEvaluationErrorKind::Cancelled,
                    "cancel inside source query",
                ))
            } else {
                Ok(())
            }
        }
        fn consume_iterations(&self, _: u64) -> Result<(), GeometryEvaluationError> {
            Ok(())
        }
        fn consume_allocation_bytes(&self, _: u64) -> Result<(), GeometryEvaluationError> {
            Ok(())
        }
    }
    let cancelled = CancelAfter(AtomicU64::new(0));
    assert!(lookup.candidates(&owner, point, &cancelled).is_err());
    assert!(cancelled.0.load(Ordering::Relaxed) > 10);
}
