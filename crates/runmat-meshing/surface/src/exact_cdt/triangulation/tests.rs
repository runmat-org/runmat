use std::sync::atomic::{AtomicBool, Ordering};

use runmat_geometry_core::{GeometryEvaluationControl, GeometryEvaluationError};
use runmat_meshing_core::{MeshingCancellationSignal, NeverCancelled};
use runmat_meshing_curve::{
    discretize_shared_curves, CurveResolutionPolicy, SharedCurveDiscretizationOptions,
    UniformCurveMetric,
};

use super::*;

#[test]
fn exact_circle_has_one_deterministic_cocircular_topology() {
    let (boundary, pslg) = fixture();
    let options = ExactFaceDelaunayOptions::default();
    let first = triangulate_exact_face_pslg(&pslg, &boundary, &NeverCancelled, options).unwrap();
    let second = triangulate_exact_face_pslg(&pslg, &boundary, &NeverCancelled, options).unwrap();

    assert_eq!(first, second);
    assert_eq!(first.triangles.len(), pslg.vertices.len() - 2);
    validate_exact_face_delaunay(&first, &pslg, &boundary, &NeverCancelled, options).unwrap();
}

#[test]
fn construction_enforces_cancellation_and_predicate_budget() {
    let (boundary, pslg) = fixture();
    let cancelled = Cancelled(AtomicBool::new(true));
    let error = triangulate_exact_face_pslg(
        &pslg,
        &boundary,
        &cancelled,
        ExactFaceDelaunayOptions::default(),
    )
    .unwrap_err();
    assert_eq!(error.kind, ExactFaceDelaunayErrorKind::Cancelled);

    let error = triangulate_exact_face_pslg(
        &pslg,
        &boundary,
        &NeverCancelled,
        ExactFaceDelaunayOptions {
            maximum_predicate_evaluations: 1,
            ..ExactFaceDelaunayOptions::default()
        },
    )
    .unwrap_err();
    assert_eq!(error.kind, ExactFaceDelaunayErrorKind::SearchWorkLimit);
}

#[test]
fn topology_is_invariant_under_exact_power_of_two_uv_scaling() {
    let (boundary, pslg) = fixture();
    let options = ExactFaceDelaunayOptions::default();
    let reference =
        triangulate_exact_face_pslg(&pslg, &boundary, &NeverCancelled, options).unwrap();
    let mut transformed_boundary = boundary.clone();
    for segment in std::iter::once(&mut transformed_boundary.outer_loop)
        .chain(&mut transformed_boundary.inner_loops)
        .flat_map(|loop_boundary| &mut loop_boundary.segments)
    {
        for uv in &mut segment.node_uv {
            *uv = [uv[0] * 2.0, uv[1] * 2.0];
        }
    }
    let mut transformed_pslg = pslg.clone();
    for vertex in &mut transformed_pslg.vertices {
        vertex.uv = [vertex.uv[0] * 2.0, vertex.uv[1] * 2.0];
    }
    let transformed = triangulate_exact_face_pslg(
        &transformed_pslg,
        &transformed_boundary,
        &NeverCancelled,
        options,
    )
    .unwrap();

    assert_eq!(transformed.triangles, reference.triangles);
}

#[test]
fn independent_validation_rejects_orientation_tampering() {
    let (boundary, pslg) = fixture();
    let options = ExactFaceDelaunayOptions::default();
    let mut triangulation =
        triangulate_exact_face_pslg(&pslg, &boundary, &NeverCancelled, options).unwrap();
    triangulation.triangles[0].vertex_indices.swap(1, 2);

    let error =
        validate_exact_face_delaunay(&triangulation, &pslg, &boundary, &NeverCancelled, options)
            .unwrap_err();
    assert_eq!(error.kind, ExactFaceDelaunayErrorKind::InvalidTopology);
}

pub(super) fn fixture() -> (crate::ExactFaceBoundary, crate::ExactFacePslg) {
    let (document, topology, registry) = runmat_geometry_fixtures::exact_circle();
    let runmat_geometry_core::GeometryModel::ExactBRep { model } = &document.model else {
        panic!("fixture must be exact")
    };
    let evaluator =
        runmat_geometry_core::PortableExactEvaluator::new(&registry, &topology, model).unwrap();
    let curves = discretize_shared_curves(
        &topology,
        &evaluator,
        &evaluator,
        &evaluator,
        &UniformCurveMetric::from_target_size_m(0.5).unwrap(),
        &Control,
        SharedCurveDiscretizationOptions {
            resolution: CurveResolutionPolicy {
                maximum_chordal_deviation_m: runmat_types::physical::Length::new(0.01),
                maximum_tangent_change_rad: 0.2,
                minimum_metric_edge_length: 0.1,
                maximum_metric_edge_length: 1.0,
            },
            maximum_nodes_per_edge: 1_024,
            maximum_subdivision_depth: 20,
            geometry_absolute_error_m: runmat_types::physical::Length::new(1.0e-10),
            pcurve_absolute_error: 1.0e-10,
            arc_length_absolute_error_m: runmat_types::physical::Length::new(1.0e-10),
        },
    )
    .unwrap();
    let surface_boundary = crate::build_exact_surface_boundary(&topology, &curves).unwrap();
    let boundary = surface_boundary.faces[0].clone();
    let pslg = crate::build_exact_face_pslg(&boundary).unwrap();
    (boundary, pslg)
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

struct Cancelled(AtomicBool);

impl MeshingCancellationSignal for Cancelled {
    fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

// witness: FEA-MESH-DISCRETIZATION-001/surface_discretization @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
//   boundary: Regression: a point just inside a hull edge keeps the hull edge in the exact face
//   triangulation; the dense-surface and many-trim cases are separate tests.
#[test]
fn a_point_just_inside_a_hull_edge_keeps_the_hull_edge() {
    let (_, pslg) = fixture();
    let segment = pslg.segments[0].vertex_indices;
    let [a, b] = segment.map(|index| pslg.vertices[index as usize].uv);
    let centroid = pslg.vertices.iter().fold([0.0, 0.0], |sum, vertex| {
        [
            sum[0] + vertex.uv[0] / pslg.vertices.len() as f64,
            sum[1] + vertex.uv[1] / pslg.vertices.len() as f64,
        ]
    });
    let middle = [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5];
    // A millionth of the edge length inside: the sliver's circumcircle is far larger than the
    // finite super triangle.
    let inward = [centroid[0] - middle[0], centroid[1] - middle[1]];
    let length = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
    let scale = 1.0e-6 * length / (inward[0].powi(2) + inward[1].powi(2)).sqrt();
    let uv = [middle[0] + inward[0] * scale, middle[1] + inward[1] * scale];
    let addition = crate::ExactFacePslgVertex {
        node_id: runmat_meshing_core::StableDigest::from_bytes([7; 32]),
        seam_image: None,
        uv,
    };
    let (refined, _) = super::super::pslg::insert_pslg_vertices(&pslg, &[addition]).unwrap();
    let options = ExactFaceDelaunayOptions::default();
    let triangulation =
        triangulate_validated_face_pslg(&refined, &NeverCancelled, options).unwrap();
    let hull_edge = {
        let mut edge = refined.segments[0].vertex_indices;
        edge.sort_unstable();
        edge
    };
    assert!(triangulation.triangles.iter().any(|triangle| {
        let [x, y, z] = triangle.vertex_indices;
        [[x, y], [y, z], [z, x]].into_iter().any(|mut edge| {
            edge.sort_unstable();
            edge == hull_edge
        })
    }));
    // Every original vertex is on the hull; the new one is interior.
    assert_eq!(triangulation.triangles.len(), refined.vertices.len() - 1);
}
