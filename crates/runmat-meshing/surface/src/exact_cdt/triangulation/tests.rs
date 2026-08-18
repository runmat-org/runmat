use std::sync::atomic::{AtomicBool, Ordering};

use runmat_geometry_core::{GeometryEvaluationControl, GeometryEvaluationError};
use runmat_meshing_core::{MeshingCancellationSignal, NeverCancelled};
use runmat_meshing_curve::{
    discretize_shared_curves, CurveResolutionPolicy, SharedCurveDiscretizationOptions,
    UniformCurveMetric,
};

use super::*;

#[test]
fn exact_circle_has_one_deterministic_strict_delaunay_topology() {
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

fn fixture() -> (crate::ExactFaceBoundary, crate::ExactFacePslg) {
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
        &UniformCurveMetric::from_target_size_m(0.5).unwrap(),
        &Control,
        SharedCurveDiscretizationOptions {
            resolution: CurveResolutionPolicy {
                maximum_chordal_deviation_m: 0.01,
                maximum_tangent_change_rad: 0.2,
                minimum_metric_edge_length: 0.1,
                maximum_metric_edge_length: 1.0,
            },
            maximum_nodes_per_edge: 1_024,
            maximum_subdivision_depth: 20,
            geometry_absolute_error_m: 1.0e-10,
            pcurve_absolute_error: 1.0e-10,
            arc_length_absolute_error_m: 1.0e-10,
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
