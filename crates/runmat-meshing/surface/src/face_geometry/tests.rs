use runmat_geometry_core::{
    GeometryEvaluationControl, GeometryEvaluationError, GeometryEvaluationErrorKind, GeometryModel,
    GeometryTransform, PortableExactEvaluator,
};
use runmat_meshing_core::{
    MetricCombinationRule, MetricFieldRequest, MetricTensor3, NeverCancelled,
};
use runmat_meshing_curve::{
    discretize_shared_curves, CurveResolutionPolicy, SharedCurveDiscretizationOptions,
    UniformCurveMetric,
};

use super::*;

#[test]
fn exact_plane_geometry_measures_metric_and_physical_triangle_quality() {
    let fixture = fixture(None);
    let evaluator = evaluator(&fixture);
    let geometry = evaluate_exact_face_geometry(
        &fixture.trimmed,
        &fixture.pslg,
        &fixture.topology,
        &request(),
        &evaluator,
        &Control,
    )
    .unwrap();

    assert_eq!(geometry.vertices.len(), fixture.pslg.vertices.len());
    assert_eq!(geometry.triangles.len(), fixture.trimmed.triangles.len());
    assert!(geometry.maximum_metric_edge_length > 0.0);
    assert!(geometry.minimum_metric_angle_rad > 0.0);
    assert!(geometry.maximum_physical_aspect_ratio >= 1.0);
    assert!(geometry.maximum_chordal_deviation_m < 1.0e-14);
    assert_eq!(geometry.maximum_normal_deviation_rad, 0.0);
    assert!(geometry
        .vertices
        .iter()
        .all(|vertex| vertex.unit_normal == [0.0, 0.0, 1.0]));
    validate_exact_face_geometry(
        &geometry,
        &fixture.trimmed,
        &fixture.pslg,
        &fixture.topology,
        &request(),
        &evaluator,
        &Control,
    )
    .unwrap();
}

#[test]
fn occurrence_scaling_is_reflected_in_physical_and_metric_edges() {
    let transform = GeometryTransform([
        2.0, 0.0, 0.0, 1.0, 0.0, 3.0, 0.0, -2.0, 0.0, 0.0, 1.0, 4.0, 0.0, 0.0, 0.0, 1.0,
    ]);
    let fixture = fixture(Some(transform));
    let evaluator = evaluator(&fixture);
    let geometry = evaluate_exact_face_geometry(
        &fixture.trimmed,
        &fixture.pslg,
        &fixture.topology,
        &request(),
        &evaluator,
        &Control,
    )
    .unwrap();

    let triangle = &geometry.triangles[0];
    let corners = triangle
        .triangle
        .vertex_indices
        .map(|index| geometry.vertices[index as usize].evaluation.point_m);
    let physical = [
        distance(corners[0], corners[1]),
        distance(corners[1], corners[2]),
        distance(corners[2], corners[0]),
    ];
    for (metric, physical) in triangle.metric_edge_lengths.iter().zip(physical) {
        assert!((metric - physical).abs() < 1.0e-12);
    }
    validate_exact_face_geometry(
        &geometry,
        &fixture.trimmed,
        &fixture.pslg,
        &fixture.topology,
        &request(),
        &evaluator,
        &Control,
    )
    .unwrap();
}

#[test]
fn independent_validation_rejects_triangle_and_summary_tampering() {
    let fixture = fixture(None);
    let evaluator = evaluator(&fixture);
    let geometry = evaluate_exact_face_geometry(
        &fixture.trimmed,
        &fixture.pslg,
        &fixture.topology,
        &request(),
        &evaluator,
        &Control,
    )
    .unwrap();
    let validate = |candidate: &ExactFaceGeometry| {
        validate_exact_face_geometry(
            candidate,
            &fixture.trimmed,
            &fixture.pslg,
            &fixture.topology,
            &request(),
            &evaluator,
            &Control,
        )
    };

    let mut altered_triangle = geometry.clone();
    altered_triangle.triangles[0].chordal_deviation_m = 1.0;
    assert_eq!(
        validate(&altered_triangle).unwrap_err().kind,
        ExactFaceGeometryErrorKind::InvalidEvaluation
    );

    let mut altered_summary = geometry;
    altered_summary.maximum_metric_edge_length += 1.0;
    assert_eq!(
        validate(&altered_summary).unwrap_err().kind,
        ExactFaceGeometryErrorKind::InvalidEvaluation
    );
}

#[test]
fn evaluation_rejects_invalid_topology_and_preserves_cancellation() {
    let fixture = fixture(None);
    let evaluator = evaluator(&fixture);
    let mut invalid = fixture.trimmed.clone();
    invalid.triangles[0].vertex_indices[0] = fixture.pslg.vertices.len() as u32;
    let error = evaluate_exact_face_geometry(
        &invalid,
        &fixture.pslg,
        &fixture.topology,
        &request(),
        &evaluator,
        &Control,
    )
    .unwrap_err();
    assert_eq!(error.kind, ExactFaceGeometryErrorKind::InvalidInput);

    let error = evaluate_exact_face_geometry(
        &fixture.trimmed,
        &fixture.pslg,
        &fixture.topology,
        &request(),
        &evaluator,
        &CancelledControl,
    )
    .unwrap_err();
    assert_eq!(
        error.kind,
        ExactFaceGeometryErrorKind::Metric(crate::ExactFaceMetricErrorKind::GeometryEvaluation(
            GeometryEvaluationErrorKind::Cancelled
        ))
    );
}

struct Fixture {
    document: runmat_geometry_core::GeometryDocument,
    topology: runmat_geometry_core::ExactBRepTopology,
    registry: runmat_geometry_core::ExactEvaluatorRegistry,
    pslg: crate::ExactFacePslg,
    trimmed: crate::ExactFaceTrimmedDelaunay,
}

fn fixture(transform: Option<GeometryTransform>) -> Fixture {
    let (document, mut topology, registry) = runmat_geometry_fixtures::exact_circle();
    if let Some(transform) = transform {
        topology.instances[0].transform = transform;
    }
    let GeometryModel::ExactBRep { model } = &document.model else {
        panic!("fixture must be exact")
    };
    let evaluator = PortableExactEvaluator::new(&registry, &topology, model).unwrap();
    let curves = discretize_shared_curves(
        &topology,
        &evaluator,
        &evaluator,
        &UniformCurveMetric::from_target_size_m(0.5).unwrap(),
        &Control,
        curve_options(),
    )
    .unwrap();
    let boundary = crate::build_exact_surface_boundary(&topology, &curves)
        .unwrap()
        .faces
        .remove(0);
    let pslg = crate::build_exact_face_pslg(&boundary).unwrap();
    let options = crate::ExactFaceDelaunayOptions::default();
    let delaunay =
        crate::triangulate_exact_face_pslg(&pslg, &boundary, &NeverCancelled, options).unwrap();
    let constrained =
        crate::recover_exact_face_segments(&delaunay, &pslg, &boundary, &NeverCancelled, options)
            .unwrap();
    let trimmed =
        crate::carve_exact_face_domain(&constrained, &pslg, &boundary, &NeverCancelled, options)
            .unwrap();
    Fixture {
        document,
        topology,
        registry,
        pslg,
        trimmed,
    }
}

fn evaluator(fixture: &Fixture) -> PortableExactEvaluator<'_> {
    let GeometryModel::ExactBRep { model } = &fixture.document.model else {
        panic!("fixture must be exact")
    };
    PortableExactEvaluator::new(&fixture.registry, &fixture.topology, model).unwrap()
}

fn request() -> MetricFieldRequest {
    MetricFieldRequest {
        combination: MetricCombinationRule::MostRestrictiveIntersection,
        global_metric: MetricTensor3::isotropic_length_m(1.0).unwrap(),
        maximum_grading_ratio: 2.0,
        contributions: Vec::new(),
    }
}

fn curve_options() -> SharedCurveDiscretizationOptions {
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
    }
}

fn distance(left: [f64; 3], right: [f64; 3]) -> f64 {
    left.into_iter()
        .zip(right)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt()
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

struct CancelledControl;

impl GeometryEvaluationControl for CancelledControl {
    fn checkpoint(&self) -> Result<(), GeometryEvaluationError> {
        Err(GeometryEvaluationError::new(
            GeometryEvaluationErrorKind::Cancelled,
            "cancelled",
        ))
    }

    fn consume_iterations(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
        self.checkpoint()
    }

    fn consume_search_work(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
        self.checkpoint()
    }

    fn consume_allocation_bytes(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
        self.checkpoint()
    }
}
