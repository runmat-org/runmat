use super::*;

use runmat_geometry_core::{GeometryEvaluationControl, GeometryEvaluationError};
use runmat_meshing_curve::{
    discretize_shared_curves, CurveResolutionPolicy, SharedCurveDiscretizationOptions,
    UniformCurveMetric,
};

#[test]
fn planar_exact_face_builds_a_canonical_closed_pslg() {
    let (topology, curves) = fixture();
    let boundary = crate::build_exact_surface_boundary(&topology, &curves).unwrap();
    let pslg = build_exact_face_pslg(&boundary.faces[0]).unwrap();

    assert_eq!(pslg.loops.len(), 1);
    assert_eq!(pslg.segments.len(), curves.edges[0].nodes.len() - 1);
    assert!(pslg
        .segments
        .windows(2)
        .all(|pair| pair[0].vertex_indices[1] == pair[1].vertex_indices[0]));
    assert_eq!(
        pslg.segments.last().unwrap().vertex_indices[1],
        pslg.segments[0].vertex_indices[0]
    );
    assert_eq!(
        pslg.segments[0].edge_parameters,
        boundary.faces[0].outer_loop.segments[0].edge_parameters
    );
}

#[test]
fn independent_pslg_admission_rejects_edge_parameter_substitution() {
    let (topology, curves) = fixture();
    let boundary = crate::build_exact_surface_boundary(&topology, &curves).unwrap();
    let mut pslg = build_exact_face_pslg(&boundary.faces[0]).unwrap();
    pslg.segments[0].edge_parameters[1] += 1.0e-6;

    let error = validate_exact_face_pslg(&pslg, &boundary.faces[0]).unwrap_err();
    assert_eq!(error.kind, ExactFacePslgErrorKind::InvalidBoundary);
}

#[test]
fn distinct_uv_images_of_one_node_require_explicit_chart_work() {
    let (topology, curves) = fixture();
    let mut boundary = crate::build_exact_surface_boundary(&topology, &curves).unwrap();
    let last = boundary.faces[0].outer_loop.segments.len() - 1;
    boundary.faces[0].outer_loop.segments[last].node_uv[1][0] += 1.0;

    let error = build_exact_face_pslg(&boundary.faces[0]).unwrap_err();
    assert_eq!(error.kind, ExactFacePslgErrorKind::InvalidTopology);
}

fn fixture() -> (
    runmat_geometry_core::ExactBRepTopology,
    runmat_meshing_curve::SharedCurveMesh,
) {
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
    (topology, curves)
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
