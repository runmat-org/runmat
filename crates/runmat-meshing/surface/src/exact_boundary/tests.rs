use super::*;

use runmat_geometry_core::{
    GeometryEvaluationControl, GeometryEvaluationError, TopologicalOrientation,
};
use runmat_meshing_curve::{
    discretize_shared_curves, CurveResolutionPolicy, SharedCurveDiscretizationOptions,
    UniformCurveMetric,
};

#[test]
fn exact_wires_consume_the_shared_curve_by_node_identity() {
    let (topology, curves) = fixture();
    let boundary = build_exact_surface_boundary(&topology, &curves).unwrap();

    assert_eq!(boundary.faces.len(), 1);
    let segments = &boundary.faces[0].outer_loop.segments;
    assert_eq!(segments.len(), curves.edges[0].nodes.len() - 1);
    assert!(segments
        .windows(2)
        .all(|pair| pair[0].node_ids[1] == pair[1].node_ids[0]));
    assert_eq!(
        segments.last().unwrap().node_ids[1],
        segments[0].node_ids[0]
    );
    assert_eq!(segments[0].source_edge_id, topology.edges[0].id);
}

#[test]
fn coedge_orientation_reverses_nodes_and_face_local_uv_together() {
    let (mut topology, _) = fixture();
    topology.coedges[0].orientation = TopologicalOrientation::Reversed;
    let curves = curves(&topology);
    let boundary = build_exact_surface_boundary(&topology, &curves).unwrap();
    let first = &boundary.faces[0].outer_loop.segments[0];
    let curve = &curves.edges[0];
    let face_use = &curve.face_uses[0];

    assert_eq!(first.node_ids[0], curve.nodes.last().unwrap().node_id);
    assert_eq!(
        first.node_ids[1],
        curve.nodes[curve.nodes.len() - 2].node_id
    );
    assert_eq!(first.node_uv[0], *face_use.node_uv.last().unwrap());
    assert_eq!(
        first.node_uv[1],
        face_use.node_uv[face_use.node_uv.len() - 2]
    );
}

#[test]
fn independent_admission_rejects_segment_substitution() {
    let (topology, curves) = fixture();
    let mut boundary = build_exact_surface_boundary(&topology, &curves).unwrap();
    boundary.faces[0].outer_loop.segments[0].node_uv[0][0] += 1.0e-6;

    let error = validate_exact_surface_boundary(&boundary, &topology, &curves).unwrap_err();
    assert_eq!(error.kind, ExactSurfaceBoundaryErrorKind::InvalidContract);
}

#[test]
fn independent_admission_rejects_seam_image_substitution() {
    let (topology, curves) = fixture();
    let mut boundary = build_exact_surface_boundary(&topology, &curves).unwrap();
    boundary.faces[0].outer_loop.segments[0].seam_image = Some(7);

    let error = validate_exact_surface_boundary(&boundary, &topology, &curves).unwrap_err();
    assert_eq!(error.kind, ExactSurfaceBoundaryErrorKind::InvalidContract);
}

fn fixture() -> (
    runmat_geometry_core::ExactBRepTopology,
    runmat_meshing_curve::SharedCurveMesh,
) {
    let (_, topology, _) = runmat_geometry_fixtures::exact_circle();
    let curves = curves(&topology);
    (topology, curves)
}

fn curves(
    topology: &runmat_geometry_core::ExactBRepTopology,
) -> runmat_meshing_curve::SharedCurveMesh {
    let (document, _, registry) = runmat_geometry_fixtures::exact_circle();
    let runmat_geometry_core::GeometryModel::ExactBRep { model } = &document.model else {
        panic!("fixture must be exact")
    };
    let evaluator =
        runmat_geometry_core::PortableExactEvaluator::new(&registry, topology, model).unwrap();
    discretize_shared_curves(
        topology,
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
    .unwrap()
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
