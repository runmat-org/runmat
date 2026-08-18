use runmat_geometry_core::{
    ExactSurfaceDefinition, ExactSurfaceImplementation, GeometryEvaluationControl,
    GeometryEvaluationError, GeometryModel, ParameterRange, PortableExactEvaluator,
    TopologicalOrientation,
};
use runmat_meshing_core::StableDigest;

use crate::{ExactFaceBoundary, ExactFaceBoundaryLoop, ExactFaceBoundarySegment};

use super::*;

#[test]
fn periodic_seam_images_lift_into_one_canonical_chart() {
    let (document, mut topology, mut registry) = runmat_geometry_fixtures::exact_circle();
    topology.faces[0].periodic_u = true;
    registry.surfaces[0].implementation = ExactSurfaceImplementation::Portable {
        definition: ExactSurfaceDefinition::Cylinder {
            origin_m: [0.0; 3],
            x_axis: [1.0, 0.0, 0.0],
            y_axis: [0.0, 1.0, 0.0],
            axis_m_per_v: [0.0, 0.0, 1.0],
            radius_m: 1.0,
            domains: [
                range(-std::f64::consts::PI, std::f64::consts::PI),
                range(-2.0, 2.0),
            ],
        },
    };
    let GeometryModel::ExactBRep { model } = &document.model else {
        panic!("fixture must be exact")
    };
    let evaluator = PortableExactEvaluator::new(&registry, &topology, model).unwrap();
    let source = boundary(
        &topology,
        [[3.0, 0.0], [-3.0, 0.0], [-3.0, 1.0], [3.0, 1.0]],
    );

    let charts = build_exact_face_charts(
        &source,
        &topology,
        &evaluator,
        &Control,
        ExactFaceChartOptions::default(),
    )
    .unwrap();
    assert_eq!(charts.source_face_id, source.source_face_id);
    assert_eq!(charts.charts.len(), 1);
    assert_ne!(charts.charts[0].chart_id, StableDigest::ZERO);
    let chart = &charts.charts[0];
    assert!(chart
        .boundary
        .outer_loop
        .segments
        .windows(2)
        .all(|pair| pair[0].node_uv[1] == pair[1].node_uv[0]));
    assert_eq!(
        chart.boundary.outer_loop.segments.last().unwrap().node_uv[1],
        chart.boundary.outer_loop.segments[0].node_uv[0]
    );
    assert_eq!(chart.pslg.vertices.len(), 4);
    validate_exact_face_charts(
        &charts,
        &source,
        &topology,
        &evaluator,
        &Control,
        ExactFaceChartOptions::default(),
    )
    .unwrap();

    let winding = boundary(
        &topology,
        [
            [0.0, 0.0],
            [std::f64::consts::FRAC_PI_2, 0.0],
            [std::f64::consts::PI, 0.0],
            [-std::f64::consts::FRAC_PI_2, 0.0],
        ],
    );
    assert_eq!(
        build_exact_face_charts(
            &winding,
            &topology,
            &evaluator,
            &Control,
            ExactFaceChartOptions::default(),
        )
        .unwrap_err()
        .kind,
        ExactFaceChartErrorKind::RequiresMultipleCharts
    );
}

#[test]
fn nonperiodic_chart_is_identity_and_tamper_evident() {
    let (document, topology, registry) = runmat_geometry_fixtures::exact_circle();
    let GeometryModel::ExactBRep { model } = &document.model else {
        panic!("fixture must be exact")
    };
    let evaluator = PortableExactEvaluator::new(&registry, &topology, model).unwrap();
    let source = boundary(&topology, [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
    let options = ExactFaceChartOptions::default();
    let charts =
        build_exact_face_charts(&source, &topology, &evaluator, &Control, options).unwrap();
    let chart = &charts.charts[0];
    assert_eq!(chart.boundary, source);
    assert_eq!(chart.periodicity, [None, None]);

    let mut tampered = charts;
    tampered.charts[0].boundary.outer_loop.segments[0].node_uv[0][0] += 0.25;
    assert_eq!(
        validate_exact_face_charts(&tampered, &source, &topology, &evaluator, &Control, options,)
            .unwrap_err()
            .kind,
        ExactFaceChartErrorKind::InvalidInput
    );

    let invalid_options = ExactFaceChartOptions {
        maximum_charts_per_face: 0,
        ..options
    };
    assert_eq!(
        build_exact_face_charts(&source, &topology, &evaluator, &Control, invalid_options,)
            .unwrap_err()
            .kind,
        ExactFaceChartErrorKind::InvalidOptions
    );
}

fn boundary(
    topology: &runmat_geometry_core::ExactBRepTopology,
    uv: [[f64; 2]; 4],
) -> ExactFaceBoundary {
    let nodes = [node(1), node(2), node(3), node(4)];
    ExactFaceBoundary {
        source_face_id: topology.faces[0].id.clone(),
        outer_loop: ExactFaceBoundaryLoop {
            source_wire_id: topology.wires[0].id.clone(),
            orientation: TopologicalOrientation::Forward,
            segments: (0..4)
                .map(|index| ExactFaceBoundarySegment {
                    source_coedge_id: topology.coedges[0].id.clone(),
                    source_edge_id: topology.edges[0].id.clone(),
                    seam_image: Some(0),
                    node_ids: [nodes[index], nodes[(index + 1) % 4]],
                    edge_parameters: [index as f64, index as f64 + 1.0],
                    node_uv: [uv[index], uv[(index + 1) % 4]],
                })
                .collect(),
        },
        inner_loops: Vec::new(),
    }
}

fn range(start: f64, end: f64) -> ParameterRange {
    ParameterRange { start, end }
}

fn node(value: u8) -> StableDigest {
    StableDigest::from_bytes([value; 32])
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
