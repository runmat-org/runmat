use runmat_meshing_cad::{SourceTopologyEdge, SourceTopologyModel, SourceTopologyVertex};

use crate::{
    discretize_topology_curves, validate_curve_discretization, CurveDiscretization,
    CurveDiscretizationOptions, CurveElement, CurveNode, CurveValidationError,
    CurveValidationOptions,
};

#[test]
fn validates_recovered_curve_endpoints_and_growth_evidence() {
    let topology = line_topology(1.0);
    let curves = discretize_topology_curves(
        &topology,
        CurveDiscretizationOptions {
            target_size_m: 0.25,
            min_segments_per_edge: 1,
            max_segments_per_edge: 16,
        },
    )
    .expect("curves should discretize");

    let report =
        validate_curve_discretization(&topology, &curves, CurveValidationOptions::default())
            .expect("curve discretization should validate");

    assert_eq!(report.source_edge_count, 1);
    assert_eq!(report.curve_node_count, 5);
    assert_eq!(report.curve_element_count, 4);
    assert_eq!(report.max_endpoint_error_m, 0.0);
    assert_eq!(report.max_projection_error_m, 0.0);
    assert_eq!(report.max_length_error_m, 0.0);
    assert_eq!(report.max_adjacent_length_ratio, 1.0);
    assert_eq!(report.max_parameter_gap, 0.0);
    assert!(report.max_segment_length_m <= 0.250000000001);
}

#[test]
fn rejects_curve_endpoint_drift_before_surface_meshing() {
    let topology = line_topology(1.0);
    let curves = CurveDiscretization {
        nodes: vec![
            CurveNode {
                node_id: 0,
                source_edge_id: 0,
                parameter: 0.0,
                coordinates_m: [0.01, 0.0, 0.0],
            },
            CurveNode {
                node_id: 1,
                source_edge_id: 0,
                parameter: 1.0,
                coordinates_m: [1.0, 0.0, 0.0],
            },
        ],
        elements: vec![CurveElement {
            element_id: 0,
            source_edge_id: 0,
            node_ids: [0, 1],
            length_m: 0.99,
        }],
    };

    let err = validate_curve_discretization(
        &topology,
        &curves,
        CurveValidationOptions {
            max_endpoint_error_m: 1.0e-4,
            max_projection_error_m: 1.0,
            max_growth_ratio: 2.0,
            ..CurveValidationOptions::default()
        },
    )
    .expect_err("endpoint drift should fail");

    assert!(matches!(
        err,
        CurveValidationError::EndpointDrift {
            source_edge_id: 0,
            parameter: 0.0,
            ..
        }
    ));
}

#[test]
fn rejects_curve_node_projection_drift_before_surface_meshing() {
    let topology = line_topology(1.0);
    let curves = CurveDiscretization {
        nodes: vec![
            CurveNode {
                node_id: 0,
                source_edge_id: 0,
                parameter: 0.0,
                coordinates_m: [0.0, 0.0, 0.0],
            },
            CurveNode {
                node_id: 1,
                source_edge_id: 0,
                parameter: 0.5,
                coordinates_m: [0.5, 0.01, 0.0],
            },
            CurveNode {
                node_id: 2,
                source_edge_id: 0,
                parameter: 1.0,
                coordinates_m: [1.0, 0.0, 0.0],
            },
        ],
        elements: vec![
            CurveElement {
                element_id: 0,
                source_edge_id: 0,
                node_ids: [0, 1],
                length_m: (0.5_f64.powi(2) + 0.01_f64.powi(2)).sqrt(),
            },
            CurveElement {
                element_id: 1,
                source_edge_id: 0,
                node_ids: [1, 2],
                length_m: (0.5_f64.powi(2) + 0.01_f64.powi(2)).sqrt(),
            },
        ],
    };

    let err = validate_curve_discretization(
        &topology,
        &curves,
        CurveValidationOptions {
            max_projection_error_m: 1.0e-4,
            ..CurveValidationOptions::default()
        },
    )
    .expect_err("interior curve node drift should fail");

    assert!(matches!(
        err,
        CurveValidationError::NodeProjectionDrift {
            node_id: 1,
            source_edge_id: 0,
            ..
        }
    ));
}

#[test]
fn rejects_curve_element_length_mismatch() {
    let topology = line_topology(1.0);
    let curves = CurveDiscretization {
        nodes: vec![
            CurveNode {
                node_id: 0,
                source_edge_id: 0,
                parameter: 0.0,
                coordinates_m: [0.0, 0.0, 0.0],
            },
            CurveNode {
                node_id: 1,
                source_edge_id: 0,
                parameter: 1.0,
                coordinates_m: [1.0, 0.0, 0.0],
            },
        ],
        elements: vec![CurveElement {
            element_id: 0,
            source_edge_id: 0,
            node_ids: [0, 1],
            length_m: 0.75,
        }],
    };

    let err = validate_curve_discretization(
        &topology,
        &curves,
        CurveValidationOptions {
            max_length_error_m: 1.0e-4,
            ..CurveValidationOptions::default()
        },
    )
    .expect_err("reported element length must match node coordinates");

    assert!(matches!(
        err,
        CurveValidationError::ElementLengthMismatch {
            element_id: 0,
            reported_length_m: 0.75,
            measured_length_m: 1.0,
            ..
        }
    ));
}

#[test]
fn rejects_excessive_adjacent_curve_growth() {
    let topology = line_topology(1.0);
    let curves = CurveDiscretization {
        nodes: vec![
            CurveNode {
                node_id: 0,
                source_edge_id: 0,
                parameter: 0.0,
                coordinates_m: [0.0, 0.0, 0.0],
            },
            CurveNode {
                node_id: 1,
                source_edge_id: 0,
                parameter: 0.1,
                coordinates_m: [0.1, 0.0, 0.0],
            },
            CurveNode {
                node_id: 2,
                source_edge_id: 0,
                parameter: 1.0,
                coordinates_m: [1.0, 0.0, 0.0],
            },
        ],
        elements: vec![
            CurveElement {
                element_id: 0,
                source_edge_id: 0,
                node_ids: [0, 1],
                length_m: 0.1,
            },
            CurveElement {
                element_id: 1,
                source_edge_id: 0,
                node_ids: [1, 2],
                length_m: 0.9,
            },
        ],
    };

    let err = validate_curve_discretization(
        &topology,
        &curves,
        CurveValidationOptions {
            max_endpoint_error_m: 1.0e-8,
            max_growth_ratio: 2.0,
            ..CurveValidationOptions::default()
        },
    )
    .expect_err("adjacent growth should fail");

    assert!(matches!(
        err,
        CurveValidationError::ExcessiveGrowth {
            source_edge_id: 0,
            left_element_id: 0,
            right_element_id: 1,
            ..
        }
    ));
}

#[test]
fn rejects_curve_parameter_chain_gaps_before_surface_meshing() {
    let topology = line_topology(1.0);
    let curves = CurveDiscretization {
        nodes: vec![
            CurveNode {
                node_id: 0,
                source_edge_id: 0,
                parameter: 0.0,
                coordinates_m: [0.0, 0.0, 0.0],
            },
            CurveNode {
                node_id: 1,
                source_edge_id: 0,
                parameter: 0.4,
                coordinates_m: [0.4, 0.0, 0.0],
            },
            CurveNode {
                node_id: 2,
                source_edge_id: 0,
                parameter: 0.6,
                coordinates_m: [0.6, 0.0, 0.0],
            },
            CurveNode {
                node_id: 3,
                source_edge_id: 0,
                parameter: 1.0,
                coordinates_m: [1.0, 0.0, 0.0],
            },
        ],
        elements: vec![
            CurveElement {
                element_id: 0,
                source_edge_id: 0,
                node_ids: [0, 1],
                length_m: 0.4,
            },
            CurveElement {
                element_id: 1,
                source_edge_id: 0,
                node_ids: [2, 3],
                length_m: 0.4,
            },
        ],
    };

    let err = validate_curve_discretization(
        &topology,
        &curves,
        CurveValidationOptions {
            max_growth_ratio: 2.0,
            ..CurveValidationOptions::default()
        },
    )
    .expect_err("parameter chain gaps should fail");

    assert!(matches!(
        err,
        CurveValidationError::ElementParameterGap {
            source_edge_id: 0,
            left_element_id: Some(0),
            right_element_id: Some(1),
            ..
        }
    ));
}

#[test]
fn rejects_curve_parameter_chain_overlap_before_surface_meshing() {
    let topology = line_topology(1.0);
    let curves = CurveDiscretization {
        nodes: vec![
            CurveNode {
                node_id: 0,
                source_edge_id: 0,
                parameter: 0.0,
                coordinates_m: [0.0, 0.0, 0.0],
            },
            CurveNode {
                node_id: 1,
                source_edge_id: 0,
                parameter: 0.7,
                coordinates_m: [0.7, 0.0, 0.0],
            },
            CurveNode {
                node_id: 2,
                source_edge_id: 0,
                parameter: 0.5,
                coordinates_m: [0.5, 0.0, 0.0],
            },
            CurveNode {
                node_id: 3,
                source_edge_id: 0,
                parameter: 1.0,
                coordinates_m: [1.0, 0.0, 0.0],
            },
        ],
        elements: vec![
            CurveElement {
                element_id: 0,
                source_edge_id: 0,
                node_ids: [0, 1],
                length_m: 0.7,
            },
            CurveElement {
                element_id: 1,
                source_edge_id: 0,
                node_ids: [2, 3],
                length_m: 0.5,
            },
        ],
    };

    let err = validate_curve_discretization(
        &topology,
        &curves,
        CurveValidationOptions {
            max_growth_ratio: 2.0,
            ..CurveValidationOptions::default()
        },
    )
    .expect_err("parameter chain overlaps should fail");

    assert!(matches!(
        err,
        CurveValidationError::ElementParameterOverlap {
            source_edge_id: 0,
            left_element_id: 0,
            right_element_id: 1,
            ..
        }
    ));
}

#[test]
fn rejects_reversed_curve_elements_before_surface_meshing() {
    let topology = line_topology(1.0);
    let curves = CurveDiscretization {
        nodes: vec![
            CurveNode {
                node_id: 0,
                source_edge_id: 0,
                parameter: 0.0,
                coordinates_m: [0.0, 0.0, 0.0],
            },
            CurveNode {
                node_id: 1,
                source_edge_id: 0,
                parameter: 0.5,
                coordinates_m: [0.5, 0.0, 0.0],
            },
            CurveNode {
                node_id: 2,
                source_edge_id: 0,
                parameter: 1.0,
                coordinates_m: [1.0, 0.0, 0.0],
            },
        ],
        elements: vec![
            CurveElement {
                element_id: 0,
                source_edge_id: 0,
                node_ids: [0, 2],
                length_m: 1.0,
            },
            CurveElement {
                element_id: 1,
                source_edge_id: 0,
                node_ids: [1, 0],
                length_m: 0.5,
            },
        ],
    };

    let err = validate_curve_discretization(
        &topology,
        &curves,
        CurveValidationOptions {
            max_growth_ratio: 2.0,
            ..CurveValidationOptions::default()
        },
    )
    .expect_err("reversed curve elements should fail");

    assert!(matches!(
        err,
        CurveValidationError::NonIncreasingElementParameter {
            source_edge_id: 0,
            element_id: 1,
            ..
        }
    ));
}

fn line_topology(length_m: f64) -> SourceTopologyModel {
    SourceTopologyModel {
        mesh_id: "line".to_string(),
        source_geometry_id: "generic-line".to_string(),
        source_geometry_revision: 1,
        source_geometry_sha256: None,
        vertices: vec![
            SourceTopologyVertex {
                vertex_id: 0,
                coordinates_m: [0.0, 0.0, 0.0],
            },
            SourceTopologyVertex {
                vertex_id: 1,
                coordinates_m: [length_m, 0.0, 0.0],
            },
        ],
        edges: vec![SourceTopologyEdge {
            edge_id: 0,
            node_ids: [0, 1],
            adjacent_face_ids: vec![0, 1],
            region_ids: vec!["edge".to_string()],
            length_m,
        }],
        faces: Vec::new(),
        bounds_min_m: [0.0, 0.0, 0.0],
        bounds_max_m: [length_m, 0.0, 0.0],
        region_ids: vec!["edge".to_string()],
    }
}
