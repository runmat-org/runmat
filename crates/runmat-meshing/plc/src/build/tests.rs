use super::*;
use crate::validate::PlcValidationError;
use runmat_meshing_core::curve::CurveValidationReport;
use runmat_meshing_core::surface::{
    SurfaceDiscretization, SurfaceElement, SurfaceLoopCoverageReport, SurfaceNode,
    INTERNAL_SOURCE_EDGE_ID,
};

#[test]
fn builds_valid_plc_from_closed_tetra_surface() {
    let plc = build_protected_boundary_complex(&tetra_surface())
        .expect("closed tetra surface should build a PLC");

    assert!(plc.validation.valid_for_volume_meshing());
    assert_eq!(plc.nodes.len(), 4);
    assert_eq!(plc.facets.len(), 4);
    assert_eq!(plc.protected_edges.len(), 6);
    assert_eq!(plc.evidence.entity_counts["facets"], 4);
    assert_eq!(plc.evidence.entity_counts["validated_curve_elements"], 6);
    assert_eq!(plc.evidence.entity_counts["surface_boundary_loops"], 4);
    assert_eq!(
        plc.evidence.entity_counts["recovered_surface_source_edges"],
        6
    );
    assert_eq!(plc.evidence.entity_counts["boundary_components"], 1);
    assert_eq!(
        plc.evidence.entity_counts["max_boundary_component_nodes"],
        4
    );
    assert_eq!(plc.evidence.entity_counts["shell_nesting_classified"], 1);
    assert_eq!(plc.evidence.entity_counts["outer_shells"], 1);
    assert_eq!(plc.evidence.entity_counts["nested_shells"], 0);
}

#[test]
fn rejects_surface_with_protected_edges_without_curve_boundary_evidence() {
    let mut surface = tetra_surface();
    surface.curve_boundary_validation = None;

    assert_eq!(
        build_protected_boundary_complex(&surface),
        Err(PlcBuildError::MissingCurveBoundaryValidation)
    );
}

#[test]
fn rejects_surface_with_protected_edges_without_loop_coverage_evidence() {
    let mut surface = tetra_surface();
    surface.loop_coverage = None;

    assert_eq!(
        build_protected_boundary_complex(&surface),
        Err(PlcBuildError::MissingSurfaceLoopCoverage)
    );
}

#[test]
fn rejects_surface_with_inconsistent_loop_coverage_evidence() {
    let mut surface = tetra_surface();
    let mut loop_coverage = loop_coverage();
    loop_coverage.recovered_source_edge_count = 5;
    surface.loop_coverage = Some(loop_coverage);

    assert_eq!(
        build_protected_boundary_complex(&surface),
        Err(PlcBuildError::InconsistentSurfaceLoopCoverage {
            recovered_face_count: 4,
            surface_source_face_count: 4,
            recovered_source_edge_count: 5,
            protected_source_edge_count: 6,
        })
    );
}

#[test]
fn rejects_open_surface_before_volume_meshing() {
    let mut surface = tetra_surface();
    surface.elements.pop();

    let err =
        build_protected_boundary_complex(&surface).expect_err("open surface must not become a PLC");

    assert!(matches!(err, PlcBuildError::OpenBoundaryEdge { .. }));
}

#[test]
fn rejects_nonmanifold_surface_edge_before_volume_meshing() {
    let err = build_protected_boundary_complex(&edge_shared_tetrahedra_surface())
        .expect_err("nonmanifold edge incidence must not become a PLC");

    assert!(matches!(err, PlcBuildError::NonManifoldBoundaryEdge { .. }));
}

#[test]
fn rejects_duplicate_surface_facets() {
    let mut surface = tetra_surface();
    surface.elements[1] = surface.elements[0].clone();
    surface.elements[1].element_id = 99;

    assert_eq!(
        build_protected_boundary_complex(&surface),
        Err(PlcBuildError::DuplicateFacet { element_id: 99 })
    );
}

#[test]
fn rejects_inconsistent_surface_orientation_before_returning_plc() {
    let mut surface = tetra_surface();
    surface.elements[0].node_ids = [0, 1, 2];

    assert!(matches!(
        build_protected_boundary_complex(&surface),
        Err(PlcBuildError::ProtectedBoundaryValidation(
            PlcValidationError::InconsistentBoundaryEdgeOrientation { .. }
                | PlcValidationError::DuplicateProtectedBoundarySegment { .. }
        ))
    ));
}

fn tetra_surface() -> SurfaceDiscretization {
    SurfaceDiscretization {
        nodes: vec![
            node(0, [0.0, 0.0, 0.0]),
            node(1, [1.0, 0.0, 0.0]),
            node(2, [0.0, 1.0, 0.0]),
            node(3, [0.0, 0.0, 1.0]),
        ],
        elements: vec![
            element(0, [0, 2, 1], [2, 1, 0]),
            element(1, [0, 1, 3], [0, 4, 3]),
            element(2, [1, 2, 3], [1, 5, 4]),
            element(3, [2, 0, 3], [2, 3, 5]),
        ],
        curve_boundary_validation: Some(CurveValidationReport {
            source_edge_count: 6,
            curve_node_count: 12,
            curve_element_count: 6,
            max_endpoint_error_m: 0.0,
            max_projection_error_m: 0.0,
            max_length_error_m: 0.0,
            max_segment_length_m: 1.0,
            max_adjacent_length_ratio: 1.0,
        }),
        loop_coverage: Some(loop_coverage()),
        exact_cad_sample_node_count: 0,
        rejected_exact_cad_sample_count: 0,
    }
}

fn edge_shared_tetrahedra_surface() -> SurfaceDiscretization {
    SurfaceDiscretization {
        nodes: vec![
            node(0, [0.0, 0.0, 0.0]),
            node(1, [1.0, 0.0, 0.0]),
            node(2, [0.0, 1.0, 0.0]),
            node(3, [0.0, 0.0, 1.0]),
            node(4, [0.0, -1.0, 0.0]),
            node(5, [0.0, 0.0, -1.0]),
        ],
        elements: vec![
            element(0, [0, 2, 1], internal_source_edges()),
            element(1, [0, 1, 3], internal_source_edges()),
            element(2, [1, 2, 3], internal_source_edges()),
            element(3, [2, 0, 3], internal_source_edges()),
            element(4, [0, 1, 4], internal_source_edges()),
            element(5, [0, 5, 1], internal_source_edges()),
            element(6, [1, 5, 4], internal_source_edges()),
            element(7, [5, 0, 4], internal_source_edges()),
        ],
        curve_boundary_validation: None,
        loop_coverage: None,
        exact_cad_sample_node_count: 0,
        rejected_exact_cad_sample_count: 0,
    }
}

fn internal_source_edges() -> [u32; 3] {
    [
        INTERNAL_SOURCE_EDGE_ID,
        INTERNAL_SOURCE_EDGE_ID,
        INTERNAL_SOURCE_EDGE_ID,
    ]
}

fn loop_coverage() -> SurfaceLoopCoverageReport {
    SurfaceLoopCoverageReport {
        source_face_count: 4,
        recovered_face_count: 4,
        boundary_loop_count: 4,
        recovered_source_edge_count: 6,
        boundary_segment_count: 12,
        max_loops_per_face: 1,
    }
}

fn node(node_id: u32, coordinates_m: [f64; 3]) -> SurfaceNode {
    SurfaceNode {
        node_id,
        source_vertex_id: node_id,
        coordinates_m,
    }
}

fn element(element_id: u32, node_ids: [u32; 3], source_edge_ids: [u32; 3]) -> SurfaceElement {
    SurfaceElement {
        element_id,
        source_face_id: element_id,
        cad_face_id: None,
        source_edge_ids,
        node_ids,
        parametric_node_uv: [[0.0, 0.0]; 3],
        max_projection_error_m: 0.0,
        region_ids: vec!["body".to_string()],
        area_m2: 0.5,
        unit_normal: [0.0, 0.0, 1.0],
    }
}
