use super::*;
use std::collections::BTreeSet;

use runmat_meshing_core::contracts::{ProtectedBoundaryComplex, TopologyEntityId};
use runmat_meshing_core::quality::predicate::{
    tetrahedron_scaled_jacobian, tetrahedron_signed_volume,
};
use runmat_meshing_plc::validate::PlcValidationError;

mod fixtures;
mod recovery;

use fixtures::*;

#[test]
fn generates_positive_tetrahedra_from_validated_tetra_plc() {
    let mesh = generate_initial_tetrahedron_mesh_from_plc(&tetra_plc())
        .expect("validated tetra PLC should generate an initial Tetrahedron mesh");

    assert_eq!(mesh.nodes.len(), 5);
    assert_eq!(mesh.tetrahedron_generation_family, "initial_plc");
    assert_eq!(mesh.elements.len(), 4);
    assert_eq!(mesh.boundary_faces.len(), 4);
    assert!(!mesh.recovery_complete);
    assert!(!mesh.quality_optimized);
    assert_eq!(mesh.evidence.entity_counts["tetrahedron4_elements"], 4);
    assert_eq!(
        mesh.evidence.entity_counts["input_plc_boundary_components"],
        1
    );
    assert_eq!(mesh.evidence.entity_counts["input_plc_outer_shells"], 1);
    assert_eq!(mesh.evidence.entity_counts["input_plc_material_regions"], 1);
    assert_eq!(
        mesh.evidence.entity_counts["input_plc_material_region_facets"],
        4
    );
    assert_eq!(
        mesh.evidence.entity_counts["tetrahedron_material_regions"],
        1
    );
    assert_eq!(
        mesh.evidence.entity_counts["unclassified_tetrahedron_material_elements"],
        0
    );
    assert!(mesh
        .elements
        .iter()
        .all(|element| element.material_region_id == "body"));
    let min_scaled_jacobian = mesh.evidence.min_scaled_jacobian.expect("quality evidence");
    assert!(min_scaled_jacobian > 0.0);
    assert!((min_scaled_jacobian - min_generated_scaled_jacobian(&mesh)).abs() <= f64::EPSILON);
}

#[test]
fn initial_generation_keeps_ambiguous_material_ownership_unclassified() {
    let plc = with_split_material_ids(tetra_plc());

    let mesh = generate_initial_tetrahedron_mesh_from_plc(&plc)
        .expect("validated PLC should generate initial Tetrahedron mesh");

    assert!(mesh
        .elements
        .iter()
        .all(|element| element.material_region_id == "unclassified"));
    assert_eq!(mesh.evidence.entity_counts["input_plc_material_regions"], 2);
    assert_eq!(
        mesh.evidence.entity_counts["unclassified_tetrahedron_material_elements"],
        mesh.elements.len()
    );
}

#[test]
fn generated_tetrahedron_mesh_preserves_input_plc_cad_curve_evidence() {
    let mut plc = tetra_plc();
    plc.evidence
        .entity_counts
        .insert("cad_curve_boundary_source_edges".to_string(), 2);
    plc.evidence
        .entity_counts
        .insert("cad_curve_boundary_segments".to_string(), 3);
    plc.evidence
        .entity_counts
        .insert("cad_curve_imported_edges".to_string(), 1);
    plc.evidence
        .entity_counts
        .insert("cad_curve_evaluator_edges".to_string(), 2);
    plc.evidence
        .entity_counts
        .insert("cad_curve_evaluator_samples".to_string(), 5);
    plc.evidence
        .entity_counts
        .insert("cad_curve_live_query_edges".to_string(), 1);
    plc.evidence
        .entity_counts
        .insert("cad_curve_live_query_samples".to_string(), 4);
    plc.evidence
        .entity_counts
        .insert("cad_curve_rejected_evaluator_samples".to_string(), 1);
    plc.evidence
        .entity_counts
        .insert("cad_curve_curvature_sized_edges".to_string(), 1);
    plc.evidence
        .entity_counts
        .insert("cad_curve_curvature_samples".to_string(), 2);

    let mesh = generate_initial_tetrahedron_mesh_from_plc(&plc)
        .expect("validated PLC should generate initial Tetrahedron mesh");

    assert_eq!(
        mesh.evidence.entity_counts["input_plc_cad_curve_boundary_source_edges"],
        2
    );
    assert_eq!(
        mesh.evidence.entity_counts["input_plc_cad_curve_boundary_segments"],
        3
    );
    assert_eq!(
        mesh.evidence.entity_counts["input_plc_cad_curve_imported_edges"],
        1
    );
    assert_eq!(
        mesh.evidence.entity_counts["input_plc_cad_curve_evaluator_edges"],
        2
    );
    assert_eq!(
        mesh.evidence.entity_counts["input_plc_cad_curve_evaluator_samples"],
        5
    );
    assert_eq!(
        mesh.evidence.entity_counts["input_plc_cad_curve_live_query_edges"],
        1
    );
    assert_eq!(
        mesh.evidence.entity_counts["input_plc_cad_curve_live_query_samples"],
        4
    );
    assert_eq!(
        mesh.evidence.entity_counts["input_plc_cad_curve_rejected_evaluator_samples"],
        1
    );
    assert_eq!(
        mesh.evidence.entity_counts["input_plc_cad_curve_curvature_sized_edges"],
        1
    );
    assert_eq!(
        mesh.evidence.entity_counts["input_plc_cad_curve_curvature_samples"],
        2
    );
}

#[test]
fn rejects_unvalidated_plc_before_tetrahedron_generation() {
    let mut plc = tetra_plc();
    plc.validation.watertight = false;

    assert!(matches!(
        generate_initial_tetrahedron_mesh_from_plc(&plc),
        Err(TetrahedronGenerationError::InvalidProtectedBoundaryComplex { .. })
    ));
}

#[test]
fn rejects_nested_shell_plc_before_initial_tetrahedron_generation() {
    assert_eq!(
        generate_initial_tetrahedron_mesh_from_plc(&nested_tetrahedron_shells_plc()),
        Err(TetrahedronGenerationError::UnsupportedNestedShellPlc {
            outer_shell_count: 1,
            nested_shell_count: 1,
            max_nesting_depth: 1,
        })
    );
}

#[test]
fn rejects_nested_shell_plc_before_solver_tetrahedron_generation() {
    assert_eq!(
        generate_solver_tetrahedron_mesh_from_plc(&nested_tetrahedron_shells_plc()),
        Err(TetrahedronGenerationError::UnsupportedNestedShellPlc {
            outer_shell_count: 1,
            nested_shell_count: 1,
            max_nesting_depth: 1,
        })
    );
}

#[test]
fn rejects_degenerate_plc_facet() {
    let mut plc = tetra_plc();
    plc.nodes[2].coordinates_m = plc.nodes[1].coordinates_m;

    assert!(matches!(
        generate_initial_tetrahedron_mesh_from_plc(&plc),
        Err(
            TetrahedronGenerationError::InvalidProtectedBoundaryComplex {
                error: PlcValidationError::DegenerateFacet { .. }
            }
        )
    ));
}

#[test]
fn initial_generation_rejects_nonconvex_boundary_facet() {
    let mut plc = octahedron_plc();
    plc.nodes[1].coordinates_m = [-0.1, 0.0, 0.1];

    assert_eq!(
        generate_initial_tetrahedron_mesh_from_plc(&plc),
        Err(TetrahedronGenerationError::UnsupportedConvexPolyhedronPlc)
    );
}

#[test]
fn generates_structured_box_tetrahedra_from_validated_plc_bounds() {
    let mesh = generate_structured_box_tetrahedron_mesh_from_plc(&box_plc())
        .expect("validated box PLC should generate structured Tetrahedron mesh");

    assert_eq!(mesh.nodes.len(), 8);
    assert_eq!(mesh.tetrahedron_generation_family, "structured_box");
    assert_eq!(mesh.elements.len(), 6);
    assert_eq!(mesh.boundary_faces.len(), 12);
    assert_eq!(mesh.evidence.entity_counts["plc_boundary_nodes"], 8);
    assert!(mesh.evidence.min_scaled_jacobian.expect("quality") >= 0.15);
    for element in &mesh.elements {
        let points = element.node_ids.clone().map(|node_id| {
            mesh.nodes
                .iter()
                .find(|node| node.node_id == node_id)
                .expect("node exists")
                .coordinates_m
        });
        assert!(tetrahedron_signed_volume(points) > 0.0);
    }
    assert!(mesh.boundary_faces.iter().all(|face| {
        mesh.elements.iter().any(|element| {
            face.node_ids
                .iter()
                .all(|node_id| element.node_ids.contains(node_id))
        })
    }));
}

#[test]
fn structured_box_generation_keeps_ambiguous_material_ownership_unclassified() {
    let plc = with_split_material_ids(box_plc());

    let mesh = generate_structured_box_tetrahedron_mesh_from_plc(&plc)
        .expect("validated box PLC should generate initial Tetrahedron mesh");

    assert_eq!(mesh.elements.len(), 6);
    assert!(mesh
        .elements
        .iter()
        .all(|element| element.material_region_id == "unclassified"));
}

#[test]
fn boundary_conforming_box_generation_keeps_ambiguous_material_ownership_unclassified() {
    let plc = with_split_material_ids(split_edge_box_plc());

    let mesh = generate_structured_box_tetrahedron_mesh_from_plc(&plc)
        .expect("subdivided box PLC should generate boundary-conforming Tetrahedron mesh");

    assert_eq!(
        mesh.mesh_id,
        "structured_box_boundary_conforming_tetrahedron_mesh"
    );
    assert_eq!(
        mesh.tetrahedron_generation_family,
        "boundary_conforming_box"
    );
    assert!(mesh
        .elements
        .iter()
        .all(|element| element.material_region_id == "unclassified"));
}

#[test]
fn boundary_conforming_box_recovery_assigns_material_ownership_from_plc_facets() {
    let plc = with_split_material_ids(split_edge_box_plc());
    let mesh = generate_structured_box_tetrahedron_mesh_from_plc(&plc)
        .expect("subdivided box PLC should generate boundary-conforming Tetrahedron mesh");
    assert!(mesh
        .elements
        .iter()
        .all(|element| element.material_region_id == "unclassified"));

    let result = crate::recover::recover_tetrahedron_mesh_from_plc(&plc, mesh)
        .expect("boundary-owned PLC materials should recover generated element ownership");

    assert!(result.tetrahedron_mesh.recovery_complete);
    assert!(result
        .tetrahedron_mesh
        .elements
        .iter()
        .all(|element| element.material_region_id != "unclassified"));
    let material_region_ids = result
        .tetrahedron_mesh
        .elements
        .iter()
        .map(|element| element.material_region_id.as_str())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        material_region_ids,
        BTreeSet::from(["region_a", "region_b"])
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["boundary_owned_material_interface_recovery_input_items"],
        2
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["recovered_boundary_owned_material_interface_items"],
        2
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["repaired_material_interface_elements"],
        plc.facets.len()
    );
}

#[test]
fn structured_box_generation_preserves_split_protected_source_edges() {
    let plc = split_edge_box_plc();

    let mesh = generate_structured_box_tetrahedron_mesh_from_plc(&plc)
        .expect("subdivided box PLC should generate boundary-conforming Tetrahedron mesh");

    assert_eq!(
        mesh.mesh_id,
        "structured_box_boundary_conforming_tetrahedron_mesh"
    );
    assert_eq!(
        mesh.tetrahedron_generation_family,
        "boundary_conforming_box"
    );
    assert_eq!(mesh.nodes.len(), plc.nodes.len() + 1);
    assert_eq!(mesh.elements.len(), plc.facets.len());
    assert_eq!(mesh.boundary_faces.len(), plc.facets.len());
    assert_eq!(
        mesh.evidence.entity_counts["boundary_conforming_box_facets"],
        plc.facets.len()
    );
    for protected_edge in &plc.protected_edges {
        let recovered = mesh.boundary_faces.iter().any(|face| {
            protected_edge
                .node_ids
                .iter()
                .all(|node_id| face.node_ids.contains(node_id))
        });
        assert!(
            recovered,
            "protected edge {} should be represented by a solver boundary face",
            protected_edge.edge_id.id
        );
    }
    for facet in &plc.facets {
        assert!(mesh.boundary_faces.iter().any(|face| {
            face.source_face_id == facet.source_face_id
                && sorted_face_ids(face.node_ids.clone()) == sorted_face_ids(facet.node_ids.clone())
        }));
    }
}

#[test]
fn generates_single_tetrahedron_mesh_from_tetrahedron_plc() {
    let mesh = generate_single_tetrahedron_mesh_from_plc(&tetra_plc())
        .expect("tetrahedron PLC should generate one solver Tetrahedron4");

    assert_eq!(mesh.nodes.len(), 4);
    assert_eq!(mesh.elements.len(), 1);
    assert_eq!(mesh.boundary_faces.len(), 4);
    assert_eq!(mesh.evidence.entity_counts["tetrahedron4_elements"], 1);
    assert!(mesh.evidence.min_scaled_jacobian.expect("quality") >= 0.15);
    let element = &mesh.elements[0];
    let points = element.node_ids.clone().map(|node_id| {
        mesh.nodes
            .iter()
            .find(|node| node.node_id == node_id)
            .expect("node exists")
            .coordinates_m
    });
    assert!(tetrahedron_signed_volume(points) > 0.0);
}

#[test]
fn single_tetrahedron_generation_keeps_ambiguous_material_ownership_unclassified() {
    let plc = with_split_material_ids(tetra_plc());

    let mesh = generate_single_tetrahedron_mesh_from_plc(&plc)
        .expect("tetrahedron PLC should generate one solver Tetrahedron4");

    assert_eq!(mesh.elements[0].material_region_id, "unclassified");
}

fn sorted_face_ids(mut node_ids: [TopologyEntityId; 3]) -> [TopologyEntityId; 3] {
    node_ids.sort();
    node_ids
}

fn min_generated_scaled_jacobian(mesh: &TetrahedronMesh) -> f64 {
    mesh.elements
        .iter()
        .map(|element| {
            let points = element.node_ids.clone().map(|node_id| {
                mesh.nodes
                    .iter()
                    .find(|node| node.node_id == node_id)
                    .expect("element node exists")
                    .coordinates_m
            });
            tetrahedron_scaled_jacobian(points)
        })
        .fold(f64::INFINITY, f64::min)
}

#[test]
fn solver_generation_supports_box_and_single_tetrahedron_plcs() {
    let box_mesh = generate_solver_tetrahedron_mesh_from_plc(&box_plc())
        .expect("box PLC should use structured box solver generation");
    let tetrahedron_mesh = generate_solver_tetrahedron_mesh_from_plc(&tetra_plc())
        .expect("tetrahedron PLC should use single Tetrahedron solver generation");

    assert_eq!(box_mesh.elements.len(), 6);
    assert_eq!(box_mesh.tetrahedron_generation_family, "structured_box");
    assert_eq!(tetrahedron_mesh.elements.len(), 1);
    assert_eq!(
        tetrahedron_mesh.tetrahedron_generation_family,
        "single_tetrahedron"
    );
}

#[test]
fn generates_convex_polyhedron_tetrahedron_mesh_from_octahedron_plc() {
    let mesh = generate_convex_polyhedron_tetrahedron_mesh_from_plc(&octahedron_plc())
        .expect("convex octahedron PLC should generate one Tetrahedron4 per boundary facet");

    assert_eq!(mesh.nodes.len(), 7);
    assert_eq!(mesh.tetrahedron_generation_family, "convex_polyhedron");
    assert_eq!(mesh.elements.len(), 8);
    assert_eq!(mesh.boundary_faces.len(), 8);
    assert_eq!(mesh.evidence.entity_counts["interior_nodes"], 1);
    assert!(
        mesh.evidence
            .entity_counts
            .get("interior_smoothing_candidate_points")
            .copied()
            .unwrap_or_default()
            > 1
    );
    assert!(mesh
        .evidence
        .entity_counts
        .contains_key("interior_smoothing_accepted_points"));
    assert_eq!(mesh.evidence.entity_counts["tetrahedron4_elements"], 8);
    assert_eq!(
        mesh.evidence.entity_counts["input_plc_boundary_components"],
        1
    );
    assert_eq!(mesh.evidence.entity_counts["input_plc_outer_shells"], 1);
    assert!(mesh.evidence.min_scaled_jacobian.expect("quality") >= 0.15);
    for element in &mesh.elements {
        let points = element.node_ids.clone().map(|node_id| {
            mesh.nodes
                .iter()
                .find(|node| node.node_id == node_id)
                .expect("node exists")
                .coordinates_m
        });
        assert!(tetrahedron_signed_volume(points) > 0.0);
    }
}

#[test]
fn convex_polyhedron_generation_keeps_ambiguous_material_ownership_unclassified() {
    let plc = with_split_material_ids(octahedron_plc());

    let mesh = generate_convex_polyhedron_tetrahedron_mesh_from_plc(&plc)
        .expect("convex octahedron PLC should generate one Tetrahedron4 per boundary facet");

    assert!(mesh
        .elements
        .iter()
        .all(|element| element.material_region_id == "unclassified"));
}

#[test]
fn solver_generation_supports_convex_polyhedron_plcs() {
    let mesh = generate_solver_tetrahedron_mesh_from_plc(&octahedron_plc())
        .expect("convex octahedron PLC should use convex polyhedron solver generation");

    assert_eq!(mesh.mesh_id, "convex_polyhedron_tetrahedron_mesh");
    assert_eq!(mesh.tetrahedron_generation_family, "convex_polyhedron");
    assert_eq!(mesh.elements.len(), 8);
}

#[test]
fn generates_star_shaped_polyhedron_tetrahedron_mesh_from_dented_corner_box_plc() {
    let plc = dented_corner_box_plc();

    assert_eq!(
        generate_convex_polyhedron_tetrahedron_mesh_from_plc(&plc),
        Err(TetrahedronGenerationError::UnsupportedConvexPolyhedronPlc)
    );
    let mesh = generate_star_shaped_polyhedron_tetrahedron_mesh_from_plc(&plc)
        .expect("star-shaped dented-corner box PLC should generate a Tetrahedron mesh");

    assert_eq!(mesh.mesh_id, "star_shaped_polyhedron_tetrahedron_mesh");
    assert_eq!(mesh.tetrahedron_generation_family, "star_shaped_polyhedron");
    assert_eq!(mesh.nodes.len(), plc.nodes.len() + 1);
    assert_eq!(mesh.elements.len(), plc.facets.len());
    assert_eq!(mesh.boundary_faces.len(), plc.facets.len());
    assert_eq!(
        mesh.evidence.entity_counts["star_shaped_polyhedron_facets"],
        plc.facets.len()
    );
    assert_eq!(
        mesh.evidence.entity_counts["input_plc_facets"],
        plc.facets.len()
    );
    assert!(mesh.evidence.min_scaled_jacobian.expect("quality") > 0.0);
    for element in &mesh.elements {
        let points = element.node_ids.clone().map(|node_id| {
            mesh.nodes
                .iter()
                .find(|node| node.node_id == node_id)
                .expect("node exists")
                .coordinates_m
        });
        assert!(tetrahedron_signed_volume(points) > 0.0);
    }
}

#[test]
fn star_shaped_polyhedron_generation_keeps_ambiguous_material_ownership_unclassified() {
    let plc = with_split_material_ids(dented_corner_box_plc());

    let mesh = generate_star_shaped_polyhedron_tetrahedron_mesh_from_plc(&plc)
        .expect("star-shaped dented-corner PLC should generate a Tetrahedron mesh");

    assert!(mesh
        .elements
        .iter()
        .all(|element| element.material_region_id == "unclassified"));
    assert_eq!(mesh.evidence.entity_counts["input_plc_material_regions"], 2);
    assert_eq!(
        mesh.evidence.entity_counts["unclassified_tetrahedron_material_elements"],
        mesh.elements.len()
    );
}

#[test]
fn star_shaped_polyhedron_recovery_assigns_material_ownership_from_plc_facets() {
    let plc = with_split_material_ids(dented_corner_box_plc());
    let mesh = generate_star_shaped_polyhedron_tetrahedron_mesh_from_plc(&plc)
        .expect("star-shaped dented-corner PLC should generate a Tetrahedron mesh");
    assert!(mesh
        .elements
        .iter()
        .all(|element| element.material_region_id == "unclassified"));

    let result = crate::recover::recover_tetrahedron_mesh_from_plc(&plc, mesh)
        .expect("star-shaped PLC material ownership should recover from boundary facets");

    assert!(result.tetrahedron_mesh.recovery_complete);
    assert!(result
        .tetrahedron_mesh
        .elements
        .iter()
        .all(|element| element.material_region_id != "unclassified"));
    let material_region_ids = result
        .tetrahedron_mesh
        .elements
        .iter()
        .map(|element| element.material_region_id.as_str())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        material_region_ids,
        BTreeSet::from(["region_a", "region_b"])
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["boundary_owned_material_interface_recovery_input_items"],
        2
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["recovered_boundary_owned_material_interface_items"],
        2
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["repaired_material_interface_elements"],
        plc.facets.len()
    );
}

#[test]
fn solver_generation_supports_star_shaped_polyhedron_plcs() {
    let mesh = generate_solver_tetrahedron_mesh_from_plc(&dented_corner_box_plc())
        .expect("dented-corner box PLC should use star-shaped polyhedron generation");

    assert_eq!(mesh.mesh_id, "star_shaped_polyhedron_tetrahedron_mesh");
    assert_eq!(mesh.tetrahedron_generation_family, "star_shaped_polyhedron");
    assert_eq!(mesh.elements.len(), 12);
}

#[test]
fn solver_generation_rejects_unreferenced_plc_nodes_before_shape_selection() {
    assert!(matches!(
        generate_solver_tetrahedron_mesh_from_plc(&octahedron_with_extra_interior_node_plc()),
        Err(TetrahedronGenerationError::InvalidProtectedBoundaryComplex { .. })
    ));
}

#[test]
fn solver_generation_rejects_open_plc_even_when_summary_claims_ready() {
    let mut plc = tetra_plc();
    plc.facets.pop();

    assert!(matches!(
        generate_solver_tetrahedron_mesh_from_plc(&plc),
        Err(TetrahedronGenerationError::InvalidProtectedBoundaryComplex { .. })
    ));
}

#[test]
fn single_tetrahedron_generation_rejects_non_tetrahedron_plc() {
    assert_eq!(
        generate_single_tetrahedron_mesh_from_plc(&box_plc()),
        Err(TetrahedronGenerationError::UnsupportedSingleTetrahedronPlc)
    );
}

#[test]
fn structured_box_generation_rejects_degenerate_bounds() {
    let mut plc = tetra_plc();
    for node in &mut plc.nodes {
        node.coordinates_m[2] = 0.0;
    }

    assert!(matches!(
        generate_structured_box_tetrahedron_mesh_from_plc(&plc),
        Err(
            TetrahedronGenerationError::InvalidProtectedBoundaryComplex {
                error: PlcValidationError::DegenerateFacet { .. }
            }
        )
    ));
}

#[test]
fn structured_box_generation_rejects_non_box_plc() {
    assert_eq!(
        generate_structured_box_tetrahedron_mesh_from_plc(&tetra_plc()),
        Err(TetrahedronGenerationError::UnsupportedStructuredBoxPlc)
    );
}

#[test]
fn convex_polyhedron_generation_rejects_nonconvex_boundary_facet() {
    let mut plc = octahedron_plc();
    plc.nodes[1].coordinates_m = [-0.1, 0.0, 0.1];

    assert_eq!(
        generate_convex_polyhedron_tetrahedron_mesh_from_plc(&plc),
        Err(TetrahedronGenerationError::UnsupportedConvexPolyhedronPlc)
    );
}

fn with_split_material_ids(mut plc: ProtectedBoundaryComplex) -> ProtectedBoundaryComplex {
    let split_index = plc.facets.len() / 2;
    for (facet_index, facet) in plc.facets.iter_mut().enumerate() {
        facet.material_interface_ids = vec![if facet_index < split_index {
            "region_a".to_string()
        } else {
            "region_b".to_string()
        }];
    }
    plc
}
