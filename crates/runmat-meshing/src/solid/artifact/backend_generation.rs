use runmat_meshing_core::contracts::MeshBackendSummary;
use runmat_meshing_tetrahedron::generate::TetrahedronMesh;

use super::backend_counts::tetrahedron_entity_count;

pub(super) fn plc_input_and_generation_summary(
    tetrahedron_mesh: &TetrahedronMesh,
    base: MeshBackendSummary,
) -> MeshBackendSummary {
    MeshBackendSummary {
        plc_input_node_count: tetrahedron_entity_count(tetrahedron_mesh, "input_plc_nodes"),
        plc_input_facet_count: tetrahedron_entity_count(tetrahedron_mesh, "input_plc_facets"),
        plc_input_protected_edge_count: tetrahedron_entity_count(
            tetrahedron_mesh,
            "input_plc_protected_edges",
        ),
        plc_input_boundary_component_count: tetrahedron_entity_count(
            tetrahedron_mesh,
            "input_plc_boundary_components",
        ),
        plc_input_boundary_component_node_count: tetrahedron_entity_count(
            tetrahedron_mesh,
            "input_plc_boundary_component_nodes",
        ),
        plc_input_max_boundary_component_node_count: tetrahedron_entity_count(
            tetrahedron_mesh,
            "input_plc_max_boundary_component_nodes",
        ),
        plc_input_shell_nesting_classified: tetrahedron_entity_count(
            tetrahedron_mesh,
            "input_plc_shell_nesting_classified",
        ) > 0,
        plc_input_outer_shell_count: tetrahedron_entity_count(
            tetrahedron_mesh,
            "input_plc_outer_shells",
        ),
        plc_input_nested_shell_count: tetrahedron_entity_count(
            tetrahedron_mesh,
            "input_plc_nested_shells",
        ),
        plc_input_max_shell_nesting_depth: tetrahedron_entity_count(
            tetrahedron_mesh,
            "input_plc_max_shell_nesting_depth",
        ),
        plc_input_material_region_count: tetrahedron_entity_count(
            tetrahedron_mesh,
            "input_plc_material_regions",
        ),
        plc_input_material_region_facet_count: tetrahedron_entity_count(
            tetrahedron_mesh,
            "input_plc_material_region_facets",
        ),
        plc_input_cad_curve_boundary_source_edge_count: tetrahedron_entity_count(
            tetrahedron_mesh,
            "input_plc_cad_curve_boundary_source_edges",
        ),
        plc_input_cad_curve_boundary_segment_count: tetrahedron_entity_count(
            tetrahedron_mesh,
            "input_plc_cad_curve_boundary_segments",
        ),
        plc_input_cad_curve_imported_edge_count: tetrahedron_entity_count(
            tetrahedron_mesh,
            "input_plc_cad_curve_imported_edges",
        ),
        plc_input_cad_curve_evaluator_edge_count: tetrahedron_entity_count(
            tetrahedron_mesh,
            "input_plc_cad_curve_evaluator_edges",
        ),
        plc_input_cad_curve_evaluator_sample_count: tetrahedron_entity_count(
            tetrahedron_mesh,
            "input_plc_cad_curve_evaluator_samples",
        ),
        plc_input_cad_curve_live_query_edge_count: tetrahedron_entity_count(
            tetrahedron_mesh,
            "input_plc_cad_curve_live_query_edges",
        ),
        plc_input_cad_curve_live_query_sample_count: tetrahedron_entity_count(
            tetrahedron_mesh,
            "input_plc_cad_curve_live_query_samples",
        ),
        plc_input_cad_curve_rejected_evaluator_sample_count: tetrahedron_entity_count(
            tetrahedron_mesh,
            "input_plc_cad_curve_rejected_evaluator_samples",
        ),
        plc_input_cad_curve_curvature_sized_edge_count: tetrahedron_entity_count(
            tetrahedron_mesh,
            "input_plc_cad_curve_curvature_sized_edges",
        ),
        plc_input_cad_curve_curvature_sample_count: tetrahedron_entity_count(
            tetrahedron_mesh,
            "input_plc_cad_curve_curvature_samples",
        ),
        plc_input_surface_boundary_node_count: tetrahedron_entity_count(
            tetrahedron_mesh,
            "input_plc_surface_boundary_nodes",
        ),
        tetrahedron_generation_family: tetrahedron_mesh.tetrahedron_generation_family.clone(),
        tetrahedron_generation_attempted_family_count: tetrahedron_entity_count(
            tetrahedron_mesh,
            "solver_generation_attempted_families",
        ),
        tetrahedron_generation_rejected_family_count: tetrahedron_entity_count(
            tetrahedron_mesh,
            "solver_generation_rejected_families",
        ),
        tetrahedron_generation_selected_family_index: tetrahedron_entity_count(
            tetrahedron_mesh,
            "solver_generation_selected_family_index",
        ),
        tetrahedron_generation_interior_support_candidate_count: tetrahedron_entity_count(
            tetrahedron_mesh,
            "interior_support_candidate_points",
        ),
        tetrahedron_generation_interior_support_accepted_count: tetrahedron_entity_count(
            tetrahedron_mesh,
            "interior_support_accepted_points",
        ),
        tetrahedron_generation_nested_shell_outer_node_count: tetrahedron_entity_count(
            tetrahedron_mesh,
            "nested_tetrahedron_shell_outer_nodes",
        ),
        tetrahedron_generation_nested_shell_inner_node_count: tetrahedron_entity_count(
            tetrahedron_mesh,
            "nested_tetrahedron_shell_inner_nodes",
        ),
        tetrahedron_generation_nested_shell_generated_node_count: tetrahedron_entity_count(
            tetrahedron_mesh,
            "nested_tetrahedron_shell_generated_nodes",
        ),
        tetrahedron_generation_nested_shell_refill_boundary_face_count: tetrahedron_entity_count(
            tetrahedron_mesh,
            "nested_tetrahedron_shell_refill_boundary_faces",
        ),
        tetrahedron_generation_nested_shell_boundary_centroid_refinement_attempt_count:
            tetrahedron_entity_count(
                tetrahedron_mesh,
                "nested_tetrahedron_shell_boundary_centroid_refinement_attempts",
            ),
        tetrahedron_generation_nested_shell_boundary_centroid_refinement_rejected_count:
            tetrahedron_entity_count(
                tetrahedron_mesh,
                "nested_tetrahedron_shell_boundary_centroid_refinement_rejected",
            ),
        tetrahedron_generation_nested_shell_boundary_exact_cover_refill_count:
            tetrahedron_entity_count(
                tetrahedron_mesh,
                "nested_tetrahedron_shell_boundary_exact_cover_refills",
            ),
        tetrahedron_generation_nested_shell_boundary_centroid_refinement_refill_count:
            tetrahedron_entity_count(
                tetrahedron_mesh,
                "nested_tetrahedron_shell_boundary_centroid_refinement_refills",
            ),
        tetrahedron_generation_nested_shell_barycentric_partition_refill_count:
            tetrahedron_entity_count(
                tetrahedron_mesh,
                "nested_tetrahedron_shell_barycentric_partition_refills",
            ),
        tetrahedron_generation_nested_shell_outer_facet_count: tetrahedron_entity_count(
            tetrahedron_mesh,
            "nested_tetrahedron_shell_outer_facets",
        ),
        tetrahedron_generation_nested_shell_inner_facet_count: tetrahedron_entity_count(
            tetrahedron_mesh,
            "nested_tetrahedron_shell_inner_facets",
        ),
        ..base
    }
}
