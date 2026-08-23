use crate::contracts::AnalysisMeshArtifact;

use super::AnalysisMeshValidationError;

const SOLID_BACKEND: &str = "solid";

pub(super) fn validate_plc_input_evidence(
    mesh: &AnalysisMeshArtifact,
) -> Result<(), AnalysisMeshValidationError> {
    if mesh.backend.backend != SOLID_BACKEND {
        return Ok(());
    }
    if mesh.backend.plc_input_node_count == 0 {
        return missing("missing_plc_nodes");
    }
    if mesh.backend.plc_input_facet_count == 0 {
        return missing("missing_plc_facets");
    }
    if mesh.backend.plc_input_protected_edge_count == 0 {
        return missing("missing_plc_protected_edges");
    }
    if mesh.backend.plc_input_boundary_component_count == 0 {
        return missing("missing_plc_boundary_components");
    }
    if mesh.backend.plc_input_boundary_component_count
        != mesh.backend.plc_input_outer_shell_count + mesh.backend.plc_input_nested_shell_count
    {
        return missing("inconsistent_plc_shell_component_count");
    }
    if mesh.backend.plc_input_boundary_component_node_count == 0 {
        return missing("missing_plc_boundary_component_nodes");
    }
    if mesh.backend.plc_input_max_boundary_component_node_count == 0 {
        return missing("missing_plc_max_boundary_component_nodes");
    }
    if mesh.backend.plc_input_surface_boundary_node_count == 0 {
        return missing("missing_plc_surface_boundary_nodes");
    }
    if mesh.backend.plc_input_surface_boundary_node_count > mesh.backend.plc_input_node_count {
        return missing("inconsistent_plc_surface_boundary_nodes");
    }
    if mesh.backend.plc_input_surface_boundary_node_count
        > mesh.backend.plc_input_boundary_component_node_count
    {
        return missing("inconsistent_plc_surface_boundary_component_nodes");
    }
    if !mesh.backend.plc_input_shell_nesting_classified {
        return missing("unclassified_plc_shell_nesting");
    }
    if mesh.backend.plc_input_outer_shell_count != 1 {
        return missing("unsupported_plc_outer_shell_count");
    }
    if mesh.backend.tetrahedron_material_region_count == 0 {
        return missing("missing_tetrahedron_material_region_evidence");
    }
    if mesh.backend.tetrahedron_unclassified_material_element_count > 0 {
        return missing("unclassified_tetrahedron_material_ownership");
    }
    if mesh.backend.plc_input_material_region_count == 1
        && mesh.backend.tetrahedron_material_region_count != 1
    {
        return missing("inconsistent_single_material_region_ownership");
    }
    if mesh.backend.plc_input_material_region_count > 0
        && mesh.backend.plc_input_material_region_facet_count == 0
    {
        return missing("missing_plc_material_region_facet_evidence");
    }
    validate_plc_input_cad_curve_evidence(mesh)?;
    Ok(())
}

fn validate_plc_input_cad_curve_evidence(
    mesh: &AnalysisMeshArtifact,
) -> Result<(), AnalysisMeshValidationError> {
    let backend = &mesh.backend;
    let source_edge_count = backend.plc_input_cad_curve_boundary_source_edge_count;
    let boundary_segment_count = backend.plc_input_cad_curve_boundary_segment_count;
    let imported_edge_count = backend.plc_input_cad_curve_imported_edge_count;
    let evaluator_edge_count = backend.plc_input_cad_curve_evaluator_edge_count;
    let evaluator_sample_count = backend.plc_input_cad_curve_evaluator_sample_count;
    let live_query_edge_count = backend.plc_input_cad_curve_live_query_edge_count;
    let live_query_sample_count = backend.plc_input_cad_curve_live_query_sample_count;
    let rejected_sample_count = backend.plc_input_cad_curve_rejected_evaluator_sample_count;
    let curvature_sized_edge_count = backend.plc_input_cad_curve_curvature_sized_edge_count;
    let curvature_sample_count = backend.plc_input_cad_curve_curvature_sample_count;

    if source_edge_count == 0 {
        let any_cad_curve_evidence = boundary_segment_count
            + imported_edge_count
            + evaluator_edge_count
            + evaluator_sample_count
            + live_query_edge_count
            + live_query_sample_count
            + rejected_sample_count
            + curvature_sized_edge_count
            + curvature_sample_count;
        if any_cad_curve_evidence > 0 {
            return missing("missing_plc_cad_curve_boundary_source_edges");
        }
        return Ok(());
    }

    if source_edge_count > backend.plc_input_protected_edge_count {
        return missing("inconsistent_plc_cad_curve_source_edge_count");
    }
    if boundary_segment_count < source_edge_count {
        return missing("inconsistent_plc_cad_curve_boundary_segment_count");
    }
    if imported_edge_count > source_edge_count {
        return missing("inconsistent_plc_cad_curve_imported_edge_count");
    }
    if evaluator_edge_count > source_edge_count {
        return missing("inconsistent_plc_cad_curve_evaluator_edge_count");
    }
    if live_query_edge_count > evaluator_edge_count {
        return missing("inconsistent_plc_cad_curve_live_query_edge_count");
    }
    if curvature_sized_edge_count > source_edge_count {
        return missing("inconsistent_plc_cad_curve_curvature_sized_edge_count");
    }
    if evaluator_sample_count > 0 && imported_edge_count + evaluator_edge_count == 0 {
        return missing("missing_plc_cad_curve_sample_edge_evidence");
    }
    if live_query_sample_count > 0 && live_query_edge_count == 0 {
        return missing("missing_plc_cad_curve_live_query_edge_evidence");
    }
    if rejected_sample_count > 0 && imported_edge_count + evaluator_edge_count == 0 {
        return missing("missing_plc_cad_curve_rejected_sample_edge_evidence");
    }
    if curvature_sample_count > 0 && source_edge_count == 0 {
        return missing("missing_plc_cad_curve_curvature_source_edge_evidence");
    }

    Ok(())
}

fn missing<T>(reason: &str) -> Result<T, AnalysisMeshValidationError> {
    Err(AnalysisMeshValidationError::MissingPlcInputEvidence {
        reason: reason.to_string(),
    })
}
