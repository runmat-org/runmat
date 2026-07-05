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
    if mesh.backend.plc_input_boundary_component_count != 1 {
        return missing("unsupported_plc_boundary_component_count");
    }
    if mesh.backend.plc_input_boundary_component_node_count == 0 {
        return missing("missing_plc_boundary_component_nodes");
    }
    if mesh.backend.plc_input_max_boundary_component_node_count == 0 {
        return missing("missing_plc_max_boundary_component_nodes");
    }
    if !mesh.backend.plc_input_shell_nesting_classified {
        return missing("unclassified_plc_shell_nesting");
    }
    if mesh.backend.plc_input_outer_shell_count != 1 {
        return missing("unsupported_plc_outer_shell_count");
    }
    if mesh.backend.plc_input_nested_shell_count > 0 {
        return missing("unsupported_nested_plc_shell");
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
    Ok(())
}

fn missing<T>(reason: &str) -> Result<T, AnalysisMeshValidationError> {
    Err(AnalysisMeshValidationError::MissingPlcInputEvidence {
        reason: reason.to_string(),
    })
}
