use runmat_meshing_core::contracts::{ProtectedBoundaryComplex, StageEvidence};
use runmat_meshing_plc::validate::{classify_boundary_components, classify_shell_nesting};

pub(super) fn record_input_plc_evidence(
    plc: &ProtectedBoundaryComplex,
    evidence: &mut StageEvidence,
) {
    evidence
        .entity_counts
        .insert("input_plc_nodes".to_string(), plc.nodes.len());
    evidence
        .entity_counts
        .insert("input_plc_facets".to_string(), plc.facets.len());
    evidence.entity_counts.insert(
        "input_plc_protected_edges".to_string(),
        plc.protected_edges.len(),
    );

    let component_report = classify_boundary_components(plc);
    evidence.entity_counts.insert(
        "input_plc_boundary_components".to_string(),
        component_report.component_count,
    );
    evidence.entity_counts.insert(
        "input_plc_boundary_component_nodes".to_string(),
        component_report.referenced_node_count,
    );
    evidence.entity_counts.insert(
        "input_plc_max_boundary_component_nodes".to_string(),
        component_report.max_component_node_count,
    );

    let shell_classification = classify_shell_nesting(&component_report);
    evidence.entity_counts.insert(
        "input_plc_shell_nesting_classified".to_string(),
        usize::from(shell_classification.shell_nesting_classified),
    );
    evidence.entity_counts.insert(
        "input_plc_outer_shells".to_string(),
        shell_classification.outer_shell_count,
    );
    evidence.entity_counts.insert(
        "input_plc_nested_shells".to_string(),
        shell_classification.nested_shell_count,
    );
    evidence.entity_counts.insert(
        "input_plc_max_shell_nesting_depth".to_string(),
        shell_classification.max_nesting_depth,
    );
}
