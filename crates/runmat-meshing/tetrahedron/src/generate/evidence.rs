use std::collections::BTreeSet;

use runmat_meshing_core::contracts::{
    ProtectedBoundaryComplex, StageEvidence, Tetrahedron4Element, UNCLASSIFIED_MATERIAL_REGION_ID,
};
use runmat_meshing_plc::validate::{classify_boundary_components, classify_shell_nesting};

const INPUT_PLC_CAD_CURVE_EVIDENCE_KEYS: &[&str] = &[
    "cad_curve_boundary_source_edges",
    "cad_curve_boundary_segments",
    "cad_curve_imported_edges",
    "cad_curve_evaluator_edges",
    "cad_curve_evaluator_samples",
    "cad_curve_live_query_edges",
    "cad_curve_live_query_samples",
    "cad_curve_rejected_evaluator_samples",
    "cad_curve_curvature_sized_edges",
    "cad_curve_curvature_samples",
];

const INPUT_PLC_SURFACE_EVIDENCE_KEYS: &[&str] = &["surface_boundary_nodes"];

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

    let shell_classification = classify_shell_nesting(plc, &component_report);
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

    let plc_material_region_ids = plc
        .facets
        .iter()
        .flat_map(|facet| facet.material_interface_ids.iter().map(String::as_str))
        .collect::<BTreeSet<_>>();
    let material_region_facet_count = plc
        .facets
        .iter()
        .filter(|facet| !facet.material_interface_ids.is_empty())
        .count();
    evidence.entity_counts.insert(
        "input_plc_material_regions".to_string(),
        plc_material_region_ids.len(),
    );
    evidence.entity_counts.insert(
        "input_plc_material_region_facets".to_string(),
        material_region_facet_count,
    );
    for key in INPUT_PLC_CAD_CURVE_EVIDENCE_KEYS {
        if let Some(count) = plc.evidence.entity_counts.get(*key) {
            evidence
                .entity_counts
                .insert(format!("input_plc_{key}"), *count);
        }
    }
    for key in INPUT_PLC_SURFACE_EVIDENCE_KEYS {
        if let Some(count) = plc.evidence.entity_counts.get(*key) {
            evidence
                .entity_counts
                .insert(format!("input_plc_{key}"), *count);
        }
    }
}

pub(super) fn record_tetrahedron_material_evidence(
    elements: &[Tetrahedron4Element],
    evidence: &mut StageEvidence,
) {
    let material_region_ids = elements
        .iter()
        .map(|element| element.material_region_id.as_str())
        .collect::<BTreeSet<_>>();
    let unclassified_material_element_count = elements
        .iter()
        .filter(|element| element.material_region_id == UNCLASSIFIED_MATERIAL_REGION_ID)
        .count();
    evidence.entity_counts.insert(
        "tetrahedron_material_regions".to_string(),
        material_region_ids.len(),
    );
    evidence.entity_counts.insert(
        "unclassified_tetrahedron_material_elements".to_string(),
        unclassified_material_element_count,
    );
}
