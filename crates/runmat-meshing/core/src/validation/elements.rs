use std::collections::BTreeSet;

use crate::contracts::{AnalysisMeshArtifact, UNCLASSIFIED_MATERIAL_REGION_ID};

use super::AnalysisMeshValidationError;

pub(super) fn validate_volume_elements(
    mesh: &AnalysisMeshArtifact,
    node_ids: &BTreeSet<u32>,
) -> Result<BTreeSet<String>, AnalysisMeshValidationError> {
    let mut element_ids = BTreeSet::<String>::new();
    for element in &mesh.volume_elements {
        if !element_ids.insert(element.element_id.clone()) {
            return Err(AnalysisMeshValidationError::DuplicateElementId {
                element_id: element.element_id.clone(),
            });
        }
        if !element.kind.is_supported_for_solid_solve() {
            return Err(AnalysisMeshValidationError::UnsupportedVolumeElementKind {
                element_id: element.element_id.clone(),
            });
        }
        let expected = element.kind.node_count();
        if element.node_ids.len() != expected {
            return Err(AnalysisMeshValidationError::WrongVolumeElementNodeCount {
                element_id: element.element_id.clone(),
                expected,
                actual: element.node_ids.len(),
            });
        }
        let mut local_nodes = BTreeSet::<u32>::new();
        for node_id in &element.node_ids {
            if !node_ids.contains(node_id) {
                return Err(AnalysisMeshValidationError::UnknownVolumeElementNode {
                    element_id: element.element_id.clone(),
                    node_id: *node_id,
                });
            }
            if !local_nodes.insert(*node_id) {
                return Err(AnalysisMeshValidationError::RepeatedVolumeElementNode {
                    element_id: element.element_id.clone(),
                });
            }
        }
        if element.material_region_id.trim().is_empty() {
            return Err(AnalysisMeshValidationError::MissingMaterialRegion {
                element_id: element.element_id.clone(),
            });
        }
        if element.material_region_id == UNCLASSIFIED_MATERIAL_REGION_ID {
            return Err(AnalysisMeshValidationError::UnclassifiedMaterialRegion {
                element_id: element.element_id.clone(),
            });
        }
    }
    Ok(element_ids)
}
