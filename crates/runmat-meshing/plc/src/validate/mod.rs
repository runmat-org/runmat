use std::collections::{BTreeMap, BTreeSet};

mod components;
mod errors;
pub use components::{
    classify_boundary_components, classify_shell_nesting, PlcBoundaryComponentReport,
    PlcShellClassificationReport,
};
pub use errors::PlcValidationError;
pub use runmat_meshing_core::contracts::PlcValidationSummary;
use runmat_meshing_core::contracts::{MeshingStage, ProtectedBoundaryComplex, TopologyEntityId};

pub const MODULE_PURPOSE: &str =
    "watertightness, manifold incidence, shell nesting, and material interfaces";

#[cfg(test)]
mod tests;

pub fn validate_protected_boundary_complex(
    plc: &ProtectedBoundaryComplex,
) -> Result<PlcValidationSummary, PlcValidationError> {
    if !plc.validation.valid_for_volume_meshing() {
        return Err(PlcValidationError::ValidationSummaryNotVolumeReady {
            summary: plc.validation,
        });
    }
    if plc.nodes.is_empty() {
        return Err(PlcValidationError::EmptyNodes);
    }
    if plc.facets.is_empty() {
        return Err(PlcValidationError::EmptyFacets);
    }

    let mut node_ids = BTreeSet::<TopologyEntityId>::new();
    for node in &plc.nodes {
        if !node_ids.insert(node.node_id.clone()) {
            return Err(PlcValidationError::DuplicateNode {
                node_id: node.node_id.clone(),
            });
        }
        if !node
            .coordinates_m
            .iter()
            .all(|coordinate| coordinate.is_finite())
        {
            return Err(PlcValidationError::NonFiniteNode {
                node_id: node.node_id.clone(),
            });
        }
    }

    let mut edge_incidence = BTreeMap::<[TopologyEntityId; 2], usize>::new();
    let mut edge_orientation_balance = BTreeMap::<[TopologyEntityId; 2], i32>::new();
    let mut referenced_node_ids = BTreeSet::<TopologyEntityId>::new();
    let mut facet_ids = BTreeSet::<TopologyEntityId>::new();
    let mut facet_id_by_nodes = BTreeMap::<[TopologyEntityId; 3], TopologyEntityId>::new();
    for facet in &plc.facets {
        if !facet_ids.insert(facet.facet_id.clone()) {
            return Err(PlcValidationError::DuplicateFacet {
                facet_id: facet.facet_id.clone(),
            });
        }
        validate_facet_source_face(facet.facet_id.clone(), &facet.source_face_id)?;
        validate_facet_nodes(facet.facet_id.clone(), facet.node_ids.as_ref(), &node_ids)?;
        validate_facet_material_interfaces(facet.facet_id.clone(), &facet.material_interface_ids)?;
        let facet_key = sorted_facet(facet.node_ids.clone());
        if let Some(first_facet_id) =
            facet_id_by_nodes.insert(facet_key.clone(), facet.facet_id.clone())
        {
            return Err(PlcValidationError::DuplicateBoundaryFacet {
                first_facet_id,
                second_facet_id: facet.facet_id.clone(),
                node_ids: facet_key,
            });
        }
        referenced_node_ids.extend(facet.node_ids.iter().cloned());
        for edge_index in 0..3 {
            let left = facet.node_ids[edge_index].clone();
            let right = facet.node_ids[(edge_index + 1) % 3].clone();
            let edge = sorted_edge(left.clone(), right.clone());
            *edge_incidence.entry(edge.clone()).or_insert(0) += 1;
            *edge_orientation_balance.entry(edge).or_insert(0) +=
                directed_edge_orientation(left, right);
        }
    }

    for node_id in &node_ids {
        if !referenced_node_ids.contains(node_id) {
            return Err(PlcValidationError::UnreferencedNode {
                node_id: node_id.clone(),
            });
        }
    }

    let mut protected_edge_ids = BTreeSet::<TopologyEntityId>::new();
    let mut protected_edge_id_by_segment =
        BTreeMap::<[TopologyEntityId; 2], TopologyEntityId>::new();
    for protected_edge in &plc.protected_edges {
        if !protected_edge_ids.insert(protected_edge.edge_id.clone()) {
            return Err(PlcValidationError::DuplicateProtectedEdge {
                edge_id: protected_edge.edge_id.clone(),
            });
        }
        if protected_edge.node_ids[0] == protected_edge.node_ids[1] {
            return Err(PlcValidationError::ProtectedEdgeHasRepeatedNode {
                edge_id: protected_edge.edge_id.clone(),
            });
        }
        if protected_edge.source_edge_id.id.is_empty() {
            return Err(PlcValidationError::ProtectedEdgeHasEmptySourceEdgeId {
                edge_id: protected_edge.edge_id.clone(),
            });
        }
        if protected_edge.source_edge_id.stage != MeshingStage::CurveMesh {
            return Err(PlcValidationError::ProtectedEdgeSourceEdgeStageMismatch {
                edge_id: protected_edge.edge_id.clone(),
                source_edge_id: protected_edge.source_edge_id.clone(),
            });
        }
        for node_id in &protected_edge.node_ids {
            if !node_ids.contains(node_id) {
                return Err(PlcValidationError::ProtectedEdgeReferencesUnknownNode {
                    edge_id: protected_edge.edge_id.clone(),
                    node_id: node_id.clone(),
                });
            }
        }
        let edge = sorted_edge(
            protected_edge.node_ids[0].clone(),
            protected_edge.node_ids[1].clone(),
        );
        if let Some(first_edge_id) =
            protected_edge_id_by_segment.insert(edge.clone(), protected_edge.edge_id.clone())
        {
            return Err(PlcValidationError::DuplicateProtectedBoundarySegment {
                first_edge_id,
                second_edge_id: protected_edge.edge_id.clone(),
                node_ids: edge,
            });
        }
        if !edge_incidence.contains_key(&edge) {
            return Err(PlcValidationError::ProtectedEdgeNotOnBoundary {
                edge_id: protected_edge.edge_id.clone(),
                node_ids: edge,
            });
        }
    }

    for (edge, incidence_count) in edge_incidence {
        if incidence_count < 2 {
            return Err(PlcValidationError::OpenBoundaryEdge {
                node_ids: edge,
                incidence_count,
            });
        }
        if incidence_count > 2 {
            return Err(PlcValidationError::NonManifoldBoundaryEdge {
                node_ids: edge,
                incidence_count,
            });
        }
        if edge_orientation_balance
            .get(&edge)
            .copied()
            .unwrap_or_default()
            != 0
        {
            return Err(PlcValidationError::InconsistentBoundaryEdgeOrientation { node_ids: edge });
        }
    }
    let component_report = classify_boundary_components(plc);
    let shell_classification = classify_shell_nesting(&component_report);
    if !shell_classification.shell_nesting_classified {
        return Err(PlcValidationError::DisconnectedBoundaryComponents {
            component_count: component_report.component_count,
        });
    }

    Ok(PlcValidationSummary {
        watertight: true,
        manifold: true,
        shell_nesting_classified: shell_classification.shell_nesting_classified,
        material_interfaces_classified: true,
    })
}

fn validate_facet_nodes(
    facet_id: TopologyEntityId,
    facet_node_ids: &[TopologyEntityId],
    known_node_ids: &BTreeSet<TopologyEntityId>,
) -> Result<(), PlcValidationError> {
    let mut unique_node_ids = BTreeSet::<TopologyEntityId>::new();
    for node_id in facet_node_ids {
        if !known_node_ids.contains(node_id) {
            return Err(PlcValidationError::FacetReferencesUnknownNode {
                facet_id,
                node_id: node_id.clone(),
            });
        }
        if !unique_node_ids.insert(node_id.clone()) {
            return Err(PlcValidationError::FacetHasRepeatedNode { facet_id });
        }
    }
    Ok(())
}

fn validate_facet_source_face(
    facet_id: TopologyEntityId,
    source_face_id: &TopologyEntityId,
) -> Result<(), PlcValidationError> {
    if source_face_id.id.is_empty() {
        return Err(PlcValidationError::FacetHasEmptySourceFaceId { facet_id });
    }
    if source_face_id.stage != MeshingStage::SurfaceMesh {
        return Err(PlcValidationError::FacetSourceFaceStageMismatch {
            facet_id,
            source_face_id: source_face_id.clone(),
        });
    }
    Ok(())
}

fn validate_facet_material_interfaces(
    facet_id: TopologyEntityId,
    material_interface_ids: &[String],
) -> Result<(), PlcValidationError> {
    let mut unique_material_interface_ids = BTreeSet::<&str>::new();
    for material_interface_id in material_interface_ids {
        if material_interface_id.is_empty() {
            return Err(PlcValidationError::FacetHasEmptyMaterialInterfaceId { facet_id });
        }
        if !unique_material_interface_ids.insert(material_interface_id.as_str()) {
            return Err(PlcValidationError::FacetHasRepeatedMaterialInterfaceId {
                facet_id,
                material_interface_id: material_interface_id.clone(),
            });
        }
    }
    Ok(())
}

fn sorted_edge(left: TopologyEntityId, right: TopologyEntityId) -> [TopologyEntityId; 2] {
    if left <= right {
        [left, right]
    } else {
        [right, left]
    }
}

fn sorted_facet(mut node_ids: [TopologyEntityId; 3]) -> [TopologyEntityId; 3] {
    node_ids.sort();
    node_ids
}

fn directed_edge_orientation(left: TopologyEntityId, right: TopologyEntityId) -> i32 {
    if left <= right {
        1
    } else {
        -1
    }
}
