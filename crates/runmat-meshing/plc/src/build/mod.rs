use std::collections::{BTreeMap, BTreeSet};

mod errors;
pub use errors::PlcBuildError;
pub use runmat_meshing_core::contracts::{
    PlcFacet, PlcNode, PlcProtectedEdge, ProtectedBoundaryComplex,
};
use runmat_meshing_core::{
    contracts::{MeshingStage, StageEvidence, TopologyEntityId},
    surface::{SurfaceDiscretization, INTERNAL_SOURCE_EDGE_ID},
};

use crate::validate::{classify_boundary_components, validate_protected_boundary_complex};

pub const MODULE_PURPOSE: &str = "oriented protected boundary complex construction";

#[cfg(test)]
mod tests;

pub fn build_protected_boundary_complex(
    surface: &SurfaceDiscretization,
) -> Result<ProtectedBoundaryComplex, PlcBuildError> {
    if surface.elements.is_empty() {
        return Err(PlcBuildError::EmptySurface);
    }
    let surface_source_face_count = surface
        .elements
        .iter()
        .map(|element| element.source_face_id)
        .collect::<BTreeSet<_>>()
        .len();
    let protected_source_edge_count = surface
        .elements
        .iter()
        .flat_map(|element| element.source_edge_ids)
        .filter(|source_edge_id| *source_edge_id != INTERNAL_SOURCE_EDGE_ID)
        .collect::<BTreeSet<_>>()
        .len();
    let has_protected_source_edges = protected_source_edge_count > 0;
    if has_protected_source_edges && surface.curve_boundary_validation.is_none() {
        return Err(PlcBuildError::MissingCurveBoundaryValidation);
    }
    if has_protected_source_edges && surface.loop_coverage.is_none() {
        return Err(PlcBuildError::MissingSurfaceLoopCoverage);
    }

    let surface_nodes = surface
        .nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    for node in &surface.nodes {
        if !node
            .coordinates_m
            .iter()
            .all(|coordinate| coordinate.is_finite())
        {
            return Err(PlcBuildError::NonFiniteSurfaceNode {
                node_id: node.node_id,
            });
        }
    }

    let mut facets = Vec::<PlcFacet>::with_capacity(surface.elements.len());
    let mut protected_edges = BTreeMap::<(u32, u32, u32), PlcProtectedEdge>::new();
    let mut edge_incidence = BTreeMap::<[u32; 2], usize>::new();
    let mut facet_keys = BTreeSet::<[u32; 3]>::new();
    for element in &surface.elements {
        if !element.area_m2.is_finite() || !element.max_projection_error_m.is_finite() {
            return Err(PlcBuildError::NonFiniteSurfaceElement {
                element_id: element.element_id,
            });
        }
        for node_id in element.node_ids {
            if !surface_nodes.contains_key(&node_id) {
                return Err(PlcBuildError::MissingSurfaceNode {
                    element_id: element.element_id,
                    node_id,
                });
            }
        }
        let mut facet_key = element.node_ids;
        facet_key.sort_unstable();
        if !facet_keys.insert(facet_key) {
            return Err(PlcBuildError::DuplicateFacet {
                element_id: element.element_id,
            });
        }

        for edge_index in 0..3 {
            let left = element.node_ids[edge_index];
            let right = element.node_ids[(edge_index + 1) % 3];
            let edge = sorted_edge(left, right);
            *edge_incidence.entry(edge).or_insert(0) += 1;

            let source_edge_id = element.source_edge_ids[edge_index];
            if source_edge_id != INTERNAL_SOURCE_EDGE_ID {
                protected_edges
                    .entry((source_edge_id, edge[0], edge[1]))
                    .or_insert_with(|| PlcProtectedEdge {
                        edge_id: topology_entity_id(
                            MeshingStage::ProtectedBoundaryComplex,
                            format!(
                                "plc_protected_edge_{source_edge_id}_{}_{}",
                                edge[0], edge[1]
                            ),
                        ),
                        node_ids: [
                            topology_entity_id(MeshingStage::ProtectedBoundaryComplex, edge[0]),
                            topology_entity_id(MeshingStage::ProtectedBoundaryComplex, edge[1]),
                        ],
                        source_edge_id: topology_entity_id(MeshingStage::CurveMesh, source_edge_id),
                    });
            }
        }

        facets.push(PlcFacet {
            facet_id: topology_entity_id(
                MeshingStage::ProtectedBoundaryComplex,
                element.element_id,
            ),
            node_ids: [
                topology_entity_id(MeshingStage::ProtectedBoundaryComplex, element.node_ids[0]),
                topology_entity_id(MeshingStage::ProtectedBoundaryComplex, element.node_ids[1]),
                topology_entity_id(MeshingStage::ProtectedBoundaryComplex, element.node_ids[2]),
            ],
            source_face_id: topology_entity_id(MeshingStage::SurfaceMesh, element.source_face_id),
            material_interface_ids: element.region_ids.clone(),
        });
    }

    for (edge, incidence_count) in edge_incidence {
        if incidence_count < 2 {
            return Err(PlcBuildError::OpenBoundaryEdge {
                node_ids: edge,
                incidence_count,
            });
        }
        if incidence_count > 2 {
            return Err(PlcBuildError::NonManifoldBoundaryEdge {
                node_ids: edge,
                incidence_count,
            });
        }
    }
    if let Some(loop_coverage) = &surface.loop_coverage {
        if loop_coverage.recovered_face_count != surface_source_face_count
            || loop_coverage.recovered_source_edge_count < protected_source_edge_count
        {
            return Err(PlcBuildError::InconsistentSurfaceLoopCoverage {
                recovered_face_count: loop_coverage.recovered_face_count,
                surface_source_face_count,
                recovered_source_edge_count: loop_coverage.recovered_source_edge_count,
                protected_source_edge_count,
            });
        }
    }

    let mut evidence = StageEvidence::complete(MeshingStage::ProtectedBoundaryComplex);
    evidence
        .entity_counts
        .insert("nodes".to_string(), surface.nodes.len());
    evidence
        .entity_counts
        .insert("facets".to_string(), facets.len());
    evidence
        .entity_counts
        .insert("protected_edges".to_string(), protected_edges.len());
    if let Some(curve_boundary_validation) = &surface.curve_boundary_validation {
        evidence.entity_counts.insert(
            "validated_curve_source_edges".to_string(),
            curve_boundary_validation.source_edge_count,
        );
        evidence.entity_counts.insert(
            "validated_curve_nodes".to_string(),
            curve_boundary_validation.curve_node_count,
        );
        evidence.entity_counts.insert(
            "validated_curve_elements".to_string(),
            curve_boundary_validation.curve_element_count,
        );
    }
    if let Some(loop_coverage) = &surface.loop_coverage {
        evidence.entity_counts.insert(
            "surface_loop_faces".to_string(),
            loop_coverage.recovered_face_count,
        );
        evidence.entity_counts.insert(
            "surface_boundary_loops".to_string(),
            loop_coverage.boundary_loop_count,
        );
        evidence.entity_counts.insert(
            "surface_boundary_segments".to_string(),
            loop_coverage.boundary_segment_count,
        );
        evidence.entity_counts.insert(
            "recovered_surface_source_edges".to_string(),
            loop_coverage.recovered_source_edge_count,
        );
    }

    let mut plc = ProtectedBoundaryComplex {
        complex_id: "plc_surface_boundary".to_string(),
        nodes: surface
            .nodes
            .iter()
            .map(|node| PlcNode {
                node_id: topology_entity_id(MeshingStage::ProtectedBoundaryComplex, node.node_id),
                coordinates_m: node.coordinates_m,
            })
            .collect(),
        facets,
        protected_edges: protected_edges.into_values().collect(),
        validation: runmat_meshing_core::contracts::PlcValidationSummary {
            watertight: true,
            manifold: true,
            shell_nesting_classified: true,
            material_interfaces_classified: true,
        },
        evidence,
    };
    let component_report = classify_boundary_components(&plc);
    plc.evidence.entity_counts.insert(
        "boundary_components".to_string(),
        component_report.component_count,
    );
    plc.evidence.entity_counts.insert(
        "boundary_component_nodes".to_string(),
        component_report.referenced_node_count,
    );
    plc.evidence.entity_counts.insert(
        "min_boundary_component_nodes".to_string(),
        component_report.min_component_node_count,
    );
    plc.evidence.entity_counts.insert(
        "max_boundary_component_nodes".to_string(),
        component_report.max_component_node_count,
    );
    plc.validation = validate_protected_boundary_complex(&plc)
        .map_err(PlcBuildError::ProtectedBoundaryValidation)?;
    Ok(plc)
}

fn sorted_edge(left: u32, right: u32) -> [u32; 2] {
    if left <= right {
        [left, right]
    } else {
        [right, left]
    }
}

fn topology_entity_id(stage: MeshingStage, id: impl ToString) -> TopologyEntityId {
    TopologyEntityId {
        stage,
        id: id.to_string(),
    }
}
