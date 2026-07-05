use runmat_meshing_cad::SourceTopologyModel;
use runmat_meshing_core::{
    CurveMesh, CurveMeshElement, CurveMeshNode, MeshingStage, StageEvidence, TopologyEntityId,
};
use runmat_meshing_curve::{
    validate_curve_discretization, CurveDiscretization, CurveValidationError,
    CurveValidationOptions,
};

pub fn build_curve_mesh_contract(
    mesh_id: impl Into<String>,
    topology: &SourceTopologyModel,
    curves: &CurveDiscretization,
    validation_options: CurveValidationOptions,
) -> Result<CurveMesh, CurveValidationError> {
    let validation = validate_curve_discretization(topology, curves, validation_options)?;
    let source_edges_by_id = topology
        .edges
        .iter()
        .map(|edge| (edge.edge_id, edge))
        .collect::<std::collections::BTreeMap<_, _>>();
    let mut evidence = StageEvidence::complete(MeshingStage::CurveMesh);
    evidence
        .entity_counts
        .insert("source_edges".to_string(), validation.source_edge_count);
    evidence
        .entity_counts
        .insert("nodes".to_string(), validation.curve_node_count);
    evidence
        .entity_counts
        .insert("elements".to_string(), validation.curve_element_count);
    evidence
        .entity_counts
        .insert("parameter_chain_gaps".to_string(), 0);
    evidence
        .entity_counts
        .insert("endpoint_nodes".to_string(), topology.edges.len() * 2);
    evidence.max_projection_error_m = Some(validation.max_projection_error_m);

    Ok(CurveMesh {
        mesh_id: mesh_id.into(),
        nodes: curves
            .nodes
            .iter()
            .map(|node| CurveMeshNode {
                node_id: curve_entity_id(node.node_id),
                source_edge_id: curve_entity_id(node.source_edge_id),
                source_vertex_id: source_edges_by_id
                    .get(&node.source_edge_id)
                    .and_then(|edge| source_vertex_for_parameter(edge.node_ids, node.parameter)),
                parameter: node.parameter,
                coordinates_m: node.coordinates_m,
            })
            .collect(),
        elements: curves
            .elements
            .iter()
            .map(|element| CurveMeshElement {
                element_id: curve_entity_id(element.element_id),
                source_edge_id: curve_entity_id(element.source_edge_id),
                node_ids: [
                    curve_entity_id(element.node_ids[0]),
                    curve_entity_id(element.node_ids[1]),
                ],
                length_m: element.length_m,
            })
            .collect(),
        evidence,
    })
}

fn source_vertex_for_parameter(node_ids: [u32; 2], parameter: f64) -> Option<TopologyEntityId> {
    if (parameter - 0.0).abs() <= 1.0e-12 {
        Some(cad_topology_entity_id(node_ids[0]))
    } else if (parameter - 1.0).abs() <= 1.0e-12 {
        Some(cad_topology_entity_id(node_ids[1]))
    } else {
        None
    }
}

fn cad_topology_entity_id(id: impl ToString) -> TopologyEntityId {
    TopologyEntityId {
        stage: MeshingStage::CadTopology,
        id: id.to_string(),
    }
}

fn curve_entity_id(id: impl ToString) -> TopologyEntityId {
    TopologyEntityId {
        stage: MeshingStage::CurveMesh,
        id: id.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use runmat_meshing_cad::{SourceTopologyEdge, SourceTopologyModel, SourceTopologyVertex};
    use runmat_meshing_core::{MeshingStage, StageEvidenceStatus};
    use runmat_meshing_curve::{
        discretize_topology_curves, CurveDiscretization, CurveDiscretizationOptions, CurveElement,
        CurveNode, CurveValidationError, CurveValidationOptions,
    };

    use super::build_curve_mesh_contract;

    #[test]
    fn curve_contract_preserves_source_edge_provenance_and_validation_evidence() {
        let topology = line_topology(1.0);
        let curves = discretize_topology_curves(
            &topology,
            CurveDiscretizationOptions {
                target_size_m: 0.25,
                min_segments_per_edge: 1,
                max_segments_per_edge: 16,
            },
        )
        .expect("curves should discretize");

        let contract = build_curve_mesh_contract(
            "curve",
            &topology,
            &curves,
            CurveValidationOptions::default(),
        )
        .expect("valid curve discretization should produce a contract");

        assert_eq!(contract.mesh_id, "curve");
        assert_eq!(contract.evidence.stage, MeshingStage::CurveMesh);
        assert_eq!(contract.evidence.status, StageEvidenceStatus::Complete);
        assert_eq!(
            contract.evidence.entity_counts.get("source_edges"),
            Some(&1)
        );
        assert_eq!(contract.evidence.entity_counts.get("nodes"), Some(&5));
        assert_eq!(contract.evidence.entity_counts.get("elements"), Some(&4));
        assert_eq!(
            contract.evidence.entity_counts.get("parameter_chain_gaps"),
            Some(&0)
        );
        assert_eq!(
            contract.evidence.entity_counts.get("endpoint_nodes"),
            Some(&2)
        );
        assert_eq!(contract.evidence.max_projection_error_m, Some(0.0));
        assert!(contract
            .nodes
            .iter()
            .all(|node| node.node_id.stage == MeshingStage::CurveMesh
                && node.source_edge_id.stage == MeshingStage::CurveMesh
                && node.source_edge_id.id == "0"));
        assert!(contract
            .elements
            .iter()
            .all(
                |element| element.element_id.stage == MeshingStage::CurveMesh
                    && element.source_edge_id.stage == MeshingStage::CurveMesh
                    && element.source_edge_id.id == "0"
            ));
        assert_eq!(contract.nodes[0].parameter, 0.0);
        assert_eq!(contract.nodes[4].parameter, 1.0);
        assert_eq!(
            contract.nodes[0]
                .source_vertex_id
                .as_ref()
                .map(|source_vertex_id| (source_vertex_id.stage, source_vertex_id.id.as_str())),
            Some((MeshingStage::CadTopology, "0"))
        );
        assert!(contract.nodes[1].source_vertex_id.is_none());
        assert_eq!(
            contract.nodes[4]
                .source_vertex_id
                .as_ref()
                .map(|source_vertex_id| (source_vertex_id.stage, source_vertex_id.id.as_str())),
            Some((MeshingStage::CadTopology, "1"))
        );
    }

    #[test]
    fn curve_contract_rejects_invalid_boundary_before_promotion() {
        let topology = line_topology(1.0);
        let curves = CurveDiscretization {
            nodes: vec![
                CurveNode {
                    node_id: 0,
                    source_edge_id: 0,
                    parameter: 0.0,
                    coordinates_m: [0.0, 0.0, 0.0],
                },
                CurveNode {
                    node_id: 1,
                    source_edge_id: 0,
                    parameter: 0.5,
                    coordinates_m: [0.5, 0.0, 0.0],
                },
                CurveNode {
                    node_id: 2,
                    source_edge_id: 0,
                    parameter: 1.0,
                    coordinates_m: [1.0, 0.0, 0.0],
                },
            ],
            elements: vec![CurveElement {
                element_id: 0,
                source_edge_id: 0,
                node_ids: [0, 1],
                length_m: 0.5,
            }],
        };

        let err = build_curve_mesh_contract(
            "curve",
            &topology,
            &curves,
            CurveValidationOptions::default(),
        )
        .expect_err("incomplete curve chain should not produce a contract");

        assert!(matches!(
            err,
            CurveValidationError::ElementParameterGap {
                source_edge_id: 0,
                left_element_id: Some(0),
                right_element_id: None,
                ..
            }
        ));
    }

    fn line_topology(length_m: f64) -> SourceTopologyModel {
        SourceTopologyModel {
            mesh_id: "line".to_string(),
            source_geometry_id: "generic-line".to_string(),
            source_geometry_revision: 1,
            source_geometry_sha256: None,
            vertices: vec![
                SourceTopologyVertex {
                    vertex_id: 0,
                    coordinates_m: [0.0, 0.0, 0.0],
                },
                SourceTopologyVertex {
                    vertex_id: 1,
                    coordinates_m: [length_m, 0.0, 0.0],
                },
            ],
            edges: vec![SourceTopologyEdge {
                edge_id: 0,
                node_ids: [0, 1],
                adjacent_face_ids: vec![0, 1],
                region_ids: vec!["edge".to_string()],
                length_m,
            }],
            faces: Vec::new(),
            bounds_min_m: [0.0, 0.0, 0.0],
            bounds_max_m: [length_m, 0.0, 0.0],
            region_ids: vec!["edge".to_string()],
        }
    }
}
