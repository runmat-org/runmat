use runmat_meshing_core::{AnalysisMeshArtifact, VolumeElementKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SolidAssemblyTopology {
    pub dof_count: usize,
    pub node_count: usize,
    pub volume_element_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SolidAssemblyError {
    EmptyNodes,
    EmptyVolumeElements,
    UnsupportedVolumeElementKind { element_id: String },
}

pub fn solid_topology_from_analysis_mesh(
    mesh: &AnalysisMeshArtifact,
    base_dof_count: usize,
) -> Result<SolidAssemblyTopology, SolidAssemblyError> {
    if mesh.nodes.is_empty() {
        return Err(SolidAssemblyError::EmptyNodes);
    }
    if mesh.volume_elements.is_empty() {
        return Err(SolidAssemblyError::EmptyVolumeElements);
    }
    for element in &mesh.volume_elements {
        if !matches!(element.kind, VolumeElementKind::Tet4) {
            return Err(SolidAssemblyError::UnsupportedVolumeElementKind {
                element_id: element.element_id.clone(),
            });
        }
    }
    Ok(SolidAssemblyTopology {
        dof_count: mesh.nodes.len().saturating_mul(3).max(base_dof_count),
        node_count: mesh.nodes.len(),
        volume_element_count: mesh.volume_elements.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_meshing_core::{
        AnalysisMeshNode, AnalysisMeshProvenance, AnalysisMeshQualityReport, AnalysisVolumeElement,
        MeshSizingField,
    };

    fn mesh(kind: VolumeElementKind) -> AnalysisMeshArtifact {
        AnalysisMeshArtifact {
            schema_version: "analysis-mesh/v1".to_string(),
            mesh_id: "mesh".to_string(),
            nodes: vec![
                AnalysisMeshNode {
                    node_id: 1,
                    coordinates_m: [0.0, 0.0, 0.0],
                    provenance: Vec::new(),
                },
                AnalysisMeshNode {
                    node_id: 2,
                    coordinates_m: [1.0, 0.0, 0.0],
                    provenance: Vec::new(),
                },
                AnalysisMeshNode {
                    node_id: 3,
                    coordinates_m: [0.0, 1.0, 0.0],
                    provenance: Vec::new(),
                },
                AnalysisMeshNode {
                    node_id: 4,
                    coordinates_m: [0.0, 0.0, 1.0],
                    provenance: Vec::new(),
                },
            ],
            volume_elements: vec![AnalysisVolumeElement {
                element_id: "tet_1".to_string(),
                kind,
                node_ids: vec![1, 2, 3, 4],
                material_region_id: "region".to_string(),
                provenance: Vec::new(),
            }],
            boundary_faces: Vec::new(),
            boundary_edges: Vec::new(),
            quality: AnalysisMeshQualityReport::default(),
            sizing: MeshSizingField::default(),
            provenance: AnalysisMeshProvenance {
                algorithm: "test".to_string(),
                source_geometry_id: "geo".to_string(),
                source_geometry_revision: 1,
                source_geometry_sha256: None,
            },
        }
    }

    #[test]
    fn solid_topology_uses_analysis_mesh_nodes_and_tets() {
        let topology =
            solid_topology_from_analysis_mesh(&mesh(VolumeElementKind::Tet4), 3).unwrap();
        assert_eq!(topology.dof_count, 12);
        assert_eq!(topology.node_count, 4);
        assert_eq!(topology.volume_element_count, 1);
    }

    #[test]
    fn solid_topology_rejects_unsupported_volume_elements() {
        let err = solid_topology_from_analysis_mesh(&mesh(VolumeElementKind::Hex8), 3)
            .expect_err("hex solid assembly is not supported yet");
        assert_eq!(
            err,
            SolidAssemblyError::UnsupportedVolumeElementKind {
                element_id: "tet_1".to_string()
            }
        );
    }
}
