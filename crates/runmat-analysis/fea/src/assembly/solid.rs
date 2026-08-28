use std::collections::BTreeMap;

use runmat_meshing_core::{AnalysisMeshArtifact, VolumeElementKind};

use crate::operator::CsrMatrix;

use super::elements::solid::{
    global_stiffness_matrix as tetrahedron4_global_stiffness_matrix, SolidMaterial,
    Tetrahedron4ElementGeometry, TETRAHEDRON4_ELEMENT_DOF_COUNT, TETRAHEDRON4_NODE_DOF_COUNT,
};
use super::solid_matrix::{empty_rows, rows_to_csr, scatter_csr};

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
    UnknownElementNode { element_id: String, node_id: u32 },
    InvalidElementNodeCount { element_id: String, actual: usize },
    ElementStiffness { element_id: String, message: String },
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
        if !matches!(element.kind, VolumeElementKind::Tetrahedron4) {
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

pub fn assemble_solid_stiffness_dense(
    mesh: &AnalysisMeshArtifact,
    material: SolidMaterial,
    base_dof_count: usize,
) -> Result<Vec<f64>, SolidAssemblyError> {
    let topology = solid_topology_from_analysis_mesh(mesh, base_dof_count)?;
    let mut node_offsets = BTreeMap::<u32, usize>::new();
    for (index, node) in mesh.nodes.iter().enumerate() {
        node_offsets.insert(node.node_id, index * TETRAHEDRON4_NODE_DOF_COUNT);
    }

    let mut dense = vec![0.0_f64; topology.dof_count * topology.dof_count];
    for element in &mesh.volume_elements {
        if element.node_ids.len() != 4 {
            return Err(SolidAssemblyError::InvalidElementNodeCount {
                element_id: element.element_id.clone(),
                actual: element.node_ids.len(),
            });
        }
        let mut nodes_m = [[0.0_f64; 3]; 4];
        let mut dof_offsets = [0_usize; 4];
        for (local_index, node_id) in element.node_ids.iter().copied().enumerate() {
            let node_index = mesh
                .nodes
                .iter()
                .position(|node| node.node_id == node_id)
                .ok_or_else(|| SolidAssemblyError::UnknownElementNode {
                    element_id: element.element_id.clone(),
                    node_id,
                })?;
            nodes_m[local_index] = mesh.nodes[node_index].coordinates_m;
            dof_offsets[local_index] = *node_offsets.get(&node_id).ok_or_else(|| {
                SolidAssemblyError::UnknownElementNode {
                    element_id: element.element_id.clone(),
                    node_id,
                }
            })?;
        }
        let element_stiffness =
            tetrahedron4_global_stiffness_matrix(material, Tetrahedron4ElementGeometry { nodes_m })
                .map_err(|err| SolidAssemblyError::ElementStiffness {
                    element_id: element.element_id.clone(),
                    message: err.to_string(),
                })?;
        scatter_tetrahedron4(
            &mut dense,
            topology.dof_count,
            dof_offsets,
            &element_stiffness,
        );
    }
    Ok(dense)
}

pub fn assemble_solid_stiffness_csr(
    mesh: &AnalysisMeshArtifact,
    material: SolidMaterial,
    base_dof_count: usize,
) -> Result<CsrMatrix, SolidAssemblyError> {
    assemble_solid_stiffness_csr_with_materials(mesh, material, &BTreeMap::new(), base_dof_count)
}

pub fn assemble_solid_stiffness_csr_with_materials(
    mesh: &AnalysisMeshArtifact,
    default_material: SolidMaterial,
    materials_by_region: &BTreeMap<String, SolidMaterial>,
    base_dof_count: usize,
) -> Result<CsrMatrix, SolidAssemblyError> {
    let topology = solid_topology_from_analysis_mesh(mesh, base_dof_count)?;
    let mut node_offsets = BTreeMap::<u32, usize>::new();
    for (index, node) in mesh.nodes.iter().enumerate() {
        node_offsets.insert(node.node_id, index * TETRAHEDRON4_NODE_DOF_COUNT);
    }

    let mut rows = empty_rows(topology.dof_count);
    for element in &mesh.volume_elements {
        if element.node_ids.len() != 4 {
            return Err(SolidAssemblyError::InvalidElementNodeCount {
                element_id: element.element_id.clone(),
                actual: element.node_ids.len(),
            });
        }
        let mut nodes_m = [[0.0_f64; 3]; 4];
        let mut dof_offsets = [0_usize; 4];
        for (local_index, node_id) in element.node_ids.iter().copied().enumerate() {
            let node_index = mesh
                .nodes
                .iter()
                .position(|node| node.node_id == node_id)
                .ok_or_else(|| SolidAssemblyError::UnknownElementNode {
                    element_id: element.element_id.clone(),
                    node_id,
                })?;
            nodes_m[local_index] = mesh.nodes[node_index].coordinates_m;
            dof_offsets[local_index] = *node_offsets.get(&node_id).ok_or_else(|| {
                SolidAssemblyError::UnknownElementNode {
                    element_id: element.element_id.clone(),
                    node_id,
                }
            })?;
        }
        let material = materials_by_region
            .get(element.material_region_id.as_str())
            .copied()
            .unwrap_or(default_material);
        let element_stiffness =
            tetrahedron4_global_stiffness_matrix(material, Tetrahedron4ElementGeometry { nodes_m })
                .map_err(|err| SolidAssemblyError::ElementStiffness {
                    element_id: element.element_id.clone(),
                    message: err.to_string(),
                })?;
        scatter_csr(&mut rows, &dof_offsets, &element_stiffness);
    }
    Ok(rows_to_csr(rows))
}

fn scatter_tetrahedron4(
    dense: &mut [f64],
    dof_count: usize,
    dof_offsets: [usize; 4],
    element_stiffness: &[[f64; TETRAHEDRON4_ELEMENT_DOF_COUNT]; TETRAHEDRON4_ELEMENT_DOF_COUNT],
) {
    for local_row_node in 0..4 {
        for local_row_axis in 0..TETRAHEDRON4_NODE_DOF_COUNT {
            let local_row = local_row_node * TETRAHEDRON4_NODE_DOF_COUNT + local_row_axis;
            let global_row = dof_offsets[local_row_node] + local_row_axis;
            for (local_col_node, global_col_offset) in dof_offsets.iter().enumerate() {
                for local_col_axis in 0..TETRAHEDRON4_NODE_DOF_COUNT {
                    let local_col = local_col_node * TETRAHEDRON4_NODE_DOF_COUNT + local_col_axis;
                    let global_col = global_col_offset + local_col_axis;
                    dense[global_row * dof_count + global_col] +=
                        element_stiffness[local_row][local_col];
                }
            }
        }
    }
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
                element_id: "tetrahedron_1".to_string(),
                kind,
                node_ids: vec![1, 2, 3, 4],
                material_region_id: "region".to_string(),
                provenance: Vec::new(),
            }],
            boundary_faces: Vec::new(),
            boundary_edges: Vec::new(),
            quality: AnalysisMeshQualityReport::default(),
            sizing: MeshSizingField::default(),
            field_topology: Vec::new(),
            backend: Default::default(),
            adaptive_iterations: Vec::new(),
            provenance: AnalysisMeshProvenance {
                algorithm: "test".to_string(),
                source_geometry_id: "geo".to_string(),
                source_geometry_revision: 1,
                source_geometry_sha256: None,
            },
        }
    }

    #[test]
    fn solid_topology_uses_analysis_mesh_nodes_and_tetrahedron4_elements() {
        let topology =
            solid_topology_from_analysis_mesh(&mesh(VolumeElementKind::Tetrahedron4), 3).unwrap();
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
                element_id: "tetrahedron_1".to_string()
            }
        );
    }

    #[test]
    fn solid_stiffness_scatter_assembles_tetrahedron4_dense_matrix() {
        let mesh = mesh(VolumeElementKind::Tetrahedron4);
        let dense = assemble_solid_stiffness_dense(
            &mesh,
            SolidMaterial {
                youngs_modulus_pa: 200.0e9,
                poisson_ratio: 0.3,
            },
            3,
        )
        .expect("Tetrahedron4 stiffness should assemble");
        let dof_count = 12;
        assert_eq!(dense.len(), dof_count * dof_count);
        for row in 0..dof_count {
            assert!(dense[row * dof_count + row] > 0.0);
            for col in 0..dof_count {
                assert!(
                    (dense[row * dof_count + col] - dense[col * dof_count + row]).abs() < 1.0e-5
                );
            }
        }
    }

    #[test]
    fn solid_stiffness_scatter_assembles_tetrahedron4_csr_matrix() {
        let mesh = mesh(VolumeElementKind::Tetrahedron4);
        let csr = assemble_solid_stiffness_csr(
            &mesh,
            SolidMaterial {
                youngs_modulus_pa: 200.0e9,
                poisson_ratio: 0.3,
            },
            3,
        )
        .expect("Tetrahedron4 stiffness should assemble");
        let dof_count = 12;
        assert_eq!(csr.row_offsets.len(), dof_count + 1);
        assert_eq!(csr.row_offsets.last().copied(), Some(csr.values.len()));
        assert_eq!(csr.column_indices.len(), csr.values.len());
        assert!(csr.values.len() <= dof_count * dof_count);
        for row in 0..dof_count {
            let start = csr.row_offsets[row];
            let end = csr.row_offsets[row + 1];
            assert!(csr.column_indices[start..end].binary_search(&row).is_ok());
        }
    }
}
