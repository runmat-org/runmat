use std::collections::BTreeMap;

use runmat_meshing_core::{ElementOrder, SolverMeshArtifact};

use crate::operator::CsrMatrix;

use super::{
    elements::solid::{
        global_stiffness_matrix as tetrahedron4_stiffness, tetrahedron10_global_stiffness_matrix,
        SolidMaterial, Tetrahedron10ElementGeometry, Tetrahedron4ElementGeometry,
    },
    solid_matrix::{empty_rows, rows_to_csr, scatter_csr},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SolverSolidTopology {
    pub dof_count: usize,
    pub node_count: usize,
    pub volume_element_count: usize,
    pub order: ElementOrder,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SolverSolidAssemblyError {
    InvalidArtifact(String),
    UnknownElementNode { element_id: u64, node_id: u64 },
    ElementStiffness { element_id: u64, message: String },
}

pub fn solver_solid_topology(
    artifact: &SolverMeshArtifact,
    base_dof_count: usize,
) -> Result<SolverSolidTopology, SolverSolidAssemblyError> {
    artifact
        .validate()
        .map_err(|failure| SolverSolidAssemblyError::InvalidArtifact(failure.to_string()))?;
    Ok(SolverSolidTopology {
        dof_count: artifact
            .topology
            .nodes
            .len()
            .saturating_mul(3)
            .max(base_dof_count),
        node_count: artifact.topology.nodes.len(),
        volume_element_count: artifact.topology.volume_elements.len(),
        order: artifact.resolved_request.element_order,
    })
}

pub fn assemble_solver_solid_stiffness_csr(
    artifact: &SolverMeshArtifact,
    default_material: SolidMaterial,
    materials_by_id: &BTreeMap<String, SolidMaterial>,
    base_dof_count: usize,
) -> Result<CsrMatrix, SolverSolidAssemblyError> {
    let topology = solver_solid_topology(artifact, base_dof_count)?;
    let nodes = artifact
        .topology
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.node_id, (node.coordinates_m, index * 3)))
        .collect::<BTreeMap<_, _>>();
    let mut rows = empty_rows(topology.dof_count);
    for element in &artifact.topology.volume_elements {
        let material = materials_by_id
            .get(&element.material_id)
            .copied()
            .unwrap_or(default_material);
        match element.order {
            ElementOrder::Tet4 => {
                let (coordinates, offsets) =
                    element_geometry::<4>(element.element_id, &element.node_ids, &nodes)?;
                let stiffness = tetrahedron4_stiffness(
                    material,
                    Tetrahedron4ElementGeometry {
                        nodes_m: coordinates,
                    },
                )
                .map_err(|failure| element_failure(element.element_id, failure))?;
                scatter_csr(&mut rows, &offsets, &stiffness);
            }
            ElementOrder::Tet10 => {
                let (coordinates, offsets) =
                    element_geometry::<10>(element.element_id, &element.node_ids, &nodes)?;
                let stiffness = tetrahedron10_global_stiffness_matrix(
                    material,
                    Tetrahedron10ElementGeometry {
                        nodes_m: coordinates,
                    },
                )
                .map_err(|failure| element_failure(element.element_id, failure))?;
                scatter_csr(&mut rows, &offsets, &stiffness);
            }
        }
    }
    Ok(rows_to_csr(rows))
}

type NodeIndex = BTreeMap<u64, ([f64; 3], usize)>;

fn element_geometry<const N: usize>(
    element_id: u64,
    node_ids: &[u64],
    nodes: &NodeIndex,
) -> Result<([[f64; 3]; N], [usize; N]), SolverSolidAssemblyError> {
    let mut coordinates = [[0.0; 3]; N];
    let mut offsets = [0; N];
    for (local, node_id) in node_ids.iter().copied().enumerate() {
        let (point, offset) =
            nodes
                .get(&node_id)
                .copied()
                .ok_or(SolverSolidAssemblyError::UnknownElementNode {
                    element_id,
                    node_id,
                })?;
        coordinates[local] = point;
        offsets[local] = offset;
    }
    Ok((coordinates, offsets))
}

fn element_failure(element_id: u64, failure: impl std::fmt::Display) -> SolverSolidAssemblyError {
    SolverSolidAssemblyError::ElementStiffness {
        element_id,
        message: failure.to_string(),
    }
}

#[cfg(test)]
#[path = "solver_solid/tests.rs"]
mod tests;
