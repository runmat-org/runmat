use std::collections::BTreeMap;

use runmat_meshing_core::{ElementOrder, SolverMeshArtifact};

use crate::algebra::{OperatorRole, RealSparseMatrix};

use super::{
    elements::solid::{
        tetrahedron10_consistent_mass_matrix, tetrahedron4_consistent_mass_matrix, SolidMaterial,
        Tetrahedron10ElementGeometry, Tetrahedron4ElementGeometry,
    },
    solid_matrix::{empty_rows, rows_to_sparse, scatter_csr},
    solver_solid::{element_geometry, solver_solid_topology, SolverSolidAssemblyError},
};

#[derive(Debug, Clone, PartialEq)]
pub struct SolverSolidMassAssembly {
    pub consistent: RealSparseMatrix,
    pub lumped_diag: Vec<f64>,
    pub total_mass_kg: f64,
}

pub fn assemble_solver_solid_mass(
    artifact: &SolverMeshArtifact,
    default_material: Option<SolidMaterial>,
    materials_by_region: &BTreeMap<String, SolidMaterial>,
) -> Result<SolverSolidMassAssembly, SolverSolidAssemblyError> {
    let topology = solver_solid_topology(artifact)?;
    let nodes = artifact
        .topology
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.node_id, (node.coordinates_m, index * 3)))
        .collect::<BTreeMap<_, _>>();
    let mut rows = empty_rows(topology.dof_count);
    let mut lumped_diag = vec![0.0; topology.dof_count];
    let mut total_mass_kg = 0.0;

    for element in &artifact.topology.volume_elements {
        let material = materials_by_region
            .get(&element.region_id.source_topology_id)
            .copied()
            .or(default_material)
            .ok_or_else(|| SolverSolidAssemblyError::UnassignedRegion {
                element_id: element.element_id,
                region_id: element.region_id.source_topology_id.clone(),
            })?;
        match element.order {
            ElementOrder::Tet4 => {
                let (coordinates, offsets) =
                    element_geometry::<4>(element.element_id, &element.node_ids, &nodes)?;
                let mass = tetrahedron4_consistent_mass_matrix(
                    material,
                    Tetrahedron4ElementGeometry {
                        nodes_m: coordinates,
                    },
                )
                .map_err(|failure| mass_failure(element.element_id, failure))?;
                add_element_mass(
                    &mut rows,
                    &mut lumped_diag,
                    &offsets,
                    &mass,
                    &mut total_mass_kg,
                )?;
            }
            ElementOrder::Tet10 => {
                let (coordinates, offsets) =
                    element_geometry::<10>(element.element_id, &element.node_ids, &nodes)?;
                let mass = tetrahedron10_consistent_mass_matrix(
                    material,
                    Tetrahedron10ElementGeometry {
                        nodes_m: coordinates,
                    },
                )
                .map_err(|failure| mass_failure(element.element_id, failure))?;
                add_element_mass(
                    &mut rows,
                    &mut lumped_diag,
                    &offsets,
                    &mass,
                    &mut total_mass_kg,
                )?;
            }
        }
    }

    Ok(SolverSolidMassAssembly {
        consistent: rows_to_sparse(rows, OperatorRole::Mass)
            .map_err(|error| SolverSolidAssemblyError::SparseOperator(format!("{error:?}")))?,
        lumped_diag,
        total_mass_kg,
    })
}

fn add_element_mass<const N: usize>(
    rows: &mut [BTreeMap<usize, f64>],
    lumped_diag: &mut [f64],
    offsets: &[usize],
    mass: &[[f64; N]; N],
    total_mass_kg: &mut f64,
) -> Result<(), SolverSolidAssemblyError> {
    scatter_csr(rows, offsets, mass);
    let element_mass_kg = mass
        .iter()
        .step_by(3)
        .map(|row| row.iter().step_by(3).sum::<f64>())
        .sum::<f64>();
    let diagonal_sum = offsets
        .iter()
        .enumerate()
        .map(|(node, _)| mass[node * 3][node * 3])
        .sum::<f64>();
    if !element_mass_kg.is_finite()
        || element_mass_kg <= 0.0
        || !diagonal_sum.is_finite()
        || diagonal_sum <= 0.0
    {
        return Err(SolverSolidAssemblyError::InvalidMassOperator);
    }
    for (node, &offset) in offsets.iter().enumerate() {
        let nodal_mass = element_mass_kg * mass[node * 3][node * 3] / diagonal_sum;
        for axis in 0..3 {
            lumped_diag[offset + axis] += nodal_mass;
        }
    }
    *total_mass_kg += element_mass_kg;
    Ok(())
}

fn mass_failure(element_id: u64, failure: impl std::fmt::Display) -> SolverSolidAssemblyError {
    SolverSolidAssemblyError::ElementMass {
        element_id,
        message: failure.to_string(),
    }
}

#[cfg(test)]
#[path = "solver_solid_mass/tests.rs"]
mod tests;
