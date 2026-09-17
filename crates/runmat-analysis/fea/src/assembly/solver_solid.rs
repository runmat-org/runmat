use std::collections::BTreeMap;

use runmat_analysis_core::MechanicalDampingModel;
use runmat_meshing_core::{ElementOrder, SolverMeshArtifact, SolverVolumeElement};

use crate::algebra::{OperatorRole, RealSparseMatrix};
use crate::{
    physics::structural::material::StructuralElementMaterial, properties::CompiledPropertyMap,
};

use super::{
    elements::solid::{
        global_stiffness_matrix as tetrahedron4_stiffness, tetrahedron10_consistent_mass_matrix,
        tetrahedron10_global_stiffness_matrix, tetrahedron4_consistent_mass_matrix, SolidMaterial,
        Tetrahedron10ElementGeometry, Tetrahedron4ElementGeometry,
    },
    solid_matrix::{empty_rows, rows_to_sparse, scatter_csr},
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
    InvalidMaterialCoverage(String),
    InvalidElementNodeCount {
        element_id: u64,
        expected: usize,
        actual: usize,
    },
    UnknownElementNode {
        element_id: u64,
        node_id: u64,
    },
    UnassignedRegion {
        element_id: u64,
        region_id: String,
    },
    AmbiguousRegion {
        element_id: u64,
        region_id: String,
        assignment_count: usize,
    },
    ElementStiffness {
        element_id: u64,
        message: String,
    },
    ElementMass {
        element_id: u64,
        message: String,
    },
    InvalidMassOperator,
    SparseOperator(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct SolverSolidOperators {
    pub stiffness: RealSparseMatrix,
    pub consistent_mass: RealSparseMatrix,
    pub damping: RealSparseMatrix,
    pub lumped_mass_diag: Vec<f64>,
    pub total_mass_kg: f64,
}

struct SolidOperatorRows {
    stiffness: Vec<BTreeMap<usize, f64>>,
    mass: Vec<BTreeMap<usize, f64>>,
    damping: Vec<BTreeMap<usize, f64>>,
    lumped_mass_diag: Vec<f64>,
    total_mass_kg: f64,
}

impl SolidOperatorRows {
    fn new(dof_count: usize) -> Self {
        Self {
            stiffness: empty_rows(dof_count),
            mass: empty_rows(dof_count),
            damping: empty_rows(dof_count),
            lumped_mass_diag: vec![0.0; dof_count],
            total_mass_kg: 0.0,
        }
    }

    fn add_element<const D: usize>(
        &mut self,
        offsets: &[usize],
        stiffness: &[[f64; D]; D],
        mass: &[[f64; D]; D],
        damping: MechanicalDampingModel,
    ) -> Result<(), SolverSolidAssemblyError> {
        scatter_csr(&mut self.stiffness, offsets, stiffness);
        scatter_csr(&mut self.mass, offsets, mass);
        let (mass_coefficient, stiffness_coefficient) = match damping {
            MechanicalDampingModel::Undamped => (0.0, 0.0),
            MechanicalDampingModel::Rayleigh {
                mass_coefficient,
                stiffness_coefficient,
            } => (mass_coefficient.per_second, stiffness_coefficient.seconds),
        };
        let element_damping: [[f64; D]; D] = std::array::from_fn(|row| {
            std::array::from_fn(|column| {
                mass_coefficient * mass[row][column]
                    + stiffness_coefficient * stiffness[row][column]
            })
        });
        scatter_csr(&mut self.damping, offsets, &element_damping);
        add_lumped_mass(
            &mut self.lumped_mass_diag,
            offsets,
            mass,
            &mut self.total_mass_kg,
        )
    }
}

pub fn solver_solid_topology(
    artifact: &SolverMeshArtifact,
) -> Result<SolverSolidTopology, SolverSolidAssemblyError> {
    artifact
        .validate()
        .map_err(|failure| SolverSolidAssemblyError::InvalidArtifact(failure.to_string()))?;
    Ok(SolverSolidTopology {
        dof_count: artifact.topology.nodes.len().saturating_mul(3),
        node_count: artifact.topology.nodes.len(),
        volume_element_count: artifact.topology.volume_elements.len(),
        order: artifact.resolved_request.element_order,
    })
}

pub fn assemble_solver_solid_operators(
    artifact: &SolverMeshArtifact,
    properties: &CompiledPropertyMap,
) -> Result<SolverSolidOperators, SolverSolidAssemblyError> {
    let topology = solver_solid_topology(artifact)?;
    if properties.identity.mesh_digest != Some(artifact.canonical_digest) {
        return Err(SolverSolidAssemblyError::InvalidMaterialCoverage(
            "compiled property map belongs to a different solver mesh".into(),
        ));
    }
    let nodes = artifact
        .topology
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.node_id, (node.coordinates_m, index * 3)))
        .collect::<BTreeMap<_, _>>();
    let mut rows = SolidOperatorRows::new(topology.dof_count);
    for element in &artifact.topology.volume_elements {
        let material = resolve_element_material(element, properties)?;
        let elasticity = solid_material(material);
        match element.order {
            ElementOrder::Tet4 => {
                let (coordinates, offsets) =
                    element_geometry::<4>(element.element_id, &element.node_ids, &nodes)?;
                let stiffness = tetrahedron4_stiffness(
                    elasticity,
                    Tetrahedron4ElementGeometry {
                        nodes_m: coordinates,
                    },
                )
                .map_err(|failure| element_failure(element.element_id, failure))?;
                let mass = tetrahedron4_consistent_mass_matrix(
                    elasticity,
                    Tetrahedron4ElementGeometry {
                        nodes_m: coordinates,
                    },
                )
                .map_err(|failure| mass_failure(element.element_id, failure))?;
                rows.add_element(&offsets, &stiffness, &mass, material.damping)?;
            }
            ElementOrder::Tet10 => {
                let (coordinates, offsets) =
                    element_geometry::<10>(element.element_id, &element.node_ids, &nodes)?;
                let stiffness = tetrahedron10_global_stiffness_matrix(
                    elasticity,
                    Tetrahedron10ElementGeometry {
                        nodes_m: coordinates,
                    },
                )
                .map_err(|failure| element_failure(element.element_id, failure))?;
                let mass = tetrahedron10_consistent_mass_matrix(
                    elasticity,
                    Tetrahedron10ElementGeometry {
                        nodes_m: coordinates,
                    },
                )
                .map_err(|failure| mass_failure(element.element_id, failure))?;
                rows.add_element(&offsets, &stiffness, &mass, material.damping)?;
            }
        }
    }
    Ok(SolverSolidOperators {
        stiffness: sparse(rows.stiffness, OperatorRole::Stiffness)?,
        consistent_mass: sparse(rows.mass, OperatorRole::Mass)?,
        damping: sparse(rows.damping, OperatorRole::Damping)?,
        lumped_mass_diag: rows.lumped_mass_diag,
        total_mass_kg: rows.total_mass_kg,
    })
}

pub(super) fn resolve_element_material(
    element: &SolverVolumeElement,
    properties: &CompiledPropertyMap,
) -> Result<StructuralElementMaterial, SolverSolidAssemblyError> {
    properties
        .solid_material(element.stable_identity)
        .ok_or_else(|| SolverSolidAssemblyError::UnassignedRegion {
            element_id: element.element_id,
            region_id: element.region_id.source_topology_id.clone(),
        })
        .and_then(|entry| {
            StructuralElementMaterial::from_model(&entry.material).map_err(|error| {
                SolverSolidAssemblyError::InvalidMaterialCoverage(error.to_string())
            })
        })
}

fn solid_material(material: StructuralElementMaterial) -> SolidMaterial {
    SolidMaterial {
        youngs_modulus_pa: material.youngs_modulus_pa,
        poisson_ratio: material.poisson_ratio,
        density_kg_per_m3: material.density_kg_per_m3,
    }
}

fn add_lumped_mass<const D: usize>(
    lumped_mass_diag: &mut [f64],
    offsets: &[usize],
    mass: &[[f64; D]; D],
    total_mass_kg: &mut f64,
) -> Result<(), SolverSolidAssemblyError> {
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
            lumped_mass_diag[offset + axis] += nodal_mass;
        }
    }
    *total_mass_kg += element_mass_kg;
    Ok(())
}

fn sparse(
    rows: Vec<BTreeMap<usize, f64>>,
    role: OperatorRole,
) -> Result<RealSparseMatrix, SolverSolidAssemblyError> {
    rows_to_sparse(rows, role)
        .map_err(|error| SolverSolidAssemblyError::SparseOperator(format!("{error:?}")))
}

pub(super) type NodeIndex = BTreeMap<u64, ([f64; 3], usize)>;

pub(super) fn element_geometry<const N: usize>(
    element_id: u64,
    node_ids: &[u64],
    nodes: &NodeIndex,
) -> Result<([[f64; 3]; N], [usize; N]), SolverSolidAssemblyError> {
    if node_ids.len() != N {
        return Err(SolverSolidAssemblyError::InvalidElementNodeCount {
            element_id,
            expected: N,
            actual: node_ids.len(),
        });
    }
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

fn mass_failure(element_id: u64, failure: impl std::fmt::Display) -> SolverSolidAssemblyError {
    SolverSolidAssemblyError::ElementMass {
        element_id,
        message: failure.to_string(),
    }
}

#[cfg(test)]
#[path = "solver_solid/tests.rs"]
pub(crate) mod tests;
