mod energy;
pub(in crate::assembly) use energy::assemble_preparing;
pub(crate) use energy::SolidEnergyPreparation;
use std::collections::BTreeMap;

use runmat_analysis_core::MechanicalDampingModel;
use runmat_meshing_core::{ElementOrder, SolverMeshArtifact, SolverVolumeElement};

use crate::algebra::{OperatorRole, RealSparseMatrix};
use crate::{
    physics::structural::material::StructuralElementMaterial,
    planning::resource::LocalExecutionPlan,
    properties::CompiledPropertyMap,
};

use super::{
    elements::solid::{
        global_stiffness_matrix as tetrahedron4_stiffness, tetrahedron10_consistent_mass_matrix,
        tetrahedron10_global_stiffness_matrix, tetrahedron4_consistent_mass_matrix, SolidMaterial,
        Tetrahedron10ElementGeometry, Tetrahedron4ElementGeometry,
    },
};
use crate::algebra::{
    element_assembly::{
        assemble_elements, AssembledElements, ElementAssemblyError, ElementDofMap,
        ElementOutputShape, ElementOutputs, ElementView,
    },
    MatrixProperties, RealSparseAssembler,
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
    LocalExecution(String),
    Cancelled,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SolverSolidOperators {
    pub stiffness: RealSparseMatrix,
    pub consistent_mass: RealSparseMatrix,
    pub damping: RealSparseMatrix,
    pub lumped_mass_diag: Vec<f64>,
    pub total_mass_kg: f64,
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
    assemble_solver_solid_operators_with_execution(
        artifact,
        properties,
        LocalExecutionPlan::serial(),
    )
}

/// Computes bounded waves of element contributions under the admitted local CPU plan, then
/// scatters them in canonical mesh order. No worker owns a replica of the global sparse rows.
pub fn assemble_solver_solid_operators_with_execution(
    artifact: &SolverMeshArtifact,
    properties: &CompiledPropertyMap,
    local_execution: LocalExecutionPlan,
) -> Result<SolverSolidOperators, SolverSolidAssemblyError> {
    assemble_solver_solid_operators_capturing(artifact, properties, local_execution, None)
}

pub(in crate::assembly) fn assemble_solver_solid_operators_capturing(
    artifact: &SolverMeshArtifact,
    properties: &CompiledPropertyMap,
    local_execution: LocalExecutionPlan,
    capture: Option<&mut SolidEnergyPreparation>,
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
        .map(|(index, node)| {
            (
                node.node_id,
                (node.coordinates_m.canonical_value(), index * 3),
            )
        })
        .collect::<BTreeMap<_, _>>();
    assemble_solid_operators(
        &artifact.topology.volume_elements,
        properties,
        &nodes,
        topology.dof_count,
        local_execution,
        capture,
    )
}

const STIFFNESS: usize = 0;
const MASS: usize = 1;
const DAMPING: usize = 2;
const LUMPED_MASS: usize = 0;
const ELEMENT_MASS: usize = 0;

/// Element kernels run on the admitted CPU pool and are scattered into one shared symbolic
/// pattern in canonical element order, so operators are bitwise independent of thread count.
fn assemble_solid_operators(
    elements: &[SolverVolumeElement],
    properties: &CompiledPropertyMap,
    nodes: &NodeIndex,
    dof_count: usize,
    local_execution: LocalExecutionPlan,
    mut capture: Option<&mut SolidEnergyPreparation>,
) -> Result<SolverSolidOperators, SolverSolidAssemblyError> {
    let map = ElementDofMap::from_elements(dof_count, elements.len(), local_execution, |index, out| {
        let element = &elements[index];
        for node_id in &element.node_ids {
            let (_, offset) = nodes.get(node_id).copied().ok_or(
                SolverSolidAssemblyError::UnknownElementNode {
                    element_id: element.element_id,
                    node_id: *node_id,
                },
            )?;
            out.extend((0..3).map(|axis| Some(offset + axis)));
        }
        Ok(())
    })
    .map_err(element_assembly_error)?;
    let pattern = map
        .symbolic_pattern(local_execution)
        .map_err(|error| element_assembly_error(error.map_element(|never| match never {})))?;
    let damped = elements.iter().try_fold(false, |damped, element| {
        Ok::<_, SolverSolidAssemblyError>(
            damped
                || !matches!(
                    resolve_element_material(element, properties)?.damping,
                    MechanicalDampingModel::Undamped
                ),
        )
    })?;
    let shape = ElementOutputShape {
        matrices: if damped { 3 } else { 2 },
        vectors: 1,
        scalars: 1,
    };
    let assembled = assemble_elements(
        &map,
        &pattern,
        shape,
        local_execution,
        |index, _, out: &mut ElementOutputs<'_, f64>| {
            element_contribution(&elements[index], properties, nodes, damped, out)
        },
        |view: &ElementView<'_, f64>| match capture.as_deref_mut() {
            Some(capture) => capture.capture(view.dofs, view.matrix(STIFFNESS)),
            None => Ok(()),
        },
    )
    .map_err(element_assembly_error)?;
    let AssembledElements {
        matrices,
        vectors,
        scalars,
    } = assembled;
    let mut matrices = matrices.into_iter();
    let stiffness = matrices.next().unwrap_or_default();
    let mass = matrices.next().unwrap_or_default();
    let damping = matrices.next();
    Ok(SolverSolidOperators {
        stiffness: sparse(&pattern, stiffness, OperatorRole::Stiffness)?,
        consistent_mass: sparse(&pattern, mass, OperatorRole::Mass)?,
        damping: match damping {
            Some(values) => sparse(&pattern, values, OperatorRole::Damping)?,
            None => finish_operator(
                RealSparseAssembler::square(
                    dof_count,
                    OperatorRole::Damping,
                    MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
                ),
                OperatorRole::Damping,
            )?,
        },
        lumped_mass_diag: vectors.into_iter().nth(LUMPED_MASS).unwrap_or_default(),
        total_mass_kg: scalars[ELEMENT_MASS],
    })
}

fn element_assembly_error(
    error: ElementAssemblyError<SolverSolidAssemblyError>,
) -> SolverSolidAssemblyError {
    match error {
        ElementAssemblyError::Element { error, .. } => error,
        ElementAssemblyError::Cancelled => SolverSolidAssemblyError::Cancelled,
        ElementAssemblyError::Execution(reason) => SolverSolidAssemblyError::LocalExecution(reason),
        other => SolverSolidAssemblyError::SparseOperator(other.to_string()),
    }
}

fn element_contribution(
    element: &SolverVolumeElement,
    properties: &CompiledPropertyMap,
    nodes: &NodeIndex,
    damped: bool,
    out: &mut ElementOutputs<'_, f64>,
) -> Result<(), SolverSolidAssemblyError> {
    let material = resolve_element_material(element, properties)?;
    let elasticity = solid_material(material);
    match element.order {
        ElementOrder::Tet4 => {
            let (coordinates, _) =
                element_geometry::<4>(element.element_id, &element.node_ids, nodes)?;
            let geometry = Tetrahedron4ElementGeometry {
                nodes_m: coordinates,
            };
            let stiffness = tetrahedron4_stiffness(elasticity, geometry)
                .map_err(|error| element_failure(element.element_id, error))?;
            let mass = tetrahedron4_consistent_mass_matrix(elasticity, geometry)
                .map_err(|error| mass_failure(element.element_id, error))?;
            write_element(out, &stiffness, &mass, material.damping, damped)
        }
        ElementOrder::Tet10 => {
            let (coordinates, _) =
                element_geometry::<10>(element.element_id, &element.node_ids, nodes)?;
            let geometry = Tetrahedron10ElementGeometry {
                nodes_m: coordinates,
            };
            let stiffness = tetrahedron10_global_stiffness_matrix(elasticity, geometry)
                .map_err(|error| element_failure(element.element_id, error))?;
            let mass = tetrahedron10_consistent_mass_matrix(elasticity, geometry)
                .map_err(|error| mass_failure(element.element_id, error))?;
            write_element(out, &stiffness, &mass, material.damping, damped)
        }
    }
}

fn write_element<const D: usize>(
    out: &mut ElementOutputs<'_, f64>,
    stiffness: &[[f64; D]; D],
    mass: &[[f64; D]; D],
    damping: MechanicalDampingModel,
    damped: bool,
) -> Result<(), SolverSolidAssemblyError> {
    for (target, row) in out.matrix(STIFFNESS).chunks_exact_mut(D).zip(stiffness) {
        target.copy_from_slice(row);
    }
    for (target, row) in out.matrix(MASS).chunks_exact_mut(D).zip(mass) {
        target.copy_from_slice(row);
    }
    if damped {
        let (mass_coefficient, stiffness_coefficient) = match damping {
            MechanicalDampingModel::Undamped => (0.0, 0.0),
            MechanicalDampingModel::Rayleigh {
                mass_coefficient,
                stiffness_coefficient,
            } => (mass_coefficient.canonical_value(), stiffness_coefficient.canonical_value()),
        };
        let target = out.matrix(DAMPING);
        for row in 0..D {
            for column in 0..D {
                target[row * D + column] = mass_coefficient * mass[row][column]
                    + stiffness_coefficient * stiffness[row][column];
            }
        }
    }
    // Row-sum-scaled nodal lumping: each node's three DOFs receive its share of element mass.
    let element_mass_kg = mass
        .iter()
        .step_by(3)
        .map(|row| row.iter().step_by(3).sum::<f64>())
        .sum::<f64>();
    let diagonal_sum = (0..D / 3).map(|node| mass[node * 3][node * 3]).sum::<f64>();
    if !element_mass_kg.is_finite()
        || element_mass_kg <= 0.0
        || !diagonal_sum.is_finite()
        || diagonal_sum <= 0.0
    {
        return Err(SolverSolidAssemblyError::InvalidMassOperator);
    }
    let lumped = out.vector(LUMPED_MASS);
    for node in 0..D / 3 {
        let nodal_mass = element_mass_kg * mass[node * 3][node * 3] / diagonal_sum;
        lumped[node * 3..node * 3 + 3].fill(nodal_mass);
    }
    *out.scalar(ELEMENT_MASS) = element_mass_kg;
    Ok(())
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
        poisson_ratio: runmat_types::physical::PoissonRatio::new(
            material.poisson_ratio.canonical_value(),
        ),
        density_kg_per_m3: runmat_types::physical::MassDensity::new(
            material.density_kg_per_m3.canonical_value(),
        ),
    }
}

fn sparse(
    pattern: &crate::algebra::SparsePattern,
    values: Vec<f64>,
    role: OperatorRole,
) -> Result<RealSparseMatrix, SolverSolidAssemblyError> {
    let assembler = RealSparseAssembler::from_planned_values(
        pattern.clone(),
        values,
        role,
        MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
    )
    .map_err(|error| SolverSolidAssemblyError::SparseOperator(format!("{error:?}")))?;
    finish_operator(assembler, role)
}

fn finish_operator(
    mut assembler: RealSparseAssembler,
    role: OperatorRole,
) -> Result<RealSparseMatrix, SolverSolidAssemblyError> {
    assembler.add_contribution(super::solid_matrix::solid_operator_provenance(role));
    assembler
        .finish()
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
