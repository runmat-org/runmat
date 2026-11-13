use runmat_analysis_core::{
    MechanicalDampingModel, ShellElementModel, StructuralElement, StructuralModel,
};

use super::{
    laminate_property, shell_element_dof_indices, shell_section_from_model, structural_node_index,
};
use crate::{
    assembly::{
        elements::shell::{
            admitted_homogeneous_constitutive, global_lumped_mass_matrix,
            global_matrix_for_constitutive_matrices, global_stiffness_matrix,
            global_stiffness_matrix_for_constitutive, homogeneous_mass_moments,
            ShellElementGeometry, ShellMassMoments, ShellMaterial, SHELL_ELEMENT_DOF_COUNT,
        },
        ShellRecoveryElementSummary, StructuralAssemblyError, StructuralDofLayout,
    },
    physics::structural::{
        material::StructuralElementMaterial, topology::explicit_element_identity,
    },
    properties::{CompiledPropertyMap, CompiledShellPropertyOwner},
};

type ShellMatrix = [[f64; SHELL_ELEMENT_DOF_COUNT]; SHELL_ELEMENT_DOF_COUNT];

pub(super) struct ShellElementContribution {
    pub dofs: [usize; SHELL_ELEMENT_DOF_COUNT],
    pub stiffness: ShellMatrix,
    pub mass: ShellMatrix,
    pub damping: Option<ShellMatrix>,
    pub recovery: ShellRecoveryElementSummary,
}

pub(super) fn compute(
    structural: &StructuralModel,
    nodes: &super::ExplicitNodeIndex,
    element: &StructuralElement,
    shell: &ShellElementModel,
    properties: &CompiledPropertyMap,
    layout: &StructuralDofLayout,
    canonical_node_indices: &[usize],
    topology_digest: runmat_execution::Digest,
) -> Result<ShellElementContribution, StructuralAssemblyError> {
    let mut explicit_nodes = [0_usize; 3];
    for (local, node_id) in shell.node_ids.iter().copied().enumerate() {
        explicit_nodes[local] = structural_node_index(nodes, node_id, element)?;
    }
    let node_indices = explicit_nodes.map(|node| canonical_node_indices[node]);
    let element_identity = explicit_element_identity(
        topology_digest,
        &element.element_id,
        &element.region_id,
        &element.kind,
    )
    .map_err(|error| StructuralAssemblyError::InvalidMaterialCoverage(error.to_string()))?;
    let property = properties.shell_property(element_identity).ok_or_else(|| {
        StructuralAssemblyError::UnassignedExplicitRegion {
            element_id: element.element_id.clone(),
            region_id: element.region_id.clone(),
        }
    })?;
    let geometry = ShellElementGeometry {
        nodes_m: explicit_nodes.map(|node| structural.nodes[node].coordinates_m.canonical_value()),
        reference_axis: shell.reference_axis,
    };
    let frame = geometry
        .local_frame()
        .map_err(|error| StructuralAssemblyError::ShellElement {
            element_id: element.element_id.clone(),
            error,
        })?;
    let dofs = shell_element_dof_indices(layout, node_indices).ok_or_else(|| {
        StructuralAssemblyError::MissingRotationalDof {
            element_id: element.element_id.clone(),
        }
    })?;
    let (stiffness, mass, damping, constitutive) = match &property.owner {
        CompiledShellPropertyOwner::Homogeneous { material, section } => {
            let section = shell_section_from_model(section);
            let material = StructuralElementMaterial::from_model(material).map_err(|error| {
                StructuralAssemblyError::InvalidMaterialCoverage(error.to_string())
            })?;
            let shell_material = ShellMaterial {
                youngs_modulus_pa: material.youngs_modulus_pa,
                poisson_ratio: runmat_types::physical::PoissonRatio::new(
                    material.poisson_ratio.canonical_value(),
                ),
                shear_modulus_pa: material.shear_modulus_pa(),
            };
            let constitutive =
                admitted_homogeneous_constitutive(section, shell_material).map_err(|error| {
                    StructuralAssemblyError::ShellElement {
                        element_id: element.element_id.clone(),
                        error,
                    }
                })?;
            let stiffness =
                global_stiffness_matrix(section, shell_material, geometry).map_err(|error| {
                    StructuralAssemblyError::ShellElement {
                        element_id: element.element_id.clone(),
                        error,
                    }
                })?;
            let mass_moments = homogeneous_mass_moments(section, material.density_kg_per_m3)
                .map_err(|error| StructuralAssemblyError::ShellElement {
                    element_id: element.element_id.clone(),
                    error,
                })?;
            let mass = global_lumped_mass_matrix(mass_moments, frame).map_err(|error| {
                StructuralAssemblyError::ShellElement {
                    element_id: element.element_id.clone(),
                    error,
                }
            })?;
            let damping = if let MechanicalDampingModel::Rayleigh {
                mass_coefficient,
                stiffness_coefficient,
            } = material.damping
            {
                let mass_damping = global_lumped_mass_matrix(
                    mass_moments
                        .proportional_damping(mass_coefficient)
                        .map_err(|error| StructuralAssemblyError::ShellElement {
                            element_id: element.element_id.clone(),
                            error,
                        })?,
                    frame,
                )
                .map_err(|error| StructuralAssemblyError::ShellElement {
                    element_id: element.element_id.clone(),
                    error,
                })?;
                Some(combined_damping(
                    &mass_damping,
                    &stiffness,
                    stiffness_coefficient.canonical_value(),
                ))
            } else {
                None
            };
            (stiffness, mass, damping, constitutive)
        }
        CompiledShellPropertyOwner::Laminate { section, plies } => {
            let response = laminate_property::definition(section, plies)
                .map_err(StructuralAssemblyError::InvalidMaterialCoverage)?
                .integrate()
                .map_err(|error| StructuralAssemblyError::ShellElement {
                    element_id: element.element_id.clone(),
                    error,
                })?;
            let stiffness = global_stiffness_matrix_for_constitutive(&response.elastic, geometry)
                .map_err(|error| StructuralAssemblyError::ShellElement {
                element_id: element.element_id.clone(),
                error,
            })?;
            let (mass_moments, mass_damping_moments) =
                laminate_property::mass_moments(response.mass);
            let mass = global_lumped_mass_matrix(mass_moments, frame).map_err(|error| {
                StructuralAssemblyError::ShellElement {
                    element_id: element.element_id.clone(),
                    error,
                }
            })?;
            let stiffness_damping =
                global_matrix_for_constitutive_matrices(&response.stiffness_damping, geometry)
                    .map_err(|error| StructuralAssemblyError::ShellElement {
                        element_id: element.element_id.clone(),
                        error,
                    })?;
            let mass_damping =
                global_lumped_mass_matrix(mass_damping_moments, frame).map_err(|error| {
                    StructuralAssemblyError::ShellElement {
                        element_id: element.element_id.clone(),
                        error,
                    }
                })?;
            let has_damping = mass_damping_moments != ShellMassMoments::<true>::zero()
                || stiffness_damping
                    .iter()
                    .flatten()
                    .any(|value| *value != 0.0);
            let damping =
                has_damping.then(|| combined_damping(&mass_damping, &stiffness_damping, 1.0));
            (stiffness, mass, damping, response.elastic)
        }
    };
    Ok(ShellElementContribution {
        dofs,
        stiffness,
        mass,
        damping,
        recovery: ShellRecoveryElementSummary {
            element_id: element.element_id.clone(),
            region_id: element.region_id.clone(),
            node_indices,
            geometry,
            constitutive,
        },
    })
}

fn combined_damping(mass: &ShellMatrix, stiffness: &ShellMatrix, scale: f64) -> ShellMatrix {
    let mut damping = [[0.0; SHELL_ELEMENT_DOF_COUNT]; SHELL_ELEMENT_DOF_COUNT];
    for row in 0..SHELL_ELEMENT_DOF_COUNT {
        for column in 0..SHELL_ELEMENT_DOF_COUNT {
            damping[row][column] = mass[row][column] + scale * stiffness[row][column];
        }
    }
    damping
}
