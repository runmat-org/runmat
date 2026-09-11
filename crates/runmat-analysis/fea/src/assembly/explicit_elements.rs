use std::collections::BTreeSet;

use runmat_analysis_core::{
    BeamSectionModel, ShellSectionModel, StructuralElementKind, StructuralFormulationRealization,
    StructuralModel, StructuralOperatorImplementation, StructuralRecoveryImplementation,
};

use super::{
    dofs::{StructuralDofKind, StructuralDofLayout},
    elements::{
        beam::{
            global_stiffness_matrix as beam_global_stiffness_matrix, transformation_matrix,
            BeamElementGeometry, BeamMaterial, BeamSection, BeamTransform12,
            BEAM_ELEMENT_DOF_COUNT,
        },
        shell::{
            global_stiffness_matrix as shell_global_stiffness_matrix, ShellElementGeometry,
            ShellMaterial, ShellSection, SHELL_ELEMENT_DOF_COUNT, SHELL_NODE_DOF_COUNT,
        },
    },
    BeamRecoveryElementSummary, ShellRecoveryElementSummary, StructuralAssemblyError,
};
use crate::{
    algebra::{
        ContributionDomain, ContributionId, ContributionProvenance, MatrixProperties, OperatorRole,
        RealSparseAssembler, RealSparseMatrix,
    },
    physics::structural::material::StructuralMaterialInventory,
};

pub(super) struct ExplicitElementAssembly {
    pub stiffness: RealSparseMatrix,
    pub damping: RealSparseMatrix,
    pub lumped_mass: Vec<f64>,
    pub beam_recovery: Vec<BeamRecoveryElementSummary>,
    pub shell_recovery: Vec<ShellRecoveryElementSummary>,
    pub formulations: Vec<StructuralFormulationRealization>,
    pub beam_count: usize,
    pub shell_count: usize,
}

pub(super) fn assemble_explicit_elements(
    structural: &StructuralModel,
    materials: &StructuralMaterialInventory,
    layout: &StructuralDofLayout,
    canonical_node_indices: &[usize],
) -> Result<ExplicitElementAssembly, StructuralAssemblyError> {
    validate_topology(structural, canonical_node_indices)?;
    let beam_count = structural
        .elements
        .iter()
        .filter(|element| matches!(element.kind, StructuralElementKind::Beam(_)))
        .count();
    let shell_count = structural
        .elements
        .iter()
        .filter(|element| matches!(element.kind, StructuralElementKind::Shell(_)))
        .count();
    if beam_count == 0 && shell_count == 0 {
        return Err(StructuralAssemblyError::MissingDiscretization);
    }

    let dof_count = layout.total_dof_count();
    let mut stiffness = RealSparseAssembler::square(
        dof_count,
        OperatorRole::Stiffness,
        MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
    );
    stiffness.add_contribution(ContributionProvenance {
        domain: ContributionDomain::Structural,
        contribution_id: ContributionId("explicit_beam_shell_stiffness".into()),
    });
    let mut damping = RealSparseAssembler::square(
        dof_count,
        OperatorRole::Damping,
        MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
    );
    let mut has_rayleigh_damping = false;
    let mut lumped_mass = vec![0.0; dof_count];
    let mut beam_recovery = Vec::with_capacity(beam_count);
    let mut shell_recovery = Vec::with_capacity(shell_count);

    for element in &structural.elements {
        match &element.kind {
            StructuralElementKind::Beam(beam) => {
                let explicit_i = structural_node_index(structural, beam.node_ids[0], element)?;
                let explicit_j = structural_node_index(structural, beam.node_ids[1], element)?;
                let node_i = canonical_node_indices[explicit_i];
                let node_j = canonical_node_indices[explicit_j];
                let section =
                    structural_beam_section(structural, &beam.section_id).ok_or_else(|| {
                        StructuralAssemblyError::UnknownBeamSection {
                            element_id: element.element_id.clone(),
                            section_id: beam.section_id.clone(),
                        }
                    })?;
                let material = materials.resolve(&element.region_id).ok_or_else(|| {
                    StructuralAssemblyError::UnassignedExplicitRegion {
                        element_id: element.element_id.clone(),
                        region_id: element.region_id.clone(),
                    }
                })?;
                let beam_material = BeamMaterial {
                    youngs_modulus_pa: material.youngs_modulus_pa,
                    shear_modulus_pa: material.shear_modulus_pa(),
                };
                let geometry = BeamElementGeometry {
                    node_i_m: structural.nodes[explicit_i].coordinates_m,
                    node_j_m: structural.nodes[explicit_j].coordinates_m,
                    reference_axis: beam.reference_axis,
                };
                let frame = geometry.local_frame().map_err(|error| {
                    StructuralAssemblyError::BeamElement {
                        element_id: element.element_id.clone(),
                        error,
                    }
                })?;
                let transform_global_to_local = transformation_matrix(frame);
                let element_stiffness =
                    beam_global_stiffness_matrix(section, beam_material, geometry).map_err(
                        |error| StructuralAssemblyError::BeamElement {
                            element_id: element.element_id.clone(),
                            error,
                        },
                    )?;
                let element_dofs =
                    beam_element_dof_indices(layout, node_i, node_j).ok_or_else(|| {
                        StructuralAssemblyError::MissingRotationalDof {
                            element_id: element.element_id.clone(),
                        }
                    })?;
                scatter(&mut stiffness, &element_dofs, &element_stiffness);
                let element_mass = lumped_beam_mass(
                    section,
                    material.density_kg_per_m3,
                    frame.length_m,
                    &transform_global_to_local,
                );
                scatter_lumped_mass(&mut lumped_mass, &element_dofs, &element_mass);
                has_rayleigh_damping |= add_rayleigh_element_damping(
                    &mut damping,
                    &element_dofs,
                    &element_stiffness,
                    &element_mass,
                    material.damping,
                );
                beam_recovery.push(BeamRecoveryElementSummary {
                    element_id: element.element_id.clone(),
                    region_id: element.region_id.clone(),
                    node_i_index: node_i,
                    node_j_index: node_j,
                    length_m: frame.length_m,
                    section,
                    material: beam_material,
                    transform_global_to_local,
                });
            }
            StructuralElementKind::Shell(shell) => {
                let mut explicit_nodes = [0_usize; 3];
                for (local, node_id) in shell.node_ids.iter().copied().enumerate() {
                    explicit_nodes[local] = structural_node_index(structural, node_id, element)?;
                }
                let node_indices = explicit_nodes.map(|node| canonical_node_indices[node]);
                let section =
                    structural_shell_section(structural, &shell.section_id).ok_or_else(|| {
                        StructuralAssemblyError::UnknownShellSection {
                            element_id: element.element_id.clone(),
                            section_id: shell.section_id.clone(),
                        }
                    })?;
                let material = materials.resolve(&element.region_id).ok_or_else(|| {
                    StructuralAssemblyError::UnassignedExplicitRegion {
                        element_id: element.element_id.clone(),
                        region_id: element.region_id.clone(),
                    }
                })?;
                let shell_material = ShellMaterial {
                    youngs_modulus_pa: material.youngs_modulus_pa,
                    poisson_ratio: material.poisson_ratio,
                    shear_modulus_pa: material.shear_modulus_pa(),
                };
                let geometry = ShellElementGeometry {
                    nodes_m: explicit_nodes.map(|node| structural.nodes[node].coordinates_m),
                    reference_axis: shell.reference_axis,
                };
                let frame = geometry.local_frame().map_err(|error| {
                    StructuralAssemblyError::ShellElement {
                        element_id: element.element_id.clone(),
                        error,
                    }
                })?;
                let element_stiffness =
                    shell_global_stiffness_matrix(section, shell_material, geometry).map_err(
                        |error| StructuralAssemblyError::ShellElement {
                            element_id: element.element_id.clone(),
                            error,
                        },
                    )?;
                let element_dofs =
                    shell_element_dof_indices(layout, node_indices).ok_or_else(|| {
                        StructuralAssemblyError::MissingRotationalDof {
                            element_id: element.element_id.clone(),
                        }
                    })?;
                scatter(&mut stiffness, &element_dofs, &element_stiffness);
                let element_mass =
                    lumped_shell_mass(section, material.density_kg_per_m3, frame.area_m2);
                scatter_lumped_mass(&mut lumped_mass, &element_dofs, &element_mass);
                has_rayleigh_damping |= add_rayleigh_element_damping(
                    &mut damping,
                    &element_dofs,
                    &element_stiffness,
                    &element_mass,
                    material.damping,
                );
                shell_recovery.push(ShellRecoveryElementSummary {
                    element_id: element.element_id.clone(),
                    region_id: element.region_id.clone(),
                    node_indices,
                    geometry,
                    section,
                    material: shell_material,
                });
            }
        }
    }

    damping.add_contribution(ContributionProvenance {
        domain: ContributionDomain::Structural,
        contribution_id: ContributionId(if has_rayleigh_damping {
            "explicit_element_rayleigh_damping".into()
        } else {
            "undamped_explicit_structure".into()
        }),
    });
    let mut formulations = Vec::new();
    if beam_count > 0 {
        formulations.push(StructuralFormulationRealization::new(
            StructuralOperatorImplementation::EulerBernoulliBeam3d,
            Some(StructuralRecoveryImplementation::BeamSectionResultants),
        ));
    }
    if shell_count > 0 {
        formulations.push(StructuralFormulationRealization::new(
            StructuralOperatorImplementation::Mitc3TriangularShell,
            Some(StructuralRecoveryImplementation::Mitc3ShellStressResultants),
        ));
    }
    Ok(ExplicitElementAssembly {
        stiffness: finish(stiffness)?,
        damping: finish(damping)?,
        lumped_mass,
        beam_recovery,
        shell_recovery,
        formulations,
        beam_count,
        shell_count,
    })
}

fn validate_topology(
    structural: &StructuralModel,
    canonical_node_indices: &[usize],
) -> Result<(), StructuralAssemblyError> {
    if structural.nodes.is_empty() {
        return Err(StructuralAssemblyError::MissingNodes);
    }
    if canonical_node_indices.len() != structural.nodes.len() {
        return Err(StructuralAssemblyError::InvalidCanonicalNodeMap {
            expected: structural.nodes.len(),
            actual: canonical_node_indices.len(),
        });
    }
    let mut node_ids = BTreeSet::new();
    for node in &structural.nodes {
        if !node_ids.insert(node.node_id) {
            return Err(StructuralAssemblyError::DuplicateNodeId {
                node_id: node.node_id,
            });
        }
    }
    let mut element_ids = BTreeSet::new();
    for element in &structural.elements {
        if !element_ids.insert(element.element_id.as_str()) {
            return Err(StructuralAssemblyError::DuplicateElementId {
                element_id: element.element_id.clone(),
            });
        }
    }
    Ok(())
}

fn structural_node_index(
    structural: &StructuralModel,
    node_id: u32,
    element: &runmat_analysis_core::StructuralElement,
) -> Result<usize, StructuralAssemblyError> {
    structural
        .nodes
        .iter()
        .position(|node| node.node_id == node_id)
        .ok_or_else(|| StructuralAssemblyError::UnknownNode {
            element_id: element.element_id.clone(),
            node_id,
        })
}

fn structural_beam_section(structural: &StructuralModel, section_id: &str) -> Option<BeamSection> {
    structural
        .beam_sections
        .iter()
        .find(|section| section.section_id == section_id)
        .map(section_from_model)
}

fn structural_shell_section(
    structural: &StructuralModel,
    section_id: &str,
) -> Option<ShellSection> {
    structural
        .shell_sections
        .iter()
        .find(|section| section.section_id == section_id)
        .map(shell_section_from_model)
}

fn section_from_model(section: &BeamSectionModel) -> BeamSection {
    BeamSection {
        area_m2: section.area_m2,
        iy_m4: section.iy_m4,
        iz_m4: section.iz_m4,
        torsion_j_m4: section.torsion_j_m4,
        outer_fiber_y_m: section.outer_fiber_y_m,
        outer_fiber_z_m: section.outer_fiber_z_m,
        torsion_outer_radius_m: section.torsion_outer_radius_m,
    }
}

fn shell_section_from_model(section: &ShellSectionModel) -> ShellSection {
    ShellSection {
        thickness_m: section.thickness_m,
        shear_correction: section.shear_correction,
        drilling_stiffness_scale: section.drilling_stiffness_scale,
    }
}

fn beam_element_dof_indices(
    layout: &StructuralDofLayout,
    node_i_index: usize,
    node_j_index: usize,
) -> Option<[usize; BEAM_ELEMENT_DOF_COUNT]> {
    let mut indices = [0usize; BEAM_ELEMENT_DOF_COUNT];
    for (local, kind) in StructuralDofKind::ORDER.iter().copied().enumerate() {
        indices[local] = layout.index(node_i_index, kind)?;
        indices[local + 6] = layout.index(node_j_index, kind)?;
    }
    Some(indices)
}

fn shell_element_dof_indices(
    layout: &StructuralDofLayout,
    node_indices: [usize; 3],
) -> Option<[usize; SHELL_ELEMENT_DOF_COUNT]> {
    let mut indices = [0usize; SHELL_ELEMENT_DOF_COUNT];
    for (node_offset, node_index) in node_indices.iter().enumerate() {
        for (component, kind) in StructuralDofKind::ORDER.iter().enumerate() {
            indices[node_offset * SHELL_NODE_DOF_COUNT + component] =
                layout.index(*node_index, *kind)?;
        }
    }
    Some(indices)
}

fn scatter<const D: usize>(
    assembler: &mut RealSparseAssembler,
    element_dofs: &[usize; D],
    element: &[[f64; D]; D],
) {
    for (local_row, global_row) in element_dofs.iter().copied().enumerate() {
        for (local_column, global_column) in element_dofs.iter().copied().enumerate() {
            assembler.add(global_row, global_column, element[local_row][local_column]);
        }
    }
}

fn scatter_lumped_mass<const D: usize>(
    global: &mut [f64],
    element_dofs: &[usize; D],
    element: &[f64; D],
) {
    for (local, global_dof) in element_dofs.iter().copied().enumerate() {
        global[global_dof] += element[local];
    }
}

fn lumped_beam_mass(
    section: BeamSection,
    density_kg_per_m3: f64,
    length_m: f64,
    transform_global_to_local: &BeamTransform12,
) -> [f64; BEAM_ELEMENT_DOF_COUNT] {
    let nodal_mass = density_kg_per_m3 * section.area_m2 * length_m / 2.0;
    let local_rotational_inertia = [
        density_kg_per_m3 * section.torsion_j_m4 * length_m / 2.0,
        density_kg_per_m3 * section.iy_m4 * length_m / 2.0,
        density_kg_per_m3 * section.iz_m4 * length_m / 2.0,
    ];
    let mut element_mass = [0.0; BEAM_ELEMENT_DOF_COUNT];
    for node_offset in [0usize, 6] {
        for component in 0..3 {
            element_mass[node_offset + component] = nodal_mass;
        }
        for global_component in 0..3 {
            let local_col = node_offset + 3 + global_component;
            element_mass[node_offset + 3 + global_component] = local_rotational_inertia
                .iter()
                .enumerate()
                .map(|(local_component, local_inertia)| {
                    let row = node_offset + 3 + local_component;
                    local_inertia * transform_global_to_local[row][local_col].powi(2)
                })
                .sum();
        }
    }
    element_mass
}

fn lumped_shell_mass(
    section: ShellSection,
    density_kg_per_m3: f64,
    area_m2: f64,
) -> [f64; SHELL_ELEMENT_DOF_COUNT] {
    let nodal_mass = density_kg_per_m3 * section.thickness_m * area_m2 / 3.0;
    let nodal_rotary_inertia = nodal_mass * section.thickness_m.powi(2) / 12.0;
    let mut element_mass = [0.0; SHELL_ELEMENT_DOF_COUNT];
    for node_offset in [0usize, 6, 12] {
        for component in 0..3 {
            element_mass[node_offset + component] = nodal_mass;
        }
        for component in 3..6 {
            element_mass[node_offset + component] = nodal_rotary_inertia;
        }
    }
    element_mass
}

fn add_rayleigh_element_damping<const D: usize>(
    damping: &mut RealSparseAssembler,
    element_dofs: &[usize; D],
    stiffness: &[[f64; D]; D],
    lumped_mass: &[f64; D],
    law: runmat_analysis_core::MechanicalDampingModel,
) -> bool {
    let runmat_analysis_core::MechanicalDampingModel::Rayleigh {
        mass_coefficient,
        stiffness_coefficient,
    } = law
    else {
        return false;
    };
    for (local_row, global_row) in element_dofs.iter().copied().enumerate() {
        for (local_column, global_column) in element_dofs.iter().copied().enumerate() {
            let mass = if local_row == local_column {
                lumped_mass[local_row]
            } else {
                0.0
            };
            damping.add(
                global_row,
                global_column,
                mass_coefficient.per_second * mass
                    + stiffness_coefficient.seconds * stiffness[local_row][local_column],
            );
        }
    }
    true
}

fn finish(assembler: RealSparseAssembler) -> Result<RealSparseMatrix, StructuralAssemblyError> {
    assembler
        .finish()
        .map_err(|error| StructuralAssemblyError::SparseOperator(format!("{error:?}")))
}
