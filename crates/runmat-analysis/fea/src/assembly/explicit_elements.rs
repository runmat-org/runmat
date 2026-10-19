use std::collections::BTreeSet;

mod beam_contribution;
mod beam_inertia;
mod laminate_property;
mod shell_contribution;
#[cfg(test)]
mod tests;

use beam_contribution::BeamElementContribution;
use beam_inertia::add_rayleigh_element_damping;
use shell_contribution::ShellElementContribution;

use runmat_analysis_core::{
    BeamSectionModel, HomogeneousShellReferenceSurface, HomogeneousShellSectionModel,
    StructuralElementKind, StructuralFormulationRealization, StructuralModel,
    StructuralOperatorImplementation, StructuralRecoveryImplementation,
};

use super::{
    dofs::{StructuralDofKind, StructuralDofLayout},
    elements::{
        beam::{BeamSection, BEAM_ELEMENT_DOF_COUNT},
        shell::{ShellMassMoments, ShellSection, SHELL_ELEMENT_DOF_COUNT, SHELL_NODE_DOF_COUNT},
    },
    BeamRecoveryElementSummary, ShellRecoveryElementSummary, StructuralAssemblyError,
};
use crate::{
    algebra::{
        ContributionDomain, ContributionId, ContributionProvenance, MatrixProperties, OperatorRole,
        RealSparseAssembler, RealSparseMatrix,
    },
    planning::resource::{execute_scoped_chunks, LocalExecutionPlan, ScopedLocalExecutionError},
    properties::CompiledPropertyMap,
};

enum ExplicitElementContribution {
    Beam(Box<BeamElementContribution>),
    Shell(Box<ShellElementContribution>),
}

pub(super) struct ExplicitElementAssembly {
    pub stiffness: RealSparseMatrix,
    pub damping: RealSparseMatrix,
    pub mass: RealSparseMatrix,
    pub beam_recovery: Vec<BeamRecoveryElementSummary>,
    pub shell_recovery: Vec<ShellRecoveryElementSummary>,
    pub formulations: Vec<StructuralFormulationRealization>,
    pub beam_count: usize,
    pub shell_count: usize,
}

pub(super) fn assemble_explicit_elements(
    structural: &StructuralModel,
    properties: &CompiledPropertyMap,
    layout: &StructuralDofLayout,
    canonical_node_indices: &[usize],
    local_execution: LocalExecutionPlan,
) -> Result<ExplicitElementAssembly, StructuralAssemblyError> {
    validate_explicit_topology(structural, canonical_node_indices)?;
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
    let topology_digest = properties
        .identity
        .explicit_topology_digest
        .ok_or_else(|| {
            StructuralAssemblyError::InvalidMaterialCoverage(
                "compiled property map has no explicit structural topology".into(),
            )
        })?;

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
    let mut mass = RealSparseAssembler::square(
        dof_count,
        OperatorRole::Mass,
        MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
    );
    mass.add_contribution(ContributionProvenance {
        domain: ContributionDomain::Structural,
        contribution_id: ContributionId("explicit_beam_shell_lumped_mass".into()),
    });
    let mut has_rayleigh_damping = false;
    let mut beam_recovery = Vec::with_capacity(beam_count);
    let mut shell_recovery = Vec::with_capacity(shell_count);

    // Workers retain bounded element-local contributions. Every K/M/C scatter still
    // follows the authored element order, independently of worker completion order.
    const ELEMENTS_PER_WORKER_WAVE: usize = 8;
    let wave_capacity = usize::from(local_execution.thread_count().get())
        .checked_mul(ELEMENTS_PER_WORKER_WAVE)
        .ok_or_else(|| {
            StructuralAssemblyError::LocalExecution("explicit wave size overflow".into())
        })?;
    let cancellation = crate::progress::cancellation_token();
    for wave in structural.elements.chunks(wave_capacity) {
        if cancellation.is_cancelled() {
            return Err(StructuralAssemblyError::Cancelled);
        }
        let contributions = execute_scoped_chunks(local_execution, wave.len() as u64, |chunk| {
            let mut contributions = Vec::with_capacity(chunk.len() as usize);
            for element in &wave[chunk.start() as usize..chunk.end_exclusive() as usize] {
                if cancellation.is_cancelled() {
                    return Err(StructuralAssemblyError::Cancelled);
                }
                contributions.push(match &element.kind {
                    StructuralElementKind::Beam(beam) => {
                        ExplicitElementContribution::Beam(Box::new(beam_contribution::compute(
                            structural,
                            element,
                            beam,
                            properties,
                            layout,
                            canonical_node_indices,
                            topology_digest,
                        )?))
                    }
                    StructuralElementKind::Shell(shell) => {
                        ExplicitElementContribution::Shell(Box::new(shell_contribution::compute(
                            structural,
                            element,
                            shell,
                            properties,
                            layout,
                            canonical_node_indices,
                            topology_digest,
                        )?))
                    }
                });
            }
            Ok(contributions)
        })
        .map_err(|error| match error {
            ScopedLocalExecutionError::Work { source, .. } => source,
            other => StructuralAssemblyError::LocalExecution(other.to_string()),
        })?;
        for contribution in contributions.into_iter().flatten() {
            match contribution {
                ExplicitElementContribution::Beam(contribution) => {
                    scatter(&mut stiffness, &contribution.dofs, &contribution.stiffness);
                    scatter_diagonal(&mut mass, &contribution.dofs, &contribution.mass);
                    has_rayleigh_damping |= add_rayleigh_element_damping(
                        &mut damping,
                        &contribution.dofs,
                        &contribution.stiffness,
                        &contribution.mass,
                        contribution.damping,
                    );
                    beam_recovery.push(contribution.recovery);
                }
                ExplicitElementContribution::Shell(contribution) => {
                    if let Some(element_damping) = &contribution.damping {
                        scatter(&mut damping, &contribution.dofs, element_damping);
                        has_rayleigh_damping = true;
                    }
                    scatter(&mut stiffness, &contribution.dofs, &contribution.stiffness);
                    scatter(&mut mass, &contribution.dofs, &contribution.mass);
                    shell_recovery.push(contribution.recovery);
                }
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
        mass: finish(mass)?,
        beam_recovery,
        shell_recovery,
        formulations,
        beam_count,
        shell_count,
    })
}

pub(super) fn validate_explicit_topology(
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
        let referenced_node_ids: &[u32] = match &element.kind {
            StructuralElementKind::Beam(beam) => &beam.node_ids,
            StructuralElementKind::Shell(shell) => &shell.node_ids,
        };
        if let Some(node_id) = referenced_node_ids
            .iter()
            .find(|node_id| !node_ids.contains(node_id))
        {
            return Err(StructuralAssemblyError::UnknownNode {
                element_id: element.element_id.clone(),
                node_id: *node_id,
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

fn shell_section_from_model(section: &HomogeneousShellSectionModel) -> ShellSection {
    let half_thickness = 0.5 * section.thickness_m;
    ShellSection {
        thickness_m: section.thickness_m,
        reference_offset_m: match section.reference_surface {
            HomogeneousShellReferenceSurface::MidSurface => 0.0,
            HomogeneousShellReferenceSurface::NegativeNormalSurface => -half_thickness,
            HomogeneousShellReferenceSurface::PositiveNormalSurface => half_thickness,
            HomogeneousShellReferenceSurface::OffsetFromMidSurface(offset) => offset,
        },
        shear_correction: section.shear_correction,
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

fn scatter_diagonal<const D: usize>(
    global: &mut RealSparseAssembler,
    element_dofs: &[usize; D],
    element: &[f64; D],
) {
    for (local, global_dof) in element_dofs.iter().copied().enumerate() {
        global.add(global_dof, global_dof, element[local]);
    }
}

fn scale_mass_moments(moments: ShellMassMoments, scale: f64) -> ShellMassMoments {
    ShellMassMoments {
        zeroth_kg_per_m2: scale * moments.zeroth_kg_per_m2,
        first_kg_per_m: scale * moments.first_kg_per_m,
        second_kg: scale * moments.second_kg,
    }
}

fn finish(assembler: RealSparseAssembler) -> Result<RealSparseMatrix, StructuralAssemblyError> {
    assembler
        .finish()
        .map_err(|error| StructuralAssemblyError::SparseOperator(format!("{error:?}")))
}
