use std::collections::BTreeSet;

mod beam_contribution;
mod beam_inertia;
mod laminate_property;
mod shell_contribution;
#[cfg(test)]
mod tests;


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
    algebra::element_assembly::{
        assemble_elements, no_visit, ElementAssemblyError, ElementDofMap, ElementOutputShape,
        ElementOutputs,
    },
    planning::resource::LocalExecutionPlan,
    properties::CompiledPropertyMap,
};

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
    let node_index: ExplicitNodeIndex = structural
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.node_id, index))
        .collect();
    let elements = &structural.elements;
    // Element DOFs; an unresolvable element gets no DOFs and its kernel reports the error.
    let map = ElementDofMap::from_elements(dof_count, elements.len(), local_execution, |index, out| {
        let element = &elements[index];
        let canonical = |node_id: &u32| {
            node_index
                .get(node_id)
                .and_then(|&explicit| canonical_node_indices.get(explicit).copied())
        };
        match &element.kind {
            StructuralElementKind::Beam(beam) => {
                if let (Some(i), Some(j)) = (canonical(&beam.node_ids[0]), canonical(&beam.node_ids[1])) {
                    if let Some(dofs) = beam_element_dof_indices(layout, i, j) {
                        out.extend(dofs.iter().map(|&dof| Some(dof)));
                    }
                }
            }
            StructuralElementKind::Shell(shell) => {
                let nodes = [
                    canonical(&shell.node_ids[0]),
                    canonical(&shell.node_ids[1]),
                    canonical(&shell.node_ids[2]),
                ];
                if let [Some(a), Some(b), Some(c)] = nodes {
                    if let Some(dofs) = shell_element_dof_indices(layout, [a, b, c]) {
                        out.extend(dofs.iter().map(|&dof| Some(dof)));
                    }
                }
            }
        }
        Ok::<_, StructuralAssemblyError>(())
    })
    .map_err(explicit_assembly_error)?;
    let pattern = map
        .symbolic_pattern(local_execution)
        .map_err(|error| explicit_assembly_error(error.map_element(|never| match never {})))?;
    let recovery = (0..elements.len())
        .map(|_| std::sync::OnceLock::new())
        .collect::<Vec<std::sync::OnceLock<ExplicitRecovery>>>();
    const K: usize = 0;
    const M: usize = 1;
    const C: usize = 2;
    let assembled = assemble_elements(
        &map,
        &pattern,
        ElementOutputShape {
            matrices: 3,
            vectors: 0,
            scalars: 1,
        },
        local_execution,
        |index, dofs, out: &mut ElementOutputs<'_, f64>| {
            let element = &elements[index];
            match &element.kind {
                StructuralElementKind::Beam(beam) => {
                    let contribution = beam_contribution::compute(
                        structural,
                        &node_index,
                        element,
                        beam,
                        properties,
                        layout,
                        canonical_node_indices,
                        topology_digest,
                    )?;
                    check_element_dofs(element, dofs, &contribution.dofs)?;
                    write_matrix(out.matrix(K), &contribution.stiffness);
                    let mass = out.matrix(M);
                    for (local, value) in contribution.mass.iter().enumerate() {
                        mass[local * BEAM_ELEMENT_DOF_COUNT + local] = *value;
                    }
                    if let runmat_analysis_core::MechanicalDampingModel::Rayleigh {
                        mass_coefficient,
                        stiffness_coefficient,
                    } = contribution.damping
                    {
                        let damping = out.matrix(C);
                        for row in 0..BEAM_ELEMENT_DOF_COUNT {
                            for column in 0..BEAM_ELEMENT_DOF_COUNT {
                                let mass = if row == column { contribution.mass[row] } else { 0.0 };
                                damping[row * BEAM_ELEMENT_DOF_COUNT + column] = mass_coefficient
                                    .per_second
                                    * mass
                                    + stiffness_coefficient.seconds
                                        * contribution.stiffness[row][column];
                            }
                        }
                        *out.scalar(0) = 1.0;
                    }
                    let _ = recovery[index].set(ExplicitRecovery::Beam(contribution.recovery));
                }
                StructuralElementKind::Shell(shell) => {
                    let contribution = shell_contribution::compute(
                        structural,
                        &node_index,
                        element,
                        shell,
                        properties,
                        layout,
                        canonical_node_indices,
                        topology_digest,
                    )?;
                    check_element_dofs(element, dofs, &contribution.dofs)?;
                    write_matrix(out.matrix(K), &contribution.stiffness);
                    write_matrix(out.matrix(M), &contribution.mass);
                    if let Some(damping) = &contribution.damping {
                        write_matrix(out.matrix(C), damping);
                        *out.scalar(0) = 1.0;
                    }
                    let _ = recovery[index].set(ExplicitRecovery::Shell(contribution.recovery));
                }
            }
            Ok(())
        },
        no_visit,
    )
    .map_err(explicit_assembly_error)?;
    let has_rayleigh_damping = assembled.scalars[0] > 0.0;
    let mut beam_recovery = Vec::with_capacity(beam_count);
    let mut shell_recovery = Vec::with_capacity(shell_count);
    for slot in recovery {
        match slot.into_inner() {
            Some(ExplicitRecovery::Beam(summary)) => beam_recovery.push(summary),
            Some(ExplicitRecovery::Shell(summary)) => shell_recovery.push(summary),
            None => {
                return Err(StructuralAssemblyError::LocalExecution(
                    "explicit element recovery summary missing".into(),
                ))
            }
        }
    }
    let mut matrices = assembled.matrices.into_iter();
    let planned = |values: Vec<f64>, role| {
        RealSparseAssembler::from_planned_values(
            pattern.clone(),
            values,
            role,
            MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
        )
        .map_err(|error| StructuralAssemblyError::SparseOperator(format!("{error:?}")))
    };
    let mut stiffness = planned(matrices.next().unwrap_or_default(), OperatorRole::Stiffness)?;
    stiffness.add_contribution(ContributionProvenance {
        domain: ContributionDomain::Structural,
        contribution_id: ContributionId("explicit_beam_shell_stiffness".into()),
    });
    let mut mass = planned(matrices.next().unwrap_or_default(), OperatorRole::Mass)?;
    mass.add_contribution(ContributionProvenance {
        domain: ContributionDomain::Structural,
        contribution_id: ContributionId("explicit_beam_shell_lumped_mass".into()),
    });
    let mut damping = planned(matrices.next().unwrap_or_default(), OperatorRole::Damping)?;

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

/// Explicit structural node id -> position in `StructuralModel::nodes`.
pub(super) type ExplicitNodeIndex = std::collections::BTreeMap<u32, usize>;

enum ExplicitRecovery {
    Beam(BeamRecoveryElementSummary),
    Shell(ShellRecoveryElementSummary),
}

fn structural_node_index(
    nodes: &ExplicitNodeIndex,
    node_id: u32,
    element: &runmat_analysis_core::StructuralElement,
) -> Result<usize, StructuralAssemblyError> {
    nodes
        .get(&node_id)
        .copied()
        .ok_or_else(|| StructuralAssemblyError::UnknownNode {
            element_id: element.element_id.clone(),
            node_id,
        })
}

fn check_element_dofs(
    element: &runmat_analysis_core::StructuralElement,
    mapped: &[u32],
    computed: &[usize],
) -> Result<(), StructuralAssemblyError> {
    if mapped.len() != computed.len()
        || mapped
            .iter()
            .zip(computed)
            .any(|(&mapped, &computed)| mapped as usize != computed)
    {
        return Err(StructuralAssemblyError::LocalExecution(format!(
            "explicit element {} DOFs differ from the assembly map",
            element.element_id
        )));
    }
    Ok(())
}

fn write_matrix<const D: usize>(target: &mut [f64], element: &[[f64; D]; D]) {
    for (row, values) in target.chunks_exact_mut(D).zip(element) {
        row.copy_from_slice(values);
    }
}

fn explicit_assembly_error(
    error: ElementAssemblyError<StructuralAssemblyError>,
) -> StructuralAssemblyError {
    match error {
        ElementAssemblyError::Element { error, .. } => error,
        ElementAssemblyError::Cancelled => StructuralAssemblyError::Cancelled,
        other => StructuralAssemblyError::LocalExecution(other.to_string()),
    }
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
