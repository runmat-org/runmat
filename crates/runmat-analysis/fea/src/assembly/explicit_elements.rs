use std::{cell::RefCell, collections::BTreeSet, future::Future, rc::Rc};

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
        shell::{ShellSection, SHELL_ELEMENT_DOF_COUNT, SHELL_NODE_DOF_COUNT},
    },
    BeamRecoveryElementSummary, ShellRecoveryElementSummary, StructuralAssemblyError,
};
use crate::{
    algebra::element_assembly::{
        assemble_elements, assemble_elements_with_async_owner, no_visit, AsyncElementKernel,
        ElementAssemblyError, ElementDofMap, ElementFillFuture, ElementOutputShape, ElementOutputs,
    },
    algebra::{
        ContributionDomain, ContributionId, ContributionProvenance, MatrixProperties, OperatorRole,
        RealSparseAssembler, RealSparseMatrix,
    },
    planning::resource::LocalExecutionPlan,
    properties::CompiledPropertyMap,
};
use runmat_simulation_usage::OperationCount;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ExplicitCapacityPhase {
    Element { index: usize },
    Scatter { elements: std::ops::Range<usize> },
}

struct ExplicitAwaitedKernel<'a, K, O, F> {
    kernel: &'a K,
    before: Rc<RefCell<O>>,
    element_work: &'a [OperationCount],
    _future: std::marker::PhantomData<fn() -> F>,
}

impl<K, O, F, A> AsyncElementKernel<f64, StructuralAssemblyError, A>
    for ExplicitAwaitedKernel<'_, K, O, F>
where
    K: Fn(usize, &[u32], &mut ElementOutputs<'_, f64>) -> Result<(), StructuralAssemblyError>,
    O: FnMut(ExplicitCapacityPhase, OperationCount) -> F,
    F: Future<Output = Result<(), A>>,
{
    fn fill<'a>(
        &'a mut self,
        index: usize,
        dofs: &'a [u32],
        values: &'a mut [f64],
        shape: ElementOutputShape,
    ) -> ElementFillFuture<'a, StructuralAssemblyError, A> {
        Box::pin(async move {
            let future = (self.before.borrow_mut())(
                ExplicitCapacityPhase::Element { index },
                self.element_work[index],
            );
            future.await.map_err(ElementAssemblyError::Admission)?;
            let mut out = ElementOutputs::from_buffer(dofs.len(), shape, values);
            (self.kernel)(index, dofs, &mut out).map_err(|error| ElementAssemblyError::Element {
                element: index,
                error,
            })
        })
    }
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

fn explicit_primitive_work(
    element: &runmat_analysis_core::StructuralElement,
    properties: &CompiledPropertyMap,
    topology_digest: runmat_execution::Digest,
) -> Result<OperationCount, StructuralAssemblyError> {
    let operations = match &element.kind {
        StructuralElementKind::Beam(_) => 2 * 43 + 64 + 6_624,
        StructuralElementKind::Shell(_) => {
            let identity = crate::physics::structural::topology::explicit_element_identity(
                topology_digest,
                &element.element_id,
                &element.region_id,
                &element.kind,
            )
            .map_err(|error| StructuralAssemblyError::InvalidMaterialCoverage(error.to_string()))?;
            let property = properties.shell_property(identity).ok_or_else(|| {
                StructuralAssemblyError::UnassignedExplicitRegion {
                    element_id: element.element_id.clone(),
                    region_id: element.region_id.clone(),
                }
            })?;
            // Frame construction uses the documented maximum of its two valid axes.
            // The remaining calls have fixed reference schedules for the selected property.
            let matrix = 62 + 68 + 597 + 74_491;
            match &property.owner {
                crate::properties::CompiledShellPropertyOwner::Homogeneous { .. } => {
                    62 + 55 + 55 + matrix
                }
                crate::properties::CompiledShellPropertyOwner::Laminate { .. } => 62 + 2 * matrix,
            }
        }
    };
    Ok(OperationCount::from_operations(operations))
}

#[cfg(test)]
pub(super) fn assemble_explicit_elements(
    structural: &StructuralModel,
    properties: &CompiledPropertyMap,
    layout: &StructuralDofLayout,
    canonical_node_indices: &[usize],
    local_execution: LocalExecutionPlan,
) -> Result<ExplicitElementAssembly, StructuralAssemblyError> {
    let future = assemble_explicit_elements_with_owner(
        structural,
        properties,
        layout,
        canonical_node_indices,
        local_execution,
        false,
        |_, _| std::future::ready(Ok::<(), StructuralAssemblyError>(())),
    );
    let mut future = std::pin::pin!(future);
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    match Future::poll(future.as_mut(), &mut context) {
        std::task::Poll::Ready(result) => result,
        std::task::Poll::Pending => unreachable!("immediate explicit assembly cannot suspend"),
    }
}

pub(super) async fn assemble_explicit_elements_with_owner<O, F, A>(
    structural: &StructuralModel,
    properties: &CompiledPropertyMap,
    layout: &StructuralDofLayout,
    canonical_node_indices: &[usize],
    local_execution: LocalExecutionPlan,
    awaited_elements: bool,
    before: O,
) -> Result<ExplicitElementAssembly, A>
where
    O: FnMut(ExplicitCapacityPhase, OperationCount) -> F,
    F: Future<Output = Result<(), A>>,
    A: From<StructuralAssemblyError> + Send,
{
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
        return Err(StructuralAssemblyError::MissingDiscretization.into());
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
    let map =
        ElementDofMap::from_elements(dof_count, elements.len(), local_execution, |index, out| {
            let element = &elements[index];
            let canonical = |node_id: &u32| {
                node_index
                    .get(node_id)
                    .and_then(|&explicit| canonical_node_indices.get(explicit).copied())
            };
            match &element.kind {
                StructuralElementKind::Beam(beam) => {
                    if let (Some(i), Some(j)) =
                        (canonical(&beam.node_ids[0]), canonical(&beam.node_ids[1]))
                    {
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
    let pattern = map.symbolic_pattern(local_execution).map_err(|error| {
        explicit_assembly_error(
            error
                .map_element(|never| match never {})
                .map_admission(|never| match never {}),
        )
    })?;
    let recovery = (0..elements.len())
        .map(|_| std::sync::OnceLock::new())
        .collect::<Vec<std::sync::OnceLock<ExplicitRecovery>>>();
    let element_work = if awaited_elements {
        elements
            .iter()
            .map(|element| explicit_primitive_work(element, properties, topology_digest))
            .collect::<Result<Vec<_>, _>>()?
    } else {
        Vec::new()
    };
    const K: usize = 0;
    const M: usize = 1;
    const C: usize = 2;
    let shape = ElementOutputShape {
        matrices: 3,
        vectors: 0,
        scalars: 1,
    };
    let kernel = |index: usize, dofs: &[u32], out: &mut ElementOutputs<'_, f64>| {
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
                            let mass = if row == column {
                                contribution.mass[row]
                            } else {
                                0.0
                            };
                            damping[row * BEAM_ELEMENT_DOF_COUNT + column] =
                                mass_coefficient.canonical_value() * mass
                                    + stiffness_coefficient.canonical_value()
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
    };
    let assembled = if awaited_elements {
        let before = Rc::new(RefCell::new(before));
        let mut awaited_kernel = ExplicitAwaitedKernel::<_, _, F> {
            kernel: &kernel,
            before: Rc::clone(&before),
            element_work: &element_work,
            _future: std::marker::PhantomData,
        };
        assemble_elements_with_async_owner(
            &map,
            &pattern,
            shape,
            local_execution,
            1,
            &kernel,
            &mut awaited_kernel,
            no_visit,
            |elements, count| {
                let future =
                    (before.borrow_mut())(ExplicitCapacityPhase::Scatter { elements }, count);
                async move { future.await.map_err(ElementAssemblyError::Admission) }
            },
        )
        .await
        .map_err(explicit_owned_error)?
    } else {
        assemble_elements(&map, &pattern, shape, local_execution, kernel, no_visit)
            .map_err(explicit_assembly_error)?
    };
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
                )
                .into())
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
        ElementAssemblyError::Admission(error) => error,
        ElementAssemblyError::Accounting(error) => StructuralAssemblyError::Accounting(error),
        ElementAssemblyError::Cancelled => StructuralAssemblyError::Cancelled,
        other => StructuralAssemblyError::LocalExecution(other.to_string()),
    }
}

fn explicit_owned_error<A: From<StructuralAssemblyError>>(
    error: ElementAssemblyError<StructuralAssemblyError, A>,
) -> A {
    match error {
        ElementAssemblyError::Admission(error) => error,
        ElementAssemblyError::Element { error, .. } => error.into(),
        ElementAssemblyError::Accounting(error) => {
            StructuralAssemblyError::Accounting(error).into()
        }
        ElementAssemblyError::Cancelled => StructuralAssemblyError::Cancelled.into(),
        ElementAssemblyError::DofOutOfRange {
            element,
            dof,
            dof_count,
        } => StructuralAssemblyError::LocalExecution(format!(
            "element {element} references degree of freedom {dof} outside 0..{dof_count}"
        ))
        .into(),
        ElementAssemblyError::DofIndexOverflow(count) => StructuralAssemblyError::LocalExecution(
            format!("degree-of-freedom count {count} exceeds the 32-bit assembly index"),
        )
        .into(),
        ElementAssemblyError::OutsidePattern { row, column } => {
            StructuralAssemblyError::LocalExecution(format!(
                "assembly entry ({row}, {column}) is outside the symbolic pattern"
            ))
            .into()
        }
        ElementAssemblyError::PatternMismatch(reason) => {
            StructuralAssemblyError::LocalExecution(reason.into()).into()
        }
        ElementAssemblyError::Pattern(error) => {
            StructuralAssemblyError::LocalExecution(error.to_string()).into()
        }
        ElementAssemblyError::Allocation(count) => StructuralAssemblyError::LocalExecution(
            format!("assembly storage allocation failed for {count} entries"),
        )
        .into(),
        ElementAssemblyError::Execution(reason) => {
            StructuralAssemblyError::LocalExecution(reason).into()
        }
    }
}

fn section_from_model(section: &BeamSectionModel) -> BeamSection {
    BeamSection {
        area_m2: runmat_types::physical::Area::new(section.area_m2.canonical_value()),
        iy_m4: runmat_types::physical::SecondMomentOfArea::new(section.iy_m4.canonical_value()),
        iz_m4: runmat_types::physical::SecondMomentOfArea::new(section.iz_m4.canonical_value()),
        torsion_j_m4: runmat_types::physical::SecondMomentOfArea::new(
            section.torsion_j_m4.canonical_value(),
        ),
        outer_fiber_y_m: runmat_types::physical::Length::new(
            section.outer_fiber_y_m.canonical_value(),
        ),
        outer_fiber_z_m: runmat_types::physical::Length::new(
            section.outer_fiber_z_m.canonical_value(),
        ),
        torsion_outer_radius_m: runmat_types::physical::Length::new(
            section.torsion_outer_radius_m.canonical_value(),
        ),
    }
}

fn shell_section_from_model(section: &HomogeneousShellSectionModel) -> ShellSection {
    let half_thickness = 0.5 * section.thickness_m.canonical_value();
    ShellSection {
        thickness_m: runmat_types::physical::Length::new(section.thickness_m.canonical_value()),
        reference_offset_m: runmat_types::physical::Length::new(match section.reference_surface {
            HomogeneousShellReferenceSurface::MidSurface => 0.0,
            HomogeneousShellReferenceSurface::NegativeNormalSurface => -half_thickness,
            HomogeneousShellReferenceSurface::PositiveNormalSurface => half_thickness,
            HomogeneousShellReferenceSurface::OffsetFromMidSurface(offset) => offset,
        }),
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

fn finish(assembler: RealSparseAssembler) -> Result<RealSparseMatrix, StructuralAssemblyError> {
    assembler
        .finish()
        .map_err(|error| StructuralAssemblyError::SparseOperator(format!("{error:?}")))
}
