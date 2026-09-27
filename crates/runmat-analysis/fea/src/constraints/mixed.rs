use runmat_analysis_core::{
    AnalysisModel, StructuralConstraint, StructuralConstraintKind, StructuralRotationDof,
    TargetCardinality,
};
use runmat_meshing_core::SolverMeshArtifact;

use crate::physics::structural::topology::{
    explicit_target_node_indices,
    node_space::CanonicalStructuralNodeSpace,
    target::{resolve_mixed_structural_target, MixedStructuralTargetOwnership},
};

use super::{
    compilation::CompiledConstraint,
    contract::{
        CanonicalStructuralConstraintTerm, CanonicalStructuralDof, CanonicalStructuralDofOwner,
        ConstraintCouplingKind, ConstraintPlanError, ResolvedConstraintTarget,
    },
    frame::require_global,
    mixed_surface,
    pairing::{pair_mixed_targets, pairing_rows},
    row,
    target::resolve_solid_target,
};

pub(super) fn requires_mixed_compilation(
    constraint: &StructuralConstraint,
    mesh: &SolverMeshArtifact,
    structural: &runmat_analysis_core::StructuralModel,
    topology: &CanonicalStructuralNodeSpace,
) -> Result<bool, ConstraintPlanError> {
    let mut mixed = false;
    visit_targets(&constraint.kind, |target| {
        if mixed {
            return;
        }
        mixed = !matches!(
            resolve_mixed_structural_target(target, mesh, structural, topology),
            Ok(MixedStructuralTargetOwnership::Solid { .. })
        );
    });
    if matches!(constraint.kind, StructuralConstraintKind::Equation(_)) {
        require_global(constraint)?;
    }
    Ok(mixed)
}

pub(super) fn compile_node_constraint(
    model: &AnalysisModel,
    mesh: Option<&SolverMeshArtifact>,
    structural: &runmat_analysis_core::StructuralModel,
    topology: &CanonicalStructuralNodeSpace,
    constraint: &StructuralConstraint,
) -> Result<CompiledConstraint, ConstraintPlanError> {
    match &constraint.kind {
        StructuralConstraintKind::Fixed { target } => {
            require_global(constraint)?;
            let (resolved, nodes) = resolve_target(mesh, structural, topology, constraint, target)?;
            let mut rows = Vec::new();
            for node in nodes {
                for dof in available_dofs(topology, node) {
                    rows.push(row::new_singleton(
                        &constraint.constraint_id,
                        topology.nodes[node].identity,
                        dof,
                        1.0,
                        0.0,
                    )?);
                }
            }
            Ok(CompiledConstraint {
                targets: vec![resolved],
                rows,
                pairings: Vec::new(),
                coupling: None,
            })
        }
        StructuralConstraintKind::Prescribed(values) => {
            mixed_surface::compile_prescribed(model, mesh, structural, topology, constraint, values)
        }
        StructuralConstraintKind::PrescribedMotion(motion) => {
            require_global(constraint)?;
            let (resolved, nodes) =
                resolve_target(mesh, structural, topology, constraint, &motion.target)?;
            let mut rows = Vec::new();
            for node in nodes {
                for translation in &motion.translations {
                    let dof = CanonicalStructuralDof::from(translation.dof);
                    if !available_dofs(topology, node).contains(&dof) {
                        return Err(ConstraintPlanError::UnavailableRotationalDof {
                            constraint_id: constraint.constraint_id.clone(),
                        });
                    }
                    rows.push(row::new_singleton(
                        &constraint.constraint_id,
                        topology.nodes[node].identity,
                        dof,
                        1.0,
                        0.0,
                    )?);
                }
            }
            Ok(CompiledConstraint {
                targets: vec![resolved],
                rows,
                pairings: Vec::new(),
                coupling: None,
            })
        }
        StructuralConstraintKind::Equation(equation) => {
            require_global(constraint)?;
            if equation.terms.is_empty() {
                return Err(ConstraintPlanError::Invalid(format!(
                    "constraint `{}` has no equation terms",
                    constraint.constraint_id
                )));
            }
            let mut targets = Vec::new();
            let mut terms = Vec::new();
            for term in &equation.terms {
                if !term.coefficient.is_finite() || term.coefficient == 0.0 {
                    return Err(ConstraintPlanError::Invalid(format!(
                        "constraint `{}` contains a zero or non-finite equation coefficient",
                        constraint.constraint_id
                    )));
                }
                let (resolved, nodes) =
                    resolve_target(mesh, structural, topology, constraint, &term.target)?;
                for node in nodes {
                    let dof = CanonicalStructuralDof::from(term.dof);
                    if !available_dofs(topology, node).contains(&dof) {
                        return Err(ConstraintPlanError::UnavailableRotationalDof {
                            constraint_id: constraint.constraint_id.clone(),
                        });
                    }
                    terms.push(CanonicalStructuralConstraintTerm {
                        owner: CanonicalStructuralDofOwner::StructuralNode(
                            topology.nodes[node].identity,
                        ),
                        dof,
                        coefficient: term.coefficient,
                    });
                }
                targets.push(resolved);
            }
            if !equation.right_hand_side.meters.is_finite() {
                return Err(ConstraintPlanError::NonFiniteValue {
                    constraint_id: constraint.constraint_id.clone(),
                });
            }
            Ok(CompiledConstraint {
                targets,
                rows: vec![row::new_sparse(
                    &constraint.constraint_id,
                    terms,
                    equation.right_hand_side.meters,
                )?],
                pairings: Vec::new(),
                coupling: None,
            })
        }
        StructuralConstraintKind::Tie(pair) => compile_pair(
            mesh,
            structural,
            topology,
            constraint,
            pair,
            &runmat_analysis_core::PeriodicTransform::Identity,
            false,
        ),
        StructuralConstraintKind::Periodic(periodic) => compile_pair(
            mesh,
            structural,
            topology,
            constraint,
            &periodic.pair,
            &periodic.transform,
            true,
        ),
        StructuralConstraintKind::NormalDisplacement { target, value } => {
            mixed_surface::compile_normal(
                model, mesh, structural, topology, constraint, target, value,
            )
        }
        StructuralConstraintKind::Symmetry { target } => {
            mixed_surface::compile_directions(mesh, structural, topology, constraint, target, false)
        }
        StructuralConstraintKind::Antisymmetry { target } => {
            mixed_surface::compile_directions(mesh, structural, topology, constraint, target, true)
        }
        StructuralConstraintKind::RigidCoupling(coupling) => mixed_surface::compile_coupling(
            mesh,
            structural,
            topology,
            constraint,
            coupling,
            ConstraintCouplingKind::Rigid,
        ),
        StructuralConstraintKind::DistributingCoupling(coupling) => {
            mixed_surface::compile_coupling(
                mesh,
                structural,
                topology,
                constraint,
                coupling,
                ConstraintCouplingKind::Distributing,
            )
        }
    }
}

fn compile_pair(
    mesh: Option<&SolverMeshArtifact>,
    structural: &runmat_analysis_core::StructuralModel,
    topology: &CanonicalStructuralNodeSpace,
    constraint: &StructuralConstraint,
    pair: &runmat_analysis_core::PairedConstraint,
    transform: &runmat_analysis_core::PeriodicTransform,
    transform_displacement: bool,
) -> Result<CompiledConstraint, ConstraintPlanError> {
    require_global(constraint)?;
    let Some(mesh) = mesh else {
        return Err(ConstraintPlanError::UnsupportedTarget {
            constraint_id: constraint.constraint_id.clone(),
            reason:
                "explicit-only tie and periodic pairing requires a qualified geometric topology",
        });
    };
    let (primary, _) = resolve_target(Some(mesh), structural, topology, constraint, &pair.primary)?;
    let (secondary, _) = resolve_target(
        Some(mesh),
        structural,
        topology,
        constraint,
        &pair.secondary,
    )?;
    let pairings = pair_mixed_targets(
        mesh,
        topology,
        &primary,
        &secondary,
        &pair.pairing,
        transform,
        &constraint.constraint_id,
    )?;
    let rows = pairing_rows(
        &constraint.constraint_id,
        &pairings,
        transform,
        transform_displacement,
    )?;
    Ok(CompiledConstraint {
        targets: vec![primary, secondary],
        rows,
        pairings,
        coupling: None,
    })
}

pub(super) fn resolve_target(
    mesh: Option<&SolverMeshArtifact>,
    structural: &runmat_analysis_core::StructuralModel,
    topology: &CanonicalStructuralNodeSpace,
    constraint: &StructuralConstraint,
    target: &runmat_analysis_core::RegionTarget,
) -> Result<(ResolvedConstraintTarget, Vec<usize>), ConstraintPlanError> {
    let ownership = if let Some(mesh) = mesh {
        resolve_mixed_structural_target(target, mesh, structural, topology).map_err(|error| {
            ConstraintPlanError::Invalid(format!(
                "constraint `{}` target resolution failed: {error}",
                constraint.constraint_id
            ))
        })?
    } else {
        let canonical_nodes = explicit_target_node_indices(structural, &target.region_id)
            .into_iter()
            .map(|node| topology.explicit_node_indices[node])
            .collect::<Vec<_>>();
        if canonical_nodes.is_empty() {
            return Err(ConstraintPlanError::EmptyTarget {
                constraint_id: constraint.constraint_id.clone(),
                region_id: target.region_id.clone(),
            });
        }
        MixedStructuralTargetOwnership::Explicit { canonical_nodes }
    };
    let nodes = ownership.canonical_nodes().to_vec();
    let resolved = match ownership {
        MixedStructuralTargetOwnership::Solid { .. }
        | MixedStructuralTargetOwnership::Shared { .. } => resolve_solid_target(
            mesh.expect("solid target ownership requires a solver mesh"),
            target,
            &constraint.constraint_id,
        )?,
        MixedStructuralTargetOwnership::Explicit { .. } => {
            let mut identities = nodes
                .iter()
                .map(|node| topology.nodes[*node].identity)
                .collect::<Vec<_>>();
            identities.sort_unstable();
            identities.dedup();
            if target.cardinality == TargetCardinality::One && identities.len() != 1 {
                return Err(ConstraintPlanError::Cardinality {
                    constraint_id: constraint.constraint_id.clone(),
                    actual: identities.len(),
                });
            }
            let selection_digest = super::plan::canonical_digest(
                "resolved-constraint-target",
                &(
                    target,
                    Vec::<runmat_meshing_core::PersistentEntityId>::new(),
                    &identities,
                ),
            )?;
            ResolvedConstraintTarget {
                authored: target.clone(),
                matched_entities: Vec::new(),
                matched_node_identities: identities,
                selection_digest,
            }
        }
    };
    Ok((resolved, nodes))
}

fn available_dofs(
    topology: &CanonicalStructuralNodeSpace,
    node: usize,
) -> Vec<CanonicalStructuralDof> {
    let mut dofs = CanonicalStructuralDof::TRANSLATIONS.to_vec();
    if topology.nodes[node].owns_explicit_rotation {
        dofs.extend([
            CanonicalStructuralDof::Rotation(StructuralRotationDof::Rx),
            CanonicalStructuralDof::Rotation(StructuralRotationDof::Ry),
            CanonicalStructuralDof::Rotation(StructuralRotationDof::Rz),
        ]);
    }
    dofs
}

fn visit_targets(
    kind: &StructuralConstraintKind,
    mut visit: impl FnMut(&runmat_analysis_core::RegionTarget),
) {
    match kind {
        StructuralConstraintKind::Fixed { target }
        | StructuralConstraintKind::NormalDisplacement { target, .. }
        | StructuralConstraintKind::Symmetry { target }
        | StructuralConstraintKind::Antisymmetry { target } => visit(target),
        StructuralConstraintKind::Prescribed(values) => visit(&values.target),
        StructuralConstraintKind::PrescribedMotion(values) => visit(&values.target),
        StructuralConstraintKind::RigidCoupling(coupling)
        | StructuralConstraintKind::DistributingCoupling(coupling) => visit(&coupling.target),
        StructuralConstraintKind::Tie(pair) => {
            visit(&pair.primary);
            visit(&pair.secondary);
        }
        StructuralConstraintKind::Periodic(periodic) => {
            visit(&periodic.pair.primary);
            visit(&periodic.pair.secondary);
        }
        StructuralConstraintKind::Equation(equation) => {
            for term in &equation.terms {
                visit(&term.target);
            }
        }
    }
}

#[cfg(test)]
mod tests;
