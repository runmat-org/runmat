use runmat_analysis_core::{
    AnalysisModel, StructuralConstraint, StructuralConstraintKind, StructuralRotationDof,
};
use runmat_meshing_core::SolverMeshArtifact;

use crate::physics::structural::topology::node_space::CanonicalStructuralNodeSpace;

use super::{
    compilation::CompiledConstraint,
    contract::{
        CanonicalStructuralConstraintTerm, CanonicalStructuralDof, CanonicalStructuralDofOwner,
        ConstraintCouplingKind, ConstraintPlanError,
    },
    frame::require_global,
    lowering, mixed_surface,
    pairing::{pair_mixed_targets, pairing_rows},
    prepared::PreparedTargets,
    row,
    workspace::{count_add, multiply, reserve},
};

pub(super) fn compile_node_constraint(
    model: &AnalysisModel,
    mesh: Option<&SolverMeshArtifact>,
    structural: &runmat_analysis_core::StructuralModel,
    topology: &CanonicalStructuralNodeSpace,
    constraint: &StructuralConstraint,
    prepared: &mut PreparedTargets<'_>,
) -> Result<CompiledConstraint, ConstraintPlanError> {
    match &constraint.kind {
        StructuralConstraintKind::Fixed { target } => {
            require_global(constraint)?;
            let (resolved, nodes) = prepared.take(target)?;
            let count = nodes.iter().try_fold(0usize, |sum, node| {
                count_add(
                    sum,
                    if topology.nodes[*node].owns_explicit_rotation {
                        6
                    } else {
                        3
                    },
                )
            })?;
            let mut rows = lowering::rows(count, 1, &constraint.constraint_id)?;
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
                targets: lowering::targets([resolved])?,
                rows,
                pairings: Vec::new(),
                coupling: None,
            })
        }
        StructuralConstraintKind::Prescribed(values) => mixed_surface::compile_prescribed(
            model, structural, topology, constraint, prepared, values,
        ),
        StructuralConstraintKind::PrescribedMotion(motion) => {
            require_global(constraint)?;
            let (resolved, nodes) = prepared.take(&motion.target)?;
            let mut rows = lowering::rows(
                multiply(nodes.len(), motion.translations.len())?,
                1,
                &constraint.constraint_id,
            )?;
            for node in nodes {
                for translation in &motion.translations {
                    let dof = CanonicalStructuralDof::from(translation.dof);
                    if !available_dofs(topology, node).any(|candidate| candidate == dof) {
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
                targets: lowering::targets([resolved])?,
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
            let mut targets = reserve(equation.terms.len())?;
            let term_count = prepared.remaining_node_count()?;
            let mut rows = lowering::rows(1, term_count, &constraint.constraint_id)?;
            let mut terms = reserve(term_count)?;
            for term in &equation.terms {
                if !term.coefficient.is_finite() || term.coefficient == 0.0 {
                    return Err(ConstraintPlanError::Invalid(format!(
                        "constraint `{}` contains a zero or non-finite equation coefficient",
                        constraint.constraint_id
                    )));
                }
                let (resolved, nodes) = prepared.take(&term.target)?;
                for node in nodes {
                    let dof = CanonicalStructuralDof::from(term.dof);
                    if !available_dofs(topology, node).any(|candidate| candidate == dof) {
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
            if !equation.right_hand_side.canonical_value().is_finite() {
                return Err(ConstraintPlanError::NonFiniteValue {
                    constraint_id: constraint.constraint_id.clone(),
                });
            }
            rows.push(row::new_sparse(
                &constraint.constraint_id,
                terms,
                equation.right_hand_side.canonical_value(),
            )?);
            Ok(CompiledConstraint::rows(targets, rows))
        }
        StructuralConstraintKind::Tie(pair) => compile_pair(
            mesh,
            topology,
            constraint,
            prepared,
            pair,
            &runmat_analysis_core::PeriodicTransform::Identity,
            false,
        ),
        StructuralConstraintKind::Periodic(periodic) => compile_pair(
            mesh,
            topology,
            constraint,
            prepared,
            &periodic.pair,
            &periodic.transform,
            true,
        ),
        StructuralConstraintKind::NormalDisplacement { target, value } => {
            mixed_surface::compile_normal(
                model, structural, topology, constraint, prepared, target, value,
            )
        }
        StructuralConstraintKind::Symmetry { target } => mixed_surface::compile_directions(
            structural, topology, constraint, prepared, target, false,
        ),
        StructuralConstraintKind::Antisymmetry { target } => mixed_surface::compile_directions(
            structural, topology, constraint, prepared, target, true,
        ),
        StructuralConstraintKind::RigidCoupling(coupling) => mixed_surface::compile_coupling(
            structural,
            topology,
            constraint,
            prepared,
            coupling,
            ConstraintCouplingKind::Rigid,
        ),
        StructuralConstraintKind::DistributingCoupling(coupling) => {
            mixed_surface::compile_coupling(
                structural,
                topology,
                constraint,
                prepared,
                coupling,
                ConstraintCouplingKind::Distributing,
            )
        }
    }
}

fn compile_pair(
    mesh: Option<&SolverMeshArtifact>,
    topology: &CanonicalStructuralNodeSpace,
    constraint: &StructuralConstraint,
    prepared: &mut PreparedTargets<'_>,
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
    let (primary, _) = prepared.take(&pair.primary)?;
    let (secondary, _) = prepared.take(&pair.secondary)?;
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
        targets: lowering::targets([primary, secondary])?,
        rows,
        pairings,
        coupling: None,
    })
}

fn available_dofs(
    topology: &CanonicalStructuralNodeSpace,
    node: usize,
) -> impl Iterator<Item = CanonicalStructuralDof> {
    let rotations = topology.nodes[node].owns_explicit_rotation;
    CanonicalStructuralDof::TRANSLATIONS.into_iter().chain(
        [
            CanonicalStructuralDof::Rotation(StructuralRotationDof::Rx),
            CanonicalStructuralDof::Rotation(StructuralRotationDof::Ry),
            CanonicalStructuralDof::Rotation(StructuralRotationDof::Rz),
        ]
        .into_iter()
        .filter(move |_| rotations),
    )
}

#[cfg(test)]
mod tests;
