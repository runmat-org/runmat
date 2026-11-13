mod assignment;
mod contract;
mod fields;
mod reconcile;
mod sampling;
mod target;

use runmat_analysis_core::{
    AnalysisModel, InitialCondition, PhysicalSignal, ProcedureStep, StructuralModel,
};
use runmat_meshing_core::SolverMeshArtifact;

use crate::{
    assembly::dofs::{StructuralDofKind, StructuralDofLayout},
    contracts::{
        AppliedStructuralInitialConditionEvidence, StructuralInitialConditionQuantity,
        StructuralInitialStateEvidence, StructuralInitialStateEvidenceBasis,
        StructuralInitialValueSourceEvidence,
    },
    physics::structural::topology::{
        node_space::CanonicalStructuralNodeSpace, CanonicalStructuralTopologyIdentity,
    },
};

use assignment::{apply_condition, Assignment, QuantityMapping, RealizationContext};
pub(crate) use contract::{
    RealizedStructuralInitialState, StructuralInitialConstraintView, StructuralInitialStateError,
    StructuralInitialStateRealizationBasis,
};
use reconcile::{reconcile_constraints, ConstraintReconciliationEvidence};

#[cfg(test)]
pub(crate) fn realize(
    model: &AnalysisModel,
    mesh: Option<&SolverMeshArtifact>,
    layout: &StructuralDofLayout,
    basis: StructuralInitialStateRealizationBasis,
    constraints: StructuralInitialConstraintView<'_>,
    procedure_step_id: &str,
) -> Result<RealizedStructuralInitialState, StructuralInitialStateError> {
    realize_with_fields(
        model,
        mesh,
        layout,
        basis,
        constraints,
        procedure_step_id,
        &[],
    )
}

pub(crate) fn realize_with_fields(
    model: &AnalysisModel,
    mesh: Option<&SolverMeshArtifact>,
    layout: &StructuralDofLayout,
    basis: StructuralInitialStateRealizationBasis,
    constraints: StructuralInitialConstraintView<'_>,
    procedure_step_id: &str,
    fields: &[crate::execution::InitialFieldBinding],
) -> Result<RealizedStructuralInitialState, StructuralInitialStateError> {
    let step = unique_step(model, procedure_step_id)?;
    let applicable_conditions = model
        .initial_conditions
        .iter()
        .filter(|condition| condition.applies_to_step(step))
        .collect::<Vec<_>>();
    if let StructuralInitialStateRealizationBasis::OperatorBoundZeroDefault {
        dof_count,
        operator_identity,
    } = basis
    {
        if dof_count != layout.total_dof_count() {
            return Err(StructuralInitialStateError::OperatorBoundDofCountMismatch {
                supplied_dof_count: dof_count,
                layout_dof_count: layout.total_dof_count(),
            });
        }
        if let Some(condition) = applicable_conditions.first() {
            return Err(
                StructuralInitialStateError::OperatorBoundInitialConditionRequiresTopology {
                    initial_condition_id: condition.initial_condition_id().to_string(),
                },
            );
        }
        let mut displacement = vec![None; dof_count];
        let mut velocity = vec![None; dof_count];
        let constraint_evidence =
            reconcile_constraints(&mut displacement, &mut velocity, constraints)?;
        if displacement
            .iter()
            .flatten()
            .any(|value| value.value != 0.0)
            || velocity.iter().flatten().any(|value| value.value != 0.0)
        {
            return Err(StructuralInitialStateError::OperatorBoundZeroStateConstraintConflict);
        }
        return Ok(finish_state(
            step,
            StructuralInitialStateEvidenceBasis::OperatorBoundZeroDefault {
                dof_count,
                operator_identity,
            },
            dof_count,
            displacement,
            velocity,
            Vec::new(),
            constraint_evidence,
        ));
    }
    let empty_structural = StructuralModel {
        nodes: Vec::new(),
        elements: Vec::new(),
        beam_sections: Vec::new(),
        shell_sections: Vec::new(),
    };
    let structural = model.structural.as_ref().unwrap_or(&empty_structural);
    let topology = match mesh {
        Some(mesh) => CanonicalStructuralNodeSpace::derive(mesh, structural),
        None if !structural.nodes.is_empty() => {
            CanonicalStructuralNodeSpace::derive_explicit(structural)
        }
        None => return Err(StructuralInitialStateError::MissingDiscretization),
    }
    .map_err(|error| StructuralInitialStateError::InvalidTopology {
        detail: error.to_string(),
    })?;
    if layout.node_count() != topology.nodes.len() {
        return Err(StructuralInitialStateError::LayoutTopologyMismatch {
            layout_node_count: layout.node_count(),
            topology_node_count: topology.nodes.len(),
        });
    }
    let topology_identity = match mesh {
        Some(mesh) if structural.nodes.is_empty() => CanonicalStructuralTopologyIdentity::Solid {
            mesh_digest: mesh.canonical_digest,
        },
        _ => topology.identity,
    };

    if !fields.is_empty() {
        crate::execution::validate_initial_field_selection(
            model,
            step,
            topology_identity.discretization_digest(),
            topology.nodes.len(),
            fields,
        )
        .map_err(|error| StructuralInitialStateError::InvalidTopology {
            detail: error.to_string(),
        })?;
    }
    let mut displacement = vec![None; layout.total_dof_count()];
    let mut velocity = vec![None; layout.total_dof_count()];
    let mut applied_conditions = Vec::new();
    let context = RealizationContext {
        model,
        mesh,
        structural,
        topology: &topology,
        layout,
        fields,
        topology_digest: topology_identity.discretization_digest(),
    };
    for condition in applicable_conditions {
        match condition {
            InitialCondition::Displacement(condition) => apply_condition(
                &context,
                condition,
                QuantityMapping {
                    quantity: StructuralInitialConditionQuantity::Displacement,
                    dofs: [
                        StructuralDofKind::Ux,
                        StructuralDofKind::Uy,
                        StructuralDofKind::Uz,
                    ],
                },
                &mut displacement,
                |signal| match signal {
                    PhysicalSignal::Displacement(signal) => Some(signal),
                    _ => None,
                },
                |value| value.canonical_value(),
                &mut applied_conditions,
            )?,
            InitialCondition::Velocity(condition) => apply_condition(
                &context,
                condition,
                QuantityMapping {
                    quantity: StructuralInitialConditionQuantity::Velocity,
                    dofs: [
                        StructuralDofKind::Ux,
                        StructuralDofKind::Uy,
                        StructuralDofKind::Uz,
                    ],
                },
                &mut velocity,
                |signal| match signal {
                    PhysicalSignal::Velocity(signal) => Some(signal),
                    _ => None,
                },
                |value| value.canonical_value(),
                &mut applied_conditions,
            )?,
            InitialCondition::AngularVelocity(condition) => apply_condition(
                &context,
                condition,
                QuantityMapping {
                    quantity: StructuralInitialConditionQuantity::AngularVelocity,
                    dofs: [
                        StructuralDofKind::Rx,
                        StructuralDofKind::Ry,
                        StructuralDofKind::Rz,
                    ],
                },
                &mut velocity,
                |signal| match signal {
                    PhysicalSignal::AngularVelocity(signal) => Some(signal),
                    _ => None,
                },
                |value| value.canonical_value(),
                &mut applied_conditions,
            )?,
            _ => {
                return Err(StructuralInitialStateError::UnsupportedInitialCondition {
                    initial_condition_id: condition.initial_condition_id().to_string(),
                });
            }
        }
    }
    let constraint_evidence = reconcile_constraints(&mut displacement, &mut velocity, constraints)?;
    Ok(finish_state(
        step,
        StructuralInitialStateEvidenceBasis::CanonicalTopology {
            discretization_identity: topology_identity.discretization_digest(),
            mesh_digest: topology_identity.mesh_digest(),
        },
        layout.total_dof_count(),
        displacement,
        velocity,
        applied_conditions,
        constraint_evidence,
    ))
}

fn finish_state(
    step: &ProcedureStep,
    basis: StructuralInitialStateEvidenceBasis,
    total_dof_count: usize,
    displacement: Vec<Option<Assignment>>,
    velocity: Vec<Option<Assignment>>,
    mut applied_conditions: Vec<AppliedStructuralInitialConditionEvidence>,
    constraint_evidence: ConstraintReconciliationEvidence,
) -> RealizedStructuralInitialState {
    applied_conditions
        .sort_by(|left, right| left.initial_condition_id.cmp(&right.initial_condition_id));
    let authored_displacement_dof_count = displacement
        .iter()
        .flatten()
        .filter(|assignment| assignment.authored_initial_condition_id().is_some())
        .count();
    let constraint_derived_displacement_dof_count = displacement
        .iter()
        .flatten()
        .filter(|assignment| assignment.authored_initial_condition_id().is_none())
        .count();
    let authored_velocity_dof_count = velocity
        .iter()
        .flatten()
        .filter(|assignment| assignment.authored_initial_condition_id().is_some())
        .count();
    let constraint_derived_velocity_dof_count = velocity
        .iter()
        .flatten()
        .filter(|assignment| assignment.authored_initial_condition_id().is_none())
        .count();
    RealizedStructuralInitialState {
        displacement: displacement
            .into_iter()
            .map(|entry| entry.map_or(0.0, |assignment| assignment.value))
            .collect(),
        velocity: velocity
            .into_iter()
            .map(|entry| entry.map_or(0.0, |assignment| assignment.value))
            .collect(),
        evidence: StructuralInitialStateEvidence {
            procedure_step_id: step.step_id.0.clone(),
            basis,
            total_dof_count,
            authored_displacement_dof_count,
            constraint_derived_displacement_dof_count,
            zero_default_displacement_dof_count: total_dof_count
                - authored_displacement_dof_count
                - constraint_derived_displacement_dof_count,
            authored_velocity_dof_count,
            constraint_derived_velocity_dof_count,
            zero_default_velocity_dof_count: total_dof_count
                - authored_velocity_dof_count
                - constraint_derived_velocity_dof_count,
            direct_constrained_dof_count: constraint_evidence.direct_constrained_dof_count,
            constraint_equation_count: constraint_evidence.constraint_equation_count,
            maximum_displacement_constraint_residual_bits: constraint_evidence
                .maximum_displacement_residual
                .to_bits(),
            maximum_velocity_constraint_residual_bits: constraint_evidence
                .maximum_velocity_residual
                .to_bits(),
            applied_conditions,
        },
    }
}

fn unique_step<'a>(
    model: &'a AnalysisModel,
    procedure_step_id: &str,
) -> Result<&'a ProcedureStep, StructuralInitialStateError> {
    let mut matches = model
        .sequence
        .steps
        .iter()
        .filter(|step| step.step_id.as_str() == procedure_step_id);
    let step = matches
        .next()
        .ok_or_else(|| StructuralInitialStateError::MissingStep {
            procedure_step_id: procedure_step_id.to_string(),
        })?;
    if matches.next().is_some() {
        return Err(StructuralInitialStateError::DuplicateStep {
            procedure_step_id: procedure_step_id.to_string(),
        });
    }
    Ok(step)
}

#[cfg(test)]
mod tests;
