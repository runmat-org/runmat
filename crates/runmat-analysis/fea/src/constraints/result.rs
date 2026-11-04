use std::collections::BTreeMap;

use crate::algebra::{capacity::measured, coercivity::NumericalCapacityError};
use runmat_analysis_core::{AnalysisField, AnalysisModel};
use runmat_execution::Digest;
use runmat_meshing_core::{SolverMeshArtifact, StableDigest};
use runmat_simulation_usage::{AsyncCapacityOwner, LedgerError, OperationCount, UsageCategory};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::{
    reaction::solve_constraint_multipliers_with_owner,
    result_state::{
        dof_component, nodal_vector_field, nodal_vector_field_for_count, owner_dof_index,
        resolve_model_state, translation_component, ConstraintNodeState,
    },
    CanonicalStructuralConstraintPlan, CanonicalStructuralDof, CanonicalStructuralDofOwner,
    CanonicalStructuralTopologyIdentity, ReactionRecovery,
};
use crate::contracts::{
    FEA_FIELD_STRUCTURAL_DISPLACEMENT, FEA_FIELD_STRUCTURAL_REACTION_FORCE,
    FEA_FIELD_STRUCTURAL_REACTION_MOMENT, FEA_FIELD_STRUCTURAL_ROTATION,
};
use crate::physics::structural::topology::node_space::CanonicalStructuralNodeSpace;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConstraintGroupResult {
    pub constraint_id: String,
    pub active_row_count: usize,
    pub redundant_row_count: usize,
    pub resultant_force_newtons: [f64; 3],
    pub resultant_moment_about_global_origin_newton_meters: [f64; 3],
    pub constraint_work_joules: f64,
    pub maximum_displacement_error_meters: f64,
    pub maximum_rotation_error_radians: f64,
    pub row_reactions: Vec<ConstraintRowReaction>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConstraintRowReaction {
    pub row_identity: Digest,
    pub generalized_reaction: f64,
    pub violation: ConstraintViolation,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "quantity", content = "magnitude")]
pub enum ConstraintViolation {
    DisplacementMeters(f64),
    RotationRadians(f64),
}

impl ConstraintViolation {
    fn is_valid(self) -> bool {
        let magnitude = match self {
            Self::DisplacementMeters(value) | Self::RotationRadians(value) => value,
        };
        magnitude.is_finite() && magnitude >= 0.0
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalStructuralConstraintResult {
    pub schema_version: u16,
    pub constraint_plan_digest: Digest,
    pub topology: CanonicalStructuralTopologyIdentity,
    pub reference_points: Vec<ConstraintReferencePointResult>,
    pub groups: Vec<ConstraintGroupResult>,
    pub resultant_force_newtons: [f64; 3],
    pub resultant_moment_about_global_origin_newton_meters: [f64; 3],
    pub constraint_work_joules: f64,
    pub maximum_displacement_error_meters: f64,
    pub maximum_rotation_error_radians: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConstraintReferencePointResult {
    pub identity: Digest,
    pub reference_point_id: String,
    pub translation_meters: [f64; 3],
    pub rotation_radians: [f64; 3],
    pub constraint_force_newtons: [f64; 3],
    pub constraint_moment_newton_meters: [f64; 3],
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum ConstraintResultError {
    #[error("constraint result numerical accounting failed: {0}")]
    Accounting(#[from] LedgerError),
    #[error("constraint result recovery cancelled")]
    Cancelled,
    #[error("constraint result topology differs from its canonical plan")]
    TopologyMismatch,
    #[error("required nodal field `{field_id}` is absent")]
    MissingField { field_id: &'static str },
    #[error("nodal field `{field_id}` is not a host real64 node-by-three field")]
    InvalidField { field_id: &'static str },
    #[error("constraint result references a node absent from solver topology")]
    NodeAbsent,
    #[error("constraint result references an authored group absent from its plan")]
    GroupAbsent,
    #[error("constraint result contains invalid numerical or identity evidence")]
    InvalidEvidence,
    #[error("constraint reactions could not be resolved from the independent equation rows")]
    ReactionRecoveryFailed,
    #[error("constraint result requires the complete mesh-node and reference-point solver state")]
    GeneralizedStateRequired,
    #[error("constraint plan could not be validated: {0}")]
    InvalidPlan(String),
}

impl CanonicalStructuralConstraintResult {
    pub fn recover(
        plan: &CanonicalStructuralConstraintPlan,
        mesh: &SolverMeshArtifact,
        fields: &[AnalysisField],
    ) -> Result<Self, ConstraintResultError> {
        if !plan.reference_points.is_empty() {
            return Err(ConstraintResultError::GeneralizedStateRequired);
        }
        let displacement = nodal_vector_field(fields, FEA_FIELD_STRUCTURAL_DISPLACEMENT, mesh)?;
        let reaction = nodal_vector_field(fields, FEA_FIELD_STRUCTURAL_REACTION_FORCE, mesh)?;
        Self::recover_from_solver_state(plan, mesh, displacement, reaction)
    }

    pub fn recover_for_model(
        plan: &CanonicalStructuralConstraintPlan,
        model: &AnalysisModel,
        mesh: Option<&SolverMeshArtifact>,
        fields: &[AnalysisField],
    ) -> Result<Self, ConstraintResultError> {
        if let CanonicalStructuralTopologyIdentity::Solid { .. } = plan.topology {
            return Self::recover(
                plan,
                mesh.ok_or_else(|| {
                    ConstraintResultError::InvalidPlan(
                        "solid constraint recovery requires its canonical mesh".into(),
                    )
                })?,
                fields,
            );
        }
        if !plan.reference_points.is_empty() {
            return Err(ConstraintResultError::GeneralizedStateRequired);
        }
        let state = resolve_model_state(plan, model, mesh)?;
        let node_count = state.node_order.len();
        let translations =
            nodal_vector_field_for_count(fields, FEA_FIELD_STRUCTURAL_DISPLACEMENT, node_count)?;
        let rotations =
            nodal_vector_field_for_count(fields, FEA_FIELD_STRUCTURAL_ROTATION, node_count)?;
        let forces =
            nodal_vector_field_for_count(fields, FEA_FIELD_STRUCTURAL_REACTION_FORCE, node_count)?;
        let moments =
            nodal_vector_field_for_count(fields, FEA_FIELD_STRUCTURAL_REACTION_MOMENT, node_count)?;
        let mut displacement = vec![0.0; state.layout.total_dof_count()];
        let mut reaction = vec![0.0; state.layout.total_dof_count()];
        for (node_index, identity) in state.node_order.iter().enumerate() {
            let node = state.nodes[identity];
            for (component, row) in node.dof_indices.into_iter().enumerate() {
                if let Some(row) = row {
                    let values = if component >= 3 {
                        rotations
                    } else {
                        translations
                    };
                    let reactions = if component >= 3 { moments } else { forces };
                    displacement[row] = values[node_index * 3 + component % 3];
                    reaction[row] = reactions[node_index * 3 + component % 3];
                }
            }
        }
        Self::recover_resolved(
            plan,
            &state.nodes,
            state.layout.node_dof_count(),
            &displacement,
            &reaction,
        )
    }

    pub fn recover_for_model_from_solver_state(
        plan: &CanonicalStructuralConstraintPlan,
        model: &AnalysisModel,
        mesh: Option<&SolverMeshArtifact>,
        displacement: &[f64],
        reaction: &[f64],
    ) -> Result<Self, ConstraintResultError> {
        immediate(Self::recover_for_model_from_solver_state_with_capacity::<
            std::convert::Infallible,
        >(plan, model, mesh, displacement, reaction, None))
    }

    pub(crate) async fn recover_for_model_from_solver_state_with_capacity<A>(
        plan: &CanonicalStructuralConstraintPlan,
        model: &AnalysisModel,
        mesh: Option<&SolverMeshArtifact>,
        displacement: &[f64],
        reaction: &[f64],
        owner: Option<&dyn AsyncCapacityOwner<A>>,
    ) -> Result<Self, NumericalCapacityError<ConstraintResultError, A>> {
        if let CanonicalStructuralTopologyIdentity::Solid { .. } = plan.topology {
            return Self::recover_from_solver_state_with_capacity(
                plan,
                mesh.ok_or_else(|| {
                    ConstraintResultError::InvalidPlan(
                        "solid constraint recovery requires its canonical mesh".into(),
                    )
                })?,
                displacement,
                reaction,
                owner,
            )
            .await;
        }
        plan.validate()
            .map_err(|error| ConstraintResultError::InvalidPlan(error.to_string()))?;
        let state = resolve_model_state(plan, model, mesh)?;
        if displacement.len() != state.layout.total_dof_count()
            || reaction.len() != state.layout.total_dof_count()
            || !all_finite(displacement)
            || !all_finite(reaction)
        {
            return Err(ConstraintResultError::GeneralizedStateRequired.into());
        }
        Self::recover_resolved_with_capacity(
            plan,
            &state.nodes,
            state.layout.node_dof_count(),
            displacement,
            reaction,
            owner,
        )
        .await
    }

    pub fn recover_from_solver_state(
        plan: &CanonicalStructuralConstraintPlan,
        mesh: &SolverMeshArtifact,
        displacement: &[f64],
        reaction: &[f64],
    ) -> Result<Self, ConstraintResultError> {
        immediate(Self::recover_from_solver_state_with_capacity::<
            std::convert::Infallible,
        >(plan, mesh, displacement, reaction, None))
    }

    async fn recover_from_solver_state_with_capacity<A>(
        plan: &CanonicalStructuralConstraintPlan,
        mesh: &SolverMeshArtifact,
        displacement: &[f64],
        reaction: &[f64],
        owner: Option<&dyn AsyncCapacityOwner<A>>,
    ) -> Result<Self, NumericalCapacityError<ConstraintResultError, A>> {
        plan.validate()
            .map_err(|error| ConstraintResultError::InvalidPlan(error.to_string()))?;
        mesh.validate()
            .map_err(|error| ConstraintResultError::InvalidPlan(error.to_string()))?;
        if plan.topology.mesh_digest() != Some(mesh.canonical_digest) {
            return Err(ConstraintResultError::TopologyMismatch.into());
        }
        let expected_dofs = mesh.topology.nodes.len() * 3 + plan.reference_points.len() * 6;
        if displacement.len() != expected_dofs
            || reaction.len() != expected_dofs
            || !all_finite(displacement)
            || !all_finite(reaction)
        {
            return Err(ConstraintResultError::GeneralizedStateRequired.into());
        }
        let nodes = mesh
            .topology
            .nodes
            .iter()
            .enumerate()
            .map(|(index, node)| {
                (
                    node.stable_identity,
                    ConstraintNodeState {
                        coordinates_m: runmat_types::physical::Quantity::<
                            runmat_types::physical::kind::Position,
                            _,
                        >::new(
                            node.coordinates_m.canonical_value()
                        ),
                        dof_indices: [
                            Some(index * 3),
                            Some(index * 3 + 1),
                            Some(index * 3 + 2),
                            None,
                            None,
                            None,
                        ],
                    },
                )
            })
            .collect::<BTreeMap<_, _>>();
        Self::recover_resolved_with_capacity(
            plan,
            &nodes,
            mesh.topology.nodes.len() * 3,
            displacement,
            reaction,
            owner,
        )
        .await
    }

    fn recover_resolved(
        plan: &CanonicalStructuralConstraintPlan,
        nodes: &BTreeMap<StableDigest, ConstraintNodeState>,
        reference_dof_offset: usize,
        displacement: &[f64],
        reaction: &[f64],
    ) -> Result<Self, ConstraintResultError> {
        immediate(Self::recover_resolved_with_capacity::<
            std::convert::Infallible,
        >(
            plan,
            nodes,
            reference_dof_offset,
            displacement,
            reaction,
            None,
        ))
    }

    async fn recover_resolved_with_capacity<A>(
        plan: &CanonicalStructuralConstraintPlan,
        nodes: &BTreeMap<StableDigest, ConstraintNodeState>,
        reference_dof_offset: usize,
        displacement: &[f64],
        reaction: &[f64],
        owner: Option<&dyn AsyncCapacityOwner<A>>,
    ) -> Result<Self, NumericalCapacityError<ConstraintResultError, A>> {
        let mut groups = plan
            .groups
            .iter()
            .map(|group| {
                (
                    group.constraint_id.as_str(),
                    ConstraintGroupResult {
                        constraint_id: group.constraint_id.clone(),
                        active_row_count: 0,
                        redundant_row_count: 0,
                        resultant_force_newtons: [0.0; 3],
                        resultant_moment_about_global_origin_newton_meters: [0.0; 3],
                        constraint_work_joules: 0.0,
                        maximum_displacement_error_meters: 0.0,
                        maximum_rotation_error_radians: 0.0,
                        row_reactions: Vec::new(),
                    },
                )
            })
            .collect::<BTreeMap<_, _>>();

        let active_rows = plan
            .rows
            .iter()
            .filter(|row| row.reaction_recovery != ReactionRecovery::RedundantEquation)
            .collect::<Vec<_>>();
        let row_coefficients = active_rows
            .iter()
            .map(|row| {
                row.terms
                    .iter()
                    .map(|term| {
                        Ok((
                            owner_dof_index(
                                plan,
                                nodes,
                                reference_dof_offset,
                                term.owner,
                                term.dof,
                            )?,
                            term.coefficient,
                        ))
                    })
                    .collect::<Result<BTreeMap<_, _>, ConstraintResultError>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut projection_work = OperationCount::ZERO;
        let mut scatter_work = OperationCount::ZERO;
        for coefficients in &row_coefficients {
            let length = count(coefficients.len())?;
            projection_work = projection_work
                .checked_add(
                    OperationCount::direct_real_dot(length.operations())
                        .map_err(LedgerError::from)
                        .map_err(ConstraintResultError::from)?,
                )
                .map_err(LedgerError::from)
                .map_err(ConstraintResultError::from)?;
            scatter_work = scatter_work
                .checked_add(
                    length
                        .checked_mul(2)
                        .map_err(LedgerError::from)
                        .map_err(ConstraintResultError::from)?,
                )
                .map_err(LedgerError::from)
                .map_err(ConstraintResultError::from)?;
        }
        let projected_reaction = measured(owner, UsageCategory::Solve, projection_work, || {
            let mut projected_reaction = vec![0.0; active_rows.len()];
            for (left, coefficients) in row_coefficients.iter().enumerate() {
                projected_reaction[left] = coefficients
                    .iter()
                    .map(|(dof, coefficient)| coefficient * reaction[*dof])
                    .sum();
            }
            Ok::<_, ConstraintResultError>(projected_reaction)
        })
        .await?;
        let multipliers = solve_constraint_multipliers_with_owner(
            &row_coefficients,
            &projected_reaction,
            |work| async move {
                match owner {
                    Some(owner) => owner.ensure_capacity(work).await,
                    None => Ok::<_, A>(()),
                }
            },
        )
        .await
        .map_err(|error| {
            error.map_numerical(|error| match error {
                super::ConstraintMultiplierError::Accounting(cause) => {
                    ConstraintResultError::Accounting(cause)
                }
                super::ConstraintMultiplierError::Cancelled => ConstraintResultError::Cancelled,
            })
        })?
        .ok_or(ConstraintResultError::ReactionRecoveryFailed)?;
        let predicted_reaction = measured(owner, UsageCategory::Solve, scatter_work, || {
            let mut predicted_reaction = BTreeMap::<usize, f64>::new();
            for (coefficients, multiplier) in row_coefficients.iter().zip(&multipliers) {
                for (dof, coefficient) in coefficients {
                    *predicted_reaction.entry(*dof).or_insert(0.0) += coefficient * multiplier;
                }
            }
            Ok::<_, ConstraintResultError>(predicted_reaction)
        })
        .await?;
        let reaction_scale = predicted_reaction
            .values()
            .map(|value| value.abs())
            .fold(1.0_f64, f64::max);
        if predicted_reaction
            .iter()
            .any(|(dof, predicted)| (reaction[*dof] - predicted).abs() > reaction_scale * 1.0e-7)
        {
            return Err(ConstraintResultError::ReactionRecoveryFailed.into());
        }

        for row in &plan.rows {
            let group = groups
                .get_mut(row.constraint_id.as_str())
                .ok_or(ConstraintResultError::GroupAbsent)?;
            if row.reaction_recovery == ReactionRecovery::RedundantEquation {
                group.redundant_row_count += 1;
                continue;
            }
            let mut row_work =
                OperationCount::direct_real_dot(count(row.terms.len())?.operations())
                    .and_then(|work| work.checked_add(OperationCount::from_operations(3)))
                    .map_err(LedgerError::from)
                    .map_err(ConstraintResultError::from)?;
            for term in &row.terms {
                if matches!(term.owner, CanonicalStructuralDofOwner::StructuralNode(_)) {
                    // Coefficient product, resultant accumulation, and (for a
                    // translation) the three-component moment cross product.
                    let operations = match term.dof {
                        CanonicalStructuralDof::Translation(_) => 14,
                        CanonicalStructuralDof::Rotation(_) => 2,
                    };
                    row_work = row_work
                        .checked_add(OperationCount::from_operations(operations))
                        .map_err(LedgerError::from)
                        .map_err(ConstraintResultError::from)?;
                }
            }
            measured(
                owner,
                UsageCategory::Solve,
                row_work,
                || -> Result<(), ConstraintResultError> {
                    let active_index = active_rows
                        .iter()
                        .position(|active| active.identity == row.identity)
                        .ok_or(ConstraintResultError::InvalidEvidence)?;
                    let multiplier = multipliers[active_index];
                    let equation_value = row
                        .terms
                        .iter()
                        .map(|term| {
                            Ok(term.coefficient
                                * displacement[owner_dof_index(
                                    plan,
                                    nodes,
                                    reference_dof_offset,
                                    term.owner,
                                    term.dof,
                                )?])
                        })
                        .collect::<Result<Vec<_>, ConstraintResultError>>()?
                        .into_iter()
                        .sum::<f64>();
                    let error = (equation_value - row.right_hand_side).abs();
                    group.active_row_count += 1;
                    for term in &row.terms {
                        if let CanonicalStructuralDofOwner::StructuralNode(identity) = term.owner {
                            let node = nodes
                                .get(&identity)
                                .ok_or(ConstraintResultError::NodeAbsent)?;
                            let generalized_reaction = term.coefficient * multiplier;
                            match term.dof {
                                CanonicalStructuralDof::Translation(_) => {
                                    let component = translation_component(term.dof)?;
                                    group.resultant_force_newtons[component] +=
                                        generalized_reaction;
                                    add_cross_product(
                                        &mut group
                                            .resultant_moment_about_global_origin_newton_meters,
                                        node.coordinates_m.canonical_value(),
                                        component_vector(component, generalized_reaction),
                                    );
                                }
                                CanonicalStructuralDof::Rotation(_) => {
                                    group.resultant_moment_about_global_origin_newton_meters
                                        [dof_component(term.dof) - 3] += generalized_reaction;
                                }
                            }
                        }
                    }
                    group.constraint_work_joules += multiplier * row.right_hand_side;
                    let violation = match row.residual_quantity {
                        super::ConstraintResidualQuantity::Displacement => {
                            group.maximum_displacement_error_meters =
                                group.maximum_displacement_error_meters.max(error);
                            ConstraintViolation::DisplacementMeters(error)
                        }
                        super::ConstraintResidualQuantity::Rotation => {
                            group.maximum_rotation_error_radians =
                                group.maximum_rotation_error_radians.max(error);
                            ConstraintViolation::RotationRadians(error)
                        }
                    };
                    group.row_reactions.push(ConstraintRowReaction {
                        row_identity: row.identity,
                        generalized_reaction: multiplier,
                        violation,
                    });
                    Ok(())
                },
            )
            .await?;
        }

        let reference_points = plan
            .reference_points
            .iter()
            .enumerate()
            .map(|(index, point)| {
                let base = reference_dof_offset + index * 6;
                ConstraintReferencePointResult {
                    identity: point.identity,
                    reference_point_id: point.reference_point_id.clone(),
                    translation_meters: [
                        displacement[base],
                        displacement[base + 1],
                        displacement[base + 2],
                    ],
                    rotation_radians: [
                        displacement[base + 3],
                        displacement[base + 4],
                        displacement[base + 5],
                    ],
                    constraint_force_newtons: [
                        predicted_reaction.get(&base).copied().unwrap_or(0.0),
                        predicted_reaction.get(&(base + 1)).copied().unwrap_or(0.0),
                        predicted_reaction.get(&(base + 2)).copied().unwrap_or(0.0),
                    ],
                    constraint_moment_newton_meters: [
                        predicted_reaction.get(&(base + 3)).copied().unwrap_or(0.0),
                        predicted_reaction.get(&(base + 4)).copied().unwrap_or(0.0),
                        predicted_reaction.get(&(base + 5)).copied().unwrap_or(0.0),
                    ],
                }
            })
            .collect::<Vec<_>>();
        let groups = groups.into_values().collect::<Vec<_>>();
        let mut result = Self {
            schema_version: 1,
            constraint_plan_digest: plan
                .content_digest()
                .map_err(|error| ConstraintResultError::InvalidPlan(error.to_string()))?,
            topology: plan.topology,
            reference_points,
            resultant_force_newtons: [0.0; 3],
            resultant_moment_about_global_origin_newton_meters: [0.0; 3],
            constraint_work_joules: 0.0,
            maximum_displacement_error_meters: 0.0,
            maximum_rotation_error_radians: 0.0,
            groups,
        };
        let totals_work = count(result.groups.len())?
            .checked_mul(7)
            .map_err(LedgerError::from)
            .map_err(ConstraintResultError::from)?;
        measured(owner, UsageCategory::Solve, totals_work, || {
            for group in &result.groups {
                for component in 0..3 {
                    result.resultant_force_newtons[component] +=
                        group.resultant_force_newtons[component];
                    result.resultant_moment_about_global_origin_newton_meters[component] +=
                        group.resultant_moment_about_global_origin_newton_meters[component];
                }
                result.constraint_work_joules += group.constraint_work_joules;
                result.maximum_displacement_error_meters = result
                    .maximum_displacement_error_meters
                    .max(group.maximum_displacement_error_meters);
                result.maximum_rotation_error_radians = result
                    .maximum_rotation_error_radians
                    .max(group.maximum_rotation_error_radians);
            }
            Ok::<_, ConstraintResultError>(())
        })
        .await?;
        result.validate_against(plan)?;
        Ok(result)
    }

    pub fn validate_against(
        &self,
        plan: &CanonicalStructuralConstraintPlan,
    ) -> Result<(), ConstraintResultError> {
        self.validate()?;
        if self.schema_version != 1
            || self.constraint_plan_digest
                != plan
                    .content_digest()
                    .map_err(|error| ConstraintResultError::InvalidPlan(error.to_string()))?
            || self.topology != plan.topology
            || self.groups.len() != plan.groups.len()
            || self.reference_points.len() != plan.reference_points.len()
            || self
                .reference_points
                .iter()
                .zip(&plan.reference_points)
                .any(|(result, planned)| {
                    result.identity != planned.identity
                        || result.reference_point_id != planned.reference_point_id
                })
        {
            return Err(ConstraintResultError::InvalidEvidence);
        }
        for (group, planned) in self.groups.iter().zip(&plan.groups) {
            let active_rows = plan
                .rows
                .iter()
                .filter(|row| {
                    row.constraint_id == planned.constraint_id
                        && row.reaction_recovery != ReactionRecovery::RedundantEquation
                })
                .collect::<Vec<_>>();
            let expected_active = active_rows.len();
            let expected_redundant = planned.row_count - expected_active;
            let expected_row_identities = plan
                .rows
                .iter()
                .filter(|row| {
                    row.constraint_id == planned.constraint_id
                        && row.reaction_recovery != ReactionRecovery::RedundantEquation
                })
                .map(|row| row.identity)
                .collect::<Vec<_>>();
            if group.constraint_id != planned.constraint_id
                || group.active_row_count != expected_active
                || group.redundant_row_count != expected_redundant
                || group.row_reactions.len() != expected_active
                || group
                    .row_reactions
                    .iter()
                    .map(|row| row.row_identity)
                    .collect::<Vec<_>>()
                    != expected_row_identities
            {
                return Err(ConstraintResultError::InvalidEvidence);
            }
            let mut force = [0.0; 3];
            let mut work = 0.0;
            let mut displacement_error: f64 = 0.0;
            let mut rotation_error: f64 = 0.0;
            for (row, reaction) in active_rows.into_iter().zip(&group.row_reactions) {
                for term in &row.terms {
                    if matches!(term.owner, CanonicalStructuralDofOwner::StructuralNode(_)) {
                        if let CanonicalStructuralDof::Translation(_) = term.dof {
                            force[translation_component(term.dof)?] +=
                                term.coefficient * reaction.generalized_reaction;
                        }
                    }
                }
                work += reaction.generalized_reaction * row.right_hand_side;
                match reaction.violation {
                    ConstraintViolation::DisplacementMeters(value) => {
                        displacement_error = displacement_error.max(value);
                    }
                    ConstraintViolation::RotationRadians(value) => {
                        rotation_error = rotation_error.max(value);
                    }
                }
            }
            if force != group.resultant_force_newtons
                || work != group.constraint_work_joules
                || displacement_error != group.maximum_displacement_error_meters
                || rotation_error != group.maximum_rotation_error_radians
            {
                return Err(ConstraintResultError::InvalidEvidence);
            }
        }
        Ok(())
    }

    /// Validates topology-dependent reaction moments against the exact executed model.
    ///
    /// The plan alone can rederive generalized force and work totals, but a moment about the
    /// global origin also depends on the coordinates of every structural-node owner.
    pub fn validate_against_model(
        &self,
        plan: &CanonicalStructuralConstraintPlan,
        model: &AnalysisModel,
        mesh: Option<&SolverMeshArtifact>,
    ) -> Result<(), ConstraintResultError> {
        self.validate_against(plan)?;
        let coordinates = match plan.topology {
            CanonicalStructuralTopologyIdentity::Solid { .. } => mesh
                .ok_or_else(|| {
                    ConstraintResultError::InvalidPlan(
                        "solid constraint-result validation requires its canonical mesh".into(),
                    )
                })?
                .topology
                .nodes
                .iter()
                .map(|node| (node.stable_identity, node.coordinates_m.canonical_value()))
                .collect::<BTreeMap<_, _>>(),
            CanonicalStructuralTopologyIdentity::Explicit { .. }
            | CanonicalStructuralTopologyIdentity::ConformalMixed { .. } => {
                let structural = model.structural.as_ref().ok_or_else(|| {
                    ConstraintResultError::InvalidPlan(
                        "explicit constraint-result validation requires structural topology".into(),
                    )
                })?;
                let topology = match plan.topology {
                    CanonicalStructuralTopologyIdentity::Explicit { .. } => {
                        CanonicalStructuralNodeSpace::derive_explicit(structural)
                    }
                    CanonicalStructuralTopologyIdentity::ConformalMixed { .. } => {
                        CanonicalStructuralNodeSpace::derive(
                            mesh.ok_or_else(|| {
                                ConstraintResultError::InvalidPlan(
                                    "mixed constraint-result validation requires its canonical mesh"
                                        .into(),
                                )
                            })?,
                            structural,
                        )
                    }
                    CanonicalStructuralTopologyIdentity::Solid { .. } => unreachable!(),
                }
                .map_err(|error| ConstraintResultError::InvalidPlan(error.to_string()))?;
                topology
                    .nodes
                    .into_iter()
                    .map(|node| (node.identity, node.coordinates_m.canonical_value()))
                    .collect()
            }
        };
        for group in &self.groups {
            let mut moment = [0.0; 3];
            for (row, reaction) in plan
                .rows
                .iter()
                .filter(|row| {
                    row.constraint_id == group.constraint_id
                        && row.reaction_recovery != ReactionRecovery::RedundantEquation
                })
                .zip(&group.row_reactions)
            {
                for term in &row.terms {
                    let CanonicalStructuralDofOwner::StructuralNode(identity) = term.owner else {
                        continue;
                    };
                    let generalized_reaction = term.coefficient * reaction.generalized_reaction;
                    match term.dof {
                        CanonicalStructuralDof::Translation(_) => add_cross_product(
                            &mut moment,
                            *coordinates
                                .get(&identity)
                                .ok_or(ConstraintResultError::NodeAbsent)?,
                            component_vector(
                                translation_component(term.dof)?,
                                generalized_reaction,
                            ),
                        ),
                        CanonicalStructuralDof::Rotation(_) => {
                            moment[dof_component(term.dof) - 3] += generalized_reaction;
                        }
                    }
                }
            }
            if moment != group.resultant_moment_about_global_origin_newton_meters {
                return Err(ConstraintResultError::InvalidEvidence);
            }
        }
        Ok(())
    }

    pub fn validate(&self) -> Result<(), ConstraintResultError> {
        if self.schema_version != 1
            || self.constraint_plan_digest.bytes() == &[0; 32]
            || !self.topology.has_valid_identity()
            || self.groups.iter().any(|group| !group_is_finite(group))
            || self.reference_points.iter().any(|point| {
                point.identity.bytes() == &[0; 32]
                    || point.reference_point_id.trim().is_empty()
                    || !all_finite(&point.translation_meters)
                    || !all_finite(&point.rotation_radians)
                    || !all_finite(&point.constraint_force_newtons)
                    || !all_finite(&point.constraint_moment_newton_meters)
            })
            || !self
                .reference_points
                .windows(2)
                .all(|pair| pair[0].identity < pair[1].identity)
            || !self
                .groups
                .windows(2)
                .all(|pair| pair[0].constraint_id.as_str() < pair[1].constraint_id.as_str())
            || !all_finite(&self.resultant_force_newtons)
            || !all_finite(&self.resultant_moment_about_global_origin_newton_meters)
            || !self.constraint_work_joules.is_finite()
            || !self.maximum_displacement_error_meters.is_finite()
            || self.maximum_displacement_error_meters < 0.0
            || !self.maximum_rotation_error_radians.is_finite()
            || self.maximum_rotation_error_radians < 0.0
        {
            return Err(ConstraintResultError::InvalidEvidence);
        }
        let mut force = [0.0; 3];
        let mut moment = [0.0; 3];
        let mut work = 0.0;
        let mut displacement_error: f64 = 0.0;
        let mut rotation_error: f64 = 0.0;
        for group in &self.groups {
            for component in 0..3 {
                force[component] += group.resultant_force_newtons[component];
                moment[component] +=
                    group.resultant_moment_about_global_origin_newton_meters[component];
            }
            work += group.constraint_work_joules;
            displacement_error = displacement_error.max(group.maximum_displacement_error_meters);
            rotation_error = rotation_error.max(group.maximum_rotation_error_radians);
        }
        if force != self.resultant_force_newtons
            || moment != self.resultant_moment_about_global_origin_newton_meters
            || work != self.constraint_work_joules
            || displacement_error != self.maximum_displacement_error_meters
            || rotation_error != self.maximum_rotation_error_radians
        {
            return Err(ConstraintResultError::InvalidEvidence);
        }
        Ok(())
    }
}

fn component_vector(component: usize, value: f64) -> [f64; 3] {
    let mut vector = [0.0; 3];
    vector[component] = value;
    vector
}

fn count(length: usize) -> Result<OperationCount, ConstraintResultError> {
    Ok(OperationCount::from_operations(
        u64::try_from(length).map_err(|_| LedgerError::CountOverflow)?,
    ))
}

fn immediate<T>(
    future: impl std::future::Future<
        Output = Result<T, NumericalCapacityError<ConstraintResultError, std::convert::Infallible>>,
    >,
) -> Result<T, ConstraintResultError> {
    use futures::FutureExt;
    future
        .now_or_never()
        .expect("native constraint result capacity is immediately ready")
        .map_err(|error| match error {
            NumericalCapacityError::Numerical(error) => error,
            NumericalCapacityError::Admission(never) => match never {},
        })
}

fn add_cross_product(moment: &mut [f64; 3], position: [f64; 3], force: [f64; 3]) {
    moment[0] += position[1] * force[2] - position[2] * force[1];
    moment[1] += position[2] * force[0] - position[0] * force[2];
    moment[2] += position[0] * force[1] - position[1] * force[0];
}

fn group_is_finite(group: &ConstraintGroupResult) -> bool {
    !group.constraint_id.is_empty()
        && all_finite(&group.resultant_force_newtons)
        && all_finite(&group.resultant_moment_about_global_origin_newton_meters)
        && group.constraint_work_joules.is_finite()
        && group.maximum_displacement_error_meters.is_finite()
        && group.maximum_displacement_error_meters >= 0.0
        && group.maximum_rotation_error_radians.is_finite()
        && group.maximum_rotation_error_radians >= 0.0
        && group.row_reactions.iter().all(|row| {
            row.row_identity.bytes() != &[0; 32]
                && row.generalized_reaction.is_finite()
                && row.violation.is_valid()
        })
}

fn all_finite(values: &[f64]) -> bool {
    values.iter().all(|value| value.is_finite())
}
