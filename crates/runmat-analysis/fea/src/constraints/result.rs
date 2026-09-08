use std::collections::BTreeMap;

use runmat_analysis_core::{AnalysisField, AnalysisFieldValues, StructuralTranslationDof};
use runmat_execution::Digest;
use runmat_meshing_core::{SolverMeshArtifact, StableDigest};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::{CanonicalStructuralConstraintPlan, CanonicalStructuralDof, ReactionRecovery};
use crate::contracts::{FEA_FIELD_STRUCTURAL_DISPLACEMENT, FEA_FIELD_STRUCTURAL_REACTION_FORCE};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConstraintGroupResult {
    pub constraint_id: String,
    pub active_row_count: usize,
    pub redundant_row_count: usize,
    pub resultant_force_newtons: [f64; 3],
    pub resultant_moment_about_global_origin_newton_meters: [f64; 3],
    pub constraint_work_joules: f64,
    pub maximum_prescribed_error_meters: f64,
    pub row_reactions: Vec<ConstraintRowReaction>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConstraintRowReaction {
    pub row_identity: Digest,
    pub generalized_reaction: f64,
    pub violation_meters: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalStructuralConstraintResult {
    pub schema_version: u16,
    pub constraint_plan_digest: Digest,
    pub topology_digest: StableDigest,
    pub groups: Vec<ConstraintGroupResult>,
    pub resultant_force_newtons: [f64; 3],
    pub resultant_moment_about_global_origin_newton_meters: [f64; 3],
    pub constraint_work_joules: f64,
    pub maximum_prescribed_error_meters: f64,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum ConstraintResultError {
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
    #[error("constraint plan could not be validated: {0}")]
    InvalidPlan(String),
}

impl CanonicalStructuralConstraintResult {
    pub fn recover(
        plan: &CanonicalStructuralConstraintPlan,
        mesh: &SolverMeshArtifact,
        fields: &[AnalysisField],
    ) -> Result<Self, ConstraintResultError> {
        plan.validate()
            .map_err(|error| ConstraintResultError::InvalidPlan(error.to_string()))?;
        mesh.validate()
            .map_err(|error| ConstraintResultError::InvalidPlan(error.to_string()))?;
        if plan.topology_digest != mesh.canonical_digest {
            return Err(ConstraintResultError::TopologyMismatch);
        }
        let displacement = nodal_vector_field(fields, FEA_FIELD_STRUCTURAL_DISPLACEMENT, mesh)?;
        let reaction = nodal_vector_field(fields, FEA_FIELD_STRUCTURAL_REACTION_FORCE, mesh)?;
        let nodes = mesh
            .topology
            .nodes
            .iter()
            .enumerate()
            .map(|(index, node)| (node.stable_identity, (index, node.coordinates_m)))
            .collect::<BTreeMap<_, _>>();
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
                        maximum_prescribed_error_meters: 0.0,
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
                        let (node_index, _) = nodes
                            .get(&term.node_identity)
                            .copied()
                            .ok_or(ConstraintResultError::NodeAbsent)?;
                        let component = translation_component(term.dof)?;
                        Ok((node_index * 3 + component, term.coefficient))
                    })
                    .collect::<Result<BTreeMap<_, _>, ConstraintResultError>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut projected_reaction = vec![0.0; active_rows.len()];
        for (left, coefficients) in row_coefficients.iter().enumerate() {
            projected_reaction[left] = coefficients
                .iter()
                .map(|(dof, coefficient)| coefficient * reaction[*dof])
                .sum();
        }
        let multipliers = solve_constraint_multipliers(&row_coefficients, &projected_reaction)
            .ok_or(ConstraintResultError::ReactionRecoveryFailed)?;
        let mut predicted_reaction = BTreeMap::<usize, f64>::new();
        for (coefficients, multiplier) in row_coefficients.iter().zip(&multipliers) {
            for (dof, coefficient) in coefficients {
                *predicted_reaction.entry(*dof).or_insert(0.0) += coefficient * multiplier;
            }
        }
        let reaction_scale = predicted_reaction
            .values()
            .map(|value| value.abs())
            .fold(1.0_f64, f64::max);
        if predicted_reaction
            .iter()
            .any(|(dof, predicted)| (reaction[*dof] - predicted).abs() > reaction_scale * 1.0e-7)
        {
            return Err(ConstraintResultError::ReactionRecoveryFailed);
        }

        for row in &plan.rows {
            let group = groups
                .get_mut(row.constraint_id.as_str())
                .ok_or(ConstraintResultError::GroupAbsent)?;
            if row.reaction_recovery == ReactionRecovery::RedundantEquation {
                group.redundant_row_count += 1;
                continue;
            }
            let active_index = active_rows
                .iter()
                .position(|active| active.identity == row.identity)
                .ok_or(ConstraintResultError::InvalidEvidence)?;
            let multiplier = multipliers[active_index];
            let equation_value = row
                .terms
                .iter()
                .map(|term| {
                    let (node_index, _) = nodes[&term.node_identity];
                    let component = translation_component(term.dof)?;
                    Ok(term.coefficient * displacement[node_index * 3 + component])
                })
                .collect::<Result<Vec<_>, ConstraintResultError>>()?
                .into_iter()
                .sum::<f64>();
            let error = (equation_value - row.right_hand_side).abs();
            group.active_row_count += 1;
            for term in &row.terms {
                let (_, coordinates) = nodes[&term.node_identity];
                let component = translation_component(term.dof)?;
                let force = term.coefficient * multiplier;
                group.resultant_force_newtons[component] += force;
                add_cross_product(
                    &mut group.resultant_moment_about_global_origin_newton_meters,
                    coordinates,
                    component_vector(component, force),
                );
            }
            group.constraint_work_joules += multiplier * row.right_hand_side;
            group.maximum_prescribed_error_meters =
                group.maximum_prescribed_error_meters.max(error);
            group.row_reactions.push(ConstraintRowReaction {
                row_identity: row.identity,
                generalized_reaction: multiplier,
                violation_meters: error,
            });
        }

        let groups = groups.into_values().collect::<Vec<_>>();
        let mut result = Self {
            schema_version: 1,
            constraint_plan_digest: plan
                .content_digest()
                .map_err(|error| ConstraintResultError::InvalidPlan(error.to_string()))?,
            topology_digest: mesh.canonical_digest,
            resultant_force_newtons: [0.0; 3],
            resultant_moment_about_global_origin_newton_meters: [0.0; 3],
            constraint_work_joules: 0.0,
            maximum_prescribed_error_meters: 0.0,
            groups,
        };
        for group in &result.groups {
            for component in 0..3 {
                result.resultant_force_newtons[component] +=
                    group.resultant_force_newtons[component];
                result.resultant_moment_about_global_origin_newton_meters[component] +=
                    group.resultant_moment_about_global_origin_newton_meters[component];
            }
            result.constraint_work_joules += group.constraint_work_joules;
            result.maximum_prescribed_error_meters = result
                .maximum_prescribed_error_meters
                .max(group.maximum_prescribed_error_meters);
        }
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
            || self.topology_digest != plan.topology_digest
            || self.groups.len() != plan.groups.len()
        {
            return Err(ConstraintResultError::InvalidEvidence);
        }
        for (group, planned) in self.groups.iter().zip(&plan.groups) {
            let expected_active = plan
                .rows
                .iter()
                .filter(|row| {
                    row.constraint_id == planned.constraint_id
                        && row.reaction_recovery != ReactionRecovery::RedundantEquation
                })
                .count();
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
        }
        Ok(())
    }

    pub fn validate(&self) -> Result<(), ConstraintResultError> {
        if self.schema_version != 1
            || self.constraint_plan_digest.bytes() == &[0; 32]
            || self.topology_digest == StableDigest::ZERO
            || self.groups.iter().any(|group| !group_is_finite(group))
            || !self
                .groups
                .windows(2)
                .all(|pair| pair[0].constraint_id.as_str() < pair[1].constraint_id.as_str())
            || !all_finite(&self.resultant_force_newtons)
            || !all_finite(&self.resultant_moment_about_global_origin_newton_meters)
            || !self.constraint_work_joules.is_finite()
            || !self.maximum_prescribed_error_meters.is_finite()
            || self.maximum_prescribed_error_meters < 0.0
        {
            return Err(ConstraintResultError::InvalidEvidence);
        }
        let mut force = [0.0; 3];
        let mut moment = [0.0; 3];
        let mut work = 0.0;
        let mut error: f64 = 0.0;
        for group in &self.groups {
            for component in 0..3 {
                force[component] += group.resultant_force_newtons[component];
                moment[component] +=
                    group.resultant_moment_about_global_origin_newton_meters[component];
            }
            work += group.constraint_work_joules;
            error = error.max(group.maximum_prescribed_error_meters);
        }
        if force != self.resultant_force_newtons
            || moment != self.resultant_moment_about_global_origin_newton_meters
            || work != self.constraint_work_joules
            || error != self.maximum_prescribed_error_meters
        {
            return Err(ConstraintResultError::InvalidEvidence);
        }
        Ok(())
    }
}

fn nodal_vector_field<'a>(
    fields: &'a [AnalysisField],
    field_id: &'static str,
    mesh: &SolverMeshArtifact,
) -> Result<&'a [f64], ConstraintResultError> {
    let field = fields
        .iter()
        .find(|field| field.field_id == field_id)
        .ok_or(ConstraintResultError::MissingField { field_id })?;
    let AnalysisFieldValues::HostF64(values) = &field.values else {
        return Err(ConstraintResultError::InvalidField { field_id });
    };
    if field.shape != [mesh.topology.nodes.len(), 3]
        || values.len() != mesh.topology.nodes.len() * 3
        || values.iter().any(|value| !value.is_finite())
    {
        return Err(ConstraintResultError::InvalidField { field_id });
    }
    Ok(values)
}

fn translation_component(dof: CanonicalStructuralDof) -> Result<usize, ConstraintResultError> {
    match dof {
        CanonicalStructuralDof::Translation(StructuralTranslationDof::Ux) => Ok(0),
        CanonicalStructuralDof::Translation(StructuralTranslationDof::Uy) => Ok(1),
        CanonicalStructuralDof::Translation(StructuralTranslationDof::Uz) => Ok(2),
        CanonicalStructuralDof::Rotation(_) => Err(ConstraintResultError::InvalidEvidence),
    }
}

fn component_vector(component: usize, value: f64) -> [f64; 3] {
    let mut vector = [0.0; 3];
    vector[component] = value;
    vector
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
        && group.maximum_prescribed_error_meters.is_finite()
        && group.maximum_prescribed_error_meters >= 0.0
        && group.row_reactions.iter().all(|row| {
            row.row_identity.bytes() != &[0; 32]
                && row.generalized_reaction.is_finite()
                && row.violation_meters.is_finite()
                && row.violation_meters >= 0.0
        })
}

fn solve_constraint_multipliers(
    rows: &[BTreeMap<usize, f64>],
    right_hand_side: &[f64],
) -> Option<Vec<f64>> {
    if rows.len() != right_hand_side.len() {
        return None;
    }
    if rows.is_empty() {
        return Some(Vec::new());
    }
    let inverse_diagonal = rows
        .iter()
        .map(|row| {
            let diagonal = row.values().map(|value| value * value).sum::<f64>();
            (diagonal > 1.0e-18 && diagonal.is_finite()).then_some(diagonal.recip())
        })
        .collect::<Option<Vec<_>>>()?;
    let mut solution = vec![0.0; rows.len()];
    let mut residual = right_hand_side.to_vec();
    let mut preconditioned = residual
        .iter()
        .zip(&inverse_diagonal)
        .map(|(residual, inverse)| residual * inverse)
        .collect::<Vec<_>>();
    let mut direction = preconditioned.clone();
    let mut rz = dot(&residual, &preconditioned);
    let scale = dot(right_hand_side, right_hand_side).sqrt().max(1.0);
    if dot(&residual, &residual).sqrt() / scale <= 1.0e-10 {
        return Some(solution);
    }
    for _ in 0..rows.len().saturating_mul(4).max(256) {
        let applied = apply_constraint_gram(rows, &direction);
        let denominator = dot(&direction, &applied);
        if !denominator.is_finite() || denominator <= 1.0e-18 {
            return None;
        }
        let alpha = rz / denominator;
        for index in 0..rows.len() {
            solution[index] += alpha * direction[index];
            residual[index] -= alpha * applied[index];
        }
        if dot(&residual, &residual).sqrt() / scale <= 1.0e-10 {
            return solution
                .iter()
                .all(|value| value.is_finite())
                .then_some(solution);
        }
        preconditioned = residual
            .iter()
            .zip(&inverse_diagonal)
            .map(|(residual, inverse)| residual * inverse)
            .collect();
        let next_rz = dot(&residual, &preconditioned);
        let beta = next_rz / rz;
        for (direction, preconditioned) in direction.iter_mut().zip(&preconditioned) {
            *direction = preconditioned + beta * *direction;
        }
        rz = next_rz;
    }
    None
}

fn apply_constraint_gram(rows: &[BTreeMap<usize, f64>], values: &[f64]) -> Vec<f64> {
    let mut dofs = BTreeMap::<usize, f64>::new();
    for (row, value) in rows.iter().zip(values) {
        for (dof, coefficient) in row {
            *dofs.entry(*dof).or_insert(0.0) += coefficient * value;
        }
    }
    rows.iter()
        .map(|row| {
            row.iter()
                .map(|(dof, coefficient)| coefficient * dofs.get(dof).copied().unwrap_or(0.0))
                .sum()
        })
        .collect()
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

fn all_finite(values: &[f64]) -> bool {
    values.iter().all(|value| value.is_finite())
}
