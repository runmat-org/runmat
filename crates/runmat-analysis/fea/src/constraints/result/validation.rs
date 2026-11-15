//! Consistency checks of a recovered constraint result against its plan and executed model.

use std::collections::BTreeMap;

use runmat_analysis_core::AnalysisModel;
use runmat_meshing_core::SolverMeshArtifact;

use super::super::{
    result_state::{dof_component, translation_component},
    CanonicalStructuralConstraintPlan, CanonicalStructuralDof, CanonicalStructuralDofOwner,
    CanonicalStructuralTopologyIdentity, ReactionRecovery,
};
use super::{
    add_cross_product, component_vector, CanonicalStructuralConstraintResult,
    ConstraintGroupResult, ConstraintResultError, ConstraintViolation,
};
use crate::physics::structural::topology::node_space::CanonicalStructuralNodeSpace;

impl CanonicalStructuralConstraintResult {
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
