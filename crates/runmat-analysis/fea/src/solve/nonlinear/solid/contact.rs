//! Contact coupling inside the existing material-history Newton procedure.
use super::*;
use crate::{
    algebra::{MatrixProperties, OperatorRole, RealSparseAssembler, SparsePattern},
    physics::structural::contact::{
        CanonicalContactOperator, ContactConfigurationState, ContactOperatorError,
    },
    solve::general::{GeneralSolveEnvelope, GeneralSolvePolicy},
};
mod energy;
mod preparation;
pub use energy::AcceptedContactEnergy;
pub(super) mod resources;
#[derive(Clone, Copy)]
pub(crate) struct SolidContactCoupling<'a> {
    pub operator: &'a CanonicalContactOperator<'a>,
    pub solve_policy: GeneralSolvePolicy,
    pub solve_admission: GeneralSolveEnvelope,
    pub maximum_additional_resident_bytes: u64,
    pub maximum_geometry_trials_per_advance: u64,
    pub maximum_symbolic_scalar_operations: u64,
}
pub(super) struct PreparedSolidContact<'a> {
    pub input: SolidContactCoupling<'a>,
    pub pattern: SparsePattern,
    pub reduced_pattern: SparsePattern,
    pub solve_resources: Option<crate::planning::resource::GeneralSparseSolveResourceDemand>,
}
pub(super) struct SolidTrialEvaluation {
    pub(super) evaluation: NonlinearSolidEvaluation,
    pub(super) contact: Option<ContactConfigurationState>,
    pub(super) contact_diagnostics:
        Option<crate::physics::structural::contact::ContactAssemblyDiagnostics>,
}
pub(in crate::solve::nonlinear::solid) fn contact_error(
    error: ContactOperatorError,
) -> SolidNewtonError {
    match error {
        ContactOperatorError::Cancelled => SolidNewtonError::Cancelled,
        other => SolidNewtonError::Contact(other),
    }
}
impl SolidNewtonProblem<'_> {
    pub(super) fn validate_contact_tangent(
        &self,
        tangent: &crate::algebra::RealSparseMatrix,
        factor: f64,
        work: &mut SolidAttemptWork,
    ) -> Result<(), SolidNewtonError> {
        if self.contact.is_some() {
            // The certified direct correction rejects a singular reduced tangent (an open pair
            // can leave a body unanchored); a zero load must give a zero correction.
            let zero = vec![0.; self.reduction.reduced_dof_count];
            let correction = self.correction(tangent, &zero, factor, work)?;
            if correction.iter().any(|v| *v != 0.) {
                return Err(SolidNewtonError::Evaluation(
                    "nonzero zero-load contact correction".into(),
                ));
            }
        }
        Ok(())
    }
    pub(super) fn initial_contact(
        &self,
        u: &[f64],
    ) -> Result<Option<ContactConfigurationState>, SolidNewtonError> {
        self.contact
            .as_ref()
            .map(|c| {
                let positions = self.basis.contact_positions(u).map_err(body_error)?;
                c.input
                    .operator
                    .initial_configuration(&positions)
                    .map_err(contact_error)
            })
            .transpose()
    }
    pub(super) fn validate_contact_state(
        &self,
        state: &AcceptedSolidState,
    ) -> Result<(), SolidNewtonError> {
        match (&self.contact, &state.contact) {
            (None, None) => Ok(()),
            (Some(c), Some(original)) => {
                if (state.load_factor == 0. && original.ordinal() != 0)
                    || (state.load_factor > 0. && original.ordinal() == 0)
                {
                    return Err(SolidNewtonError::Invalid(
                        "contact accepted ordinal/load differs".into(),
                    ));
                }
                let positions = self
                    .basis
                    .contact_positions(&state.displacement)
                    .map_err(body_error)?;
                if positions != original.positions_m() {
                    return Err(SolidNewtonError::Invalid(
                        "contact original configuration/body displacement differs".into(),
                    ));
                }
                c.input
                    .operator
                    .validate_configuration(original, original.ordinal())
                    .map_err(contact_error)
            }
            _ => Err(SolidNewtonError::Invalid(
                "missing or foreign contact accepted history".into(),
            )),
        }
    }
    pub(super) fn evaluate_trial(
        &self,
        u: &[f64],
        material: &NonlinearSolidState,
        contact: Option<&ContactConfigurationState>,
        factor: f64,
    ) -> Result<SolidTrialEvaluation, SolidNewtonError> {
        let mut evaluation = self.evaluate(u, material)?;
        if let Some(follower) = self.follower {
            self.add_follower_loads(follower, u, factor, &mut evaluation)?;
        }
        let Some(c) = &self.contact else {
            if contact.is_some() {
                return Err(SolidNewtonError::Invalid(
                    "foreign contact trial history".into(),
                ));
            }
            return Ok(SolidTrialEvaluation {
                evaluation,
                contact: None,
                contact_diagnostics: None,
            });
        };
        let original = contact.ok_or_else(|| {
            SolidNewtonError::Invalid("missing original contact trial history".into())
        })?;
        let positions = self.basis.contact_positions(u).map_err(body_error)?;
        let trial = c
            .input
            .operator
            .assemble_trial(original, original.ordinal(), &positions)
            .map_err(contact_error)?;
        let mut tangent = RealSparseAssembler::with_pattern(
            c.pattern.try_clone().map_err(constraint_error)?,
            OperatorRole::EffectiveTangent,
            MatrixProperties::GENERAL,
        )
        .map_err(constraint_error)?;
        // One shared planned support combines original material and contact actions.
        for operator in [&evaluation.tangent, &trial.tangent] {
            for contribution in operator.contributions() {
                tangent.add_contribution(contribution.clone());
            }
            for row in 0..operator.rows() {
                for entry in operator.pattern().row_range(row) {
                    tangent.add(
                        row,
                        operator.pattern().column(entry),
                        operator.values()[entry],
                    );
                }
            }
        }
        for (body, contact) in evaluation.internal_force.iter_mut().zip(&trial.residual) {
            *body += contact;
            if !body.is_finite() {
                return Err(SolidNewtonError::Evaluation(
                    "coupled contact force overflow".into(),
                ));
            }
        }
        evaluation.tangent = tangent.finish().map_err(constraint_error)?;
        Ok(SolidTrialEvaluation {
            evaluation,
            contact: Some(trial.configuration),
            contact_diagnostics: Some(trial.diagnostics),
        })
    }
}
impl SolidNewtonProblem<'_> {
    /// Folds `-λ f(u)` and `-λ ∂f/∂u` of the follower loads into the body evaluation, so the
    /// residual stays `f_int - λ f_dead` and constraint reactions include follower loading.
    fn add_follower_loads(
        &self,
        follower: &crate::assembly::nonlinear_solid::FollowerLoadOperator,
        u: &[f64],
        factor: f64,
        evaluation: &mut NonlinearSolidEvaluation,
    ) -> Result<(), SolidNewtonError> {
        let loads = self
            .basis
            .evaluate_follower_loads(follower, u)
            .map_err(body_error)?;
        let mut tangent = RealSparseAssembler::with_pattern(
            self.basis
                .contact_pattern()
                .try_clone()
                .map_err(constraint_error)?,
            OperatorRole::EffectiveTangent,
            MatrixProperties::GENERAL,
        )
        .map_err(constraint_error)?;
        let body = &evaluation.tangent;
        for contribution in body.contributions() {
            tangent.add_contribution(contribution.clone());
        }
        tangent.add_contribution(crate::algebra::ContributionProvenance {
            domain: crate::algebra::ContributionDomain::Structural,
            contribution_id: crate::algebra::ContributionId("follower_load_stiffness".into()),
        });
        for row in 0..body.rows() {
            for entry in body.pattern().row_range(row) {
                tangent.add(row, body.pattern().column(entry), body.values()[entry]);
            }
        }
        for block in &loads.blocks {
            let width = block.rows.len();
            for (i, &row) in block.rows.iter().enumerate() {
                for (j, &column) in block.rows.iter().enumerate() {
                    tangent.add(row, column, -factor * block.values[i * width + j]);
                }
            }
        }
        for (internal, load) in evaluation.internal_force.iter_mut().zip(&loads.force) {
            *internal -= factor * load;
            if !internal.is_finite() {
                return Err(SolidNewtonError::Evaluation(
                    "follower load force overflow".into(),
                ));
            }
        }
        evaluation.tangent = tangent.finish().map_err(constraint_error)?;
        Ok(())
    }
}

fn body_error(error: NonlinearSolidError) -> SolidNewtonError {
    match error {
        NonlinearSolidError::Cancelled => SolidNewtonError::Cancelled,
        NonlinearSolidError::Accounting(error) => SolidNewtonError::Usage(error),
        other => SolidNewtonError::Evaluation(other.to_string()),
    }
}
