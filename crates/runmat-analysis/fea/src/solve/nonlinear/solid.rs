//! Small-strain canonical-solid equilibrium. Trial point histories are never committed
//! by an iteration or a rejected line-search/cutback candidate.
use runmat_canonical_codec::{encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    assembly::{
        nonlinear_solid::{
            CanonicalNonlinearSolidBasis, NonlinearSolidEvaluation, NonlinearSolidState,
        },
        AssemblySummary,
    },
    constraints::AffineConstraintReduction,
};

#[cfg(test)]
pub(crate) mod continuum_qualification;
mod equilibrium;
mod linear_correction;
mod policy;
#[cfg(test)]
mod tests;

pub(crate) use policy::SolidNewtonPolicy;

#[derive(Debug, Error)]
pub(crate) enum SolidNewtonError {
    #[error("invalid canonical solid Newton input: {0}")]
    Invalid(String),
    #[error("canonical solid constitutive/equilibrium evaluation failed: {0}")]
    Evaluation(String),
    #[error("canonical solid Newton cancelled")]
    Cancelled,
    #[error(
        "canonical solid increment at factor {factor} failed after {cutbacks} cutbacks: {reason}"
    )]
    NotConverged {
        factor: f64,
        cutbacks: usize,
        reason: String,
    },
}

/// A converged physical state, not an unconverged guess. Scope binds the constitutive
/// basis, fixed force vector and complete affine equality constraints.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcceptedSolidState {
    pub(crate) scope_identity: Digest,
    pub(crate) load_factor: f64,
    pub(crate) displacement: Vec<f64>,
    pub(crate) constitutive: NonlinearSolidState,
    pub(crate) accumulated_plastic_dissipation_j: f64,
}

impl AcceptedSolidState {
    pub fn scope_identity(&self) -> Digest {
        self.scope_identity
    }

    pub fn basis_identity(&self) -> Digest {
        self.constitutive.basis_identity
    }

    pub fn integration_identity(&self) -> Digest {
        self.constitutive.integration_identity
    }

    pub fn load_factor(&self) -> f64 {
        self.load_factor
    }

    pub fn displacement(&self) -> &[f64] {
        &self.displacement
    }

    pub fn accumulated_plastic_dissipation_j(&self) -> f64 {
        self.accumulated_plastic_dissipation_j
    }

    /// Original ordered point history, never reconstructed from element averages.
    /// Deserialized carriers remain untrusted until validated against their physical scope.
    pub fn point_history(
        &self,
    ) -> impl Iterator<
        Item = (
            runmat_meshing_core::StableDigest,
            usize,
            &crate::physics::structural::plasticity::PlasticityPointState,
        ),
    > {
        self.constitutive
            .point_identities
            .iter()
            .zip(&self.constitutive.points)
            .map(|(identity, history)| {
                (
                    identity.element_identity,
                    identity.quadrature_ordinal,
                    history,
                )
            })
    }
}

pub(crate) struct AcceptedSolidIncrement {
    pub state: AcceptedSolidState,
    pub evaluation: NonlinearSolidEvaluation,
    pub relative_residual: f64,
    /// Undamped pending Newton correction, normalized by current/committed free motion.
    pub relative_increment: f64,
    pub correction_norm: f64,
    pub iterations: usize,
    pub backtracks: usize,
    pub cutbacks: usize,
    /// All trial work since the preceding accepted state, including rejected cutbacks.
    pub attempted_work: SolidAttemptWork,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct SolidAttemptWork {
    pub iterations: usize,
    /// Includes the unapplied convergence solve; Newton update caps are unchanged.
    pub linear_solves: usize,
    pub backtracks: usize,
    /// Constitutive assembly evaluations during trial iterations and acceptance replay.
    pub tangent_evaluations: usize,
}

pub(crate) struct SolidNewtonProblem<'a> {
    basis: &'a CanonicalNonlinearSolidBasis,
    reduction: AffineConstraintReduction,
    physical_force: &'a [f64],
    scope_identity: Digest,
    policy: SolidNewtonPolicy,
    solver_admission: Option<&'a crate::planning::resource::SolverAdmissionResourcePolicy>,
}

impl<'a> SolidNewtonProblem<'a> {
    pub(crate) fn prepare(
        basis: &'a CanonicalNonlinearSolidBasis,
        summary: &'a AssemblySummary,
        policy: SolidNewtonPolicy,
    ) -> Result<Self, SolidNewtonError> {
        policy.validate()?;
        if basis.dof_count() != summary.dof_count
            || summary.physical_load.len() != summary.dof_count
            || summary.physical_load.iter().any(|v| !v.is_finite())
        {
            return Err(SolidNewtonError::Invalid(
                "force/solid DOF inventory differs".into(),
            ));
        }
        let reduction = policy::constraint_reduction(summary)?;
        let encoded = encode_contract(
            b"runmat-fea-small-strain-solid-equilibrium/v1\0",
            "canonical-solid-fixed-load-scope",
            &(
                basis.identity(),
                &summary.physical_load,
                &reduction,
                &summary.constraint_matrix,
                &summary.spatial_projection_evidence,
                &summary.structural_sources,
                &summary.structural_boundary_transfers,
            ),
            CanonicalLimits::new(1024 * 1024 * 1024, 100_000_000, 65_536, 32),
        )
        .map_err(|e| SolidNewtonError::Invalid(e.to_string()))?;
        Ok(Self {
            basis,
            reduction,
            physical_force: &summary.physical_load,
            scope_identity: Digest::sha256(encoded),
            policy,
            solver_admission: None,
        })
    }

    pub(crate) fn with_solver_admission(
        mut self,
        admission: Option<&'a crate::planning::resource::SolverAdmissionResourcePolicy>,
    ) -> Self {
        self.solver_admission = admission;
        self
    }

    pub(crate) fn scope_identity(&self) -> Digest {
        self.scope_identity
    }

    pub(crate) fn initial_state(&self) -> Result<AcceptedSolidState, SolidNewtonError> {
        let constitutive = self.basis.zero_state();
        let displacement = self.reduction.offset.clone();
        let evaluation = self.evaluate(&displacement, &constitutive)?;
        let residual = self.residual(&evaluation, 0.0)?;
        // Fixed nonzero prescribed displacements need their own equilibrated initial
        // transition. Do not invent it by accepting a plastified trial at factor zero.
        if !norm(&residual).is_finite()
            || norm(&residual) != 0.0
            || evaluation.trial_state != constitutive
        {
            return Err(SolidNewtonError::Invalid(
                "nonzero initial lifting requires an equilibrated initial-state transition".into(),
            ));
        }
        Ok(AcceptedSolidState {
            scope_identity: self.scope_identity,
            load_factor: 0.0,
            displacement,
            constitutive,
            accumulated_plastic_dissipation_j: 0.0,
        })
    }

    pub(crate) fn validate_accepted(
        &self,
        state: &AcceptedSolidState,
    ) -> Result<(), SolidNewtonError> {
        if state.scope_identity != self.scope_identity
            || !state.load_factor.is_finite()
            || state.load_factor < 0.0
            || state.displacement.len() != self.basis.dof_count()
            || state.displacement.iter().any(|v| !v.is_finite())
            || !state.accumulated_plastic_dissipation_j.is_finite()
            || state.accumulated_plastic_dissipation_j < 0.0
        {
            return Err(SolidNewtonError::Invalid(
                "foreign/malformed accepted solid state".into(),
            ));
        }
        self.basis
            .validate_state(&state.constitutive)
            .map_err(|e| SolidNewtonError::Invalid(e.to_string()))?;
        let free = self
            .reduction
            .free_dof_indices
            .iter()
            .map(|i| state.displacement[*i])
            .collect::<Vec<_>>();
        let expanded = self.reduction.expand(&free).map_err(constraint_error)?;
        if expanded.iter().any(|v| !v.is_finite())
            || expanded.iter().zip(&state.displacement).any(|(a, b)| {
                (a - b).abs() > 128.0 * f64::EPSILON * a.abs().max(b.abs()).max(f64::MIN_POSITIVE)
            })
        {
            return Err(SolidNewtonError::Invalid(
                "accepted state violates equality constraints".into(),
            ));
        }
        let evaluation = self.evaluate(&state.displacement, &state.constitutive)?;
        let residual = self.residual(&evaluation, state.load_factor)?;
        let scale = norm(
            &self
                .reduction
                .transpose_apply(self.physical_force)
                .map_err(constraint_error)?,
        ) * state.load_factor;
        let relative_residual = relative_norm(&residual, scale);
        if !scale.is_finite()
            || !relative_residual.is_finite()
            || relative_residual > self.policy.relative_residual_tolerance
        {
            return Err(SolidNewtonError::Invalid(
                "accepted state is not in physical equilibrium".into(),
            ));
        }
        self.basis
            .validate_replayed_history(&state.displacement, &state.constitutive)
            .map_err(|e| SolidNewtonError::Invalid(e.to_string()))?;
        let derived_dissipation = self
            .basis
            .accumulated_plastic_dissipation_j(&state.constitutive)
            .map_err(|e| SolidNewtonError::Invalid(e.to_string()))?;
        let dissipation_scale = derived_dissipation.max(state.accumulated_plastic_dissipation_j);
        if (derived_dissipation - state.accumulated_plastic_dissipation_j).abs()
            > 512.0 * f64::EPSILON * dissipation_scale
        {
            return Err(SolidNewtonError::Invalid(
                "accepted accumulated dissipation differs from material-owned point history".into(),
            ));
        }
        Ok(())
    }

    /// Advance with immutable caller-owned accepted state. Exhaustion returns an error
    /// without publishing a partial result or changing the supplied displacement/history.
    #[cfg(test)]
    pub(crate) fn advance(
        &self,
        accepted: &AcceptedSolidState,
        target: f64,
    ) -> Result<Vec<AcceptedSolidIncrement>, SolidNewtonError> {
        let mut current = accepted.clone();
        let mut increments = Vec::new();
        while current.load_factor < target {
            if increments.len() == self.policy.max_accepted_subincrements {
                return Err(SolidNewtonError::NotConverged {
                    factor: target,
                    cutbacks: 0,
                    reason: "accepted subincrement inventory limit reached".into(),
                });
            }
            let increment = self.advance_next(&current, target)?;
            current = increment.state.clone();
            increments.push(increment);
        }
        if increments.is_empty() {
            return Err(SolidNewtonError::Invalid(
                "target load factor must increase finitely".into(),
            ));
        }
        Ok(increments)
    }

    /// Return just the next accepted subincrement. The orchestrator owns the global
    /// accepted inventory and consumes this evidence before requesting another step.
    /// No complete prefix of sparse tangents or trial states is retained in production.
    pub(crate) fn advance_next(
        &self,
        accepted: &AcceptedSolidState,
        target: f64,
    ) -> Result<AcceptedSolidIncrement, SolidNewtonError> {
        self.validate_accepted(accepted)?;
        if !target.is_finite() || target <= accepted.load_factor {
            return Err(SolidNewtonError::Invalid(
                "target load factor must increase finitely".into(),
            ));
        }
        let mut next = target;
        let mut cutbacks = 0;
        let mut attempted_work = SolidAttemptWork::default();
        loop {
            if crate::is_fea_cancelled() {
                return Err(SolidNewtonError::Cancelled);
            }
            match self.solve_trial(accepted, next, &mut attempted_work) {
                Ok(mut increment) => {
                    increment.cutbacks = cutbacks;
                    increment.attempted_work = attempted_work;
                    return Ok(increment);
                }
                Err(SolidNewtonError::NotConverged { reason, .. }) => {
                    if cutbacks == self.policy.max_cutbacks {
                        return Err(SolidNewtonError::NotConverged {
                            factor: next,
                            cutbacks,
                            reason,
                        });
                    }
                    next = accepted.load_factor + 0.5 * (next - accepted.load_factor);
                    cutbacks += 1;
                    if next <= accepted.load_factor
                        || next - accepted.load_factor < self.policy.minimum_load_increment
                    {
                        return Err(SolidNewtonError::NotConverged {
                            factor: next,
                            cutbacks,
                            reason: "minimum representable/permitted load increment reached".into(),
                        });
                    }
                }
                Err(error) => return Err(error),
            }
        }
    }

    fn evaluate(
        &self,
        u: &[f64],
        committed: &NonlinearSolidState,
    ) -> Result<NonlinearSolidEvaluation, SolidNewtonError> {
        self.basis
            .evaluate(u, committed)
            .map_err(|e| SolidNewtonError::Evaluation(e.to_string()))
    }

    fn residual(
        &self,
        evaluation: &NonlinearSolidEvaluation,
        factor: f64,
    ) -> Result<Vec<f64>, SolidNewtonError> {
        let full = evaluation
            .internal_force
            .iter()
            .zip(self.physical_force)
            .map(|(internal, force)| internal - factor * force)
            .collect::<Vec<_>>();
        if full.iter().any(|v| !v.is_finite()) {
            return Err(SolidNewtonError::Evaluation(
                "physical residual overflow".into(),
            ));
        }
        let reduced = self
            .reduction
            .transpose_apply(&full)
            .map_err(constraint_error)?;
        if reduced.iter().any(|v| !v.is_finite()) {
            return Err(SolidNewtonError::Evaluation(
                "reduced residual overflow".into(),
            ));
        }
        Ok(reduced)
    }
}

fn constraint_error(error: impl std::fmt::Display) -> SolidNewtonError {
    SolidNewtonError::Invalid(error.to_string())
}

// Hypot accumulation avoids norm overflow for otherwise finite inputs.
fn norm(values: &[f64]) -> f64 {
    values.iter().fold(0.0, |a, v| a.hypot(*v))
}
fn relative_norm(values: &[f64], scale: f64) -> f64 {
    let value = norm(values);
    if value == 0.0 {
        0.0
    } else if scale > 0.0 {
        value / scale
    } else {
        f64::INFINITY
    }
}
