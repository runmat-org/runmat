use super::{
    constraint_error, measured_solid, norm, solid_count, AcceptedSolidIncrement,
    AcceptedSolidState, SolidAttemptWork, SolidNewtonError, SolidNewtonProblem,
};
use runmat_simulation_usage::LedgerError;

fn measured_norm(values: &[f64]) -> Result<f64, SolidNewtonError> {
    measured_solid(|| solid_count(values.len(), 1), || Ok(norm(values)))
}

fn measured_relative_norm(values: &[f64], scale: f64) -> Result<f64, SolidNewtonError> {
    let value = measured_norm(values)?;
    if value == 0.0 {
        Ok(0.0)
    } else if scale > 0.0 {
        measured_solid(|| solid_count(1, 1), || Ok(value / scale))
    } else {
        Ok(f64::INFINITY)
    }
}

impl SolidNewtonProblem<'_> {
    pub(super) fn solve_trial(
        &self,
        committed: &AcceptedSolidState,
        factor: f64,
        work: &mut SolidAttemptWork,
    ) -> Result<AcceptedSolidIncrement, SolidNewtonError> {
        // Supports move with the load factor; free coordinates start from the accepted state.
        let delta_factor =
            measured_solid(|| solid_count(1, 1), || Ok(factor - committed.load_factor))?;
        let mut displacement = measured_solid(
            || solid_count(committed.displacement.len().min(self.prescribed.len()), 2),
            || {
                Ok(committed
                    .displacement
                    .iter()
                    .zip(&self.prescribed)
                    .map(|(u, p)| u + delta_factor * p)
                    .collect::<Vec<_>>())
            },
        )?;
        let mut evaluation = self.evaluate_trial(
            &displacement,
            &committed.constitutive,
            committed.contact.as_ref(),
            factor,
        )?;
        work.tangent_evaluations += 1;
        let mut residual = self.residual(&evaluation.evaluation, factor)?;
        let reduced_force = measured_solid(
            || solid_count(self.reduction.recovery_coefficients.len(), 2),
            || {
                self.reduction
                    .transpose_apply(&self.scale_force)
                    .map_err(constraint_error)
            },
        )?;
        let force_norm = measured_solid(
            || {
                solid_count(reduced_force.len(), 1)?
                    .checked_add(solid_count(1, 1)?)
                    .map_err(LedgerError::from)
            },
            || Ok(norm(&reduced_force) * factor),
        )?;
        let scale = force_norm.max(measured_norm(&residual)?);
        if !scale.is_finite() || !force_norm.is_finite() {
            return Err(SolidNewtonError::Evaluation(
                "physical force/residual norm overflow".into(),
            ));
        }
        let mut correction_norm = 0.0;
        let mut backtracks = 0;
        for iteration in 0..=self.policy.max_iterations {
            if crate::is_fea_cancelled() {
                return Err(SolidNewtonError::Cancelled);
            }
            let relative_residual = measured_relative_norm(&residual, scale)?;
            if iteration == self.policy.max_iterations
                && relative_residual > self.policy.relative_residual_tolerance
            {
                break;
            }
            let correction =
                self.correction(&evaluation.evaluation.tangent, &residual, factor, work)?;
            let motion_scale =
                measured_norm(&displacement)?.max(measured_norm(&committed.displacement)?);
            let relative_increment = measured_relative_norm(&correction, motion_scale)?;
            if relative_residual <= self.policy.relative_residual_tolerance
                && relative_increment <= self.policy.relative_residual_tolerance
            {
                let dissipation = measured_solid(
                    || {
                        solid_count(
                            evaluation
                                .evaluation
                                .element_evidence
                                .len()
                                .saturating_sub(1),
                            1,
                        )
                    },
                    || {
                        Ok(evaluation
                            .evaluation
                            .element_evidence
                            .iter()
                            .map(|e| e.plastic_dissipation_increment_j)
                            .sum::<f64>())
                    },
                )?;
                if !dissipation.is_finite() || dissipation < 0.0 {
                    return Err(SolidNewtonError::Evaluation(
                        "invalid plastic dissipation".into(),
                    ));
                }
                let accumulated = measured_solid(
                    || solid_count(1, 1),
                    || Ok(committed.accumulated_plastic_dissipation_j + dissipation),
                )?;
                if !accumulated.is_finite() {
                    return Err(SolidNewtonError::Evaluation(
                        "plastic dissipation overflow".into(),
                    ));
                }
                let contact_energy = match (
                    committed.contact_energy.as_ref(),
                    evaluation.contact_diagnostics,
                ) {
                    (None, None) => None,
                    (Some(previous), Some(diagnostics)) => Some(previous.advance(diagnostics)?),
                    _ => {
                        return Err(SolidNewtonError::Invalid(
                            "contact energy/trial inventory differs".into(),
                        ))
                    }
                };
                let state = AcceptedSolidState {
                    scope_identity: self.scope_identity,
                    load_factor: factor,
                    displacement,
                    constitutive: evaluation.evaluation.trial_state.clone(),
                    contact: evaluation.contact,
                    contact_energy,
                    accumulated_plastic_dissipation_j: accumulated,
                };
                self.validate_accepted_record(&state, work)?;
                work.tangent_evaluations += 1;
                return Ok(AcceptedSolidIncrement {
                    state,
                    evaluation: evaluation.evaluation,
                    relative_residual,
                    relative_increment,
                    correction_norm,
                    iterations: iteration,
                    backtracks,
                    cutbacks: 0,
                    attempted_work: *work,
                });
            }
            if iteration == self.policy.max_iterations {
                break;
            }
            work.iterations += 1;
            let old_norm = measured_norm(&residual)?;
            let mut alpha = 1.0;
            let mut candidate = None;
            let attempts = if self.policy.line_search {
                self.policy.max_backtracks
            } else {
                0
            };
            for attempt in 0..=attempts {
                let trial_u = measured_solid(
                    || solid_count(displacement.len().min(correction.len()), 2),
                    || {
                        Ok(displacement
                            .iter()
                            .zip(&correction)
                            .map(|(u, delta)| u + alpha * delta)
                            .collect::<Vec<_>>())
                    },
                )?;
                // Every line-search candidate uses the SAME accepted constitutive history.
                let trial = self.evaluate_trial(
                    &trial_u,
                    &committed.constitutive,
                    committed.contact.as_ref(),
                    factor,
                )?;
                work.tangent_evaluations += 1;
                let trial_residual = self.residual(&trial.evaluation, factor)?;
                let trial_norm = measured_norm(&trial_residual)?;
                let sufficient = if !self.policy.line_search {
                    true
                } else {
                    let threshold = measured_solid(
                        || solid_count(1, 3),
                        || Ok((1.0 - 1.0e-4 * alpha) * old_norm),
                    )?;
                    trial_norm <= threshold
                        || measured_relative_norm(&trial_residual, scale)?
                            <= self.policy.relative_residual_tolerance
                };
                if sufficient {
                    candidate = Some((trial_u, trial, trial_residual));
                    let magnitude = measured_norm(&correction)?;
                    correction_norm =
                        measured_solid(|| solid_count(1, 1), || Ok(alpha * magnitude))?;
                    break;
                }
                if attempt < attempts {
                    alpha = measured_solid(
                        || solid_count(1, 1),
                        || Ok(alpha * self.policy.backtrack_reduction),
                    )?;
                    backtracks += 1;
                    work.backtracks += 1;
                }
            }
            let Some((next_u, next_evaluation, next_residual)) = candidate else {
                return Err(SolidNewtonError::NotConverged {
                    factor,
                    cutbacks: 0,
                    reason: "physical residual line search exhausted".into(),
                });
            };
            displacement = next_u;
            evaluation = next_evaluation;
            residual = next_residual;
        }
        Err(SolidNewtonError::NotConverged {
            factor,
            cutbacks: 0,
            reason: "physical residual/increment Newton iteration limit reached".into(),
        })
    }
}
