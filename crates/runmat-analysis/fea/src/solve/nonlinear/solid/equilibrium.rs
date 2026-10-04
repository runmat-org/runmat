use super::{
    constraint_error, norm, relative_norm, AcceptedSolidIncrement, AcceptedSolidState,
    SolidAttemptWork, SolidNewtonError, SolidNewtonProblem,
};

impl SolidNewtonProblem<'_> {
    pub(super) fn solve_trial(
        &self,
        committed: &AcceptedSolidState,
        factor: f64,
        work: &mut SolidAttemptWork,
    ) -> Result<AcceptedSolidIncrement, SolidNewtonError> {
        let mut displacement = committed.displacement.clone();
        let mut evaluation = self.evaluate(&displacement, &committed.constitutive)?;
        work.tangent_evaluations += 1;
        let mut residual = self.residual(&evaluation, factor)?;
        let force_norm = norm(
            &self
                .reduction
                .transpose_apply(self.physical_force)
                .map_err(constraint_error)?,
        ) * factor;
        let scale = force_norm.max(norm(&residual));
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
            let relative_residual = relative_norm(&residual, scale);
            if iteration == self.policy.max_iterations
                && relative_residual > self.policy.relative_residual_tolerance
            {
                break;
            }
            let correction = self.correction(&evaluation.tangent, &residual, factor, work)?;
            let motion_norm = |u: &[f64]| {
                u.iter()
                    .zip(&self.reduction.offset)
                    .fold(0.0_f64, |n, (value, offset)| n.hypot(value - offset))
            };
            let motion_scale = motion_norm(&displacement).max(motion_norm(&committed.displacement));
            let relative_increment = relative_norm(&correction, motion_scale);
            if relative_residual <= self.policy.relative_residual_tolerance
                && relative_increment <= self.policy.relative_residual_tolerance
            {
                let dissipation = evaluation
                    .element_evidence
                    .iter()
                    .map(|e| e.plastic_dissipation_increment_j)
                    .sum::<f64>();
                if !dissipation.is_finite() || dissipation < 0.0 {
                    return Err(SolidNewtonError::Evaluation(
                        "invalid plastic dissipation".into(),
                    ));
                }
                let accumulated = committed.accumulated_plastic_dissipation_j + dissipation;
                if !accumulated.is_finite() {
                    return Err(SolidNewtonError::Evaluation(
                        "plastic dissipation overflow".into(),
                    ));
                }
                let state = AcceptedSolidState {
                    scope_identity: self.scope_identity,
                    load_factor: factor,
                    displacement,
                    constitutive: evaluation.trial_state.clone(),
                    accumulated_plastic_dissipation_j: accumulated,
                };
                self.validate_accepted(&state)?;
                work.tangent_evaluations += 1;
                return Ok(AcceptedSolidIncrement {
                    state,
                    evaluation,
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
            let old_norm = norm(&residual);
            let mut alpha = 1.0;
            let mut candidate = None;
            let attempts = if self.policy.line_search {
                self.policy.max_backtracks
            } else {
                0
            };
            for attempt in 0..=attempts {
                let trial_u = displacement
                    .iter()
                    .zip(&correction)
                    .map(|(u, delta)| u + alpha * delta)
                    .collect::<Vec<_>>();
                // Every line-search candidate uses the SAME accepted constitutive history.
                let trial = self.evaluate(&trial_u, &committed.constitutive)?;
                work.tangent_evaluations += 1;
                let trial_residual = self.residual(&trial, factor)?;
                let trial_norm = norm(&trial_residual);
                if !self.policy.line_search
                    || trial_norm <= (1.0 - 1.0e-4 * alpha) * old_norm
                    || relative_norm(&trial_residual, scale)
                        <= self.policy.relative_residual_tolerance
                {
                    candidate = Some((trial_u, trial, trial_residual));
                    correction_norm = alpha * norm(&correction);
                    break;
                }
                if attempt < attempts {
                    alpha *= self.policy.backtrack_reduction;
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
