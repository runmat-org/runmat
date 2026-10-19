use super::*;

impl SolidNewtonProblem<'_> {
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
        let mut attempted_work = SolidAttemptWork::default();
        self.validate_accepted_record(accepted, &mut attempted_work)?;
        if !target.is_finite() || target <= accepted.load_factor {
            return Err(SolidNewtonError::Invalid(
                "target load factor must increase finitely".into(),
            ));
        }
        let mut next = target;
        let mut cutbacks = 0;
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
}
