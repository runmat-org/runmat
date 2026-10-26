//! The nonlinear merit is separate from actual user fitness and original
//! feasibility. An augmented score is never returned as the user's fval.
use super::{invalid, ScalarCooperation, POLICY};
use crate::BuiltinResult;

#[cfg(test)]
mod tests;

pub(super) struct AugmentedLagrangian {
    multipliers: Vec<f64>,
    equality_multipliers: Vec<f64>,
    penalty: f64,
    factor: f64,
    feasibility_target: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Update {
    Multipliers,
    Penalty,
    Unrepresentable,
}

#[derive(Clone, Copy)]
pub(super) enum BarrierDomain {
    Interior,
    Exterior(f64),
    Undefined,
}

impl AugmentedLagrangian {
    /// Known maxima come from the actual starting population. None is unknown,
    /// never a fabricated measured zero. If a covering shift is unrepresentable,
    /// a finite positive seed is retained and the controller must restore the
    /// actual strict barrier domain before minimizing the logarithmic merit.
    pub async fn new(
        largest_inequalities: &[Option<f64>],
        equalities: usize,
        penalty: f64,
        factor: f64,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        if !penalty.is_finite() || penalty < 1.0 || !factor.is_finite() || factor <= 1.0 {
            return Err(invalid(
                "InitialPenalty must be at least 1 and PenaltyFactor must exceed 1",
            ));
        }
        let mut multipliers = POLICY.capacity(largest_inequalities.len(), cooperation.runtime())?;
        for &value in largest_inequalities {
            if value.is_some_and(|value| !value.is_finite()) {
                return Err(invalid("initial nonlinear inequalities must be finite"));
            }
            let maximum = value.unwrap_or(0.0).max(0.0);
            let shifted = maximum + 1.0_f64.max(maximum * f64::EPSILON.sqrt());
            let covering = penalty * shifted;
            let multiplier = if covering.is_finite() && covering / penalty > maximum {
                covering
            } else {
                penalty
            };
            multipliers.push(multiplier);
            cooperation.completed(1).await?;
        }
        let mut equality_multipliers = POLICY.capacity(equalities, cooperation.runtime())?;
        for _ in 0..equalities {
            equality_multipliers.push(0.0);
            cooperation.completed(1).await?;
        }
        Ok(Self {
            multipliers,
            equality_multipliers,
            penalty,
            factor,
            feasibility_target: penalty.powf(-0.75),
        })
    }

    pub fn domain(&self, inequalities: &[f64]) -> BarrierDomain {
        if inequalities.len() != self.multipliers.len() {
            return BarrierDomain::Undefined;
        }
        let mut exterior = false;
        let mut maximum = 0.0_f64;
        for (&constraint, &multiplier) in inequalities.iter().zip(&self.multipliers) {
            if !constraint.is_finite() {
                return BarrierDomain::Undefined;
            }
            let shift = multiplier / self.penalty;
            if constraint >= shift {
                exterior = true;
                maximum = maximum.max(constraint - shift);
            }
        }
        if exterior {
            BarrierDomain::Exterior(maximum)
        } else {
            BarrierDomain::Interior
        }
    }

    /// Shifted-barrier form with c<=0 and ceq=0. Barrier domain failures are
    /// undefined trials, not a clamped surrogate or a penalty-algorithm alias.
    pub fn score(&self, fitness: f64, inequalities: &[f64], equalities: &[f64]) -> Option<f64> {
        if !fitness.is_finite()
            || inequalities.len() != self.multipliers.len()
            || equalities.len() != self.equality_multipliers.len()
        {
            return None;
        }
        let mut score = fitness;
        for (&constraint, &multiplier) in inequalities.iter().zip(&self.multipliers) {
            let shift = multiplier / self.penalty;
            let distance = shift - constraint;
            if !constraint.is_finite() || !distance.is_finite() || distance <= 0.0 {
                return None;
            }
            score -= multiplier * shift * distance.ln();
            if !score.is_finite() {
                return None;
            }
        }
        for (&constraint, &multiplier) in equalities.iter().zip(&self.equality_multipliers) {
            if !constraint.is_finite() {
                return None;
            }
            score += constraint * multiplier + (0.5 * self.penalty * constraint) * constraint;
            if !score.is_finite() {
                return None;
            }
        }
        Some(score)
    }

    /// The outer controller calls this only after an actual fixed-parameter
    /// GA subproblem. It supplies whether that subproblem met its declared
    /// numerical stopping tolerance; this owner does not invent stationarity.
    pub async fn update(
        &mut self,
        inequalities: &[f64],
        equalities: &[f64],
        subproblem_solved: bool,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Update> {
        if inequalities.len() != self.multipliers.len()
            || equalities.len() != self.equality_multipliers.len()
        {
            return Err(invalid("nonlinear constraint extent changed"));
        }
        let mut eligible = subproblem_solved;
        for (&constraint, &multiplier) in inequalities.iter().zip(&self.multipliers) {
            let shift = multiplier / self.penalty;
            let distance = shift - constraint;
            let candidate = multiplier * (shift / distance);
            let scaled_complementarity = constraint * (candidate / multiplier);
            eligible &= constraint.is_finite()
                && distance > 0.0
                && candidate.is_finite()
                && candidate > 0.0
                && scaled_complementarity.is_finite()
                && scaled_complementarity.abs() <= self.feasibility_target;
            cooperation.completed(1).await?;
        }
        for (&constraint, &multiplier) in equalities.iter().zip(&self.equality_multipliers) {
            eligible &= constraint.is_finite()
                && constraint.abs() <= self.feasibility_target
                && (multiplier + self.penalty * constraint).is_finite();
            cooperation.completed(1).await?;
        }
        if eligible {
            // The first pass validates every new multiplier; publication does
            // not leave a partially updated merit if arithmetic is undefined.
            for (&constraint, multiplier) in inequalities.iter().zip(&mut self.multipliers) {
                let shift = *multiplier / self.penalty;
                *multiplier *= shift / (shift - constraint);
                cooperation.completed(1).await?;
            }
            for (&constraint, multiplier) in equalities.iter().zip(&mut self.equality_multipliers) {
                *multiplier += self.penalty * constraint;
                cooperation.completed(1).await?;
            }
            self.feasibility_target *= self.penalty.powf(-0.25);
            Ok(Update::Multipliers)
        } else {
            let penalty = self.penalty * self.factor;
            if !penalty.is_finite() || self.multipliers.iter().any(|&value| value / penalty <= 0.0)
            {
                return Ok(Update::Unrepresentable);
            }
            self.penalty = penalty;
            self.feasibility_target = penalty.powf(-0.75);
            Ok(Update::Penalty)
        }
    }

    pub fn complementarity(&self, inequalities: &[f64]) -> Option<f64> {
        if inequalities.len() != self.multipliers.len() {
            return None;
        }
        let mut maximum = 0.0_f64;
        for (&constraint, &multiplier) in inequalities.iter().zip(&self.multipliers) {
            let shift = multiplier / self.penalty;
            let distance = shift - constraint;
            if !constraint.is_finite() || distance <= 0.0 {
                return None;
            }
            let product = constraint * (multiplier * (shift / distance));
            if !product.is_finite() {
                return None;
            }
            maximum = maximum.max(product.abs());
        }
        Some(maximum)
    }
}

/// Deb feasibility penalty for one measured individual. The caller computes
/// worst_feasible only from actual feasible finite measurements of this same
/// population. With no feasible member, ranking is solely violation; the raw
/// fitness is still retained separately for public output and future reuse.
pub(super) fn penalty_score(
    fitness: Option<f64>,
    violation_sum: f64,
    feasible: bool,
    worst_feasible: Option<f64>,
) -> Option<f64> {
    if feasible {
        return fitness.filter(|value| value.is_finite());
    }
    if !violation_sum.is_finite() || violation_sum < 0.0 {
        return None;
    }
    let score = worst_feasible.unwrap_or(0.0) + violation_sum;
    score.is_finite().then_some(score)
}
