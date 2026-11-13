//! MaxFunctionEvaluations bounds evaluated points, not the sum of objective
//! and nonlinear-constraint callbacks at the same requested point.

#[derive(Clone, Copy, Debug, Default)]
pub(in crate::builtins::math::optim::fmincon) struct Counts {
    pub points: usize,
    pub objective_calls: usize,
    pub constraint_calls: usize,
}

pub(in crate::builtins::math::optim::fmincon) struct Budget {
    limit: usize,
    counts: Counts,
}

impl Budget {
    pub fn new(limit: usize) -> Self {
        Self {
            limit,
            counts: Counts::default(),
        }
    }
    pub fn counts(&self) -> Counts {
        self.counts
    }
    pub fn remaining(&self) -> usize {
        self.limit.saturating_sub(self.counts.points)
    }
    pub fn permits_sweep_and_trial(&self, perturbations: usize) -> bool {
        perturbations
            .checked_add(1)
            .is_some_and(|required| required <= self.remaining())
    }
    /// Caller selects at least one real callback and owns this point until
    /// all its requested outputs/derivatives have been consumed or failed.
    pub fn begin_point(&mut self) -> bool {
        if self.counts.points >= self.limit {
            return false;
        }
        self.counts.points += 1;
        true
    }
    pub fn completed_objective(&mut self) {
        self.counts.objective_calls += 1;
    }
    pub fn completed_constraints(&mut self) {
        self.counts.constraint_calls += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn one_point_can_call_both_functions_without_doubling_public_budget() {
        let mut budget = Budget::new(2);
        assert!(budget.begin_point());
        budget.completed_objective();
        budget.completed_constraints();
        assert_eq!(budget.counts().points, 1);
        assert_eq!(budget.counts().objective_calls, 1);
        assert_eq!(budget.counts().constraint_calls, 1);
        assert!(!budget.permits_sweep_and_trial(1));
        assert!(budget.begin_point());
        budget.completed_constraints();
        assert!(!budget.begin_point());
        assert_eq!(budget.counts().points, 2);
        assert_eq!(budget.counts().constraint_calls, 2);
        assert!(!Budget::new(0).begin_point());
        assert!(!budget.permits_sweep_and_trial(usize::MAX));
    }
}
