//! One checked controller-work envelope used by execution and resource planning.
use super::NonlinearSolveOptions;
use crate::FeaRunError;

#[derive(Clone, Copy)]
pub(crate) struct NonlinearWorkLimits {
    pub accepted: usize,
    pub rejected: usize,
    pub iterations: usize,
    pub linear_solves: usize,
    pub backtracks: usize,
    pub tangent_evaluations: usize,
    pub controller_evaluations: usize,
}

impl NonlinearWorkLimits {
    pub fn derive(options: &NonlinearSolveOptions) -> Result<Self, FeaRunError> {
        let accepted = options.max_accepted_increments;
        if accepted == 0 || options.increment_count == 0 || options.max_newton_iters == 0 {
            return Err(FeaRunError::InvalidModel(
                "J2 controller inventories must be positive".into(),
            ));
        }
        let trials = multiply(accepted, add(options.max_load_cutbacks, 1)?)?;
        let iterations = multiply(trials, options.max_newton_iters)?;
        let backtracks = if options.line_search {
            options.max_line_search_backtracks
        } else {
            0
        };
        Ok(Self {
            accepted,
            rejected: multiply(accepted, options.max_load_cutbacks)?,
            iterations,
            linear_solves: multiply(trials, add(options.max_newton_iters, 1)?)?,
            backtracks: multiply(iterations, backtracks)?,
            tangent_evaluations: multiply(
                trials,
                add(multiply(options.max_newton_iters, add(backtracks, 1)?)?, 2)?,
            )?,
            // Every successful block advances at least once. Fresh and cold each require two
            // prefix basis evaluations; each adds one per advance and one terminal
            // validation. Thus at most four per accepted step.
            controller_evaluations: multiply(accepted, 4)?,
        })
    }

    /// At most two block-prefix/end proofs per accepted step and two accepted
    /// state proofs per advance. Uncoupled limits remain unchanged.
    pub(crate) fn derive_contact(options: &NonlinearSolveOptions) -> Result<Self, FeaRunError> {
        let mut limits = Self::derive(options)?;
        limits.linear_solves = add(limits.linear_solves, multiply(limits.accepted, 4)?)?;
        Ok(limits)
    }

    pub fn total_evaluations(self) -> Result<usize, FeaRunError> {
        add(self.tangent_evaluations, self.controller_evaluations)
    }
}

pub(super) fn add(a: usize, b: usize) -> Result<usize, FeaRunError> {
    a.checked_add(b).ok_or_else(overflow)
}

fn multiply(a: usize, b: usize) -> Result<usize, FeaRunError> {
    a.checked_mul(b).ok_or_else(overflow)
}

fn overflow() -> FeaRunError {
    FeaRunError::InvalidModel("J2 controller work inventory overflow".into())
}
