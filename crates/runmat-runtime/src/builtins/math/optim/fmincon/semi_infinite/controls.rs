//! Actual numerical consumers of documented semi-infinite controls.
//! Canonical option-name/class decoding remains with the shared catalog owner.
use super::super::{
    active_set,
    model::{Linearization, Role},
};
use super::bernstein::invalid;
use crate::builtins::math::optim::common::{
    derivative_scaling::DerivativeScaling, ScalarCooperation,
};
use crate::BuiltinResult;
#[derive(Clone, Copy)]
pub(in crate::builtins::math::optim::fmincon) struct Controls {
    pub mesh_tolerance: f64,
    pub check_values: bool,
    pub diff_min_change: f64,
    pub diff_max_change: f64,
    pub inner_iterations: Option<usize>,
    pub inner_tolerance: f64,
    pub relative_line_bound: Option<f64>,
    pub relative_line_duration: usize,
    pub diagnostics: bool,
    pub detailed_display: bool,
}
impl Default for Controls {
    fn default() -> Self {
        Self {
            mesh_tolerance: 1e-6,
            check_values: false,
            diff_min_change: 0.0,
            diff_max_change: f64::INFINITY,
            inner_iterations: None,
            inner_tolerance: 1e-6,
            relative_line_bound: None,
            relative_line_duration: 1,
            diagnostics: false,
            detailed_display: false,
        }
    }
}
impl Controls {
    pub fn validate(&self) -> BuiltinResult<()> {
        if !self.mesh_tolerance.is_finite()
            || self.mesh_tolerance < 0.0
            || !self.inner_tolerance.is_finite()
            || self.inner_tolerance <= 0.0
        {
            return Err(invalid(
                "constraint tolerance must be nonnegative and inner SQP tolerance must be positive",
            ));
        }
        if self.inner_iterations == Some(0) || self.relative_line_duration == 0 {
            return Err(invalid(
                "inner SQP and relative line-search duration counts must be positive",
            ));
        }
        if self
            .relative_line_bound
            .is_some_and(|v| !v.is_finite() || v < 0.0)
        {
            return Err(invalid(
                "relative line-search bound must be finite and nonnegative",
            ));
        }
        self.scaling(DerivativeScaling::default())?;
        Ok(())
    }
    pub fn scaling(&self, scaling: DerivativeScaling) -> BuiltinResult<DerivativeScaling> {
        scaling.with_absolute_steps(self.diff_min_change, self.diff_max_change)
    }
    pub async fn pivots(
        &self,
        variables: usize,
        model: &Linearization,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<active_set::Limits> {
        let iterations = if let Some(value) = self.inner_iterations {
            value
        } else {
            let mut inequalities = 0usize;
            for role in &model.roles {
                if matches!(
                    role,
                    Role::InequalityLinear(_)
                        | Role::InequalityNonlinear(_)
                        | Role::Lower(_)
                        | Role::Upper(_)
                ) {
                    inequalities = inequalities
                        .checked_add(1)
                        .ok_or_else(|| invalid("inner SQP row count overflow"))?;
                }
                cooperation.completed(1).await?;
            }
            variables
                .max(inequalities)
                .checked_mul(10)
                .ok_or_else(|| invalid("default inner SQP count overflow"))?
        };
        Ok(active_set::Limits {
            tolerance: self.inner_tolerance,
            iterations: Some(iterations),
        })
    }
    pub fn line_active(&self, iteration: usize) -> bool {
        self.relative_line_bound.is_some() && iteration < self.relative_line_duration
    }
    pub async fn line_cap(
        &self,
        x: &[f64],
        direction: &[f64],
        iteration: usize,
        scaling: &DerivativeScaling,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<f64> {
        if !self.line_active(iteration) {
            return Ok(1.0);
        }
        if x.len() != direction.len() {
            return Err(invalid(
                "line-search direction has the wrong variable extent",
            ));
        }
        let relative = self.relative_line_bound.unwrap();
        let mut alpha = 1.0f64;
        for (index, (&x, &direction)) in x.iter().zip(direction).enumerate() {
            let bound = relative * x.abs().max(scaling.typical_value(index));
            // An overflowing positive bound cannot restrict alpha<=1 for
            // a finite direction. Underflow to zero permits no represented
            // coordinate displacement; it never fabricates an objective.
            if bound.is_finite() && direction != 0.0 {
                alpha = alpha.min(bound / direction.abs());
            }
            cooperation.completed(1).await?;
        }
        Ok(alpha)
    }
    pub async fn line_feasible(
        &self,
        x: &[f64],
        trial: &[f64],
        iteration: usize,
        scaling: &DerivativeScaling,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        if !self.line_active(iteration) {
            return Ok(true);
        }
        if x.len() != trial.len() {
            return Err(invalid("line-search trial has the wrong variable extent"));
        }
        let relative = self.relative_line_bound.unwrap();
        for (index, (&x, &trial)) in x.iter().zip(trial).enumerate() {
            let bound = relative * x.abs().max(scaling.typical_value(index));
            if (trial - x).abs() > bound {
                return Ok(false);
            }
            cooperation.completed(1).await?;
        }
        Ok(true)
    }
}
