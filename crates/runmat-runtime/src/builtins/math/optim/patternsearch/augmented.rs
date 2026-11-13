//! One-partition generating-set augmented Lagrangian (SAND2006-5315,
//! Algorithms 3.1/4.1 and section 7). For original c(x)<=0, nonnegative
//! slacks in c+s=0 are minimized analytically, not fabricated as callbacks.
use super::{buffer, evaluation::Point, invalid};
use crate::builtins::math::optim::common::ScalarCooperation;
use crate::BuiltinResult;

#[derive(Clone, Copy)]
pub(super) struct Parameters {
    pub initial_penalty: f64,
    pub penalty_factor: f64,
    pub mesh_tolerance: f64,
    pub constraint_tolerance: f64,
}
pub(super) struct Augmented {
    multipliers: Vec<f64>,
    inequalities: usize,
    equalities: usize,
    penalty: f64,
    omega: f64,
    eta: f64,
    inner_tolerance: f64,
    parameters: Parameters,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Update {
    Converged,
    Multipliers,
    Penalty,
}

impl Augmented {
    pub async fn new(
        point: &Point,
        parameters: Parameters,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        if !parameters.initial_penalty.is_finite()
            || parameters.initial_penalty < 1.0
            || !parameters.penalty_factor.is_finite()
            || parameters.penalty_factor <= 1.0
            || !parameters.mesh_tolerance.is_finite()
            || parameters.mesh_tolerance < 0.0
            || !parameters.constraint_tolerance.is_finite()
            || parameters.constraint_tolerance <= 0.0
        {
            return Err(invalid("invalid augmented-Lagrangian parameters"));
        }
        let count = point
            .inequality
            .len()
            .checked_add(point.equality.len())
            .ok_or_else(|| invalid("nonlinear constraint count overflow"))?;
        let mut multipliers = buffer(count)?;
        for _ in 0..count {
            multipliers.push(0.0);
            cooperation.completed(1).await?;
        }
        let omega = 1.0 / parameters.initial_penalty;
        let inner_tolerance = omega / (1.0 + parameters.initial_penalty);
        Ok(Self {
            multipliers,
            inequalities: point.inequality.len(),
            equalities: point.equality.len(),
            penalty: parameters.initial_penalty,
            omega,
            eta: omega.powf(0.1),
            inner_tolerance,
            parameters,
        })
    }

    pub fn inner_tolerance(&self) -> f64 {
        self.inner_tolerance
    }

    pub fn reconfigure(&mut self, parameters: Parameters) -> BuiltinResult<()> {
        if !parameters.penalty_factor.is_finite()
            || parameters.penalty_factor <= 1.0
            || !parameters.mesh_tolerance.is_finite()
            || parameters.mesh_tolerance < 0.0
            || !parameters.constraint_tolerance.is_finite()
            || parameters.constraint_tolerance <= 0.0
        {
            return Err(invalid("invalid augmented-Lagrangian transition controls"));
        }
        self.parameters = parameters;
        Ok(())
    }

    fn check_extents(&self, point: &Point) -> BuiltinResult<()> {
        if point.inequality.len() != self.inequalities || point.equality.len() != self.equalities {
            return Err(invalid("nonlinear constraint extent changed"));
        }
        Ok(())
    }

    pub async fn merit(
        &self,
        point: &Point,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<f64>> {
        self.merit_values(point.f, &point.inequality, &point.equality, cooperation)
            .await
    }

    /// Evaluate raw retained measurements against THIS multiplier generation.
    pub async fn merit_values(
        &self,
        f: f64,
        inequality: &[f64],
        equality: &[f64],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<f64>> {
        if inequality.len() != self.inequalities || equality.len() != self.equalities {
            return Err(invalid("nonlinear constraint extent changed"));
        }
        let mut result = f;
        let scale = (0.5 * self.penalty).sqrt();
        for (index, &constraint) in inequality.iter().chain(equality).enumerate() {
            let shifted = constraint + self.multipliers[index] / self.penalty;
            let residual = if index < self.inequalities {
                shifted.max(0.0)
            } else {
                shifted
            };
            let scaled = scale * residual;
            // Complete-square constants depend only on the outer state and
            // are omitted from ALL candidates, including the retained base.
            // This avoids subtracting two huge multiplier-squared constants.
            result += scaled * scaled;
            if !shifted.is_finite() || !result.is_finite() {
                return Ok(None);
            }
            cooperation.completed(1).await?;
        }
        Ok(Some(result))
    }

    /// Only the controller's completed unsuccessful inner poll may call this.
    /// A budget stop or numerical/representability failure never updates the
    /// multipliers or certifies a stationary subproblem.
    pub async fn completed_inner(
        &mut self,
        point: &Point,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Update>> {
        self.check_extents(point)?;
        let mut residual_norm = 0.0f64;
        let mut violation_norm = 0.0f64;
        let mut complementarity = 0.0f64;
        for (index, &constraint) in point.inequality.iter().chain(&point.equality).enumerate() {
            let inequality = index < self.inequalities;
            // Exact slack minimizer s=max(0,-c-lambda/rho) gives this
            // residual of the lifted equality, with s>=0 by construction.
            let residual = if inequality {
                constraint.max(-self.multipliers[index] / self.penalty)
            } else {
                constraint
            };
            residual_norm = residual_norm.hypot(residual);
            violation_norm = violation_norm.hypot(if inequality {
                constraint.max(0.0)
            } else {
                constraint
            });
            if inequality {
                complementarity = complementarity.hypot(self.multipliers[index] * constraint);
            }
            cooperation.completed(1).await?;
        }
        if !residual_norm.is_finite() || !violation_norm.is_finite() || !complementarity.is_finite()
        {
            return Ok(None);
        }
        if self.inner_tolerance <= self.parameters.mesh_tolerance
            && violation_norm <= self.parameters.constraint_tolerance
            && complementarity <= self.parameters.constraint_tolerance.sqrt()
        {
            return Ok(Some(Update::Converged));
        }
        let update = if residual_norm <= self.eta {
            for (index, &constraint) in point.inequality.iter().chain(&point.equality).enumerate() {
                let value = self.multipliers[index] + self.penalty * constraint;
                if !value.is_finite() {
                    return Ok(None);
                }
                self.multipliers[index] = if index < self.inequalities {
                    value.max(0.0)
                } else {
                    value
                };
                cooperation.completed(1).await?;
            }
            // InitialPenalty=1 is a documented edge. Its mu=1 cannot tighten
            // tolerances multiplicatively; use the same selected reduction
            // factor in that case, without silently changing the penalty.
            let reduction = if self.penalty == 1.0 {
                1.0 / self.parameters.penalty_factor
            } else {
                1.0 / self.penalty
            };
            self.omega *= reduction;
            self.eta *= reduction.powf(0.9);
            Update::Multipliers
        } else {
            self.penalty *= self.parameters.penalty_factor;
            self.omega = 1.0 / self.penalty;
            self.eta = self.omega.powf(0.1);
            Update::Penalty
        };
        let mut multiplier_norm = 0.0f64;
        for &value in &self.multipliers {
            multiplier_norm = multiplier_norm.hypot(value);
            cooperation.completed(1).await?;
        }
        // theta=1+||lambda||+rho is the paper's admissible scaling with one
        // constraint partition and beta_max=1. No positive tolerance floor
        // silently relaxes a caller's requested precision.
        let theta = 1.0 + multiplier_norm + self.penalty;
        self.inner_tolerance = self.omega / theta;
        if !self.penalty.is_finite()
            || !self.inner_tolerance.is_finite()
            || self.inner_tolerance <= 0.0
        {
            return Ok(None);
        }
        Ok(Some(update))
    }
}
