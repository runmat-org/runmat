//! A cooperative convex-QP candidate is not accepted merely because its kernel
//! converged. Recheck the original linearization, gradient and model Hessian.
use super::{
    model::{Linearization, Role},
    workspace,
};
use crate::builtins::math::optim::common::{conic, ScalarCooperation};
use crate::BuiltinResult;
use clarabel::{
    algebra::CscMatrix,
    solver::{DefaultInfo, SupportedConeT},
};
use runmat_builtins::BuiltinCatalogIdentity;
pub(super) const LOCAL_ITERATIONS: u32 = 200;

pub(super) struct Step {
    pub direction: Vec<f64>,
    pub multipliers: Vec<f64>,
}
pub(super) struct Constraints<'a> {
    pub a: &'a CscMatrix<f64>,
    pub b: &'a [f64],
    pub equalities: usize,
    pub cones: &'a [SupportedConeT<f64>],
}
impl<'a> From<&'a Linearization> for Constraints<'a> {
    fn from(value: &'a Linearization) -> Self {
        Self {
            a: &value.a,
            b: &value.b,
            equalities: value.equalities,
            cones: &value.cones,
        }
    }
}
pub(super) enum Result {
    Step(Step),
    InfeasibleLinearization,
    NumericalDifficulty,
}
pub(super) async fn clip_bounds(
    linearization: &Linearization,
    direction: &mut [f64],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<bool> {
    let mut adjusted = false;
    for (&role, &rhs) in linearization.roles.iter().zip(&linearization.b) {
        match role {
            Role::Lower(variable) => {
                let value = direction[variable].max(-rhs);
                adjusted |= value != direction[variable];
                direction[variable] = value;
            }
            Role::Upper(variable) => {
                let value = direction[variable].min(rhs);
                adjusted |= value != direction[variable];
                direction[variable] = value;
            }
            _ => {}
        }
        cooperation.completed(1).await?;
    }
    Ok(adjusted)
}
pub(super) async fn regular(
    hessian: &CscMatrix<f64>,
    gradient: &[f64],
    linearization: &Linearization,
    tolerance: f64,
    owner: BuiltinCatalogIdentity,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Result> {
    let constraints = Constraints::from(linearization);
    let result = solve(
        hessian,
        gradient,
        &constraints,
        tolerance,
        owner,
        cooperation,
    )
    .await?;
    match result {
        Result::Step(mut step) => {
            if clip_bounds(linearization, &mut step.direction, cooperation).await?
                && !checked(
                    hessian,
                    gradient,
                    &constraints,
                    &step.direction,
                    &step.multipliers,
                    tolerance,
                    owner,
                    cooperation,
                )
                .await?
            {
                Ok(Result::NumericalDifficulty)
            } else {
                Ok(Result::Step(step))
            }
        }
        other => Ok(other),
    }
}
struct Iterations {
    used: u32,
    accuracy: f64,
}
impl conic::solve::IterationControl for Iterations {
    fn accuracy(&self) -> f64 {
        self.accuracy
    }
    fn remaining(&self) -> u32 {
        LOCAL_ITERATIONS.saturating_sub(self.used)
    }
    fn record(
        &mut self,
        previous: &mut u32,
        info: &DefaultInfo<f64>,
        _scale: f64,
        _runtime: Option<&crate::context::RuntimeContext>,
    ) -> BuiltinResult<()> {
        self.used = self
            .used
            .saturating_add(info.iterations.saturating_sub(*previous));
        *previous = info.iterations;
        Ok(())
    }
}

pub(super) async fn solve(
    hessian: &CscMatrix<f64>,
    gradient: &[f64],
    linearization: &Constraints<'_>,
    tolerance: f64,
    owner: BuiltinCatalogIdentity,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Result> {
    let model = conic::Model {
        p: Some(hessian),
        a: linearization.a,
        b: linearization.b,
        f: gradient,
        cones: linearization.cones,
        identity: owner,
        objective_scale: 1.0,
    };
    let mut seed = workspace::capacity(gradient.len(), owner)?;
    for _ in gradient {
        seed.push(0.0);
        cooperation.completed(1).await?;
    }
    let mut iterations = Iterations {
        used: 0,
        accuracy: tolerance,
    };
    match conic::solve::candidate(&model, &seed, &mut iterations, cooperation).await? {
        conic::solve::Candidate::Converged { x, z } => {
            if checked(
                hessian,
                gradient,
                linearization,
                &x,
                &z,
                tolerance,
                owner,
                cooperation,
            )
            .await?
            {
                Ok(Result::Step(Step {
                    direction: x,
                    multipliers: z,
                }))
            } else {
                Ok(Result::NumericalDifficulty)
            }
        }
        // The exact certificate applies to this local linearization only.
        conic::solve::Candidate::CertifiedInfeasible => Ok(Result::InfeasibleLinearization),
        _ => Ok(Result::NumericalDifficulty),
    }
}

pub(super) async fn symmetric_product(
    matrix: &CscMatrix<f64>,
    x: &[f64],
    owner: BuiltinCatalogIdentity,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<Vec<f64>>> {
    let mut result = workspace::capacity(x.len(), owner)?;
    for _ in x {
        result.push(0.0);
        cooperation.completed(1).await?;
    }
    for column in 0..matrix.n {
        for source in matrix.colptr[column]..matrix.colptr[column + 1] {
            let row = matrix.rowval[source];
            let coefficient = matrix.nzval[source];
            result[row] += coefficient * x[column];
            if row != column {
                result[column] += coefficient * x[row];
            }
            cooperation.completed(1).await?;
        }
        cooperation.completed(1).await?;
    }
    if result.iter().any(|value| !value.is_finite()) {
        return Ok(None);
    }
    Ok(Some(result))
}

pub(super) async fn checked(
    hessian: &CscMatrix<f64>,
    gradient: &[f64],
    linearization: &Constraints<'_>,
    step: &[f64],
    z: &[f64],
    tolerance: f64,
    owner: BuiltinCatalogIdentity,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<bool> {
    if step.len() != gradient.len()
        || z.len() != linearization.b.len()
        || step.iter().chain(z).any(|value| !value.is_finite())
    {
        return Ok(false);
    }
    let Some(hp) = symmetric_product(hessian, step, owner, cooperation).await? else {
        return Ok(false);
    };
    let mut activity = workspace::capacity(z.len(), owner)?;
    for _ in z {
        activity.push(0.0);
        cooperation.completed(1).await?;
    }
    for column in 0..step.len() {
        let mut residual = gradient[column] + hp[column];
        let mut scale = gradient[column].abs() + hp[column].abs();
        for source in linearization.a.colptr[column]..linearization.a.colptr[column + 1] {
            let row = linearization.a.rowval[source];
            let coefficient = linearization.a.nzval[source];
            activity[row] += coefficient * step[column];
            let term = coefficient * z[row];
            residual += term;
            scale += term.abs();
            cooperation.completed(1).await?;
        }
        if !residual.is_finite() || !scale.is_finite() || residual.abs() / (1.0 + scale) > tolerance
        {
            return Ok(false);
        }
        cooperation.completed(1).await?;
    }
    for row in 0..z.len() {
        let activity = activity[row];
        let rhs = linearization.b[row];
        if !activity.is_finite() {
            return Ok(false);
        }
        let residual = activity - rhs;
        let primal = if row < linearization.equalities {
            residual.abs()
        } else {
            residual.max(0.0)
        };
        if !primal.is_finite() || primal > tolerance {
            return Ok(false);
        }
        if row >= linearization.equalities {
            let product = z[row] * (rhs - activity);
            let scale = 1.0 + (z[row] * rhs).abs() + (z[row] * activity).abs();
            if z[row] < -tolerance
                || !product.is_finite()
                || !scale.is_finite()
                || product.abs() / scale > tolerance
            {
                return Ok(false);
            }
        }
        cooperation.completed(1).await?;
    }
    Ok(true)
}
