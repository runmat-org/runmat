//! Nonlinear success authority: original constraints and the actual Lagrangian
//! gradient. A convex subproblem's status cannot stand in for this check.
use super::{
    model::{Linearization, Role},
    workspace,
};
use crate::builtins::math::optim::common::ScalarCooperation;
use crate::BuiltinResult;
use runmat_builtins::BuiltinCatalogIdentity;

pub(super) struct Checked {
    pub violation: f64,
    pub optimality: f64,
    pub valid: bool,
}
pub(super) async fn lagrangian(
    gradient: &[f64],
    constraints: &Linearization,
    multipliers: &[f64],
    owner: BuiltinCatalogIdentity,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<Vec<f64>>> {
    if multipliers.len() != constraints.b.len()
        || gradient.len() != constraints.a.n
        || multipliers
            .iter()
            .chain(gradient)
            .any(|value| !value.is_finite())
    {
        return Ok(None);
    }
    let mut result = workspace::capacity(gradient.len(), owner)?;
    for (column, &gradient) in gradient.iter().enumerate() {
        let mut sum = gradient;
        let mut compensation = 0.0;
        for source in constraints.a.colptr[column]..constraints.a.colptr[column + 1] {
            let term = constraints.a.nzval[source] * multipliers[constraints.a.rowval[source]];
            let corrected = term - compensation;
            let next = sum + corrected;
            compensation = (next - sum) - corrected;
            sum = next;
            cooperation.completed(1).await?;
        }
        if !sum.is_finite() {
            return Ok(None);
        }
        result.push(sum);
        cooperation.completed(1).await?;
    }
    Ok(Some(result))
}
pub(super) async fn check(
    gradient: &[f64],
    constraints: &Linearization,
    multipliers: &[f64],
    feasibility: f64,
    optimality: f64,
    owner: BuiltinCatalogIdentity,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<Checked>> {
    let Some(gradient) = lagrangian(gradient, constraints, multipliers, owner, cooperation).await?
    else {
        return Ok(None);
    };
    let (violation, _) = constraints.violation(cooperation).await?;
    let mut residual = 0.0f64;
    for &value in &gradient {
        residual = residual.max(value.abs());
        cooperation.completed(1).await?;
    }
    for (row, &multiplier) in multipliers.iter().enumerate().skip(constraints.equalities) {
        let product = multiplier * constraints.b[row];
        if !product.is_finite() {
            return Ok(None);
        }
        residual = residual.max((-multiplier).max(0.0)).max(product.abs());
        cooperation.completed(1).await?;
    }
    Ok(Some(Checked {
        violation,
        optimality: residual,
        valid: violation <= feasibility && residual <= optimality,
    }))
}

/// Six fixed public multiplier roles, indexed by original supplied rows or
/// variables. Inactive/infinite bounds retain their actual zero multiplier.
pub(in crate::builtins::math::optim) struct Multipliers {
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
    pub ineqlin: Vec<f64>,
    pub eqlin: Vec<f64>,
    pub ineqnonlin: Vec<f64>,
    pub eqnonlin: Vec<f64>,
}
pub(super) async fn split(
    constraints: &Linearization,
    multipliers: &[f64],
    variables: usize,
    linear_inequalities: usize,
    linear_equalities: usize,
    nonlinear_inequalities: usize,
    nonlinear_equalities: usize,
    owner: BuiltinCatalogIdentity,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Multipliers> {
    async fn zeros(
        n: usize,
        owner: BuiltinCatalogIdentity,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Vec<f64>> {
        let mut values = workspace::capacity(n, owner)?;
        for _ in 0..n {
            values.push(0.0);
            cooperation.completed(1).await?;
        }
        Ok(values)
    }
    let mut result = Multipliers {
        lower: zeros(variables, owner, cooperation).await?,
        upper: zeros(variables, owner, cooperation).await?,
        ineqlin: zeros(linear_inequalities, owner, cooperation).await?,
        eqlin: zeros(linear_equalities, owner, cooperation).await?,
        ineqnonlin: zeros(nonlinear_inequalities, owner, cooperation).await?,
        eqnonlin: zeros(nonlinear_equalities, owner, cooperation).await?,
    };
    for (&role, &value) in constraints.roles.iter().zip(multipliers) {
        match role {
            Role::Lower(row) => result.lower[row] = value,
            Role::Upper(row) => result.upper[row] = value,
            Role::EqualityLinear(row) => result.eqlin[row] = value,
            Role::InequalityLinear(row) => result.ineqlin[row] = value,
            Role::EqualityNonlinear(row) => result.eqnonlin[row] = value,
            Role::InequalityNonlinear(row) => result.ineqnonlin[row] = value,
        }
        cooperation.completed(1).await?;
    }
    Ok(result)
}
