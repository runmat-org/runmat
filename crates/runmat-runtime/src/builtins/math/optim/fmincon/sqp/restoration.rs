//! Elastic SQP relaxes only constraint linearizations, never actual bounds.
//! A local elastic result cannot certify global nonlinear infeasibility.
use super::super::{
    model::{Linearization, Role},
    workspace,
};
use super::subproblem::{self, Constraints, Result, Step};
use crate::builtins::math::optim::common::ScalarCooperation;
use crate::{build_runtime_error, BuiltinResult};
use clarabel::{algebra::CscMatrix, solver::SupportedConeT};
use runmat_builtins::BuiltinCatalogIdentity;

pub(super) async fn solve(
    hessian: &CscMatrix<f64>,
    gradient: &[f64],
    linearization: &Linearization,
    penalty: f64,
    tolerance: f64,
    owner: BuiltinCatalogIdentity,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<Step>> {
    let variables = gradient.len();
    let rows = linearization
        .b
        .len()
        .checked_add(linearization.equalities)
        .and_then(|n| n.checked_add(1))
        .ok_or_else(|| {
            build_runtime_error("elastic constraint extent overflow")
                .with_builtin(owner.name)
                .build()
        })?;
    let augmented_variables = variables.checked_add(1).ok_or_else(|| {
        build_runtime_error("elastic variable extent overflow")
            .with_builtin(owner.name)
            .build()
    })?;
    let mut positive = workspace::capacity(linearization.b.len(), owner)?;
    let mut negative = workspace::capacity(linearization.equalities, owner)?;
    let mut rhs = workspace::capacity(rows, owner)?;
    for (row, &value) in linearization.b.iter().enumerate() {
        positive.push(rhs.len());
        rhs.push(value);
        if row < linearization.equalities {
            negative.push(rhs.len());
            rhs.push(-value);
        }
        cooperation.completed(1).await?;
    }
    rhs.push(0.0); // -t <= 0
    let mut offsets = workspace::capacity(
        augmented_variables.checked_add(1).ok_or_else(|| {
            build_runtime_error("elastic column extent overflow")
                .with_builtin(owner.name)
                .build()
        })?,
        owner,
    )?;
    let mut indices = Vec::new();
    let mut entries = Vec::new();
    offsets.push(0);
    for column in 0..variables {
        for source in linearization.a.colptr[column]..linearization.a.colptr[column + 1] {
            let row = linearization.a.rowval[source];
            let coefficient = linearization.a.nzval[source];
            workspace::push(&mut indices, positive[row], owner)?;
            workspace::push(&mut entries, coefficient, owner)?;
            if row < linearization.equalities {
                workspace::push(&mut indices, negative[row], owner)?;
                workspace::push(&mut entries, -coefficient, owner)?;
            }
            cooperation.completed(1).await?;
        }
        offsets.push(indices.len());
        cooperation.completed(1).await?;
    }
    for (row, role) in linearization.roles.iter().enumerate() {
        // The outer iteration stays bound-feasible; bounds are never elastic.
        if !matches!(role, Role::Lower(_) | Role::Upper(_)) {
            workspace::push(&mut indices, positive[row], owner)?;
            workspace::push(&mut entries, -1.0, owner)?;
            if row < linearization.equalities {
                workspace::push(&mut indices, negative[row], owner)?;
                workspace::push(&mut entries, -1.0, owner)?;
            }
        }
        cooperation.completed(1).await?;
    }
    workspace::push(&mut indices, rows - 1, owner)?;
    workspace::push(&mut entries, -1.0, owner)?;
    offsets.push(indices.len());
    let a = CscMatrix::new(rows, augmented_variables, offsets, indices, entries);
    // Preserve the original P numerical values. The extra elastic variable has
    // a linear penalty and an exactly zero quadratic column.
    let mut poffsets = workspace::capacity(augmented_variables + 1, owner)?;
    for &offset in &hessian.colptr {
        poffsets.push(offset);
        cooperation.completed(1).await?;
    }
    poffsets.push(hessian.nzval.len());
    let mut prows = workspace::capacity(hessian.rowval.len(), owner)?;
    let mut pvalues = workspace::capacity(hessian.nzval.len(), owner)?;
    for (&row, &value) in hessian.rowval.iter().zip(&hessian.nzval) {
        prows.push(row);
        pvalues.push(value);
        cooperation.completed(1).await?;
    }
    let p = CscMatrix::new(
        augmented_variables,
        augmented_variables,
        poffsets,
        prows,
        pvalues,
    );
    let mut f = workspace::capacity(augmented_variables, owner)?;
    for _ in gradient {
        // Do not let the original objective oppose feasibility restoration.
        f.push(0.0);
        cooperation.completed(1).await?;
    }
    f.push(penalty);
    let cones = [SupportedConeT::NonnegativeConeT(rows)];
    let constraints = Constraints {
        a: &a,
        b: &rhs,
        equalities: 0,
        cones: &cones,
    };
    let Result::Step(mut step) =
        subproblem::solve(&p, &f, &constraints, tolerance, owner, cooperation).await?
    else {
        return Ok(None);
    };
    if subproblem::clip_bounds(linearization, &mut step.direction, cooperation).await?
        && !subproblem::checked(
            &p,
            &f,
            &constraints,
            &step.direction,
            &step.multipliers,
            tolerance,
            owner,
            cooperation,
        )
        .await?
    {
        return Ok(None);
    }
    let slack = step.direction.pop().expect("elastic variable present");
    if !slack.is_finite() || slack < -tolerance {
        return Ok(None);
    }
    let mut multipliers = workspace::capacity(linearization.b.len(), owner)?;
    for row in 0..linearization.b.len() {
        let value = step.multipliers[positive[row]]
            - if row < linearization.equalities {
                step.multipliers[negative[row]]
            } else {
                0.0
            };
        multipliers.push(value);
        cooperation.completed(1).await?;
    }
    Ok(Some(Step {
        direction: step.direction,
        multipliers,
    }))
}
