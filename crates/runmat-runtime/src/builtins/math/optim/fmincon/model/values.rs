//! Original nonlinear/linear/bound values before any Jacobian work. An
//! accepted trial moves this generation into sparse Jacobian assembly.
use super::{activity, invalid, workspace, Geometry, GeometryView, Point, Role};
use crate::builtins::math::optim::common::ScalarCooperation;
use crate::BuiltinResult;
use runmat_builtins::BuiltinCatalogIdentity;
pub(in crate::builtins::math::optim::fmincon) struct ConstraintValues {
    pub rhs: Vec<f64>,
    pub roles: Vec<Role>,
    pub equalities: usize,
    pub(super) lower_rows: Vec<Option<usize>>,
    pub(super) upper_rows: Vec<Option<usize>>,
}
impl GeometryView<'_> {
    /// None denotes only nonfinite original numerical activity/residuals.
    /// Allocation, cooperation and structural failures stay outer errors.
    pub(in crate::builtins::math::optim::fmincon) async fn constraint_values(
        &self,
        x: &[f64],
        point: &Point,
        owner: BuiltinCatalogIdentity,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<ConstraintValues>> {
        let neq = self.equality.rhs.len();
        let nleq = point.equality.len();
        let nineq = self.inequality.rhs.len();
        let nlneq = point.inequality.len();
        let eq = neq
            .checked_add(nleq)
            .ok_or_else(|| invalid(owner, "constraint extent overflow"))?;
        let fixed = eq
            .checked_add(nineq)
            .and_then(|n| n.checked_add(nlneq))
            .ok_or_else(|| invalid(owner, "constraint extent overflow"))?;
        let mut roles = workspace::capacity(fixed, owner)?;
        let mut b = workspace::capacity(fixed, owner)?;
        let Some(eqactivity) = activity(&self.equality.matrix, x, owner, cooperation).await? else {
            return Ok(None);
        };
        let Some(ineqactivity) = activity(&self.inequality.matrix, x, owner, cooperation).await?
        else {
            return Ok(None);
        };
        for (row, (&rhs, &value)) in self.equality.rhs.iter().zip(&eqactivity).enumerate() {
            let rhs = rhs - value;
            if !rhs.is_finite() {
                return Ok(None);
            }
            b.push(rhs);
            roles.push(Role::EqualityLinear(row));
            cooperation.completed(1).await?;
        }
        for (row, &value) in point.equality.iter().enumerate() {
            b.push(-value);
            roles.push(Role::EqualityNonlinear(row));
            cooperation.completed(1).await?;
        }
        for (row, (&rhs, &value)) in self.inequality.rhs.iter().zip(&ineqactivity).enumerate() {
            let rhs = rhs - value;
            if !rhs.is_finite() {
                return Ok(None);
            }
            b.push(rhs);
            roles.push(Role::InequalityLinear(row));
            cooperation.completed(1).await?;
        }
        for (row, &value) in point.inequality.iter().enumerate() {
            b.push(-value);
            roles.push(Role::InequalityNonlinear(row));
            cooperation.completed(1).await?;
        }
        let mut lower_rows = workspace::capacity(x.len(), owner)?;
        let mut upper_rows = workspace::capacity(x.len(), owner)?;
        for variable in 0..x.len() {
            if self.lower[variable].is_finite() {
                lower_rows.push(Some(b.len()));
                workspace::push(&mut b, x[variable] - self.lower[variable], owner)?;
                workspace::push(&mut roles, Role::Lower(variable), owner)?;
            } else {
                lower_rows.push(None);
            }
            if self.upper[variable].is_finite() {
                upper_rows.push(Some(b.len()));
                workspace::push(&mut b, self.upper[variable] - x[variable], owner)?;
                workspace::push(&mut roles, Role::Upper(variable), owner)?;
            } else {
                upper_rows.push(None);
            }
            cooperation.completed(1).await?;
        }
        if b.iter().any(|value| !value.is_finite()) {
            return Ok(None);
        }
        Ok(Some(ConstraintValues {
            rhs: b,
            roles,
            equalities: eq,
            lower_rows,
            upper_rows,
        }))
    }
}

impl Geometry {
    pub(in crate::builtins::math::optim::fmincon) async fn constraint_values(
        &self,
        x: &[f64],
        point: &Point,
        owner: BuiltinCatalogIdentity,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<ConstraintValues>> {
        self.view()
            .constraint_values(x, point, owner, cooperation)
            .await
    }
}
