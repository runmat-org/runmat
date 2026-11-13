//! Rectangular scaled Jacobian plus leading diagonal rows; no normal matrix.
use super::*;
use crate::builtins::math::linalg::ops::sparse_factor::least_squares as qr;
use trust_quadratic::{
    residual::{self, ShiftedResidual},
    Operator,
};
struct Augmented<'a> {
    normal: normal::Normal<'a>,
    metric: &'a trust_quadratic::bounds::Metric,
    columns: usize,
    rows: usize,
    pointers: Vec<usize>,
    indices: Vec<usize>,
    values: Vec<f64>,
    rhs: Vec<f64>,
    product: Vec<f64>,
}
pub(super) async fn step(
    geometry: &Geometry,
    point: &initial::Seed,
    jacobian: &Jacobian,
    metric: &trust_quadratic::bounds::Metric,
    radius: f64,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Option<Vec<f64>>> {
    let products = jacobian.products(c).await?;
    let m = point.residual.len();
    let n = geometry.axes.len();
    let rows = m
        .checked_add(n)
        .ok_or_else(|| invalid("augmented residual extent overflow".into()))?;
    let capacity = products
        .selected_entries(&geometry.axes, c)
        .await?
        .checked_add(n)
        .ok_or_else(|| invalid("augmented Jacobian storage overflow".into()))?;
    let mut pointers = runmat_value::exact_capacity_vec(
        n.checked_add(1)
            .ok_or_else(|| invalid("Jacobian pointer overflow".into()))?,
    )
    .map_err(allocation)?;
    let mut indices = runmat_value::exact_capacity_vec(capacity).map_err(allocation)?;
    let mut values = runmat_value::exact_capacity_vec(capacity).map_err(allocation)?;
    let mut rhs = zeros_with(rows, c).await?;
    for (to, &r) in rhs[n..].iter_mut().zip(&point.residual) {
        *to = -r;
        c.completed(1).await?;
    }
    pointers.push(0);
    // Preserve the reviewed diagonal-first row permutation and zero leading
    // RHS; near-bound small residual projections are not obtained by cancellation.
    for (column, &axis) in geometry.axes.iter().enumerate() {
        indices.push(column);
        values.push(metric.diagonal[column].sqrt());
        if products
            .append_column(axis, metric.scale[column], n, &mut indices, &mut values, c)
            .await?
            .is_none()
        {
            return Ok(None);
        }
        pointers.push(values.len());
    }
    let normal = normal::Normal::new(geometry, metric, products, c).await?;
    let mut augmented = Augmented {
        normal,
        metric,
        columns: n,
        rows,
        pointers,
        indices,
        values,
        rhs,
        product: zeros_with(n, c).await?,
    };
    residual::step(&mut augmented, &metric.gradient, radius, c).await
}
impl ShiftedResidual for Augmented<'_> {
    async fn solve(
        &mut self,
        shift: f64,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Vec<f64>>> {
        // This physical RHS prefix belongs to every shifted QR attempt.
        for value in &mut self.rhs[..self.columns] {
            *value = 0.;
            c.completed(1).await?;
        }
        for column in 0..self.columns {
            let value = self.metric.diagonal[column] + shift;
            if !(value >= 0. && value.is_finite()) {
                return Ok(None);
            }
            self.values[self.pointers[column]] = value.sqrt();
            c.completed(1).await?;
        }
        let Some(solution) = qr::solve_borrowed(
            qr::BorrowedProblem {
                rows: self.rows,
                columns: self.columns,
                pointers: &self.pointers,
                indices: &self.indices,
                values: &self.values,
                rhs: &self.rhs,
            },
            c.runtime(),
            policy().identity,
        )
        .await?
        else {
            return Ok(None);
        };
        c.completed(0).await?;
        if self
            .normal
            .apply(&solution.values, &mut self.product, c)
            .await?
            .is_none()
        {
            return Ok(None);
        }
        if !self
            .normal
            .qualifies_direct(
                qr::BorrowedProblem {
                    rows: self.rows,
                    columns: self.columns,
                    pointers: &self.pointers,
                    indices: &self.indices,
                    values: &self.values,
                    rhs: &self.rhs,
                },
                &solution.values,
                &self.product,
                shift,
                c,
            )
            .await?
        {
            return Ok(None);
        }
        Ok(Some(solution.values))
    }
}
