//! Regularization rows precede residual rows, preserving the same least-squares
//! problem without subtractive Householder updates of small bound distances.
use super::*;
pub(super) struct Augmented<'a> {
    operator: products::Products<'a>,
    product: Vec<f64>,
    residual: &'a [f64],
    rows: usize,
    columns: usize,
    pointers: Vec<usize>,
    indices: Vec<usize>,
    values: Vec<f64>,
    rhs: Vec<f64>,
}
impl<'a> Augmented<'a> {
    pub(super) async fn new(
        model: &'a Model,
        point: &'a Point,
        metric: &'a trust_quadratic::bounds::Metric,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Self>> {
        let rows = model
            .c
            .m
            .checked_add(model.c.n)
            .ok_or_else(|| invalid("direct residual row extent overflow"))?;
        let entries = model
            .c
            .nzval
            .len()
            .checked_add(model.c.n)
            .ok_or_else(|| invalid("direct residual storage overflow"))?;
        let mut pointers = runmat_value::exact_capacity_vec(
            model
                .c
                .n
                .checked_add(1)
                .ok_or_else(|| invalid("direct residual pointer overflow"))?,
        )
        .map_err(allocation)?;
        let mut indices = runmat_value::exact_capacity_vec(entries).map_err(allocation)?;
        let mut values = runmat_value::exact_capacity_vec(entries).map_err(allocation)?;
        let mut rhs = zeros(rows, c).await?;
        let units = model.units.sqrt();
        for (row, &r) in point.residual.iter().enumerate() {
            rhs[model.c.n + row] = -units * model.weights[row].sqrt() * r;
            if !rhs[model.c.n + row].is_finite() {
                return Ok(None);
            }
            c.completed(1).await?;
        }
        pointers.push(0);
        for column in 0..model.c.n {
            indices.push(column);
            values.push(metric.diagonal[column].sqrt());
            for entry in model.c.colptr[column]..model.c.colptr[column + 1] {
                let row = model.c.rowval[entry];
                let value =
                    units * model.weights[row].sqrt() * model.c.nzval[entry] * metric.scale[column];
                if !value.is_finite() {
                    return Ok(None);
                }
                indices.push(model.c.n + row);
                values.push(value);
                c.completed(1).await?;
            }
            pointers.push(values.len());
            c.completed(1).await?;
        }
        Ok(Some(Self {
            operator: products::Products {
                model,
                metric,
                scratch: zeros(model.c.m, c).await?,
            },
            product: zeros(model.c.n, c).await?,
            residual: &point.residual,
            rows,
            columns: model.c.n,
            pointers,
            indices,
            values,
            rhs,
        }))
    }
    pub(super) async fn solve(
        &mut self,
        shift: f64,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Vec<f64>>> {
        for column in 0..self.columns {
            let value = self.operator.metric.diagonal[column] + shift;
            if !(value >= 0. && value.is_finite()) {
                return Ok(None);
            }
            self.values[self.pointers[column]] = value.sqrt();
            c.completed(1).await?;
        }
        let solution = least_squares::solve_borrowed(
            least_squares::BorrowedProblem {
                rows: self.rows,
                columns: self.columns,
                pointers: &self.pointers,
                indices: &self.indices,
                values: &self.values,
                rhs: &self.rhs,
            },
            c.runtime(),
            runmat_builtins::LSQLIN_CATALOG_ENTRY.identity,
        )
        .await?;
        c.completed(0).await?;
        let Some(solution) = solution else {
            return Ok(None);
        };
        if self
            .operator
            .apply(&solution.values, &mut self.product, c)
            .await?
            .is_none()
        {
            return Ok(None);
        }
        if !self
            .operator
            .qualifies(&solution.values, &self.product, self.residual, shift, c)
            .await?
        {
            return Ok(None);
        }
        Ok(Some(solution.values))
    }
}
