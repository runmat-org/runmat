//! A scaled normal product over one accepted rectangular Jacobian session.
//! O(m+n) vectors are numerical work, not a second coefficient/Jacobian tree.
use super::*;
use trust_quadratic::Operator;
mod factor;
pub(super) struct Normal<'a> {
    geometry: &'a Geometry,
    metric: &'a trust_quadratic::bounds::Metric,
    products: derivative::Products<'a>,
    input: Vec<f64>,
    residual: Vec<f64>,
    transpose: Vec<f64>,
}
impl<'a> Normal<'a> {
    pub(super) async fn new(
        geometry: &'a Geometry,
        metric: &'a trust_quadratic::bounds::Metric,
        products: derivative::Products<'a>,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let (rows, columns) = products.dimensions();
        if columns != geometry.variables {
            return Err(invalid(
                "accepted Jacobian geometry disagrees with trust subspace".into(),
            ));
        }
        Ok(Self {
            geometry,
            metric,
            products,
            input: zeros_with(columns, c).await?,
            residual: zeros_with(rows, c).await?,
            transpose: zeros_with(columns, c).await?,
        })
    }
    pub(super) async fn qualifies(
        &mut self,
        point: &[f64],
        product: &[f64],
        residual: &[f64],
        shift: f64,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        let n = self.geometry.axes.len();
        if point.len() != n || product.len() != n {
            return Err(invalid("trust-step qualification geometry mismatch".into()));
        }
        // The same physical input used by signed apply belongs to this step.
        self.products
            .absolute_forward(&self.input, &mut self.residual, c)
            .await?;
        for (column, &axis) in self.geometry.axes.iter().enumerate() {
            let (rhs, curvature) = self
                .products
                .absolute_components(axis, residual, &self.residual, c)
                .await?;
            let scale = trust_quadratic::residual::StationarityScale {
                rhs: self.metric.scale[column] * rhs,
                curvature: self.metric.scale[column] * curvature,
                diagonal: self.metric.diagonal[column] * point[column].abs(),
            };
            if !scale.qualifies(
                self.metric.gradient[column],
                product[column],
                shift * point[column],
                n,
            ) {
                return Ok(false);
            }
            c.completed(1).await?;
        }
        Ok(true)
    }
}
impl Operator for Normal<'_> {
    fn policy(&self) -> trust_quadratic::Policy {
        policy()
    }
    async fn copy_band(
        &self,
        n: usize,
        bandwidth: usize,
        output: &mut [f64],
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<trust_quadratic::Materialization> {
        if self.products.is_opaque() {
            return Ok(trust_quadratic::Materialization::Unavailable);
        }
        let stride = bandwidth
            .checked_add(1)
            .ok_or_else(|| invalid("Jacobian preconditioner extent overflow".into()))?;
        if n != self.geometry.axes.len() || n.checked_mul(stride) != Some(output.len()) {
            return Err(invalid("Jacobian preconditioner geometry mismatch".into()));
        }
        for row in 0..n {
            for column in row.saturating_sub(bandwidth)..=row {
                let mut value = self
                    .products
                    .inner_product(
                        self.geometry.axes[row],
                        self.geometry.axes[column],
                        self.metric.scale[row],
                        self.metric.scale[column],
                        c,
                    )
                    .await?;
                if row == column {
                    value += self.metric.diagonal[row];
                }
                if !value.is_finite() {
                    return Ok(trust_quadratic::Materialization::Undefined);
                }
                output[row * stride + row - column] = value;
                c.completed(1).await?;
            }
        }
        Ok(trust_quadratic::Materialization::Complete)
    }
    async fn apply(
        &mut self,
        input: &[f64],
        output: &mut [f64],
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<()>> {
        let n = self.geometry.axes.len();
        if input.len() != n || output.len() != n {
            return Err(invalid(
                "reflective normal product geometry mismatch".into(),
            ));
        }
        for (column, &axis) in self.geometry.axes.iter().enumerate() {
            self.input[axis] = self.metric.scale[column] * input[column];
            c.completed(1).await?;
        }
        if self
            .products
            .apply_normal(&self.input, &mut self.residual, &mut self.transpose, c)
            .await?
            .is_none()
        {
            return Ok(None);
        }
        for (column, (&axis, to)) in self.geometry.axes.iter().zip(output).enumerate() {
            *to = self.metric.scale[column] * self.transpose[axis]
                + self.metric.diagonal[column] * input[column];
            if !to.is_finite() {
                return Ok(None);
            }
            c.completed(1).await?;
        }
        Ok(Some(()))
    }
}
