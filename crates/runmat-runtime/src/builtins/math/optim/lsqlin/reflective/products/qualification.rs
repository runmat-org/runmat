//! Absolute original rectangular contributions for direct-step qualification.
use super::*;
impl Products<'_> {
    /// Preserve the signed original product, reusing only its retired row
    /// scratch for absolute contribution bounds. No C'C is assembled.
    pub(in crate::builtins::math::optim::lsqlin::reflective) async fn qualifies(
        &mut self,
        input: &[f64],
        product: &[f64],
        residual: &[f64],
        shift: f64,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        let model = self.model;
        if input.len() != model.c.n || product.len() != model.c.n || residual.len() != model.c.m {
            return Err(invalid("original stationarity geometry mismatch"));
        }
        for chunk in self.scratch.chunks_mut(1024) {
            chunk.fill(0.);
            c.completed(chunk.len() as u64).await?;
        }
        for (column, &z) in input.iter().enumerate() {
            let physical = (self.metric.scale[column] * z).abs();
            if !physical.is_finite() {
                return Ok(false);
            }
            for entry in model.c.colptr[column]..model.c.colptr[column + 1] {
                let row = model.c.rowval[entry];
                self.scratch[row] = model.c.nzval[entry]
                    .abs()
                    .mul_add(physical, self.scratch[row]);
                if !self.scratch[row].is_finite() {
                    return Ok(false);
                }
                c.completed(1).await?;
            }
            c.completed(1).await?;
        }
        for column in 0..model.c.n {
            let mut rhs = 0f64;
            let mut curvature = 0f64;
            for entry in model.c.colptr[column]..model.c.colptr[column + 1] {
                let row = model.c.rowval[entry];
                let root = model.weights[row].sqrt();
                let coefficient = model.c.nzval[entry].abs() * root;
                rhs = coefficient.mul_add(root * residual[row].abs(), rhs);
                curvature = coefficient.mul_add(root * self.scratch[row], curvature);
                c.completed(1).await?;
            }
            let scale = model.units * self.metric.scale[column];
            let formation = trust_quadratic::residual::StationarityScale {
                rhs: scale * rhs,
                curvature: scale * curvature,
                diagonal: self.metric.diagonal[column] * input[column].abs(),
            };
            if !formation.qualifies(
                self.metric.gradient[column],
                product[column],
                shift * input[column],
                model.c.n,
            ) {
                return Ok(false);
            }
            c.completed(1).await?;
        }
        Ok(true)
    }
}
