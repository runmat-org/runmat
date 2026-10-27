//! Direct opaque work qualifies against original products and its actual
//! formed rectangular columns, not an inferred dense normal matrix.
use super::*;
use crate::builtins::math::linalg::ops::sparse_factor::least_squares::BorrowedProblem;
impl Normal<'_> {
    pub(in crate::builtins::math::optim::least_squares::reflective) async fn qualifies_direct(
        &mut self,
        factor: BorrowedProblem<'_>,
        point: &[f64],
        product: &[f64],
        shift: f64,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        if !self.products.is_opaque() {
            return self
                .qualifies(point, product, &factor.rhs[factor.columns..], shift, c)
                .await;
        }
        let n = self.geometry.axes.len();
        let m = self.residual.len();
        if factor.columns != n
            || m.checked_add(n) != Some(factor.rows)
            || factor.rhs.len() != factor.rows
            || point.len() != n
            || product.len() != n
        {
            return Err(invalid(
                "opaque direct qualification geometry mismatch".into(),
            ));
        }
        for chunk in self.residual.chunks_mut(1024) {
            chunk.fill(0.);
            c.completed(chunk.len() as u64).await?;
        }
        // These are the actual J*D columns already formed for the selected QR.
        // Retired m-row product storage owns the uncancelled formation bound.
        for (column, &z) in point.iter().enumerate() {
            for index in factor.pointers[column]..factor.pointers[column + 1] {
                let row = factor.indices[index];
                if row >= n {
                    if row >= factor.rows {
                        return Err(invalid("opaque factor row is out of range".into()));
                    }
                    self.residual[row - n] = factor.values[index]
                        .abs()
                        .mul_add(z.abs(), self.residual[row - n]);
                }
                c.completed(1).await?;
            }
        }
        for column in 0..n {
            let mut rhs = 0.;
            let mut curvature = 0.;
            for index in factor.pointers[column]..factor.pointers[column + 1] {
                let row = factor.indices[index];
                if row >= n {
                    let coefficient = factor.values[index].abs();
                    rhs = coefficient.mul_add(factor.rhs[row].abs(), rhs);
                    curvature = coefficient.mul_add(self.residual[row - n], curvature);
                }
                c.completed(1).await?;
            }
            if !(trust_quadratic::residual::StationarityScale {
                rhs,
                curvature,
                diagonal: self.metric.diagonal[column] * point[column].abs(),
            })
            .qualifies(
                self.metric.gradient[column],
                product[column],
                shift * point[column],
                n,
            ) {
                return Ok(false);
            }
        }
        Ok(true)
    }
}
