//! Matrix-free scaled residual curvature and actual diagonal preconditioning.
use super::*;
mod qualification;
pub(super) struct Products<'a> {
    pub model: &'a Model,
    pub metric: &'a trust_quadratic::bounds::Metric,
    pub scratch: Vec<f64>,
}
impl Operator for Products<'_> {
    fn policy(&self) -> trust_quadratic::Policy {
        policy()
    }
    async fn apply(
        &mut self,
        input: &[f64],
        output: &mut [f64],
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<()>> {
        let model = self.model;
        if input.len() != model.c.n || output.len() != model.c.n {
            return Err(invalid("reflective model product geometry mismatch"));
        }
        for chunk in self.scratch.chunks_mut(1024) {
            chunk.fill(0.);
            c.completed(chunk.len() as u64).await?;
        }
        for (column, &z) in input.iter().enumerate() {
            let x = z * self.metric.scale[column];
            if !x.is_finite() {
                return Ok(None);
            }
            for entry in model.c.colptr[column]..model.c.colptr[column + 1] {
                let row = model.c.rowval[entry];
                self.scratch[row] = model.c.nzval[entry].mul_add(x, self.scratch[row]);
                if !self.scratch[row].is_finite() {
                    return Ok(None);
                }
                c.completed(1).await?;
            }
            c.completed(1).await?;
        }
        for (column, to) in output.iter_mut().enumerate() {
            let mut sum = 0f64;
            for entry in model.c.colptr[column]..model.c.colptr[column + 1] {
                let row = model.c.rowval[entry];
                let root = model.weights[row].sqrt();
                sum = (model.c.nzval[entry] * root).mul_add(root * self.scratch[row], sum);
                c.completed(1).await?;
            }
            *to = model.units * self.metric.scale[column] * sum
                + self.metric.diagonal[column] * input[column];
            if !to.is_finite() {
                return Ok(None);
            }
            c.completed(1).await?;
        }
        Ok(Some(()))
    }
    async fn copy_band(
        &self,
        n: usize,
        bandwidth: usize,
        output: &mut [f64],
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<trust_quadratic::Materialization> {
        let bandwidth = bandwidth.min(n.saturating_sub(1));
        let stride = bandwidth
            .checked_add(1)
            .ok_or_else(|| invalid("preconditioner band overflow"))?;
        if n != self.model.c.n || n.checked_mul(stride) != Some(output.len()) {
            return Err(invalid("reflective band preconditioner geometry mismatch"));
        }
        for row in 0..n {
            for column in row.saturating_sub(bandwidth)..=row {
                let mut left = self.model.c.colptr[row];
                let left_end = self.model.c.colptr[row + 1];
                let mut right = self.model.c.colptr[column];
                let right_end = self.model.c.colptr[column + 1];
                let mut value = 0f64;
                let mut length = 0f64;
                while left < left_end && right < right_end {
                    let i = self.model.c.rowval[left];
                    let j = self.model.c.rowval[right];
                    match i.cmp(&j) {
                        std::cmp::Ordering::Less => left += 1,
                        std::cmp::Ordering::Greater => right += 1,
                        std::cmp::Ordering::Equal => {
                            let root = self.model.weights[i].sqrt();
                            let a = self.model.c.nzval[left] * root * self.metric.scale[row];
                            let b = self.model.c.nzval[right] * root * self.metric.scale[column];
                            if row == column {
                                length = length.hypot(a);
                            } else {
                                value = a.mul_add(b, value);
                            }
                            left += 1;
                            right += 1;
                        }
                    }
                    c.completed(1).await?;
                }
                if row == column {
                    value = self.model.units * length.powi(2) + self.metric.diagonal[row];
                } else {
                    value *= self.model.units;
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
}
