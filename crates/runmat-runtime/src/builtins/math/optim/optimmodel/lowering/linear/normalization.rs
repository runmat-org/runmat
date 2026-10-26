//! Stable floating proposal weights; exact proof normalization is separate.
use super::*;
pub(in crate::builtins::math::optim::optimmodel::lowering) enum Normalization {
    Identity,
    Count(usize),
    Weighted { scale: Vec<f64>, sum: Vec<f64> },
}
impl Normalization {
    pub(in crate::builtins::math::optim::optimmodel::lowering) fn weight(
        &self,
        raw: f64,
        row: usize,
    ) -> f64 {
        match self {
            Self::Identity => raw,
            Self::Count(count) => raw / (*count as f64),
            Self::Weighted { scale, sum } => (raw / scale[row]) / sum[row],
        }
    }
}
impl Linear {
    pub(in crate::builtins::math::optim::optimmodel::lowering) async fn normalization(
        &mut self,
        inputs: usize,
        outputs: usize,
        policy: Policy,
        c: &mut ScalarCooperation,
    ) -> Lowered<Normalization> {
        let Self::Mean(mean) = self else {
            return Ok(Ok(Normalization::Identity));
        };
        if !mean.is_weighted() {
            let factor = mean.reduction.factor();
            return if factor == 0 && outputs != 0 {
                Ok(Err(Stop::NonFinite))
            } else {
                Ok(Ok(Normalization::Count(factor)))
            };
        }
        async fn zeros(
            count: usize,
            policy: Policy,
            c: &mut ScalarCooperation,
        ) -> BuiltinResult<Vec<f64>> {
            let mut values =
                runmat_value::exact_capacity_vec(count).map_err(|e| policy.storage(e))?;
            for _ in 0..count {
                values.push(0.);
                c.completed(1).await?;
            }
            Ok(values)
        }
        let mut scale = zeros(outputs, policy, c).await?;
        for source in 0..inputs {
            let row = mean
                .reduction
                .project(source)
                .ok_or_else(|| policy.invalid("mean source exceeds admitted projection"))?;
            let weight = mean.weight(source);
            if !weight.is_finite() {
                return Ok(Err(Stop::NonFinite));
            }
            scale[row] = scale[row].max(weight);
            c.completed(1).await?;
        }
        for &value in &scale {
            if value == 0. {
                return Ok(Err(Stop::NonFinite));
            }
            c.completed(1).await?;
        }
        let mut sum = zeros(outputs, policy, c).await?;
        for source in 0..inputs {
            let row = mean
                .reduction
                .project(source)
                .ok_or_else(|| policy.invalid("mean source exceeds admitted projection"))?;
            sum[row] += mean.weight(source) / scale[row];
            c.completed(1).await?;
        }
        Ok(Ok(Normalization::Weighted { scale, sum }))
    }
}
