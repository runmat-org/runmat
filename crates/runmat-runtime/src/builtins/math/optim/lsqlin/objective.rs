//! Residual objective data, never a formed normal-equations matrix.
use super::*;
pub(in crate::builtins::math::optim) enum Weights {
    Uniform,
    PerRow(Vec<f64>),
}
#[derive(Clone, Copy)]
pub(super) enum Weighting {
    Uniform,
    PerRow,
}
#[derive(Clone, Copy)]
pub(in crate::builtins::math::optim) enum Units {
    HalfSquared,
    OriginalSquared,
}
impl Units {
    pub(super) fn multiplier(self) -> f64 {
        match self {
            Self::HalfSquared => 1.,
            Self::OriginalSquared => 2.,
        }
    }
}
impl Weights {
    pub(super) async fn into_values(
        self,
        count: usize,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<(Vec<f64>, f64, Weighting)> {
        let weighting = match &self {
            Self::Uniform => Weighting::Uniform,
            Self::PerRow(_) => Weighting::PerRow,
        };
        let values = match self {
            Self::Uniform => {
                let mut values = runmat_value::exact_capacity_vec(count).map_err(|source| {
                    build_runtime_error("lsqlin: residual weight allocation failed")
                        .with_builtin(NAME)
                        .with_source(source)
                        .build()
                })?;
                for _ in 0..count {
                    values.push(1.);
                    c.completed(1).await?;
                }
                values
            }
            Self::PerRow(values) => values,
        };
        if values.len() != count {
            return Err(invalid("residual weights differ from actual rows"));
        }
        if matches!(weighting, Weighting::Uniform) {
            return Ok((values, 1., weighting));
        }
        let mut scale = 0f64;
        for &value in &values {
            if !value.is_finite() || value < 0. {
                return Err(invalid("residual weights must be finite and nonnegative"));
            }
            scale = scale.max(value);
            c.completed(1).await?;
        }
        Ok((values, if scale == 0. { 1. } else { scale }, weighting))
    }
}
