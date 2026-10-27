//! Original coefficient magnitudes, independent of the supplied candidate's
//! vanishing differential. LP deliberately retains its existing normalization.
use super::*;
pub(in crate::builtins::math::optim::optimmodel::lowering) struct ObjectiveScale {
    linear_abs: Vec<Rational>,
    curvature_row_abs: Vec<Rational>,
}
impl ObjectiveScale {
    pub(in crate::builtins::math::optim::optimmodel::lowering) async fn from_parts(
        linear_abs: Vec<Rational>,
        curvature_row_abs: Vec<Rational>,
        policy: Policy,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        if linear_abs.len() != curvature_row_abs.len() {
            return Err(policy.invalid("original objective scale dimensions disagree"));
        }
        for (linear, curvature) in linear_abs.iter().zip(&curvature_row_abs) {
            if linear.is_negative() || curvature.is_negative() {
                return Err(policy.invalid("original objective scale must be nonnegative"));
            }
            c.completed(1).await?;
        }
        Ok(Self {
            linear_abs,
            curvature_row_abs,
        })
    }
    pub(in crate::builtins::math::optim::optimmodel::lowering) async fn at(
        &self,
        column: usize,
        x_norm: &Rational,
        policy: Policy,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Rational> {
        let linear = self
            .linear_abs
            .get(column)
            .ok_or_else(|| policy.invalid("original objective scale column is unavailable"))?;
        let curvature = &self.curvature_row_abs[column];
        linear
            .add(
                &curvature
                    .mul(
                        &Rational::from_usize(1).add(x_norm, policy, c).await?,
                        policy,
                        c,
                    )
                    .await?,
                policy,
                c,
            )
            .await
    }
    pub(in crate::builtins::math::optim::optimmodel::lowering) async fn magnitude(
        &self,
        point: &[Rational],
        x_norm: &Rational,
        policy: Policy,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Rational> {
        if point.len() != self.linear_abs.len() {
            return Err(policy.invalid("original objective scale point has the wrong dimensions"));
        }
        let mut total = Rational::zero();
        for (column, x) in point.iter().enumerate() {
            let scale = self.at(column, x_norm, policy, c).await?;
            total = total
                .add(
                    &scale
                        .mul(
                            &Rational::from_usize(1).add(&x.abs(), policy, c).await?,
                            policy,
                            c,
                        )
                        .await?,
                    policy,
                    c,
                )
                .await?;
            c.completed(1).await?;
        }
        // Only an exactly zero source objective has no relative data scale.
        if total.is_zero() {
            total = Rational::from_usize(1);
        }
        Ok(total)
    }
}
