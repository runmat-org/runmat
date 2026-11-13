//! Evaluated residual points and stopped-before-evaluation coordinates are
//! distinct products. Only the former owns original DAG values.
use super::super::*;
use crate::builtins::math::optim::least_squares::{LeastSquaresResult, PointEvidence};
pub(in crate::builtins::math::optim::optimmodel::lowering) enum Candidate {
    Evaluated(numerical::Point),
    Supplied(validation::stopped::Candidate),
}
impl Candidate {
    pub(in crate::builtins::math::optim::optimmodel::lowering) async fn select(
        layout: &layout::Layout,
        result: &LeastSquaresResult,
        accepted: Option<numerical::Point>,
        policy: Policy,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        match result.point_evidence {
            PointEvidence::Evaluated => {
                let point = accepted.ok_or_else(|| {
                    policy.invalid("residual result has no accepted original source point")
                })?;
                if !point.matches_layout(layout, &result.x, policy, c).await? {
                    return Err(policy.invalid(
                        "residual result differs from its accepted original source point",
                    ));
                }
                Ok(Self::Evaluated(point))
            }
            PointEvidence::Supplied => {
                if accepted.is_some() || result.func_count != 0 || result.exitflag != 0 {
                    return Err(policy.invalid(
                        "unevaluated residual result contradicts original point ownership",
                    ));
                }
                let point = validation::stopped::Candidate::supplied(layout, &result.x, policy, c)
                    .await?
                    .ok_or_else(|| policy.invalid("stopped supplied coordinates are invalid"))?;
                Ok(Self::Supplied(point))
            }
        }
    }
}
