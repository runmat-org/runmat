//! O(1) active Brent samples retain their actual original-DAG evaluations.
use super::*;
use crate::builtins::math::optim::{brent::ZeroFunction, fzero::compiled};
use crate::context::host_allocation::{
    host_allocation_runtime_error, HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use runmat_execution::resource::{ExactShared, RetainedHostAllocationClaim};
use std::sync::Arc;
struct Backing {
    point: numerical::Point,
    _claim: Option<Arc<dyn RetainedHostAllocationClaim>>,
}
#[derive(Clone)]
struct Sample(ExactShared<Backing>);
impl Sample {
    fn retain(
        point: numerical::Point,
        policy: Policy,
        c: &ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let runtime = c.runtime();
        let error = |source| {
            host_allocation_runtime_error(
                runtime,
                "RunMat:OptimModel:Allocation",
                "equation sample control allocation failed",
                source,
            )
        };
        let bytes = u64::try_from(ExactShared::<Backing>::requested_bytes())
            .map_err(|_| error(HostAllocationError::Overflow))?;
        let retained = HostAllocationPlan::retained_metadata_bytes(bytes).map_err(error)?;
        let plan = HostAllocationPlan::new(0, retained).map_err(error)?;
        let receipt = HostAllocationReceipt::admit(runtime, plan).map_err(error)?;
        let claim = match (receipt.as_ref(), runtime) {
            (Some(receipt), Some(runtime)) => Some(
                receipt
                    .claim_retained_metadata(runtime, bytes)
                    .map_err(error)?,
            ),
            _ => None,
        };
        let sample = ExactShared::try_new(Backing {
            point,
            _claim: claim,
        })
        .map_err(|_| {
            policy.storage(runmat_value::ExactHostNumericAllocationError::AllocationFailed)
        })?;
        if let Some(receipt) = receipt {
            receipt.checkpoint_captured().map_err(error)?;
        }
        Ok(Self(sample))
    }
    fn into_point(self, policy: Policy) -> BuiltinResult<numerical::Point> {
        self.0
            .try_unwrap()
            .map(|backing| backing.point)
            .map_err(|_| {
                policy.invalid("selected equation sample still has an active bracket owner")
            })
    }
}
struct Original<'a> {
    model: &'a Model,
    policy: Policy,
}
impl ZeroFunction for Original<'_> {
    type Receipt = Sample;
    async fn evaluate(
        &mut self,
        x: f64,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<(f64, Sample)> {
        let mut original = super::super::residual::Original {
            source: &self.model.source,
            layout: &self.model.layout,
            projection: super::super::residual::Projection::Equations(&self.model.evidence),
            roots: &self.model.evidence.formulation.roots,
            rows: 1,
            pending: None,
            accepted: None,
            policy: self.policy,
            cooperation: c,
        };
        let values = original.probe(&[x]).await?;
        let point = original.pending.take().ok_or_else(|| {
            self.policy
                .invalid("scalar equation sample has no original evaluation")
        })?;
        drop(original);
        c.completed(0).await?;
        let sample = Sample::retain(point, self.policy, c)?;
        Ok((values[0], sample))
    }
}
pub(in crate::builtins::math::optim::optimmodel) struct Outcome {
    pub(super) result: crate::builtins::math::optim::brent::BrentZeroResult,
    pub(super) point: numerical::Point,
}
pub(super) async fn solve(
    model: &Model,
    starting: Option<&Value>,
    options: Option<&Value>,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Outcome> {
    c.completed(0).await?;
    if model.layout.columns != 1 || model.evidence.rows != 1 {
        return Err(policy.invalid(
            "fzero requires exactly one original variable coordinate and one equation residual",
        ));
    }
    let options = compiled::prepare(options, c).await?;
    let bounds = layout::continuous_bounds(&model.layout, policy, c).await?;
    if bounds.lower[0] != f64::NEG_INFINITY || bounds.upper[0] != f64::INFINITY {
        return Err(policy.invalid("fzero cannot discard original variable bounds"));
    }
    let starting = starting.ok_or_else(|| {
        policy.invalid("a scalar nonlinear equation requires an initial-point structure")
    })?;
    let x = layout::initial_point(&model.source, &model.layout, starting, policy, c).await?;
    let selected = compiled::solve(&options, x[0], Original { model, policy }, c)?.await?;
    let point = selected.receipt.into_point(policy)?;
    if !point
        .matches_layout(&model.layout, &[selected.result.x], policy, c)
        .await?
    {
        return Err(policy.invalid("scalar equation result differs from its original sample"));
    }
    Ok(Outcome {
        result: selected.result,
        point,
    })
}
