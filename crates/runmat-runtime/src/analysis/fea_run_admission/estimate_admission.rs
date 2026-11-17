//! Refusal of a run whose plan estimate exceeds the allowance its lease grants, before any work.
use super::{FeaRunDenied, FeaRunOutcome, FeaRunPermit, FeaRunRequest, FeaRunUsage};

/// Returns the permit when the request's operations estimate fits the lease's spending cap.
/// Otherwise closes the lease with no usage, so the refused run charges nothing.
pub(super) fn refuse_unaffordable(
    request: &FeaRunRequest,
    permit: Box<dyn FeaRunPermit>,
) -> Result<Box<dyn FeaRunPermit>, FeaRunDenied> {
    let available_ops = permit.budget().budget_ops;
    let Some(estimated_ops) = request.estimate_ops.filter(|ops| *ops > available_ops) else {
        return Ok(permit);
    };
    let unused = FeaRunUsage {
        completed_ops: Default::default(),
        resources: Default::default(),
    };
    permit.settle(FeaRunOutcome::Failed, &unused)?;
    Err(FeaRunDenied::AllowanceInsufficient {
        estimated_ops,
        available_ops,
    })
}
