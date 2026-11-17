//! Refusal of a planned solve whose operations estimate exceeds what the run's allowance has
//! left, after meshing and before assembly reserves any work.
use super::estimate_static_solid_operations_from_plan;
use crate::planning::resource::FeaResourcePlan;
use runmat_meshing_core::SolverMeshArtifact;
use runmat_simulation_usage::{LedgerError, OperationCount, UsageContext};

/// With the plan's exact counts, the work already completed or reserved plus this solve's
/// estimate must fit the run's spending cap. A run without a usage ledger is not metered.
pub(crate) fn admit_planned_static_operations(
    plan: &FeaResourcePlan,
    mesh: Option<&SolverMeshArtifact>,
) -> Result<(), LedgerError> {
    let (Some(usage), Some(mesh)) = (UsageContext::capture(), mesh) else {
        return Ok(());
    };
    let Some(estimate) = estimate_static_solid_operations_from_plan(plan, mesh)? else {
        return Ok(());
    };
    admit_estimate(&usage, estimate.total()?)
}

pub(crate) fn admit_estimate(
    usage: &UsageContext,
    estimate: OperationCount,
) -> Result<(), LedgerError> {
    let snapshot = usage.ledger().snapshot()?;
    let required = snapshot
        .completed
        .total()?
        .checked_add(snapshot.reserved)?
        .checked_add(estimate)?;
    if required > snapshot.budget {
        return Err(LedgerError::BudgetExceeded {
            required,
            budget: snapshot.budget,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_simulation_usage::{UsageCategory, UsageLedger};
    use std::sync::Arc;

    #[test]
    fn completed_work_plus_the_estimate_must_fit_the_spending_cap() {
        let ops = OperationCount::from_operations;
        let ledger = Arc::new(UsageLedger::new(ops(1_000), ops(1_000)).unwrap());
        let usage = UsageContext::new(Arc::clone(&ledger));
        let mut meshing = ledger.reserve(UsageCategory::Meshing, ops(300)).unwrap();
        meshing.complete(ops(300)).unwrap();
        drop(meshing);
        assert_eq!(admit_estimate(&usage, ops(700)), Ok(()));
        assert_eq!(
            admit_estimate(&usage, ops(701)),
            Err(LedgerError::BudgetExceeded {
                required: ops(1_001),
                budget: ops(1_000),
            })
        );
        assert_eq!(ledger.snapshot().unwrap().completed.assembly, ops(0));
    }
}
