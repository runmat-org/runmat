//! Admission refuses a run whose plan estimate exceeds the lease's allowance, before any work.
use super::*;
use runmat_simulation_usage::OperationCount;
use std::sync::{Arc, Mutex};

fn ops(value: u64) -> OperationCount {
    OperationCount::from_operations(value)
}

#[derive(Default)]
struct Settlements(Vec<(FeaRunOutcome, FeaRunUsage)>);
struct Host(Arc<Mutex<Settlements>>);
struct Permit(Arc<Mutex<Settlements>>);

impl FeaRunPermit for Permit {
    fn budget(&self) -> FeaRunBudget {
        FeaRunBudget {
            budget_ops: ops(50_000_000_000),
            granted_ops: ops(0),
            expires_at_unix_millis: runmat_time::unix_timestamp_ms() as u64 + 60_000,
            heartbeat_millis: 30_000,
        }
    }
    fn heartbeat(&self, _: &FeaRunUsage) -> Result<FeaRunBudget, FeaRunDenied> {
        Ok(self.budget())
    }
    fn settle(
        self: Box<Self>,
        outcome: FeaRunOutcome,
        usage: &FeaRunUsage,
    ) -> Result<(), FeaRunDenied> {
        self.0.lock().unwrap().0.push((outcome, usage.clone()));
        Ok(())
    }
}
impl FeaRunAdmission for Host {
    fn admit(&self, _: &FeaRunRequest) -> Result<Box<dyn FeaRunPermit>, FeaRunDenied> {
        Ok(Box::new(Permit(Arc::clone(&self.0))))
    }
}

fn request(estimate: Option<u64>) -> FeaRunRequest {
    let mut request = FeaRunRequest::local(
        FeaRunKind::MeshAndSolve,
        1000,
        runmat_execution::Digest::sha256(b"estimated study"),
    )
    .unwrap()
    .with_estimate(estimate.map(ops));
    request.site = FeaRunSite::Dispatched;
    request
}

#[test]
fn an_estimate_above_the_allowance_is_refused_before_work_and_charges_nothing() {
    let settled = Arc::new(Mutex::new(Settlements::default()));
    let host = Host(Arc::clone(&settled));
    let denied = FeaRunScope::admit_with(request(Some(61_250_000_000)), &host)
        .err()
        .expect("the estimate exceeds the allowance");
    assert_eq!(
        denied,
        FeaRunDenied::AllowanceInsufficient {
            estimated_ops: ops(61_250_000_000),
            available_ops: ops(50_000_000_000),
        }
    );
    assert_eq!(denied.error_code(), "RM.FEA.AllowanceInsufficient");
    assert_eq!(
        denied.to_string(),
        "This run needs about 61.3 billion operations; 50.0 billion remain in this period's \
         simulation allowance. Coarsen the mesh, or see your simulation allowance and plans at \
         https://runmat.com/pricing"
    );
    let settlements = &settled.lock().unwrap().0;
    assert_eq!(settlements.len(), 1, "the refused lease closes once");
    assert_eq!(settlements[0].0, FeaRunOutcome::Failed);
    assert_eq!(settlements[0].1.completed_ops.total().unwrap(), ops(0));
    // The typed denial crosses the operation wire with both counts.
    let wire = serde_json::to_value(&denied).unwrap();
    assert_eq!(
        wire,
        serde_json::json!({"AllowanceInsufficient": {
            "estimated_ops": "61250000000", "available_ops": "50000000000"
        }})
    );
}

#[test]
fn an_estimate_within_the_allowance_or_absent_is_admitted() {
    let settled = Arc::new(Mutex::new(Settlements::default()));
    let host = Host(Arc::clone(&settled));
    for estimate in [None, Some(50_000_000_000), Some(1)] {
        let scope = FeaRunScope::admit_with(request(estimate), &host).unwrap();
        scope.settle(FeaRunOutcome::Completed).unwrap();
    }
    assert_eq!(settled.lock().unwrap().0.len(), 3);
}

#[test]
fn the_estimate_travels_to_the_host_as_estimate_ops() {
    let wire = serde_json::to_value(request(Some(42))).unwrap();
    assert_eq!(wire["estimateOps"], "42");
    let absent = serde_json::to_value(request(None)).unwrap();
    assert!(absent.get("estimateOps").is_none());
    let decoded: FeaRunRequest = serde_json::from_value(absent).unwrap();
    assert_eq!(decoded.estimate_ops, None);
}
