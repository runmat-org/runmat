//! Operation-based local test Server. No production account is charged.
use runmat_execution::Digest;
use runmat_server_client::{
    ids::SimulationRunId,
    simulation::{SimulationClientKind, SimulationKind, SimulationOutcome},
};
use runmat_simulation_usage::{CompletedUsage, MeasurementVersion, OperationCount};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;

const AVAILABLE: u64 = 1_000_000_000_000;
const INITIAL: u64 = 1_000_000;
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Start {
    run_id: SimulationRunId,
    attempt: u32,
    measurement_version: MeasurementVersion,
    runmat_version: String,
    kind: SimulationKind,
    client: SimulationClientKind,
    study_identity: Digest,
    estimate_ops: Option<OperationCount>,
    budget_ops: Option<OperationCount>,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Report {
    measurement_version: MeasurementVersion,
    used_ops: CompletedUsage,
    grant_target_ops: Option<OperationCount>,
    other_detail: Option<String>,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Settle {
    measurement_version: MeasurementVersion,
    used_ops: CompletedUsage,
    outcome: SimulationOutcome,
    other_detail: Option<String>,
}
pub(super) struct Lease {
    request: Start,
    budget: OperationCount,
    grant: OperationCount,
    completed: CompletedUsage,
    outcome: Option<SimulationOutcome>,
    other_detail: Option<String>,
    expires_at: chrono::DateTime<chrono::Utc>,
    settled_at: Option<chrono::DateTime<chrono::Utc>>,
}
fn expiry() -> chrono::DateTime<chrono::Utc> {
    chrono::Utc::now() + chrono::Duration::hours(24)
}
fn error(status: &'static str, code: &str) -> (&'static str, Value) {
    (status, json!({"error":{"code":code}}))
}
pub(super) fn invalid() -> (&'static str, Value) {
    error("422 Unprocessable Entity", "simulation.invalid_request")
}

pub(super) fn handle(
    path: &str,
    body: Value,
    leases: &mut HashMap<String, Lease>,
) -> (&'static str, Value) {
    if path == format!("/v1/orgs/{}/simulation/leases", super::ORG) {
        let Ok(request) = serde_json::from_value::<Start>(body) else {
            return invalid();
        };
        if request.measurement_version != MeasurementVersion::reference_v1()
            || request.runmat_version.is_empty()
        {
            return invalid();
        }
        let budget = request
            .budget_ops
            .unwrap_or(OperationCount::from_operations(AVAILABLE));
        if budget.operations() == 0 {
            return error("409 Conflict", "simulation.allowance_used");
        }
        if budget.operations() > AVAILABLE {
            return (
                "409 Conflict",
                json!({"error":{"code":"simulation.budget_exceeds_available","details":{"availableOps":AVAILABLE.to_string()}}}),
            );
        }
        let id = leases
            .iter()
            .find(|(_, lease)| {
                lease.request.run_id == request.run_id && lease.request.attempt == request.attempt
            })
            .map(|(id, _)| id.clone())
            .unwrap_or_else(|| format!("simlease_{:032x}", leases.len() + 1));
        assert!(leases.len() < 128);
        let lease = leases.entry(id.clone()).or_insert_with(|| Lease {
            request: request.clone(),
            budget,
            grant: OperationCount::from_operations(INITIAL.min(budget.operations())),
            completed: CompletedUsage::default(),
            outcome: None,
            other_detail: None,
            expires_at: expiry(),
            settled_at: None,
        });
        if lease.request != request {
            return error("409 Conflict", "simulation.lease_conflict");
        }
        return (
            "200 OK",
            json!({"leaseId":id,"state":if lease.outcome.is_some(){"settled"}else if chrono::Utc::now() >= lease.expires_at {"expired"}else{"active"},"grantedOps":lease.grant,"budgetOps":lease.budget,"expiresAt":lease.expires_at,"heartbeatSeconds":30,"waived":false}),
        );
    }
    let Some((id, action)) = path
        .strip_prefix("/v1/simulation/leases/")
        .and_then(|tail| tail.split_once('/'))
    else {
        return error("404 Not Found", "simulation.lease_not_found");
    };
    let Some(lease) = leases.get_mut(id) else {
        return error("404 Not Found", "simulation.lease_not_found");
    };
    match action {
        "usage" if lease.outcome.is_none() && chrono::Utc::now() < lease.expires_at => {
            let Ok(report) = serde_json::from_value::<Report>(body) else {
                return invalid();
            };
            if report.measurement_version != lease.request.measurement_version
                || !report.used_ops.is_monotonic_since(lease.completed)
            {
                return invalid();
            }
            let Ok(total) = report.used_ops.total() else {
                return invalid();
            };
            if total > lease.grant {
                return invalid();
            }
            if let Some(target) = report.grant_target_ops {
                if target > lease.budget {
                    lease.completed = report.used_ops;
                    if report.other_detail.is_some() {
                        lease.other_detail = report.other_detail;
                    }
                    lease.expires_at = expiry();
                    return error("409 Conflict", "simulation.budget_reached");
                }
                lease.grant = lease.grant.max(target);
            }
            lease.completed = report.used_ops;
            if report.other_detail.is_some() {
                lease.other_detail = report.other_detail;
            }
            lease.expires_at = expiry();
            (
                "200 OK",
                json!({"state":"active","grantedOps":lease.grant,"budgetOps":lease.budget,"totalOps":total,"expiresAt":lease.expires_at}),
            )
        }
        "settle" => {
            let Ok(report) = serde_json::from_value::<Settle>(body) else {
                return invalid();
            };
            if lease.outcome.is_some_and(|outcome| {
                outcome != report.outcome
                    || lease.completed != report.used_ops
                    || lease.request.measurement_version != report.measurement_version
                    || (report.other_detail.is_some() && lease.other_detail != report.other_detail)
            }) {
                return error("409 Conflict", "simulation.settle_conflict");
            }
            if lease.outcome.is_none() && chrono::Utc::now() >= lease.expires_at {
                return error("404 Not Found", "simulation.lease_not_found");
            }
            if report.measurement_version != lease.request.measurement_version
                || !report.used_ops.is_monotonic_since(lease.completed)
            {
                return invalid();
            }
            let Ok(total) = report.used_ops.total() else {
                return invalid();
            };
            if total > lease.grant {
                return invalid();
            }
            lease.completed = report.used_ops;
            lease.outcome = Some(report.outcome);
            if report.other_detail.is_some() {
                lease.other_detail = report.other_detail;
            }
            let settled_at = *lease.settled_at.get_or_insert_with(chrono::Utc::now);
            (
                "200 OK",
                json!({"receiptId":format!("simreceipt_{}",id.strip_prefix("simlease_").unwrap()),"leaseId":id,"orgId":super::ORG,"runId":lease.request.run_id,"attempt":lease.request.attempt,"measurementVersion":lease.request.measurement_version,"kind":lease.request.kind,"client":lease.request.client,"outcome":report.outcome,"usedOps":lease.completed,"otherDetail":lease.other_detail,"totalOps":total,"grantedOps":lease.grant,"budgetOps":lease.budget,"estimateOps":lease.request.estimate_ops,"chargedOps":total,"waived":false,"settledAt":settled_at}),
            )
        }
        _ => error("404 Not Found", "simulation.lease_not_found"),
    }
}
