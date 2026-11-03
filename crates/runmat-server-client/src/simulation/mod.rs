//! Exact simulation-operation lease transport. The numerical owner retains the
//! run ledger, execution authority and recovery identities; this client sends
//! typed snapshots and grants without converting physical-resource telemetry.

mod errors;
mod types;
mod allowance;
pub use allowance::{
    SimulationAllowance, SimulationBudgetError, SimulationPeriod, SimulationRenewal,
};
pub use errors::{SimulationErrorCode, SimulationErrorDetails, SimulationLeaseError};
pub use types::*;

use crate::{
    ids::{OrgId, SimulationLeaseId},
    lease_http::LeaseHttp,
};
use chrono::Utc;

#[derive(Clone)]
pub struct SimulationLeaseClient {
    transport: LeaseHttp,
}

impl SimulationLeaseClient {
    pub fn new(server: &str, token: &str) -> Result<Self, SimulationLeaseError> {
        Ok(Self {
            transport: LeaseHttp::new(server, token)?,
        })
    }

    pub async fn allowance(
        &self,
        org: &OrgId,
        runmat_version: &str,
    ) -> Result<SimulationAllowance, SimulationLeaseError> {
        self.transport
            .response(
                self.transport
                    .http
                    .get(format!(
                        "{}/v1/orgs/{org}/simulation/usage",
                        self.transport.endpoint.as_str()
                    ))
                    .query(&[("runmatVersion", runmat_version)]),
                errors::api_error,
            )
            .await
    }

    /// Resolve only an unambiguous membership when no organization is selected.
    pub async fn sole_organization(&self) -> Result<OrgId, SimulationLeaseError> {
        #[derive(serde::Deserialize)]
        struct Org {
            id: OrgId,
        }
        #[derive(serde::Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct Page {
            items: Vec<Org>,
            next_cursor: Option<String>,
        }
        let page: Page = self
            .transport
            .response(
                self.transport
                    .http
                    .get(format!("{}/v1/orgs", self.transport.endpoint.as_str()))
                    .query(&[("limit", "2")]),
                errors::api_error,
            )
            .await?;
        if page.next_cursor.is_some() || page.items.len() != 1 {
            return Err(SimulationLeaseError::InvalidRequest(
                "select an organization explicitly",
            ));
        }
        Ok(page.items.into_iter().next().expect("one organization").id)
    }

    pub async fn start(
        &self,
        org: &OrgId,
        request: &StartSimulationLease,
    ) -> Result<SimulationLease, SimulationLeaseError> {
        if !(1..=1_000_000).contains(&request.attempt)
            || request
                .budget_ops
                .is_some_and(|budget| budget.operations() == 0)
            || request.runmat_version.is_empty()
            || request.runmat_version.len() > 64
        {
            return Err(SimulationLeaseError::InvalidRequest(
                "attempt, budget or runtime version",
            ));
        }
        let lease: SimulationLease = self
            .transport
            .response(
                self.transport
                    .http
                    .post(format!(
                        "{}/v1/orgs/{org}/simulation/leases",
                        self.transport.endpoint.as_str()
                    ))
                    .json(request),
                errors::api_error,
            )
            .await?;
        if lease.state != SimulationLeaseState::Active
            || request
                .budget_ops
                .is_some_and(|budget| lease.budget_ops != budget)
            || lease.granted_ops.operations() == 0
            || lease.granted_ops > lease.budget_ops
            || !(1..=120).contains(&lease.heartbeat_seconds)
            || lease.expires_at <= Utc::now()
        {
            return Err(SimulationLeaseError::InvalidResponse(
                "grant, spending cap or lease liveness",
            ));
        }
        Ok(lease)
    }

    pub async fn usage(
        &self,
        lease: &SimulationLeaseId,
        request: &ReportSimulationUsage,
    ) -> Result<SimulationGrant, SimulationLeaseError> {
        validate_detail(request.other_detail.as_deref())?;
        let grant: SimulationGrant = self
            .transport
            .response(
                self.transport
                    .http
                    .post(format!(
                        "{}/v1/simulation/leases/{lease}/usage",
                        self.transport.endpoint.as_str()
                    ))
                    .json(request),
                errors::api_error,
            )
            .await?;
        if grant.state != SimulationLeaseState::Active
            || grant.granted_ops.operations() == 0
            || grant.granted_ops > grant.budget_ops
            || grant.total_ops > runmat_simulation_usage::OperationTotal::from(grant.granted_ops)
            || grant.total_ops
                != request.used_ops.wide_total().map_err(|_| {
                    SimulationLeaseError::InvalidRequest("completed operation total")
                })?
            || grant.expires_at <= Utc::now()
        {
            return Err(SimulationLeaseError::InvalidResponse(
                "usage grant, reported total or lease liveness",
            ));
        }
        Ok(grant)
    }

    pub async fn settle(
        &self,
        lease: &SimulationLeaseId,
        request: &SettleSimulationLease,
    ) -> Result<SimulationReceipt, SimulationLeaseError> {
        validate_detail(request.other_detail.as_deref())?;
        let receipt: SimulationReceipt = self
            .transport
            .response(
                self.transport
                    .http
                    .post(format!(
                        "{}/v1/simulation/leases/{lease}/settle",
                        self.transport.endpoint.as_str()
                    ))
                    .json(request),
                errors::api_error,
            )
            .await?;
        let total = request
            .used_ops
            .wide_total()
            .map_err(|_| SimulationLeaseError::InvalidRequest("completed operation total"))?;
        let outcome = match request.outcome {
            SimulationOutcome::Completed => SimulationReceiptOutcome::Completed,
            SimulationOutcome::Cancelled => SimulationReceiptOutcome::Cancelled,
            SimulationOutcome::Failed => SimulationReceiptOutcome::Failed,
        };
        if &receipt.lease_id != lease
            || receipt.measurement_version != request.measurement_version
            || receipt.used_ops != request.used_ops
            || receipt.outcome != outcome
            || receipt.total_ops != total
            || receipt.granted_ops > receipt.budget_ops
            || total > runmat_simulation_usage::OperationTotal::from(receipt.granted_ops)
            || receipt.charged_ops
                != if receipt.waived {
                    runmat_simulation_usage::OperationTotal::ZERO
                } else {
                    total
                }
        {
            return Err(SimulationLeaseError::InvalidResponse(
                "settlement identity, completed work or charge",
            ));
        }
        Ok(receipt)
    }
}

fn validate_detail(detail: Option<&str>) -> Result<(), SimulationLeaseError> {
    if detail.is_some_and(|value| value.trim().is_empty() || value.chars().count() > 256) {
        return Err(SimulationLeaseError::InvalidRequest(
            "otherDetail must contain 1–256 characters",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
