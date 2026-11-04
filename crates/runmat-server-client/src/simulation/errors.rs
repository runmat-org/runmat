use crate::ServerLeaseTransportError;
use chrono::{DateTime, Utc};
use runmat_simulation_usage::{OperationCount, OperationTotal};
use serde::Deserialize;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize)]
pub enum SimulationErrorCode {
    #[serde(rename = "auth.invalid_token")]
    SignInRequired,
    #[serde(rename = "org.forbidden")]
    Forbidden,
    #[serde(rename = "simulation.allowance_used")]
    AllowanceUsed,
    #[serde(rename = "simulation.allowance_unpublished")]
    AllowanceUnpublished,
    #[serde(rename = "simulation.over_grant")]
    OverGrant,
    #[serde(rename = "simulation.budget_reached")]
    BudgetReached,
    #[serde(rename = "simulation.budget_exceeds_available")]
    BudgetExceedsAvailable,
    #[serde(rename = "simulation.lease_not_found")]
    LeaseNotFound,
    #[serde(rename = "simulation.receipt_not_found")]
    ReceiptNotFound,
    #[serde(rename = "simulation.lease_conflict")]
    LeaseConflict,
    #[serde(rename = "simulation.settle_conflict")]
    SettleConflict,
    #[serde(rename = "simulation.invalid_request")]
    InvalidRequest,
    #[serde(rename = "simulation.driver_authority_invalid")]
    DriverAuthorityInvalid,
    #[serde(rename = "simulation.billing_person_required")]
    BillingPersonRequired,
    #[serde(other)]
    Unknown,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SimulationErrorDetails {
    pub refills_at: Option<DateTime<Utc>>,
    pub granted_ops: Option<OperationCount>,
    pub total_ops: Option<OperationTotal>,
    pub budget_ops: Option<OperationCount>,
    pub available_ops: Option<OperationCount>,
}

#[derive(Debug, thiserror::Error)]
pub enum SimulationLeaseError {
    #[error("invalid simulation lease request: {0}")]
    InvalidRequest(&'static str),
    #[error("invalid simulation lease response: {0}")]
    InvalidResponse(&'static str),
    #[error(transparent)]
    Transport(#[from] ServerLeaseTransportError),
    #[error("RunMat Server refused simulation admission: HTTP {status}, {code:?}")]
    Rejected {
        status: u16,
        code: SimulationErrorCode,
        message: Option<String>,
        details: SimulationErrorDetails,
    },
}

#[derive(Deserialize)]
struct ErrorBody {
    error: ErrorDetail,
}
#[derive(Deserialize)]
struct ErrorDetail {
    code: SimulationErrorCode,
    message: Option<String>,
    details: Option<SimulationErrorDetails>,
}

pub(super) fn api_error(status: u16, bytes: &[u8]) -> SimulationLeaseError {
    let error = serde_json::from_slice::<ErrorBody>(bytes).map(|body| body.error);
    match error {
        Ok(error) => SimulationLeaseError::Rejected {
            status,
            code: error.code,
            message: error.message,
            details: error.details.unwrap_or_default(),
        },
        Err(_) if status == 401 => SimulationLeaseError::Rejected {
            status,
            code: SimulationErrorCode::SignInRequired,
            message: None,
            details: SimulationErrorDetails::default(),
        },
        Err(error) => SimulationLeaseError::Transport(ServerLeaseTransportError::Decode(error)),
    }
}
