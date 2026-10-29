use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum LeaseKind {
    Meshing,
    Solve,
    MeshAndSolve,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ClientSurface {
    Cli,
    Desktop,
    Browser,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum LeaseOutcome {
    Completed,
    Cancelled,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateLeaseRequest {
    pub idempotency_key: String,
    pub allocated_cpu_millicores: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub estimated_wall_seconds: Option<u64>,
    pub kind: LeaseKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub study_identity: Option<String>,
    pub client_surface: ClientSurface,
}

impl CreateLeaseRequest {
    pub(super) fn valid(&self) -> bool {
        !self.idempotency_key.is_empty()
            && self.idempotency_key.len() <= 128
            && self
                .idempotency_key
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"._:-".contains(&b))
            && (1..=4_096_000).contains(&self.allocated_cpu_millicores)
            && self
                .estimated_wall_seconds
                .is_none_or(|v| (1..=604_800).contains(&v))
            && self
                .study_identity
                .as_ref()
                .is_none_or(|v| v.chars().count() <= 256)
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageReport {
    pub wall_millis: u64,
    pub cpu_core_millis: u64,
}

impl UsageReport {
    pub(super) fn cumulative(self, other: Self) -> Self {
        Self {
            wall_millis: self.wall_millis.max(other.wall_millis),
            cpu_core_millis: self.cpu_core_millis.max(other.cpu_core_millis),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct Settlement {
    pub outcome: LeaseOutcome,
    #[serde(flatten)]
    pub usage: UsageReport,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, thiserror::Error)]
pub enum AdmissionError {
    #[error("Sign in to start an FEA run")]
    SignInRequired,
    #[error("Core-hour allowance used")]
    AllowanceUsed {
        refills_at_unix_seconds: Option<i64>,
    },
    #[error("Core-hour lease expired")]
    LeaseExpired,
    #[error("Invalid core-hour lease")]
    LicenseInvalid,
    #[error("Unable to contact RunMat Server")]
    ServerUnreachable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeasePermit {
    pub lease_id: String,
    pub idempotency_key: String,
    pub(super) org_id: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) enum LeaseStatus {
    Active,
    Settled,
    Released,
    Expired,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct LeaseResponse {
    pub lease_id: String,
    pub org_id: String,
    pub user_id: String,
    pub idempotency_key: String,
    pub status: LeaseStatus,
    pub allocated_cpu_millicores: u32,
    pub reserved_core_micro_hours: u64,
    pub used_core_micro_hours: u64,
    pub charged_core_micro_hours: Option<u64>,
    pub overage_core_micro_hours: u64,
    pub outcome: Option<LeaseOutcome>,
    pub issued_at: chrono::DateTime<chrono::Utc>,
    pub expires_at: chrono::DateTime<chrono::Utc>,
    pub refills_at: Option<chrono::DateTime<chrono::Utc>>,
    pub token: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(super) struct SigningKeys {
    pub keys: Vec<SigningKey>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct SigningKey {
    pub key_id: String,
    pub algorithm: String,
    pub public_key: String,
}

#[derive(Debug, Deserialize)]
pub(super) struct ApiErrorEnvelope {
    pub error: ApiError,
}

#[derive(Debug, Deserialize)]
pub(super) struct ApiError {
    pub code: ServerErrorCode,
    pub details: Option<ErrorDetails>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct ErrorDetails {
    pub refills_at: Option<chrono::DateTime<chrono::Utc>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub(super) enum ServerErrorCode {
    #[serde(rename = "auth.invalid_token")]
    InvalidToken,
    #[serde(rename = "policy.denied")]
    PolicyDenied,
    #[serde(rename = "core_hour_lease.user_required")]
    UserRequired,
    #[serde(rename = "core_hours_exhausted")]
    CoreHoursExhausted,
    #[serde(rename = "license.invalid")]
    LicenseInvalid,
    #[serde(rename = "core_hour_lease.invalid_request")]
    InvalidRequest,
    #[serde(rename = "core_hour_lease.idempotency_conflict")]
    IdempotencyConflict,
    #[serde(rename = "core_hour_lease.not_found")]
    NotFound,
    #[serde(rename = "core_hour_lease.signing_unavailable")]
    SigningUnavailable,
    #[serde(other)]
    Unknown,
}
