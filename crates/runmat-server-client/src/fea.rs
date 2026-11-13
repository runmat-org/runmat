//! Local FEA lease contract. Hand-written until the Server OpenAPI handoff.
use chrono::{DateTime, Utc};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

use crate::{
    ids::OrgId,
    lease_http::{LeaseHttp, ServerLeaseTransportError},
};

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum FeaLeaseKind {
    Meshing,
    Solve,
    MeshAndSolve,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StartFeaLease {
    pub run_id: String,
    pub kind: FeaLeaseKind,
    pub allocated_cpu_millicores: u64,
    pub study_identity: String,
    pub client: FeaLeaseClientKind,
}
#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum FeaLeaseClientKind {
    Desktop,
    Browser,
    Cli,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FeaLease {
    pub lease_id: String,
    pub held_core_millis: u64,
    pub expires_at: DateTime<Utc>,
    pub heartbeat_seconds: u64,
}
#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FeaLeaseUsage {
    pub cpu_core_millis: u64,
    pub wall_millis: u64,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FeaLeaseHold {
    pub held_core_millis: u64,
    pub expires_at: DateTime<Utc>,
}
#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum FeaLeaseOutcome {
    Completed,
    Cancelled,
    Failed,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FeaLeaseSettlement {
    pub charged_core_millis: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum FeaLeaseError {
    #[error("FEA requires sign-in")]
    SignInRequired,
    #[error("FEA authorization is invalid")]
    LicenseInvalid,
    #[error("FEA core-hours are used")]
    AllowanceUsed {
        refills_at_unix_seconds: Option<i64>,
    },
    #[error("FEA lease has expired")]
    LeaseExpired,
    #[error("RunMat Server is unreachable")]
    ServerUnreachable,
}

#[derive(Deserialize)]
struct ErrorBody {
    error: ErrorDetail,
}
#[derive(Deserialize)]
struct ErrorDetail {
    code: String,
    details: Option<ErrorDetails>,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ErrorDetails {
    refills_at: Option<DateTime<Utc>>,
}

fn api_error(status: u16, body: &[u8]) -> FeaLeaseError {
    let error = serde_json::from_slice::<ErrorBody>(body)
        .ok()
        .map(|b| b.error);
    match (status, error.as_ref().map(|e| e.code.as_str())) {
        (401, _) => FeaLeaseError::SignInRequired,
        (403 | 422, _) => FeaLeaseError::LicenseInvalid,
        (409, Some("fea.core_hours_used")) => FeaLeaseError::AllowanceUsed {
            refills_at_unix_seconds: error
                .and_then(|e| e.details)
                .and_then(|d| d.refills_at)
                .map(|d| d.timestamp()),
        },
        (404, Some("fea.lease_not_found")) => FeaLeaseError::LeaseExpired,
        (500..=599, _) => FeaLeaseError::ServerUnreachable,
        _ => FeaLeaseError::LicenseInvalid,
    }
}

#[derive(Clone)]
pub struct FeaLeaseClient {
    transport: LeaseHttp,
}
impl FeaLeaseClient {
    pub fn new(server: &str, token: &str) -> Result<Self, FeaLeaseError> {
        Ok(Self {
            transport: LeaseHttp::new(server, token)?,
        })
    }

    async fn response<T: DeserializeOwned>(
        &self,
        request: reqwest::RequestBuilder,
    ) -> Result<T, FeaLeaseError> {
        self.transport.response(request, api_error).await
    }

    pub async fn start(
        &self,
        org: &OrgId,
        request: &StartFeaLease,
    ) -> Result<FeaLease, FeaLeaseError> {
        self.response(
            self.transport
                .http
                .post(format!(
                    "{}/v1/orgs/{}/fea/leases",
                    self.transport.endpoint.as_str(),
                    org.as_str()
                ))
                .json(request),
        )
        .await
    }
    fn lease_url(&self, lease: &str, operation: &str) -> Result<String, FeaLeaseError> {
        // Server identifiers are opaque, never URL path fragments supplied by a caller.
        let suffix = lease
            .strip_prefix("fealease_")
            .ok_or(FeaLeaseError::LicenseInvalid)?;
        uuid::Uuid::parse_str(suffix).map_err(|_| FeaLeaseError::LicenseInvalid)?;
        Ok(format!(
            "{}/v1/fea/leases/{lease}/{operation}",
            self.transport.endpoint.as_str()
        ))
    }
    pub async fn usage(
        &self,
        lease: &str,
        usage: FeaLeaseUsage,
    ) -> Result<FeaLeaseHold, FeaLeaseError> {
        self.response(
            self.transport
                .http
                .post(self.lease_url(lease, "usage")?)
                .json(&usage),
        )
        .await
    }
    pub async fn settle(
        &self,
        lease: &str,
        outcome: FeaLeaseOutcome,
        usage: FeaLeaseUsage,
    ) -> Result<FeaLeaseSettlement, FeaLeaseError> {
        #[derive(Serialize)]
        struct Settlement {
            outcome: FeaLeaseOutcome,
            #[serde(flatten)]
            usage: FeaLeaseUsage,
        }
        self.response(
            self.transport
                .http
                .post(self.lease_url(lease, "settle")?)
                .json(&Settlement { outcome, usage }),
        )
        .await
    }

    /// With no saved/explicit organization, only an unambiguous membership is selected.
    pub async fn sole_organization(&self) -> Result<OrgId, FeaLeaseError> {
        #[derive(Deserialize)]
        struct Org {
            id: OrgId,
        }
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct Page {
            items: Vec<Org>,
            next_cursor: Option<String>,
        }
        let page: Page = self
            .response(
                self.transport
                    .http
                    .get(format!("{}/v1/orgs", self.transport.endpoint.as_str()))
                    .query(&[("limit", "2")]),
            )
            .await?;
        if page.next_cursor.is_some() || page.items.len() != 1 {
            return Err(FeaLeaseError::LicenseInvalid);
        }
        Ok(page.items.into_iter().next().expect("one organization").id)
    }
}

impl From<ServerLeaseTransportError> for FeaLeaseError {
    fn from(error: ServerLeaseTransportError) -> Self {
        match error {
            ServerLeaseTransportError::Http(_) | ServerLeaseTransportError::ServerStatus(_) => {
                Self::ServerUnreachable
            }
            ServerLeaseTransportError::Endpoint(_)
            | ServerLeaseTransportError::Credential(_)
            | ServerLeaseTransportError::BodyTooLarge
            | ServerLeaseTransportError::Decode(_) => Self::LicenseInvalid,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn typed_denials_preserve_status_code_and_refill() {
        assert_eq!(api_error(409, br#"{"error":{"code":"fea.core_hours_used","details":{"refillsAt":"2026-11-02T00:00:00Z"}}}"#), FeaLeaseError::AllowanceUsed { refills_at_unix_seconds: Some(1793577600) });
        assert_eq!(
            api_error(404, br#"{"error":{"code":"fea.lease_not_found"}}"#),
            FeaLeaseError::LeaseExpired
        );
        assert_eq!(
            api_error(409, br#"{"error":{"code":"other"}}"#),
            FeaLeaseError::LicenseInvalid
        );
        assert_eq!(api_error(401, b""), FeaLeaseError::SignInRequired);
        assert_eq!(api_error(503, b""), FeaLeaseError::ServerUnreachable);
        assert_eq!(api_error(422, b""), FeaLeaseError::LicenseInvalid);
    }
    #[test]
    fn usage_is_cumulative_cpu_and_wall_only() {
        assert_eq!(
            serde_json::to_value(FeaLeaseUsage {
                cpu_core_millis: 2500,
                wall_millis: 1250
            })
            .unwrap(),
            serde_json::json!({"cpuCoreMillis":2500,"wallMillis":1250})
        );
    }

    #[tokio::test]
    async fn lease_contract_sends_original_run_and_cumulative_usage_on_all_three_routes() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let lease = "fealease_00000000-0000-0000-0000-000000000001";
        let server = tokio::spawn(async move {
            let mut bodies = Vec::new();
            for (path, response) in [
                (
                    "/v1/orgs/org_00000000000000000000000000000001/fea/leases".to_owned(),
                    format!(
                        r#"{{"leaseId":"{lease}","heldCoreMillis":1,"expiresAt":"2099-01-01T00:00:00Z","heartbeatSeconds":30}}"#
                    ),
                ),
                (
                    format!("/v1/fea/leases/{lease}/usage"),
                    r#"{"heldCoreMillis":2,"expiresAt":"2099-01-01T00:00:00Z"}"#.to_owned(),
                ),
                (
                    format!("/v1/fea/leases/{lease}/settle"),
                    r#"{"chargedCoreMillis":2}"#.to_owned(),
                ),
            ] {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut bytes = Vec::new();
                let header_end = loop {
                    let mut chunk = [0; 1024];
                    let count = socket.read(&mut chunk).await.unwrap();
                    assert_ne!(count, 0);
                    bytes.extend_from_slice(&chunk[..count]);
                    assert!(bytes.len() < 16384);
                    if let Some(end) = bytes.windows(4).position(|v| v == b"\r\n\r\n") {
                        break end + 4;
                    }
                };
                let headers = std::str::from_utf8(&bytes[..header_end]).unwrap();
                assert!(headers.starts_with(&format!("POST {path} HTTP/1.1\r\n")));
                assert!(headers
                    .to_ascii_lowercase()
                    .contains("authorization: bearer test-key\r\n"));
                let length: usize = headers
                    .lines()
                    .find_map(|line| {
                        line.to_ascii_lowercase()
                            .strip_prefix("content-length:")
                            .map(|v| v.trim().parse().unwrap())
                    })
                    .unwrap();
                while bytes.len() < header_end + length {
                    let mut chunk = [0; 1024];
                    let count = socket.read(&mut chunk).await.unwrap();
                    assert_ne!(count, 0);
                    bytes.extend_from_slice(&chunk[..count]);
                }
                bodies.push(
                    serde_json::from_slice::<serde_json::Value>(
                        &bytes[header_end..header_end + length],
                    )
                    .unwrap(),
                );
                socket.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{response}", response.len()).as_bytes()).await.unwrap();
            }
            bodies
        });
        let client = FeaLeaseClient::new(&format!("http://{address}"), "test-key").unwrap();
        let request = StartFeaLease {
            run_id: "run_00000000000000000000000000000001".into(),
            kind: FeaLeaseKind::MeshAndSolve,
            allocated_cpu_millicores: 2000,
            study_identity: "sha256:study".into(),
            client: FeaLeaseClientKind::Cli,
        };
        let admitted = client
            .start(&OrgId::from_uuid(uuid::Uuid::from_u128(1)), &request)
            .await
            .unwrap();
        assert_eq!(admitted.held_core_millis, 1);
        let cumulative = FeaLeaseUsage {
            cpu_core_millis: 2,
            wall_millis: 1,
        };
        assert_eq!(
            client
                .usage(&admitted.lease_id, cumulative)
                .await
                .unwrap()
                .held_core_millis,
            2
        );
        assert_eq!(
            client
                .settle(&admitted.lease_id, FeaLeaseOutcome::Completed, cumulative)
                .await
                .unwrap()
                .charged_core_millis,
            2
        );
        let bodies = server.await.unwrap();
        assert_eq!(bodies[0]["runId"], request.run_id);
        assert_eq!(bodies[0]["client"], "cli");
        assert_eq!(
            bodies[1],
            serde_json::json!({"cpuCoreMillis":2,"wallMillis":1})
        );
        assert_eq!(
            bodies[2],
            serde_json::json!({"outcome":"completed","cpuCoreMillis":2,"wallMillis":1})
        );
    }
}
