//! The host owns authentication; the domain client owns the lease protocol and journal.
use super::{CreateLeaseRequest, LeaseOutcome, UsageReport};
use futures::future::BoxFuture;
use serde_json::Value;
use std::time::Duration;

/// The complete set of lease operations. Hosts cannot supply arbitrary routes or bodies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LeaseTransportRequest {
    SigningKeys,
    Create {
        org_id: String,
        request: CreateLeaseRequest,
    },
    Heartbeat {
        org_id: String,
        lease_id: String,
        usage: UsageReport,
    },
    Settle {
        org_id: String,
        lease_id: String,
        outcome: LeaseOutcome,
        usage: UsageReport,
    },
}

impl LeaseTransportRequest {
    /// Absolute API path, with identifiers encoded as individual path segments.
    pub fn path(&self) -> String {
        let mut url = url::Url::parse("https://lease.invalid").expect("constant URL");
        {
            let mut path = url.path_segments_mut().expect("hierarchical URL");
            path.clear();
            match self {
                Self::SigningKeys => {
                    path.extend(["v1", "execution", "core-hour-lease-keys"]);
                }
                Self::Create { org_id, .. } => {
                    path.extend(["v1", "orgs", org_id, "execution", "core-hour-leases"]);
                }
                Self::Heartbeat {
                    org_id, lease_id, ..
                }
                | Self::Settle {
                    org_id, lease_id, ..
                } => {
                    path.extend([
                        "v1",
                        "orgs",
                        org_id,
                        "execution",
                        "core-hour-leases",
                        lease_id,
                        if matches!(self, Self::Heartbeat { .. }) {
                            "heartbeat"
                        } else {
                            "settle"
                        },
                    ]);
                }
            }
        }
        url.path().into()
    }

    /// None means GET; every authenticated operation is POST with this exact JSON body.
    pub fn body(&self) -> Option<Value> {
        match self {
            Self::SigningKeys => None,
            Self::Create { request, .. } => {
                Some(serde_json::to_value(request).expect("lease request serialization"))
            }
            Self::Heartbeat { usage, .. } => {
                Some(serde_json::to_value(usage).expect("usage serialization"))
            }
            Self::Settle { outcome, usage, .. } => Some(serde_json::json!({
                "outcome":outcome, "wallMillis":usage.wall_millis, "cpuCoreMillis":usage.cpu_core_millis,
            })),
        }
    }

    pub fn requires_auth(&self) -> bool {
        !matches!(self, Self::SigningKeys)
    }
}

#[derive(Debug, Clone)]
pub struct LeaseHttpResponse {
    pub status: u16,
    pub body: Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum LeaseTransportError {
    #[error("Sign in to RunMat Server")]
    SignedOut,
    #[error("RunMat Server is unreachable")]
    Unreachable,
}

/// Implementations may refresh credentials for every authenticated request. They must
/// use the configured Server, enforce TLS, and must not follow cross-origin redirects.
/// Never put authentication secrets into requests, responses, or diagnostic errors here.
pub trait CoreHourLeaseTransport: Send + Sync {
    fn request(
        &self,
        request: LeaseTransportRequest,
    ) -> BoxFuture<'_, Result<LeaseHttpResponse, LeaseTransportError>>;
}

pub(super) struct ReqwestLeaseTransport {
    http: reqwest::Client,
    server: String,
    credential: String,
}

impl ReqwestLeaseTransport {
    pub fn new(server: String, credential: String) -> anyhow::Result<Self> {
        Ok(Self {
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(10))
                .redirect(reqwest::redirect::Policy::none())
                .build()?,
            server: server.trim_end_matches('/').into(),
            credential,
        })
    }
}

impl CoreHourLeaseTransport for ReqwestLeaseTransport {
    fn request(
        &self,
        request: LeaseTransportRequest,
    ) -> BoxFuture<'_, Result<LeaseHttpResponse, LeaseTransportError>> {
        Box::pin(async move {
            let url = format!("{}{}", self.server, request.path());
            let mut http = match request.body() {
                Some(body) => self.http.post(url).json(&body),
                None => self.http.get(url),
            };
            if request.requires_auth() {
                if self.credential.is_empty() {
                    return Err(LeaseTransportError::SignedOut);
                }
                http = http.bearer_auth(&self.credential);
            }
            let response = http
                .send()
                .await
                .map_err(|_| LeaseTransportError::Unreachable)?;
            let status = response.status().as_u16();
            // Malformed bodies are retained as null, so the domain rejects successful
            // admissions and does not acknowledge malformed usage receipts.
            let body = response.json().await.unwrap_or(Value::Null);
            Ok(LeaseHttpResponse { status, body })
        })
    }
}
