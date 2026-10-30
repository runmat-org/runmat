//! Local FEA core-hour admission and durable, cumulative usage delivery.
//!
//! All network work is async. Synchronous hosts should own a worker runtime rather than
//! blocking a runtime thread. After admission, reporting failures must never abort a run.
mod journal;
mod cleanup;
mod token;
mod transport;
mod types;
pub use transport::{
    CoreHourLeaseTransport, LeaseHttpResponse, LeaseTransportError, LeaseTransportRequest,
};
pub use types::{
    AdmissionError, ClientSurface, CreateLeaseRequest, LeaseKind, LeaseOutcome, LeasePermit,
    UsageReport,
};

use anyhow::Result;
use journal::{Entry, Journal};
use sha2::{Digest, Sha256};
use std::{path::PathBuf, sync::Arc};
use types::*;

#[derive(Clone)]
pub struct CoreHourLeaseClient {
    transport: Arc<dyn CoreHourLeaseTransport>,
    known_signed_out: bool,
    org: String,
    journal: Journal,
    // Prevent duplicate in-process create calls and replay races without blocking executor threads.
    admission: Arc<tokio::sync::Mutex<()>>,
    delivery: Arc<tokio::sync::Mutex<()>>,
}

impl CoreHourLeaseClient {
    /// Resolve the same credentials, organization and private-server override as other CLI commands.
    /// Never launches interactive login.
    pub async fn from_environment() -> Result<Self, AdmissionError> {
        let mut config =
            crate::auth::RemoteConfig::load().map_err(|_| AdmissionError::SignInRequired)?;
        let server = crate::auth::resolve_server_url(&config, None)
            .map_err(|_| AdmissionError::LicenseInvalid)?;
        let credential = crate::auth::resolve_auth_token(&mut config, &server)
            .await
            .map_err(|_| AdmissionError::SignInRequired)?;
        Self::from_auth_context(&config, &server, &credential)
    }

    /// Reuse an already-resolved command credential without resolving auth again.
    /// This constructor does no network I/O and never starts interactive sign-in.
    pub fn from_auth_context(
        config: &crate::auth::RemoteConfig,
        server: &str,
        credential: &str,
    ) -> Result<Self, AdmissionError> {
        let org = crate::auth::resolve_public_org_id(config, None)
            .map_err(|_| AdmissionError::SignInRequired)?
            .to_string();
        let config = crate::auth::config_path().map_err(|_| AdmissionError::LicenseInvalid)?;
        let scope = hex::encode(Sha256::digest(
            format!("{}\n{}", server.trim_end_matches('/'), org).as_bytes(),
        ));
        let path = config
            .parent()
            .ok_or(AdmissionError::LicenseInvalid)?
            .join("core-hour-usage")
            .join(format!("{scope}.json"));
        Self::new(server.to_owned(), org, credential.to_owned(), path)
            .map_err(|_| AdmissionError::LicenseInvalid)
    }

    /// Explicit context constructor, also useful for embedded hosts and contract stubs.
    pub fn new(
        server: String,
        org: String,
        credential: String,
        journal_path: PathBuf,
    ) -> Result<Self> {
        let known_signed_out = credential.is_empty();
        let transport = Arc::new(transport::ReqwestLeaseTransport::new(
            server.clone(),
            credential,
        )?);
        let mut client = Self::with_transport(server, org, journal_path, transport)?;
        client.known_signed_out = known_signed_out;
        Ok(client)
    }

    /// Hosts with a refreshing credential broker inject transport without exposing secrets.
    /// `server` and `org` identify this journal's domain; the transport must use that Server.
    pub fn with_transport(
        server: String,
        org: String,
        journal_path: PathBuf,
        transport: Arc<dyn CoreHourLeaseTransport>,
    ) -> Result<Self> {
        let url = url::Url::parse(&server)?;
        // Real Server credentials and verification keys must travel over TLS.
        // Plain HTTP is only for an explicitly loopback development/stub endpoint;
        // do not resolve names or grant exceptions to private-network addresses.
        let loopback = match url.host() {
            Some(url::Host::Domain(host)) => host == "localhost",
            Some(url::Host::Ipv4(address)) => address.is_loopback(),
            Some(url::Host::Ipv6(address)) => address.is_loopback(),
            None => false,
        };
        anyhow::ensure!(
            url.scheme() == "https" || (url.scheme() == "http" && loopback),
            "RunMat Server requires HTTPS (HTTP is only allowed for loopback development endpoints)"
        );
        anyhow::ensure!(
            !org.is_empty() && org.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_'),
            "Invalid organization id"
        );
        Ok(Self {
            transport,
            known_signed_out: false,
            org,
            journal: Journal::new(journal_path),
            admission: Arc::new(tokio::sync::Mutex::new(())),
            delivery: Arc::new(tokio::sync::Mutex::new(())),
        })
    }

    pub async fn admit(&self, request: CreateLeaseRequest) -> Result<LeasePermit, AdmissionError> {
        if self.known_signed_out {
            return Err(AdmissionError::SignInRequired);
        }
        if !request.valid() {
            return Err(AdmissionError::LicenseInvalid);
        }
        let _guard = self.admission.lock().await;
        // Opportunistic replay is independent of admission; a disconnected existing run can finish.
        let _ = self.replay_pending().await;
        let state = self
            .journal
            .read()
            .map_err(|_| AdmissionError::LicenseInvalid)?;
        if let Some(rejected) = state.rejected_creates.get(&request.idempotency_key) {
            return Err(rejected.denial.clone());
        }
        if let Some(entry) = state.runs.get(&request.idempotency_key) {
            if entry.request != request || entry.settlement.is_some() {
                return Err(AdmissionError::LicenseInvalid);
            }
            return token::verify(
                &entry.lease,
                &request,
                &self.org,
                &state.keys,
                chrono::Utc::now().timestamp(),
            );
        }
        let response = self
            .transport
            .request(LeaseTransportRequest::Create {
                org_id: self.org.clone(),
                request: request.clone(),
            })
            .await
            .map_err(map_transport_error)?;
        if !successful(response.status) {
            return Err(map_admission_error(response));
        }
        let body = response.body;
        let admitted = self
            .finish_admission(request.clone(), body.clone(), state.keys)
            .await;
        if let Err(denial) = &admitted {
            if let Some(rejected) = cleanup::RejectedCreate::recognizable(
                &body,
                &self.org,
                &request.idempotency_key,
                denial.clone(),
            ) {
                self.release_rejected_create(rejected).await;
            }
        }
        admitted
    }

    async fn finish_admission(
        &self,
        request: CreateLeaseRequest,
        body: serde_json::Value,
        mut keys: SigningKeys,
    ) -> Result<LeasePermit, AdmissionError> {
        let lease: LeaseResponse =
            serde_json::from_value(body).map_err(|_| AdmissionError::LicenseInvalid)?;
        let kid = token::key_id(
            lease
                .token
                .as_deref()
                .ok_or(AdmissionError::LicenseInvalid)?,
        )
        .ok_or(AdmissionError::LicenseInvalid)?;
        if !keys.keys.iter().any(|key| key.key_id == kid) {
            let response = self
                .transport
                .request(LeaseTransportRequest::SigningKeys)
                .await
                .map_err(map_transport_error)?;
            if !successful(response.status) {
                return Err(AdmissionError::ServerUnreachable);
            }
            keys = serde_json::from_value(response.body)
                .map_err(|_| AdmissionError::LicenseInvalid)?;
        }
        let permit = token::verify(
            &lease,
            &request,
            &self.org,
            &keys,
            chrono::Utc::now().timestamp(),
        )?;
        self.journal
            .update(|state| {
                state.keys = keys;
                state
                    .runs
                    .entry(request.idempotency_key.clone())
                    .or_insert(Entry {
                        request,
                        lease,
                        usage: UsageReport::default(),
                        heartbeat: None,
                        settlement: None,
                    });
                Ok(())
            })
            .map_err(|error| {
                eprintln!("Warning: FEA admission could not be persisted: {error}");
                AdmissionError::LicenseInvalid
            })?;
        Ok(permit)
    }

    /// Durably records usage before attempting delivery. Network errors leave it pending and return Ok.
    pub async fn heartbeat(&self, permit: &LeasePermit, usage: UsageReport) -> Result<()> {
        self.record_heartbeat(permit, usage)?;
        self.replay_pending().await
    }

    /// First outcome wins. Repeated calls and offline replay are safe.
    pub async fn settle(
        &self,
        permit: &LeasePermit,
        outcome: LeaseOutcome,
        usage: UsageReport,
    ) -> Result<()> {
        self.record_settlement(permit, outcome, usage)?;
        self.replay_pending().await
    }

    /// Persist before scheduling background delivery. This method performs no HTTP.
    pub fn record_heartbeat(&self, permit: &LeasePermit, usage: UsageReport) -> Result<()> {
        self.record(permit, usage, None)
    }

    /// Persist the first terminal outcome before scheduling background delivery. No HTTP.
    pub fn record_settlement(
        &self,
        permit: &LeasePermit,
        outcome: LeaseOutcome,
        usage: UsageReport,
    ) -> Result<()> {
        self.record(permit, usage, Some(outcome))
    }

    fn record(
        &self,
        permit: &LeasePermit,
        usage: UsageReport,
        outcome: Option<LeaseOutcome>,
    ) -> Result<()> {
        anyhow::ensure!(
            permit.org_id == self.org,
            "Lease belongs to another organization"
        );
        self.journal.update(|state| {
            // Already delivered settlements have no entry. Never recreate them.
            if let Some(entry) = state.runs.get_mut(&permit.idempotency_key) {
                anyhow::ensure!(entry.lease.lease_id == permit.lease_id, "Lease id mismatch");
                if entry.settlement.is_none() {
                    entry.usage = entry.usage.cumulative(usage);
                    match outcome {
                        Some(outcome) => {
                            entry.settlement = Some(Settlement {
                                outcome,
                                usage: entry.usage,
                            });
                        }
                        None => entry.heartbeat = Some(entry.usage),
                    }
                }
            }
            Ok(())
        })
    }

    /// Flush on the next online command. 404 drops an unknown lease; all other failures retain it.
    /// Token expiry is intentionally not checked here: an admitted run is allowed to finish offline.
    pub async fn replay_pending(&self) -> Result<()> {
        let _guard = self.delivery.lock().await;
        let state = self.journal.read()?;
        for rejected in state
            .rejected_creates
            .values()
            .filter(|entry| !entry.delivered)
        {
            if self.send_failed_cleanup(rejected).await {
                self.mark_cleanup_delivered(rejected)?;
            }
        }
        for (run, entry) in state.runs {
            // Coalesce heartbeats cumulatively but preserve the final pending heartbeat before settle.
            let mut calls = Vec::new();
            if let Some(heartbeat) = &entry.heartbeat {
                calls.push(LeaseTransportRequest::Heartbeat {
                    org_id: self.org.clone(),
                    lease_id: entry.lease.lease_id.clone(),
                    usage: *heartbeat,
                });
            }
            if let Some(settlement) = &entry.settlement {
                calls.push(LeaseTransportRequest::Settle {
                    org_id: self.org.clone(),
                    lease_id: entry.lease.lease_id.clone(),
                    outcome: settlement.outcome,
                    usage: settlement.usage,
                });
            }
            for request in calls {
                let settling = matches!(&request, LeaseTransportRequest::Settle { .. });
                let Ok(response) = self.transport.request(request).await else {
                    break;
                };
                let not_found = response.status == 404;
                if not_found {
                    let error = serde_json::from_value::<ApiErrorEnvelope>(response.body);
                    if !matches!(error, Ok(body) if body.error.code == ServerErrorCode::NotFound) {
                        break;
                    }
                } else {
                    if !successful(response.status) {
                        break;
                    }
                    // A malformed/truncated success is not an acknowledgement.
                    let Ok(receipt) = serde_json::from_value::<LeaseResponse>(response.body) else {
                        break;
                    };
                    if receipt.lease_id != entry.lease.lease_id || receipt.org_id != self.org {
                        break;
                    }
                }
                self.journal.update(|state| {
                    if not_found || settling {
                        state.runs.remove(&run);
                    } else if let Some(current) = state.runs.get_mut(&run) {
                        if current.heartbeat == entry.heartbeat {
                            current.heartbeat = None;
                        }
                    }
                    Ok(())
                })?;
                if not_found {
                    break;
                }
            }
        }
        Ok(())
    }
}

fn successful(status: u16) -> bool {
    (200..300).contains(&status)
}

fn map_transport_error(error: LeaseTransportError) -> AdmissionError {
    match error {
        LeaseTransportError::SignedOut => AdmissionError::SignInRequired,
        LeaseTransportError::Unreachable => AdmissionError::ServerUnreachable,
    }
}

fn map_admission_error(response: LeaseHttpResponse) -> AdmissionError {
    let status = response.status;
    let error = serde_json::from_value::<ApiErrorEnvelope>(response.body)
        .ok()
        .map(|body| body.error);
    let code = error
        .as_ref()
        .map(|error| error.code)
        .unwrap_or(ServerErrorCode::Unknown);
    match (status, code) {
        (401, _) | (403, ServerErrorCode::PolicyDenied | ServerErrorCode::UserRequired) => {
            AdmissionError::SignInRequired
        }
        (402, ServerErrorCode::CoreHoursExhausted) => AdmissionError::AllowanceUsed {
            refills_at_unix_seconds: error
                .and_then(|error| error.details)
                .and_then(|details| details.refills_at)
                .map(|value| value.timestamp()),
        },
        (500..=599, _) => AdmissionError::ServerUnreachable,
        _ => AdmissionError::LicenseInvalid,
    }
}

#[cfg(test)]
mod tests;
#[cfg(test)]
mod transport_tests;
#[cfg(test)]
mod cleanup_tests;
