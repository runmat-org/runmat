//! Refused admission never authorizes execution, but a recognizable reservation still needs release.
use super::{types::*, CoreHourLeaseClient, LeaseTransportRequest};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct RejectedCreate {
    pub lease_id: String,
    pub org_id: String,
    pub idempotency_key: String,
    pub denial: AdmissionError,
    pub delivered: bool,
}

impl RejectedCreate {
    pub fn recognizable(
        body: &serde_json::Value,
        org: &str,
        run: &str,
        denial: AdmissionError,
    ) -> Option<Self> {
        // Do not use any unverified token claim as cleanup authority. Only the authenticated
        // create response's context-matching identifiers can name the reservation to release.
        let lease = body.get("leaseId")?.as_str()?;
        if body.get("orgId")?.as_str()? != org
            || body.get("idempotencyKey")?.as_str()? != run
            || lease.is_empty()
            || lease.len() > 128
            || matches!(lease, "." | "..")
            || !lease
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"._:-".contains(&b))
        {
            return None;
        }
        Some(Self {
            lease_id: lease.into(),
            org_id: org.into(),
            idempotency_key: run.into(),
            denial,
            delivered: false,
        })
    }
}

impl CoreHourLeaseClient {
    pub(super) async fn release_rejected_create(&self, rejected: RejectedCreate) {
        let persisted = self.journal.update(|state| {
            state.runs.remove(&rejected.idempotency_key);
            state
                .rejected_creates
                .insert(rejected.idempotency_key.clone(), rejected.clone());
            Ok(())
        });
        if let Err(error) = &persisted {
            eprintln!("Warning: unable to persist rejected FEA lease cleanup: {error}");
        }
        if self.send_failed_cleanup(&rejected).await {
            if let Err(error) = self.mark_cleanup_delivered(&rejected) {
                eprintln!("Warning: FEA lease was released, but its cleanup acknowledgement could not be saved: {error}");
            }
        } else if persisted.is_err() {
            eprintln!("Warning: rejected FEA lease could not be released online or saved for retry; no FEA execution was admitted.");
        }
    }

    pub(super) async fn send_failed_cleanup(&self, rejected: &RejectedCreate) -> bool {
        if rejected.org_id != self.org {
            return false;
        }
        let request = LeaseTransportRequest::Settle {
            org_id: rejected.org_id.clone(),
            lease_id: rejected.lease_id.clone(),
            outcome: super::LeaseOutcome::Failed,
            usage: super::UsageReport::default(),
        };
        let Ok(response) = self.transport.request(request).await else {
            return false;
        };
        if response.status == 404 {
            return matches!(serde_json::from_value::<ApiErrorEnvelope>(response.body), Ok(body) if body.error.code == ServerErrorCode::NotFound);
        }
        if !super::successful(response.status) {
            return false;
        }
        matches!(serde_json::from_value::<LeaseResponse>(response.body), Ok(receipt)
            if receipt.lease_id == rejected.lease_id && receipt.org_id == rejected.org_id)
    }

    pub(super) fn mark_cleanup_delivered(&self, rejected: &RejectedCreate) -> anyhow::Result<()> {
        self.journal.update(|state| {
            if let Some(current) = state.rejected_creates.get_mut(&rejected.idempotency_key) {
                if current.lease_id == rejected.lease_id {
                    current.delivered = true;
                }
            }
            Ok(())
        })
    }
}
