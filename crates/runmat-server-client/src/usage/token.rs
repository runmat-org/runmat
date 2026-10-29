use super::types::*;
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use ed25519_dalek::{Signature, VerifyingKey};
use serde::Deserialize;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Claims {
    audience: String,
    lease_id: String,
    org_id: String,
    user_id: String,
    idempotency_key: String,
    allocated_cpu_millicores: u32,
    // The reservation can grow after issuance; it is not a fixed execution cap.
    #[serde(rename = "coreMicroHours")]
    _core_micro_hours: u64,
    issued_at: i64,
    expires_at: i64,
}

pub(super) fn key_id(token: &str) -> Option<&str> {
    let parts: Vec<_> = token.split('.').collect();
    if parts.len() != 4
        || parts[0] != "ed25519v1"
        || parts[1].is_empty()
        || !parts[1]
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"_-".contains(&b))
    {
        return None;
    }
    Some(parts[1])
}

pub(super) fn verify(
    lease: &LeaseResponse,
    request: &CreateLeaseRequest,
    org: &str,
    keys: &SigningKeys,
    now: i64,
) -> Result<LeasePermit, AdmissionError> {
    let invalid = || AdmissionError::LicenseInvalid;
    let token = lease.token.as_deref().ok_or_else(invalid)?;
    let kid = key_id(token).ok_or_else(invalid)?;
    let key = keys
        .keys
        .iter()
        .find(|key| key.key_id == kid && key.algorithm == "Ed25519")
        .ok_or_else(invalid)?;
    let bytes: [u8; 32] = URL_SAFE_NO_PAD
        .decode(&key.public_key)
        .map_err(|_| invalid())?
        .try_into()
        .map_err(|_| invalid())?;
    let key = VerifyingKey::from_bytes(&bytes).map_err(|_| invalid())?;
    let parts: Vec<_> = token.split('.').collect();
    let signature = URL_SAFE_NO_PAD.decode(parts[3]).map_err(|_| invalid())?;
    let signature = Signature::from_slice(&signature).map_err(|_| invalid())?;
    let signed = token.rsplit_once('.').ok_or_else(invalid)?.0;
    key.verify_strict(signed.as_bytes(), &signature)
        .map_err(|_| invalid())?;
    let claims: Claims =
        serde_json::from_slice(&URL_SAFE_NO_PAD.decode(parts[2]).map_err(|_| invalid())?)
            .map_err(|_| invalid())?;
    if claims.audience != "runmat.fea.core_hours"
        || claims.lease_id != lease.lease_id
        || claims.org_id != org
        || lease.org_id != org
        || claims.user_id != lease.user_id
        || claims.idempotency_key != request.idempotency_key
        || lease.idempotency_key != request.idempotency_key
        || claims.allocated_cpu_millicores < request.allocated_cpu_millicores
        || claims.allocated_cpu_millicores != lease.allocated_cpu_millicores
        || claims.issued_at != lease.issued_at.timestamp()
        || claims.expires_at != lease.expires_at.timestamp()
        || claims.expires_at <= claims.issued_at
        || lease.status != LeaseStatus::Active
    {
        return Err(invalid());
    }
    if claims.expires_at <= now {
        return Err(AdmissionError::LeaseExpired);
    }
    Ok(LeasePermit {
        lease_id: lease.lease_id.clone(),
        idempotency_key: request.idempotency_key.clone(),
        org_id: org.into(),
    })
}
