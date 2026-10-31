//! Node-owned proof that a particular worker process tree has retired.
//!
//! A lease release, heartbeat loss, or process-leader exit is not this proof.

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use ed25519_dalek::{Signature, Signer as _, Verifier as _, VerifyingKey};
use minicbor::Encoder;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

use super::{
    bounded_ascii, identity_fingerprint, invalid, EndpointIdentitySigner, EndpointTrustPolicy,
};
use crate::identity::{AttemptId, WorkerId};
use crate::ContractError;
use crate::{Digest, TaskId};

/// Fixed protocol metadata is bounded independently of the node's retained
/// per-allocation grant. Opaque encrypted receipts are charged to that grant;
/// the wire budget includes base64 expansion, envelope, and GET wrapper.
pub const MAX_RETIREMENT_ATTEMPTS: usize = 256;
/// 2 KiB journal + 32-byte private salt + run-object framing/auth tag.
pub const MAX_RETIREMENT_ENCRYPTED_RECEIPT_BYTES: usize = 4 * 1024;
/// Canonical base64url expands each 4 KiB object to at most 5,462 bytes.
/// Includes the GET `{"evidence":...}` wrapper and all fixed identity fields.
pub const MAX_RETIREMENT_WIRE_BYTES: usize =
    16 * 1024 + MAX_RETIREMENT_ATTEMPTS * (4 * MAX_RETIREMENT_ENCRYPTED_RECEIPT_BYTES / 3 + 16);
pub const MAX_RETIREMENT_JOURNAL_ENTRY_BYTES: usize = 2 * 1024;
/// One serialized writer owns this non-authoritative atomic-publication slot.
/// It may remain after a process is killed; only published `.receipt` files
/// establish durable admission or terminal accounting.
pub const RETIREMENT_JOURNAL_STAGING_FILE: &str = ".pending";

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RetiredResultInventoryV1 {
    pub artifact_bytes: u64,
    pub egress_bytes: u64,
    pub relay_bytes: u64,
}

/// One original worker-owned attempt interval. An absent result inventory
/// means the worker durably established that no terminal result was published;
/// it does not claim that no network traffic or transient objects existed.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RetiredAttemptAccountingV1 {
    pub attempt_id: AttemptId,
    pub task_id: TaskId,
    pub worker_id: WorkerId,
    pub ordinal: u16,
    pub driver_fencing_token: u64,
    pub request_digest: Digest,
    pub clock_domain_nonce: [u8; 16],
    pub started_monotonic_ticks: u64,
    pub ended_monotonic_ticks: u64,
    pub ticks_per_second: u64,
    pub published_result_inventory: Option<RetiredResultInventoryV1>,
}

impl RetiredAttemptAccountingV1 {
    pub fn measured_wall_millis(&self) -> Result<u64, ContractError> {
        if self.clock_domain_nonce.iter().all(|v| *v == 0)
            || self.ordinal == 0
            || self.driver_fencing_token == 0
            || self.driver_fencing_token > i64::MAX as u64
            || self.ticks_per_second == 0
            || self.started_monotonic_ticks > i64::MAX as u64
            || self.ended_monotonic_ticks > i64::MAX as u64
            || self.ticks_per_second > i64::MAX as u64
            || self.ended_monotonic_ticks <= self.started_monotonic_ticks
            || self
                .published_result_inventory
                .as_ref()
                .is_some_and(|inventory| {
                    inventory.artifact_bytes > i64::MAX as u64
                        || inventory.egress_bytes > i64::MAX as u64
                        || inventory.relay_bytes > i64::MAX as u64
                })
        {
            return Err(invalid("retired attempt clock or identity is malformed"));
        }
        let ticks = u128::from(self.ended_monotonic_ticks - self.started_monotonic_ticks);
        let frequency = u128::from(self.ticks_per_second);
        let millis = ticks
            .checked_mul(1000)
            .and_then(|value| value.checked_add(frequency - 1))
            .map(|value| value / frequency)
            .ok_or_else(|| invalid("retired attempt wall interval overflowed"))?;
        u64::try_from(millis.max(1))
            .map_err(|_| invalid("retired attempt wall interval overflowed"))
    }
}

/// Worker-authored, node-sandbox-local admission record. The worker must
/// fsync this before executing the attempt and update it before publishing a
/// terminal report. Only the process-tree owner may close an open interval.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct WorkerAttemptJournalEntryV1 {
    pub schema_version: u16,
    pub run_id: String,
    pub allocation_lease_id: String,
    pub allocation_fencing_token: u64,
    pub driver_fencing_token: u64,
    pub process_launch_nonce: [u8; 16],
    pub attempt_id: AttemptId,
    pub task_id: TaskId,
    pub worker_id: WorkerId,
    pub ordinal: u16,
    pub request_digest: Digest,
    pub clock_domain_nonce: [u8; 16],
    pub started_monotonic_ticks: u64,
    pub ticks_per_second: u64,
    pub finished_monotonic_ticks: Option<u64>,
    pub published_result_inventory: Option<RetiredResultInventoryV1>,
}

impl WorkerAttemptJournalEntryV1 {
    pub fn validate_shape(&self) -> Result<(), ContractError> {
        if self.schema_version != 1
            || !bounded_ascii(&self.run_id, 256)
            || !bounded_ascii(&self.allocation_lease_id, 256)
            || self.allocation_fencing_token == 0
            || self.driver_fencing_token == 0
            || self.process_launch_nonce.iter().all(|v| *v == 0)
            || self.clock_domain_nonce.iter().all(|v| *v == 0)
            || self.ordinal == 0
            || self.started_monotonic_ticks == 0
            || self.ticks_per_second == 0
            || self.started_monotonic_ticks > i64::MAX as u64
            || self.ticks_per_second > i64::MAX as u64
            || self
                .finished_monotonic_ticks
                .is_some_and(|v| v > i64::MAX as u64)
            || self
                .finished_monotonic_ticks
                .is_some_and(|v| v <= self.started_monotonic_ticks)
            || (self.published_result_inventory.is_some()
                && self.finished_monotonic_ticks.is_none())
            || serde_json::to_vec(self)
                .map_err(|_| invalid("attempt journal cannot be serialized"))?
                .len()
                > MAX_RETIREMENT_JOURNAL_ENTRY_BYTES
        {
            return Err(invalid("worker attempt journal entry is malformed"));
        }
        Ok(())
    }

    pub fn into_retired_claim(
        self,
        ended_monotonic_ticks: u64,
    ) -> Result<RetiredAttemptAccountingV1, ContractError> {
        self.validate_shape()?;
        let ended_monotonic_ticks = match self.finished_monotonic_ticks {
            Some(finished) if finished <= ended_monotonic_ticks => finished,
            Some(_) => return Err(invalid("attempt finished after physical tree retirement")),
            None => ended_monotonic_ticks,
        };
        let claim = RetiredAttemptAccountingV1 {
            attempt_id: self.attempt_id,
            task_id: self.task_id,
            worker_id: self.worker_id,
            ordinal: self.ordinal,
            driver_fencing_token: self.driver_fencing_token,
            request_digest: self.request_digest,
            clock_domain_nonce: self.clock_domain_nonce,
            started_monotonic_ticks: self.started_monotonic_ticks,
            ended_monotonic_ticks,
            ticks_per_second: self.ticks_per_second,
            published_result_inventory: self.published_result_inventory,
        };
        claim.measured_wall_millis()?;
        Ok(claim)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct WorkerTreeRetirementEvidenceV1 {
    pub schema_version: u16,
    pub org_id: String,
    pub cluster_id: String,
    pub node_id: String,
    pub run_id: String,
    pub allocation_lease_id: String,
    pub allocation_fencing_token: u64,
    pub driver_fencing_token: u64,
    pub process_id: u32,
    pub process_launch_nonce: [u8; 16],
    pub retired_at_unix_millis: u64,
    pub clock_domain_nonce: [u8; 16],
    pub retired_monotonic_ticks: u64,
    pub ticks_per_second: u64,
    /// Canonical base64url run-encrypted, independently salted attempt journals.
    /// Server stores these opaque objects without receiving task metadata.
    pub encrypted_attempt_receipts: Vec<String>,
    pub identity_public_key: Vec<u8>,
    pub identity_fingerprint: String,
    pub signature: Vec<u8>,
}

impl WorkerTreeRetirementEvidenceV1 {
    pub fn validate_shape(&self) -> Result<(), ContractError> {
        if self.schema_version != 1
            || [
                self.org_id.as_str(),
                self.cluster_id.as_str(),
                self.node_id.as_str(),
                self.run_id.as_str(),
                self.allocation_lease_id.as_str(),
            ]
            .iter()
            .any(|v| !bounded_ascii(v, 256))
            || self.allocation_fencing_token == 0
            || self.driver_fencing_token == 0
            || self.process_id == 0
            || self.process_launch_nonce.iter().all(|v| *v == 0)
            || self.clock_domain_nonce.iter().all(|v| *v == 0)
            || self.retired_monotonic_ticks == 0
            || self.retired_monotonic_ticks > i64::MAX as u64
            || self.ticks_per_second == 0
            || self.ticks_per_second > i64::MAX as u64
            || self.retired_at_unix_millis == 0
            || self.retired_at_unix_millis > i64::MAX as u64
            || self.allocation_fencing_token > i64::MAX as u64
            || self.driver_fencing_token > i64::MAX as u64
            || self.identity_public_key.len() != 32
            || self.identity_fingerprint != identity_fingerprint(&self.identity_public_key)
            || self.signature.len() != 64
            || self.encrypted_attempt_receipts.len() > MAX_RETIREMENT_ATTEMPTS
        {
            return Err(invalid("worker tree retirement evidence is malformed"));
        }
        let mut identities = BTreeSet::new();
        for encoded in &self.encrypted_attempt_receipts {
            if encoded.len() > 4 * MAX_RETIREMENT_ENCRYPTED_RECEIPT_BYTES.div_ceil(3) {
                return Err(invalid("retirement receipt exceeds its encoded budget"));
            }
            let bytes = URL_SAFE_NO_PAD
                .decode(encoded)
                .map_err(|_| invalid("retirement receipt is not canonical base64url"))?;
            if bytes.is_empty()
                || bytes.len() > MAX_RETIREMENT_ENCRYPTED_RECEIPT_BYTES
                || URL_SAFE_NO_PAD.encode(&bytes) != *encoded
                || !identities.insert(Digest::sha256(&bytes))
            {
                return Err(invalid(
                    "retirement receipt is oversized, repeated, or noncanonical",
                ));
            }
        }
        if serde_json::to_vec(self)
            .map_err(|_| invalid("worker retirement evidence cannot be serialized"))?
            .len()
            > MAX_RETIREMENT_WIRE_BYTES
        {
            return Err(invalid(
                "worker retirement evidence exceeds its wire budget",
            ));
        }
        Ok(())
    }

    fn signing_bytes(&self) -> Result<Vec<u8>, ContractError> {
        let mut bytes = b"runmat-worker-tree-retirement-v1\0".to_vec();
        let mut encoder = Encoder::new(&mut bytes);
        encoder.array(17).map_err(super::encoding)?;
        encoder.u16(self.schema_version).map_err(super::encoding)?;
        for value in [
            &self.org_id,
            &self.cluster_id,
            &self.node_id,
            &self.run_id,
            &self.allocation_lease_id,
        ] {
            encoder.str(value).map_err(super::encoding)?;
        }
        encoder
            .u64(self.allocation_fencing_token)
            .map_err(super::encoding)?;
        encoder
            .u64(self.driver_fencing_token)
            .map_err(super::encoding)?;
        encoder.u32(self.process_id).map_err(super::encoding)?;
        encoder
            .bytes(&self.process_launch_nonce)
            .map_err(super::encoding)?;
        encoder
            .u64(self.retired_at_unix_millis)
            .map_err(super::encoding)?;
        encoder
            .bytes(&self.clock_domain_nonce)
            .map_err(super::encoding)?;
        encoder
            .u64(self.retired_monotonic_ticks)
            .map_err(super::encoding)?;
        encoder
            .u64(self.ticks_per_second)
            .map_err(super::encoding)?;
        encoder
            .bytes(&self.identity_public_key)
            .map_err(super::encoding)?;
        encoder
            .str(&self.identity_fingerprint)
            .map_err(super::encoding)?;
        encoder
            .array(self.encrypted_attempt_receipts.len() as u64)
            .map_err(super::encoding)?;
        for encoded in &self.encrypted_attempt_receipts {
            let bytes = URL_SAFE_NO_PAD
                .decode(encoded)
                .map_err(|_| invalid("retirement receipt is not base64url"))?;
            encoder
                .bytes(Digest::sha256(&bytes).bytes())
                .map_err(super::encoding)?;
        }
        Ok(bytes)
    }
}

impl EndpointIdentitySigner {
    pub fn sign_worker_tree_retirement(
        &self,
        evidence: &mut WorkerTreeRetirementEvidenceV1,
    ) -> Result<(), ContractError> {
        if evidence.identity_public_key != self.public_key()
            || evidence.identity_fingerprint != self.fingerprint()
        {
            return Err(invalid("signer does not own worker retirement identity"));
        }
        evidence.signature = vec![0; 64];
        evidence.validate_shape()?;
        evidence.signature = self.0.sign(&evidence.signing_bytes()?).to_bytes().to_vec();
        Ok(())
    }
}

impl EndpointTrustPolicy {
    pub fn verify_worker_tree_retirement(
        &self,
        evidence: &WorkerTreeRetirementEvidenceV1,
        expected_node_fingerprint: &str,
    ) -> Result<(), ContractError> {
        evidence.validate_shape()?;
        if expected_node_fingerprint.is_empty()
            || evidence.identity_fingerprint != expected_node_fingerprint
            || !self
                .trusted_identity_fingerprints
                .contains(expected_node_fingerprint)
            || evidence.retired_at_unix_millis
                > self
                    .now_unix_millis
                    .saturating_add(self.maximum_clock_skew_millis)
        {
            return Err(invalid(
                "worker retirement identity or time is not locally trusted",
            ));
        }
        let public_key: [u8; 32] = evidence
            .identity_public_key
            .as_slice()
            .try_into()
            .map_err(|_| invalid("invalid retirement signing key length"))?;
        let key = VerifyingKey::from_bytes(&public_key)
            .map_err(|_| invalid("invalid retirement signing key"))?;
        let signature = Signature::from_slice(&evidence.signature)
            .map_err(|_| invalid("invalid worker retirement signature"))?;
        key.verify(&evidence.signing_bytes()?, &signature)
            .map_err(|_| invalid("worker retirement signature verification failed"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maximum_opaque_retirement_inventory_is_bounded_and_authenticated() {
        let signer = EndpointIdentitySigner::from_secret([23; 32]).unwrap();
        let receipts = (0..MAX_RETIREMENT_ATTEMPTS)
            .map(|index| {
                let mut bytes = vec![index as u8; MAX_RETIREMENT_ENCRYPTED_RECEIPT_BYTES];
                bytes[0] = index as u8;
                URL_SAFE_NO_PAD.encode(bytes)
            })
            .collect::<Vec<_>>();
        let mut certificate = WorkerTreeRetirementEvidenceV1 {
            schema_version: 1,
            org_id: "o".repeat(256),
            cluster_id: "c".repeat(256),
            node_id: "n".repeat(256),
            run_id: "r".repeat(256),
            allocation_lease_id: "a".repeat(256),
            allocation_fencing_token: i64::MAX as u64,
            driver_fencing_token: i64::MAX as u64,
            process_id: u32::MAX,
            process_launch_nonce: [255; 16],
            retired_at_unix_millis: i64::MAX as u64,
            clock_domain_nonce: [254; 16],
            retired_monotonic_ticks: i64::MAX as u64,
            ticks_per_second: i64::MAX as u64,
            encrypted_attempt_receipts: receipts,
            identity_public_key: signer.public_key().to_vec(),
            identity_fingerprint: signer.fingerprint(),
            signature: Vec::new(),
        };
        signer
            .sign_worker_tree_retirement(&mut certificate)
            .unwrap();
        let policy = EndpointTrustPolicy {
            permitted_tiers: Default::default(),
            trusted_identity_fingerprints: [signer.fingerprint()].into_iter().collect(),
            allowed_attestation_classes: Default::default(),
            require_pinned_identity: true,
            now_unix_millis: u64::MAX,
            maximum_clock_skew_millis: 0,
        };
        policy
            .verify_worker_tree_retirement(&certificate, &signer.fingerprint())
            .unwrap();
        // Native arrays have a larger JSON representation than the HTTP
        // base64url nonces. Include the response wrapper in the bound as well.
        let response =
            serde_json::to_vec(&serde_json::json!({ "evidence": &certificate })).unwrap();
        assert!(response.len() <= MAX_RETIREMENT_WIRE_BYTES);
        let decoded: WorkerTreeRetirementEvidenceV1 =
            serde_json::from_slice(&serde_json::to_vec(&certificate).unwrap()).unwrap();
        assert_eq!(decoded, certificate);
        let mut altered = certificate.clone();
        let mut bytes = URL_SAFE_NO_PAD
            .decode(&altered.encrypted_attempt_receipts[0])
            .unwrap();
        bytes[0] ^= 1;
        altered.encrypted_attempt_receipts[0] = URL_SAFE_NO_PAD.encode(bytes);
        assert!(policy
            .verify_worker_tree_retirement(&altered, &signer.fingerprint())
            .is_err());
        let mut over_capacity = certificate.clone();
        over_capacity
            .encrypted_attempt_receipts
            .push(URL_SAFE_NO_PAD.encode([1_u8]));
        assert!(over_capacity.validate_shape().is_err());
        let mut outside_wire_range = certificate.clone();
        outside_wire_range.allocation_fencing_token = u64::MAX;
        assert!(outside_wire_range.validate_shape().is_err());
        outside_wire_range = certificate.clone();
        outside_wire_range.encrypted_attempt_receipts[0] = "not base64!".into();
        assert!(outside_wire_range.validate_shape().is_err());
        let mut invalid_clock = certificate;
        invalid_clock.retired_monotonic_ticks = 0;
        assert!(invalid_clock.validate_shape().is_err());
    }
}
