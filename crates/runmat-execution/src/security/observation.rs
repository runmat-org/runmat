//! Node-owned monotonic wave observation across a driver replacement.
//! Neither receipt contains task identity or workload metadata.

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use ed25519_dalek::{Signature, Signer as _, Verifier as _, VerifyingKey};
use minicbor::Encoder;
use serde::{Deserialize, Serialize};

use super::{
    bounded_ascii, identity_fingerprint, invalid, EndpointIdentitySigner, EndpointTrustPolicy,
};
use crate::{ContractError, Digest};

pub const MAX_CLOCK_OBSERVATION_WIRE_BYTES: usize = 4 * 1024;
/// One live slot charges start, end, demand/index, and ID metadata (16 KiB).
/// The existing 64 MiB encrypted pool-checkpoint control budget bounds slots.
pub const MAX_CLOCK_OBSERVATIONS_PER_NODE: usize =
    64 * 1024 * 1024 / (4 * MAX_CLOCK_OBSERVATION_WIRE_BYTES);

/// Opaque local start demand. The node derives its identity and clock reading;
/// no driver-supplied tick is accepted as an observation.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct NodeClockStartRequestV1 {
    pub schema_version: u16,
    pub run_id: String,
    pub driver_allocation_lease_id: String,
    pub driver_allocation_fence: u64,
    #[serde(with = "nonce_base64")]
    pub driver_process_launch_nonce: [u8; 16],
    #[serde(with = "nonce_base64")]
    pub observation_id: [u8; 16],
    #[serde(with = "nonce_base64")]
    pub clock_domain_nonce: [u8; 16],
}

impl NodeClockStartRequestV1 {
    pub fn validate_shape(&self) -> Result<(), ContractError> {
        if self.schema_version != 1
            || !bounded_ascii(&self.run_id, 256)
            || !bounded_ascii(&self.driver_allocation_lease_id, 256)
            || self.driver_allocation_fence == 0
            || self.driver_allocation_fence > i64::MAX as u64
            || self
                .driver_process_launch_nonce
                .iter()
                .all(|byte| *byte == 0)
            || self.observation_id.iter().all(|byte| *byte == 0)
            || self.clock_domain_nonce.iter().all(|byte| *byte == 0)
            || serde_json::to_vec(self)
                .map_err(|_| invalid("clock request is not serializable"))?
                .len()
                > MAX_CLOCK_OBSERVATION_WIRE_BYTES
        {
            return Err(invalid("clock start request is malformed"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct NodeClockOwnerV1 {
    pub org_id: String,
    pub cluster_id: String,
    pub node_id: String,
    pub run_id: String,
    pub driver_allocation_lease_id: String,
    pub driver_allocation_fence: u64,
    #[serde(with = "nonce_base64")]
    pub driver_process_launch_nonce: [u8; 16],
    #[serde(with = "nonce_base64")]
    pub observation_id: [u8; 16],
    #[serde(with = "nonce_base64")]
    pub clock_domain_nonce: [u8; 16],
    pub ticks_per_second: u64,
    #[serde(with = "bytes_base64")]
    pub identity_public_key: Vec<u8>,
    pub identity_fingerprint: String,
}

impl NodeClockOwnerV1 {
    fn validate(&self) -> Result<(), ContractError> {
        if [
            self.org_id.as_str(),
            self.cluster_id.as_str(),
            self.node_id.as_str(),
            self.run_id.as_str(),
            self.driver_allocation_lease_id.as_str(),
        ]
        .iter()
        .any(|value| !bounded_ascii(value, 256))
            || self.driver_allocation_fence == 0
            || self.driver_allocation_fence > i64::MAX as u64
            || self
                .driver_process_launch_nonce
                .iter()
                .all(|byte| *byte == 0)
            || self.observation_id.iter().all(|byte| *byte == 0)
            || self.clock_domain_nonce.iter().all(|byte| *byte == 0)
            || self.ticks_per_second == 0
            || self.ticks_per_second > i64::MAX as u64
            || self.identity_public_key.len() != 32
            || self.identity_fingerprint != identity_fingerprint(&self.identity_public_key)
        {
            return Err(invalid("clock observation owner is malformed"));
        }
        Ok(())
    }

    fn encode(&self, encoder: &mut Encoder<&mut Vec<u8>>) -> Result<(), ContractError> {
        encoder.array(12).map_err(super::encoding)?;
        for value in [
            &self.org_id,
            &self.cluster_id,
            &self.node_id,
            &self.run_id,
            &self.driver_allocation_lease_id,
        ] {
            encoder.str(value).map_err(super::encoding)?;
        }
        encoder
            .u64(self.driver_allocation_fence)
            .map_err(super::encoding)?;
        encoder
            .bytes(&self.driver_process_launch_nonce)
            .map_err(super::encoding)?;
        encoder
            .bytes(&self.observation_id)
            .map_err(super::encoding)?;
        encoder
            .bytes(&self.clock_domain_nonce)
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
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct NodeClockStartV1 {
    pub schema_version: u16,
    pub owner: NodeClockOwnerV1,
    pub started_monotonic_ticks: u64,
    #[serde(with = "bytes_base64")]
    pub signature: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct NodeClockEndV1 {
    pub schema_version: u16,
    pub owner: NodeClockOwnerV1,
    pub start_receipt_digest: Digest,
    #[serde(with = "nonce_base64")]
    pub demand_challenge: [u8; 16],
    pub ended_monotonic_ticks: u64,
    #[serde(with = "bytes_base64")]
    pub signature: Vec<u8>,
}

mod nonce_base64 {
    use super::*;
    pub fn serialize<S: serde::Serializer>(
        value: &[u8; 16],
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&URL_SAFE_NO_PAD.encode(value))
    }
    pub fn deserialize<'de, D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> Result<[u8; 16], D::Error> {
        let value = String::deserialize(deserializer)?;
        let bytes = URL_SAFE_NO_PAD
            .decode(&value)
            .map_err(serde::de::Error::custom)?;
        if URL_SAFE_NO_PAD.encode(&bytes) != value {
            return Err(serde::de::Error::custom("noncanonical clock nonce"));
        }
        bytes
            .try_into()
            .map_err(|_| serde::de::Error::custom("clock nonce length"))
    }
}

mod bytes_base64 {
    use super::*;
    pub fn serialize<S: serde::Serializer>(
        value: &Vec<u8>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&URL_SAFE_NO_PAD.encode(value))
    }
    pub fn deserialize<'de, D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Vec<u8>, D::Error> {
        let value = String::deserialize(deserializer)?;
        let bytes = URL_SAFE_NO_PAD
            .decode(&value)
            .map_err(serde::de::Error::custom)?;
        if URL_SAFE_NO_PAD.encode(&bytes) != value {
            return Err(serde::de::Error::custom("noncanonical clock bytes"));
        }
        Ok(bytes)
    }
}

impl NodeClockStartV1 {
    pub fn validate_shape(&self) -> Result<(), ContractError> {
        self.owner.validate()?;
        if self.schema_version != 1
            || self.started_monotonic_ticks == 0
            || self.started_monotonic_ticks > i64::MAX as u64
            || self.signature.len() != 64
            || serde_json::to_vec(self)
                .map_err(|_| invalid("clock start is not serializable"))?
                .len()
                > MAX_CLOCK_OBSERVATION_WIRE_BYTES
        {
            return Err(invalid("clock start is malformed"));
        }
        Ok(())
    }

    pub fn receipt_digest(&self) -> Result<Digest, ContractError> {
        self.validate_shape()?;
        Ok(Digest::sha256(self.signing_bytes()?))
    }

    fn signing_bytes(&self) -> Result<Vec<u8>, ContractError> {
        let mut bytes = b"runmat-node-clock-start-v1\0".to_vec();
        let mut encoder = Encoder::new(&mut bytes);
        encoder.array(3).map_err(super::encoding)?;
        encoder.u16(self.schema_version).map_err(super::encoding)?;
        self.owner.encode(&mut encoder)?;
        encoder
            .u64(self.started_monotonic_ticks)
            .map_err(super::encoding)?;
        Ok(bytes)
    }
}

impl NodeClockEndV1 {
    pub fn receipt_digest(&self) -> Result<Digest, ContractError> {
        self.validate_shape()?;
        Ok(Digest::sha256(self.signing_bytes()?))
    }

    pub fn validate_shape(&self) -> Result<(), ContractError> {
        self.owner.validate()?;
        if self.schema_version != 1
            || self.demand_challenge.iter().all(|byte| *byte == 0)
            || self.ended_monotonic_ticks == 0
            || self.ended_monotonic_ticks > i64::MAX as u64
            || self.signature.len() != 64
            || serde_json::to_vec(self)
                .map_err(|_| invalid("clock end is not serializable"))?
                .len()
                > MAX_CLOCK_OBSERVATION_WIRE_BYTES
        {
            return Err(invalid("clock end is malformed"));
        }
        Ok(())
    }

    pub fn measured_wall_millis(&self, start: &NodeClockStartV1) -> Result<u64, ContractError> {
        self.validate_shape()?;
        start.validate_shape()?;
        if self.owner != start.owner
            || self.start_receipt_digest != start.receipt_digest()?
            || self.ended_monotonic_ticks <= start.started_monotonic_ticks
        {
            return Err(invalid("clock end differs from its original start"));
        }
        let delta = u128::from(self.ended_monotonic_ticks - start.started_monotonic_ticks);
        let frequency = u128::from(self.owner.ticks_per_second);
        let millis = delta
            .checked_mul(1000)
            .and_then(|value| value.checked_add(frequency - 1))
            .ok_or_else(|| invalid("clock interval overflowed"))?
            / frequency;
        u64::try_from(millis.max(1)).map_err(|_| invalid("clock interval overflowed"))
    }

    fn signing_bytes(&self) -> Result<Vec<u8>, ContractError> {
        let mut bytes = b"runmat-node-clock-end-v1\0".to_vec();
        let mut encoder = Encoder::new(&mut bytes);
        encoder.array(5).map_err(super::encoding)?;
        encoder.u16(self.schema_version).map_err(super::encoding)?;
        self.owner.encode(&mut encoder)?;
        encoder
            .bytes(self.start_receipt_digest.bytes())
            .map_err(super::encoding)?;
        encoder
            .bytes(&self.demand_challenge)
            .map_err(super::encoding)?;
        encoder
            .u64(self.ended_monotonic_ticks)
            .map_err(super::encoding)?;
        Ok(bytes)
    }
}

impl EndpointIdentitySigner {
    pub fn sign_node_clock_start(&self, start: &mut NodeClockStartV1) -> Result<(), ContractError> {
        self.check_clock_owner(&start.owner)?;
        start.signature = vec![0; 64];
        start.validate_shape()?;
        start.signature = self.0.sign(&start.signing_bytes()?).to_bytes().to_vec();
        Ok(())
    }

    pub fn sign_node_clock_end(&self, end: &mut NodeClockEndV1) -> Result<(), ContractError> {
        self.check_clock_owner(&end.owner)?;
        end.signature = vec![0; 64];
        end.validate_shape()?;
        end.signature = self.0.sign(&end.signing_bytes()?).to_bytes().to_vec();
        Ok(())
    }

    fn check_clock_owner(&self, owner: &NodeClockOwnerV1) -> Result<(), ContractError> {
        if owner.identity_public_key != self.public_key()
            || owner.identity_fingerprint != self.fingerprint()
        {
            return Err(invalid("clock signer does not own node identity"));
        }
        Ok(())
    }
}

impl EndpointTrustPolicy {
    pub fn verify_node_clock_start(
        &self,
        start: &NodeClockStartV1,
        pinned_node: &str,
    ) -> Result<(), ContractError> {
        start.validate_shape()?;
        self.verify_clock_signature(
            &start.owner,
            pinned_node,
            &start.signing_bytes()?,
            &start.signature,
        )
    }

    pub fn verify_node_clock_end(
        &self,
        end: &NodeClockEndV1,
        pinned_node: &str,
    ) -> Result<(), ContractError> {
        end.validate_shape()?;
        self.verify_clock_signature(
            &end.owner,
            pinned_node,
            &end.signing_bytes()?,
            &end.signature,
        )
    }

    fn verify_clock_signature(
        &self,
        owner: &NodeClockOwnerV1,
        pinned_node: &str,
        bytes: &[u8],
        signature: &[u8],
    ) -> Result<(), ContractError> {
        if pinned_node.is_empty()
            || owner.identity_fingerprint != pinned_node
            || !self.trusted_identity_fingerprints.contains(pinned_node)
        {
            return Err(invalid("clock observation node identity is not pinned"));
        }
        let public: [u8; 32] = owner
            .identity_public_key
            .as_slice()
            .try_into()
            .map_err(|_| invalid("clock observation key length is invalid"))?;
        let key = VerifyingKey::from_bytes(&public)
            .map_err(|_| invalid("clock observation key is invalid"))?;
        let signature = Signature::from_slice(signature)
            .map_err(|_| invalid("clock observation signature is invalid"))?;
        key.verify(bytes, &signature)
            .map_err(|_| invalid("clock observation signature verification failed"))
    }
}
