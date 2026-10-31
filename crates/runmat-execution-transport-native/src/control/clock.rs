//! Opaque observation control carried only on existing driver/node heartbeats.

use crate::{TransportError, TransportResult};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use runmat_execution::security::{NodeClockEndV1, NodeClockStartV1};
use runmat_execution::Digest;

pub const MAX_CLOCK_HEARTBEAT_ITEMS: usize = 128;

pub(crate) fn encode_id(value: &[u8; 16]) -> String {
    URL_SAFE_NO_PAD.encode(value)
}

pub(crate) fn decode_id(value: &str) -> TransportResult<[u8; 16]> {
    let bytes = URL_SAFE_NO_PAD
        .decode(value)
        .map_err(|_| TransportError::Integrity)?;
    if URL_SAFE_NO_PAD.encode(&bytes) != value {
        return Err(TransportError::Integrity);
    }
    bytes.try_into().map_err(|_| TransportError::Integrity)
}

pub(crate) fn bounded(items: usize) -> TransportResult<()> {
    if items > MAX_CLOCK_HEARTBEAT_ITEMS {
        Err(TransportError::Overflow)
    } else {
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClockEndDemandV1 {
    pub observation_id: [u8; 16],
    pub start_receipt_digest: Digest,
    pub demand_challenge: [u8; 16],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClockNodeDemandV1 {
    pub start_receipt: NodeClockStartV1,
    pub demand_challenge: [u8; 16],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClockObservationReleaseEntryV1 {
    pub observation_id: [u8; 16],
    pub start_receipt_digest: Digest,
    pub end_receipt_digest: Digest,
}

/// Post-ACK, retryable cleanup attestation tied to the latest accepted
/// encrypted checkpoint. The Server must verify artifact identity and fence
/// before removing any retained signed observation receipt.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClockObservationReleaseManifestV1 {
    pub checkpoint_sequence: u64,
    pub checkpoint_artifact_id: String,
    pub checkpoint_ciphertext_digest: String,
    pub driver_fence: u64,
    pub entries: Vec<ClockObservationReleaseEntryV1>,
}

#[derive(Clone, Debug, Default)]
pub struct ClockDriverHeartbeatRequestV1 {
    pub start_reservations: Vec<[u8; 16]>,
    pub start_receipts: Vec<NodeClockStartV1>,
    pub end_demands: Vec<ClockEndDemandV1>,
    pub lookup_ids: Vec<[u8; 16]>,
    pub release_manifest: Option<ClockObservationReleaseManifestV1>,
}

#[derive(Clone, Debug)]
pub struct ClockDriverHeartbeatResponseV1 {
    pub run: super::DriverHeartbeat,
    pub reserved_start_ids: Vec<[u8; 16]>,
    pub acked_start_ids: Vec<[u8; 16]>,
    pub starts: Vec<NodeClockStartV1>,
    pub ends: Vec<NodeClockEndV1>,
    pub pending_end_demands: Vec<ClockEndDemandV1>,
    pub release_accepted: bool,
    pub latest_accepted_release_manifest: Option<ClockObservationReleaseManifestV1>,
}

#[derive(Clone, Debug)]
pub struct ClockNodeHeartbeatResponseV1 {
    pub node: super::NodeStatus,
    pub open_observer_count: u32,
    pub demands: Vec<ClockNodeDemandV1>,
    pub acked_start_ids: Vec<[u8; 16]>,
    pub acked_end_ids: Vec<[u8; 16]>,
    pub acked_release_confirm_ids: Vec<[u8; 16]>,
}
