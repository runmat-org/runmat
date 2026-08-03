use std::sync::Arc;

use rand::RngCore as _;
use sha2::{Digest as _, Sha256};

use runmat_execution_transport_native::control::{
    EnrollmentRequest, NodeControlPlane, NodeInventory,
};

use super::{CredentialStore, NodeCredential};
use crate::AgentResult;

pub async fn enroll(
    control: Arc<dyn NodeControlPlane>,
    store: &CredentialStore,
    token: String,
    inventory: NodeInventory,
    heartbeat_ttl_seconds: u64,
) -> AgentResult<NodeCredential> {
    let mut identity = [0_u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut identity);
    let identity_secret = base64_url(&identity);
    let identity_fingerprint = format!("{:x}", Sha256::digest(identity));
    let enrolled = control
        .enroll(EnrollmentRequest {
            token,
            identity_fingerprint: identity_fingerprint.clone(),
            inventory,
            heartbeat_ttl_seconds,
        })
        .await?;
    let credential = NodeCredential {
        node_id: enrolled.node_id,
        cluster_id: enrolled.cluster_id,
        org_id: enrolled.org_id,
        identity_secret,
        identity_fingerprint,
        credential: enrolled.credential,
        credential_epoch: enrolled.credential_epoch,
        lease_epoch: enrolled.lease_epoch,
    };
    store.store(&credential)?;
    Ok(credential)
}

fn base64_url(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut output = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let first = chunk[0];
        let second = chunk.get(1).copied().unwrap_or(0);
        let third = chunk.get(2).copied().unwrap_or(0);
        output.push(ALPHABET[(first >> 2) as usize] as char);
        output.push(ALPHABET[(((first & 0x03) << 4) | (second >> 4)) as usize] as char);
        if chunk.len() > 1 {
            output.push(ALPHABET[(((second & 0x0f) << 2) | (third >> 6)) as usize] as char);
        }
        if chunk.len() > 2 {
            output.push(ALPHABET[(third & 0x3f) as usize] as char);
        }
    }
    output
}
