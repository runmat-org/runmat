use std::sync::Arc;

use runmat_execution_transport_native::control::{
    EnrollmentRequest, NodeControlPlane, NodeInventory,
};

use super::{CredentialStore, NodeCredential};
use crate::AgentResult;

pub async fn enroll(
    control: Arc<dyn NodeControlPlane>,
    store: &CredentialStore,
    endpoint: crate::config::ServerEndpoint,
    token: String,
    inventory: NodeInventory,
    heartbeat_ttl_seconds: u64,
) -> AgentResult<NodeCredential> {
    let mut owner = store.clone().claim()?;
    let pending =
        owner.prepare_token_enrollment(endpoint, &token, inventory, heartbeat_ttl_seconds)?;
    let enrolled = control
        .enroll(EnrollmentRequest {
            token,
            credential: pending.credential().to_owned(),
            identity_fingerprint: pending.identity_fingerprint().to_owned(),
            identity_public_key: pending.identity_public_key().to_owned(),
            inventory: pending.inventory().clone(),
            heartbeat_ttl_seconds: pending.heartbeat_ttl_seconds(),
        })
        .await?;
    pending.complete(enrolled)
}
