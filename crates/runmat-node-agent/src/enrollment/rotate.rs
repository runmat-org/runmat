use std::sync::Arc;

use runmat_execution_transport_native::control::{NodeControlPlane, NodeHeartbeat};

use super::{CredentialStore, NodeCredential};
use crate::AgentResult;

pub async fn rotate(
    control: Arc<dyn NodeControlPlane>,
    store: &CredentialStore,
    credential: &mut NodeCredential,
    mut heartbeat: NodeHeartbeat,
) -> AgentResult<()> {
    heartbeat.credential = credential.credential.clone();
    heartbeat.credential_epoch = credential.credential_epoch;
    let replacement = control.rotate_credential(&heartbeat).await?;
    credential.credential = replacement.credential;
    credential.credential_epoch = replacement.credential_epoch;
    store.store(credential)
}
