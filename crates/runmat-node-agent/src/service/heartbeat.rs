use runmat_execution_transport_native::control::{NodeHeartbeat, NodeInventory};

use crate::enrollment::NodeCredential;

pub fn heartbeat_for(
    credential: &NodeCredential,
    inventory: NodeInventory,
    heartbeat_ttl_seconds: u64,
) -> NodeHeartbeat {
    NodeHeartbeat {
        org_id: credential.org_id.clone(),
        node_id: credential.node_id.clone(),
        credential: credential.credential.clone(),
        credential_epoch: credential.credential_epoch,
        inventory,
        heartbeat_ttl_seconds,
    }
}
