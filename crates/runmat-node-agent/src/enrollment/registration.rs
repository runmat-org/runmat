//! Account authorization is used only for bootstrap; service state retains node authority.
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use runmat_execution_transport_native::control::{EnrolledNode, NodeInventory};
use runmat_server_client::public_api::types;

use super::{CredentialStore, NodeCredential, RegistrationPreparation, RegistrationTarget};
use crate::server_api::{inventory_to_api, map_error, to_i64, to_u64};
use crate::{AgentError, AgentResult};

/// Retain machine identity before the request so a lost response can be retried
/// without granting another node. Neither the account token nor its HTTP client
/// is written to the node's state directory.
pub async fn register_authenticated(
    store: &CredentialStore,
    target: RegistrationTarget,
    inventory: NodeInventory,
    heartbeat_ttl_seconds: u64,
    account_token: &str,
) -> AgentResult<NodeCredential> {
    let mut owner = store.clone().claim()?;
    let pending =
        match owner.prepare_registration(target.clone(), inventory, heartbeat_ttl_seconds)? {
            RegistrationPreparation::Registered(credential) => return Ok(*credential),
            RegistrationPreparation::Pending(pending) => pending,
        };
    let client =
        runmat_server_client::auth::build_public_client(target.endpoint().as_str(), account_token)
            .map_err(|error| {
                AgentError::Configuration(format!(
                    "cannot prepare authenticated registration: {error}"
                ))
            })?;
    let body = types::RegisterNodeRequest {
        credential: pending.credential().to_owned().try_into().map_err(|_| {
            AgentError::UnsafeCredential("pending node credential violates the API contract".into())
        })?,
        identity_fingerprint: pending.identity_fingerprint().to_owned(),
        identity_public_key: URL_SAFE_NO_PAD.encode(pending.identity_public_key()),
        inventory: inventory_to_api(pending.inventory().clone())?,
        heartbeat_ttl_seconds: to_i64(pending.heartbeat_ttl_seconds(), "heartbeat TTL")?,
    };
    let response = client
        .register_node(target.org().as_str(), target.cluster().as_str(), &body)
        .await
        .map_err(map_error)?
        .into_inner();
    pending.complete(EnrolledNode {
        node_id: response.node_id,
        cluster_id: response.cluster_id,
        org_id: response.org_id,
        credential: response.credential,
        credential_epoch: to_u64(response.credential_epoch, "credential epoch")?,
        lease_epoch: to_u64(response.lease_epoch, "lease epoch")?,
    })
}
