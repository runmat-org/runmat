//! Crash-retained registration identity. No account credential enters this record.
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use rand::RngCore as _;
use runmat_execution_transport_native::control::{EnrolledNode, NodeInventory};
use runmat_filesystem::{AtomicCreateResult, NativeFsProvider, PublicationDurability};
use runmat_server_client::ids::{ClusterId, ClusterNodeId, OrgId};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

use super::{validate_signing_identity, NodeCredential, OwnedCredentialStore};
use crate::config::ServerEndpoint;
use crate::{AgentError, AgentResult};

// This bounds local control metadata, never numerical input/workspace storage.
const MAX_REGISTRATION_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RegistrationTarget {
    endpoint: ServerEndpoint,
    org: OrgId,
    cluster: ClusterId,
}

impl RegistrationTarget {
    pub fn new(endpoint: ServerEndpoint, org: OrgId, cluster: ClusterId) -> Self {
        Self {
            endpoint,
            org,
            cluster,
        }
    }
    pub fn endpoint(&self) -> &ServerEndpoint {
        &self.endpoint
    }
    pub fn org(&self) -> &OrgId {
        &self.org
    }
    pub fn cluster(&self) -> &ClusterId {
        &self.cluster
    }

    fn matches(&self, credential: &NodeCredential) -> bool {
        credential.server_endpoint == self.endpoint
            && credential.org_id == self.org.as_str()
            && credential.cluster_id == self.cluster.as_str()
    }
}

/// Account registration knows its destination before the request. A bootstrap
/// token resolves that destination at the Server; persist only its digest here.
#[derive(Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase", deny_unknown_fields)]
enum RegistrationIntent {
    Account {
        target: RegistrationTarget,
    },
    Token {
        endpoint: ServerEndpoint,
        token_digest: [u8; 32],
    },
}

impl RegistrationIntent {
    fn endpoint(&self) -> &ServerEndpoint {
        match self {
            Self::Account { target } => target.endpoint(),
            Self::Token { endpoint, .. } => endpoint,
        }
    }

    fn accepts_destination(&self, response: &EnrolledNode) -> bool {
        match self {
            Self::Account { target } => {
                response.org_id == target.org.as_str()
                    && response.cluster_id == target.cluster.as_str()
            }
            Self::Token { .. } => true,
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Record {
    schema_version: u16,
    intent: RegistrationIntent,
    identity_secret: String,
    identity_public_key: Vec<u8>,
    identity_fingerprint: String,
    credential: String,
    inventory: NodeInventory,
    heartbeat_ttl_seconds: u64,
}

pub enum RegistrationPreparation<'a> {
    Registered(Box<NodeCredential>),
    Pending(PendingRegistration<'a>),
}

/// Borrows the exact original writer exclusively until the request settles.
/// Completion cannot accidentally publish through another state directory.
pub struct PendingRegistration<'a> {
    owner: &'a mut OwnedCredentialStore,
    record: Box<Record>,
}

impl OwnedCredentialStore {
    pub fn prepare_registration(
        &mut self,
        target: RegistrationTarget,
        inventory: NodeInventory,
        heartbeat_ttl_seconds: u64,
    ) -> AgentResult<RegistrationPreparation<'_>> {
        match self.load() {
            Ok(credential) if target.matches(&credential) => {
                return Ok(RegistrationPreparation::Registered(Box::new(credential)));
            }
            Ok(_) => return Err(target_mismatch()),
            Err(AgentError::NotEnrolled) => {}
            Err(error) => return Err(error),
        }
        self.prepare_pending(
            RegistrationIntent::Account { target },
            inventory,
            heartbeat_ttl_seconds,
        )
        .map(RegistrationPreparation::Pending)
    }

    pub fn prepare_token_enrollment(
        &mut self,
        endpoint: ServerEndpoint,
        token: &str,
        inventory: NodeInventory,
        heartbeat_ttl_seconds: u64,
    ) -> AgentResult<PendingRegistration<'_>> {
        match self.load() {
            Ok(_) => return Err(AgentError::Configuration(
                "node state is already enrolled; use its existing registration or a separate state directory".into(),
            )),
            Err(AgentError::NotEnrolled) => {},
            Err(error) => return Err(error),
        }
        if token.is_empty() || token.len() > 4096 {
            return Err(AgentError::Configuration(
                "enrollment token has an invalid length".into(),
            ));
        }
        let mut digest = Sha256::new();
        digest.update(b"runmat.node-enrollment-intent.v1\0");
        digest.update(token.as_bytes());
        self.prepare_pending(
            RegistrationIntent::Token {
                endpoint,
                token_digest: digest.finalize().into(),
            },
            inventory,
            heartbeat_ttl_seconds,
        )
    }

    fn prepare_pending(
        &mut self,
        intent: RegistrationIntent,
        inventory: NodeInventory,
        heartbeat_ttl_seconds: u64,
    ) -> AgentResult<PendingRegistration<'_>> {
        let mut identity = [0; 32];
        let mut secret = [0; 32];
        rand::rngs::OsRng.fill_bytes(&mut identity);
        rand::rngs::OsRng.fill_bytes(&mut secret);
        let signer = runmat_execution::security::EndpointIdentitySigner::from_secret(identity)
            .map_err(|error| AgentError::UnsafeCredential(error.to_string()))?;
        let candidate = Record {
            schema_version: 2,
            intent,
            identity_secret: URL_SAFE_NO_PAD.encode(identity),
            identity_public_key: signer.public_key().to_vec(),
            identity_fingerprint: signer.fingerprint(),
            credential: URL_SAFE_NO_PAD.encode(secret),
            inventory,
            heartbeat_ttl_seconds,
        };
        candidate.validate()?;
        let bytes =
            runmat_canonical_codec::encode_json_bounded(&candidate, MAX_REGISTRATION_BYTES)?;
        let path = self.store.path.with_file_name("registration.json");
        let (record, needs_retention) = match NativeFsProvider.atomic_create_private_new(
            &path,
            &bytes,
            MAX_REGISTRATION_BYTES,
        )? {
            AtomicCreateResult::Published { durability } => {
                (candidate, durability != PublicationDurability::Durable)
            }
            AtomicCreateResult::Existing(bytes) => {
                let retained: Record = serde_json::from_slice(&bytes)?;
                retained.validate()?;
                if retained.intent != candidate.intent {
                    return Err(target_mismatch());
                }
                (retained, true)
            }
        };
        if needs_retention {
            // Reaffirm a previous visible publication before spending account
            // authority. Failure leaves the exact record for the next retry.
            NativeFsProvider
                .synchronize_directory(path.parent().expect("credential state directory"))?;
        }
        Ok(PendingRegistration {
            owner: self,
            record: Box::new(record),
        })
    }
}

impl Record {
    fn validate(&self) -> AgentResult<()> {
        if self.schema_version != 2
            || !(10..=300).contains(&self.heartbeat_ttl_seconds)
            || self.inventory.cpu_millicores == 0
            || self.inventory.memory_bytes == 0
            || self.inventory.host.validate().is_err()
        {
            return Err(AgentError::UnsafeCredential(
                "pending registration metadata is invalid".into(),
            ));
        }
        validate_signing_identity(
            &self.identity_secret,
            &self.identity_public_key,
            &self.identity_fingerprint,
        )?;
        let decoded = URL_SAFE_NO_PAD.decode(&self.credential).map_err(|_| {
            AgentError::UnsafeCredential("pending node credential is malformed".into())
        })?;
        if decoded.len() != 32 {
            return Err(AgentError::UnsafeCredential(
                "pending node credential must encode 32 bytes".into(),
            ));
        }
        Ok(())
    }
}

impl PendingRegistration<'_> {
    pub fn identity_public_key(&self) -> &[u8] {
        &self.record.identity_public_key
    }
    pub fn identity_fingerprint(&self) -> &str {
        &self.record.identity_fingerprint
    }
    pub fn credential(&self) -> &str {
        &self.record.credential
    }
    pub fn inventory(&self) -> &NodeInventory {
        &self.record.inventory
    }
    pub fn heartbeat_ttl_seconds(&self) -> u64 {
        self.record.heartbeat_ttl_seconds
    }

    pub fn complete(self, response: EnrolledNode) -> AgentResult<NodeCredential> {
        let Self { owner, record } = self;
        ClusterNodeId::parse(&response.node_id).map_err(|_| {
            AgentError::UnsafeCredential("registration returned an invalid node identity".into())
        })?;
        OrgId::parse(&response.org_id).map_err(|_| {
            AgentError::UnsafeCredential(
                "registration returned an invalid organization identity".into(),
            )
        })?;
        ClusterId::parse(&response.cluster_id).map_err(|_| {
            AgentError::UnsafeCredential("registration returned an invalid cluster identity".into())
        })?;
        if !record.intent.accepts_destination(&response)
            || response.credential != record.credential
            || response.credential_epoch == 0
            || response.lease_epoch == 0
        {
            return Err(AgentError::UnsafeCredential(
                "registration response differs from original machine authority".into(),
            ));
        }
        let credential = NodeCredential {
            server_endpoint: record.intent.endpoint().clone(),
            node_id: response.node_id,
            cluster_id: response.cluster_id,
            org_id: response.org_id,
            identity_secret: record.identity_secret,
            identity_public_key: record.identity_public_key,
            identity_fingerprint: record.identity_fingerprint,
            credential: response.credential,
            credential_epoch: response.credential_epoch,
            lease_epoch: response.lease_epoch,
        };
        owner.store(&credential)?;
        let path = owner.store.path.with_file_name("registration.json");
        std::fs::remove_file(&path).map_err(AgentError::RegistrationCleanup)?;
        NativeFsProvider
            .synchronize_directory(path.parent().expect("credential state directory"))
            .map_err(AgentError::RegistrationCleanup)?;
        Ok(credential)
    }
}

fn target_mismatch() -> AgentError {
    AgentError::Configuration("node state belongs to a different Server, organization or cluster; use its existing registration or a separate state directory".into())
}
