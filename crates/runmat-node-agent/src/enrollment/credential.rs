use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{AgentError, AgentResult};

mod pending;
pub use pending::{PendingRegistration, RegistrationPreparation, RegistrationTarget};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct NodeCredential {
    pub server_endpoint: crate::config::ServerEndpoint,
    pub node_id: String,
    pub cluster_id: String,
    pub org_id: String,
    pub identity_secret: String,
    pub identity_public_key: Vec<u8>,
    pub identity_fingerprint: String,
    pub credential: String,
    pub credential_epoch: u64,
    pub lease_epoch: u64,
}

#[derive(Debug, Clone)]
pub struct CredentialStore {
    path: PathBuf,
}

/// The unique local writer for node credentials and registration state.
/// The stable kernel lease also fences a second foreground/service agent.
pub struct OwnedCredentialStore {
    store: CredentialStore,
    _state_owner: runmat_filesystem::ExclusiveFileLock,
}

impl OwnedCredentialStore {
    pub fn load(&self) -> AgentResult<NodeCredential> {
        self.store.load()
    }

    pub fn store(&mut self, credential: &NodeCredential) -> AgentResult<()> {
        self.store.persist(credential)
    }
}

impl CredentialStore {
    pub fn new(state_directory: &Path) -> Self {
        Self {
            path: state_directory.join("credential.json"),
        }
    }

    pub fn claim(self) -> AgentResult<OwnedCredentialStore> {
        let directory = self.path.parent().ok_or_else(|| {
            AgentError::UnsafeCredential("node state requires a dedicated directory".into())
        })?;
        if !directory.is_absolute() || directory.parent().is_none() {
            return Err(AgentError::UnsafeCredential(
                "node state requires an absolute non-root directory".into(),
            ));
        }
        runmat_filesystem::NativeFsProvider.create_private_directory_durable(directory)?;
        let owner =
            runmat_filesystem::ExclusiveFileLock::try_acquire(&directory.join("agent.lock"))?
                .ok_or_else(|| {
                    AgentError::Configuration(
                        "node state is already owned by another running agent or registration"
                            .into(),
                    )
                })?;
        Ok(OwnedCredentialStore {
            store: self,
            _state_owner: owner,
        })
    }

    pub fn load(&self) -> AgentResult<NodeCredential> {
        #[cfg(not(windows))]
        if !self.path.exists() {
            return Err(AgentError::NotEnrolled);
        }
        #[cfg(windows)]
        let _directory = runmat_filesystem::NativeFsProvider
            .open_owner_private_directory(
                self.path
                    .parent()
                    .ok_or_else(|| AgentError::UnsafeCredential("missing parent".into()))?,
            )
            .map_err(load_error)?;
        #[cfg(not(windows))]
        verify_private_file(&self.path)?;
        #[cfg(not(windows))]
        let bytes = std::fs::read(&self.path).map_err(load_error)?;
        #[cfg(windows)]
        let bytes = {
            use std::io::Read as _;
            let mut file = runmat_filesystem::NativeFsProvider
                .open_owner_private_file(&self.path)
                .map_err(load_error)?;
            let mut bytes = Vec::new();
            file.read_to_end(&mut bytes)?;
            bytes
        };
        let credential: NodeCredential = serde_json::from_slice(&bytes)?;
        validate(&credential)?;
        Ok(credential)
    }

    fn persist(&self, credential: &NodeCredential) -> AgentResult<()> {
        validate(credential)?;
        let parent = self
            .path
            .parent()
            .ok_or_else(|| AgentError::UnsafeCredential("missing parent".to_string()))?;
        #[cfg(windows)]
        let _directory = {
            let provider = runmat_filesystem::NativeFsProvider;
            provider.create_private_directory_durable(parent)?;
            provider.open_owner_private_directory(parent)?
        };
        #[cfg(windows)]
        match std::fs::symlink_metadata(&self.path) {
            Ok(_) => verify_private_file(&self.path)?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(AgentError::Io(error)),
        }
        #[cfg(not(windows))]
        std::fs::create_dir_all(parent)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            std::fs::set_permissions(parent, std::fs::Permissions::from_mode(0o700))?;
        }
        let temporary = parent.join(format!(".credential-{}.tmp", rand::random::<u64>()));
        let bytes = serde_json::to_vec(credential)?;
        write_private(&temporary, &bytes)?;
        std::fs::rename(&temporary, &self.path)?;
        sync_directory(parent)?;
        verify_private_file(&self.path)
    }
}

fn load_error(error: std::io::Error) -> AgentError {
    if error.kind() == std::io::ErrorKind::NotFound {
        AgentError::NotEnrolled
    } else {
        AgentError::Io(error)
    }
}

fn validate(value: &NodeCredential) -> AgentResult<()> {
    if value.node_id.is_empty()
        || value.cluster_id.is_empty()
        || value.org_id.is_empty()
        || value.identity_secret.len() < 43
        || value.identity_public_key.len() != 32
        || value.identity_fingerprint.len() != 64
        || value.credential.len() < 43
        || value.credential_epoch == 0
        || value.lease_epoch == 0
    {
        return Err(AgentError::UnsafeCredential(
            "credential record is malformed".to_string(),
        ));
    }
    validate_signing_identity(
        &value.identity_secret,
        &value.identity_public_key,
        &value.identity_fingerprint,
    )
}

fn validate_signing_identity(
    secret: &str,
    public_key: &[u8],
    fingerprint: &str,
) -> AgentResult<()> {
    let secret = decode_secret(secret)?;
    let signer = runmat_execution::security::EndpointIdentitySigner::from_secret(secret)
        .map_err(|error| AgentError::UnsafeCredential(error.to_string()))?;
    if signer.public_key().as_slice() != public_key || signer.fingerprint() != fingerprint {
        return Err(AgentError::UnsafeCredential(
            "credential signing identity does not match its public key".to_string(),
        ));
    }
    Ok(())
}

pub(crate) fn decode_secret(value: &str) -> AgentResult<[u8; 32]> {
    use base64::Engine as _;
    let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(value)
        .map_err(|_| AgentError::UnsafeCredential("identity secret is malformed".to_string()))?;
    bytes
        .try_into()
        .map_err(|_| AgentError::UnsafeCredential("identity secret has an invalid length".into()))
}

fn write_private(path: &Path, bytes: &[u8]) -> AgentResult<()> {
    #[cfg(windows)]
    let mut file = runmat_filesystem::NativeFsProvider.create_owner_private_file_new(path)?;
    #[cfg(not(windows))]
    let mut file = {
        let mut options = std::fs::OpenOptions::new();
        options.create_new(true).write(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt as _;
            options.mode(0o600);
        }
        options.open(path)?
    };
    use std::io::Write as _;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn verify_private_file(path: &Path) -> AgentResult<()> {
    #[cfg(windows)]
    {
        runmat_filesystem::NativeFsProvider.open_owner_private_file(path)?;
        Ok(())
    }
    #[cfg(not(windows))]
    {
        let metadata = std::fs::metadata(path)?;
        if !metadata.is_file() {
            return Err(AgentError::UnsafeCredential(
                "credential path is not a regular file".to_string(),
            ));
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            if metadata.permissions().mode() & 0o077 != 0 {
                return Err(AgentError::UnsafeCredential(
                    "credential file must be mode 0600".to_string(),
                ));
            }
        }
        Ok(())
    }
}

fn sync_directory(path: &Path) -> AgentResult<()> {
    runmat_filesystem::NativeFsProvider.synchronize_directory(path)?;
    Ok(())
}
