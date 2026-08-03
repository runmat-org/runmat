use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{AgentError, AgentResult};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct NodeCredential {
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

impl CredentialStore {
    pub fn new(state_directory: &Path) -> Self {
        Self {
            path: state_directory.join("credential.json"),
        }
    }

    pub fn load(&self) -> AgentResult<NodeCredential> {
        if !self.path.exists() {
            return Err(AgentError::NotEnrolled);
        }
        verify_private_file(&self.path)?;
        let bytes = std::fs::read(&self.path).map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                AgentError::NotEnrolled
            } else {
                AgentError::Io(error)
            }
        })?;
        let credential: NodeCredential = serde_json::from_slice(&bytes)?;
        validate(&credential)?;
        Ok(credential)
    }

    pub fn store(&self, credential: &NodeCredential) -> AgentResult<()> {
        validate(credential)?;
        let parent = self
            .path
            .parent()
            .ok_or_else(|| AgentError::UnsafeCredential("missing parent".to_string()))?;
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
    let secret = decode_secret(&value.identity_secret)?;
    let signer = runmat_execution::security::EndpointIdentitySigner::from_secret(secret)
        .map_err(|error| AgentError::UnsafeCredential(error.to_string()))?;
    if signer.public_key().as_slice() != value.identity_public_key
        || signer.fingerprint() != value.identity_fingerprint
    {
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
    let mut options = std::fs::OpenOptions::new();
    options.create_new(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.mode(0o600);
    }
    let mut file = options.open(path)?;
    use std::io::Write as _;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn verify_private_file(path: &Path) -> AgentResult<()> {
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

#[cfg(unix)]
fn sync_directory(path: &Path) -> AgentResult<()> {
    std::fs::File::open(path)?.sync_all()?;
    Ok(())
}

#[cfg(windows)]
fn sync_directory(_: &Path) -> AgentResult<()> {
    Ok(())
}
