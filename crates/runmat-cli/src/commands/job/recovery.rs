use std::io::Write as _;
use std::num::NonZeroU32;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use base64::Engine as _;
use runmat_execution::security::recipient_fingerprint;
use runmat_execution_artifact::encryption::{
    decode_run_key_envelope, EncryptionPurpose, ExecutionHpkeSuite, ExecutionRecipientKey,
    PortableExecutionEncryption, PortableExecutionPrivateKey,
};
use runmat_execution_artifact::ProgramExecutionResponse;
use runmat_server_client::auth::resolve_project_id;
use runmat_server_client::execution::public_error;
use runmat_server_client::public_api::types;
use serde::{Deserialize, Serialize};
use zeroize::{Zeroize as _, Zeroizing};

use crate::cli::JobRecoveryCommand;

const RECOVERY_SUITE: &str = "x25519-hkdf-sha256-aes128gcm-v1";
const MAX_ENVELOPE_BYTES: usize = 64 * 1024;
const MILLIS_PER_DAY: u64 = 24 * 60 * 60 * 1_000;

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct RecoveryKeyFile {
    schema_version: u16,
    recipient: RecoveryRecipientFile,
    private_key: String,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct RecoveryRecipientFile {
    suite: String,
    public_key: String,
    fingerprint: String,
    valid_after_unix_millis: u64,
    valid_before_unix_millis: u64,
    custodian_uri: Option<String>,
}

struct LoadedRecoveryKey {
    recipient: RecoveryRecipientFile,
    private_key: PortableExecutionPrivateKey,
}

pub(super) async fn execute(command: JobRecoveryCommand) -> Result<()> {
    match command {
        JobRecoveryCommand::Keygen {
            output,
            valid_days,
            custodian_uri,
            json,
        } => keygen(&output, valid_days, custodian_uri, json),
        JobRecoveryCommand::Configure {
            org,
            key,
            max_active_runs,
            max_active_runs_per_project,
            max_active_runs_per_principal,
            json,
        } => {
            configure(
                &org,
                &key,
                max_active_runs,
                max_active_runs_per_project,
                max_active_runs_per_principal,
                json,
            )
            .await
        }
        JobRecoveryCommand::Disable { org, json } => disable(&org, json).await,
        JobRecoveryCommand::Show { org, json } => show(&org, json).await,
        JobRecoveryCommand::Recover {
            run_id,
            project,
            key,
            json,
        } => recover(&run_id, project, &key, json).await,
    }
}

fn keygen(output: &Path, valid_days: u32, custodian_uri: Option<String>, json: bool) -> Result<()> {
    let valid_after = now_unix_millis()?;
    let valid_before = valid_after
        .checked_add(u64::from(valid_days) * MILLIS_PER_DAY)
        .context("recovery key validity exceeds the timestamp range")?;
    let (mut recipient, private_key) = PortableExecutionEncryption
        .recipient_from_entropy_with_derived_fingerprint(
            random_entropy()?,
            valid_after,
            valid_before,
        )?;
    recipient.custodian_uri = custodian_uri;
    recipient.validate()?;
    let private_bytes = Zeroizing::new(private_key.to_bytes());
    let mut document = RecoveryKeyFile {
        schema_version: 1,
        recipient: RecoveryRecipientFile::from_recipient(&recipient),
        private_key: base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(private_bytes.as_slice()),
    };
    write_new_key(output, &document)?;
    document.private_key.zeroize();
    if json {
        println!(
            "{}",
            serde_json::to_string(&serde_json::json!({
                "path": output,
                "recipient": document.recipient,
            }))?
        );
    } else {
        println!(
            "{} {}",
            crate::presentation::stdout().success("Recovery key"),
            output.display()
        );
        println!("fingerprint\t{}", document.recipient.fingerprint);
        println!(
            "{}",
            crate::presentation::stdout().muted(
                "The private key remains local. Back it up through your organization key custodian."
            )
        );
    }
    Ok(())
}

async fn configure(
    org: &str,
    key_path: &Path,
    max_active_runs: Option<u32>,
    max_active_runs_per_project: Option<u32>,
    max_active_runs_per_principal: Option<u32>,
    json: bool,
) -> Result<()> {
    let key = load_key(key_path)?;
    let recipient = key.recipient;
    let now = now_unix_millis()?;
    if now < recipient.valid_after_unix_millis || now >= recipient.valid_before_unix_millis {
        bail!("recovery key is not currently within its configured validity window");
    }
    let (client, _, _) = super::authenticated_context().await?;
    let current = client
        .api()
        .get_organization_execution_policy(org)
        .await
        .map_err(public_error)?
        .into_inner();
    let updated = client
        .api()
        .update_organization_execution_policy(
            org,
            &types::UpdateExecutionOrganizationPolicyRequest {
                expected_version: current.version,
                max_active_runs: nonzero(
                    max_active_runs.unwrap_or(u32::try_from(current.max_active_runs)?),
                    "maximum active runs",
                )?,
                max_active_runs_per_project: nonzero(
                    max_active_runs_per_project
                        .unwrap_or(u32::try_from(current.max_active_runs_per_project)?),
                    "maximum active runs per project",
                )?,
                max_active_runs_per_principal: nonzero(
                    max_active_runs_per_principal
                        .unwrap_or(u32::try_from(current.max_active_runs_per_principal)?),
                    "maximum active runs per principal",
                )?,
                recovery_recipient: Some(recipient.to_api()?),
            },
        )
        .await
        .map_err(public_error)?
        .into_inner();
    print_policy(&updated, json)
}

async fn disable(org: &str, json: bool) -> Result<()> {
    let (client, _, _) = super::authenticated_context().await?;
    let current = client
        .api()
        .get_organization_execution_policy(org)
        .await
        .map_err(public_error)?
        .into_inner();
    let updated = client
        .api()
        .update_organization_execution_policy(
            org,
            &types::UpdateExecutionOrganizationPolicyRequest {
                expected_version: current.version,
                max_active_runs: nonzero(
                    u32::try_from(current.max_active_runs)?,
                    "maximum active runs",
                )?,
                max_active_runs_per_project: nonzero(
                    u32::try_from(current.max_active_runs_per_project)?,
                    "maximum active runs per project",
                )?,
                max_active_runs_per_principal: nonzero(
                    u32::try_from(current.max_active_runs_per_principal)?,
                    "maximum active runs per principal",
                )?,
                recovery_recipient: None,
            },
        )
        .await
        .map_err(public_error)?
        .into_inner();
    print_policy(&updated, json)
}

async fn show(org: &str, json: bool) -> Result<()> {
    let (client, _, _) = super::authenticated_context().await?;
    let policy = client
        .api()
        .get_organization_execution_policy(org)
        .await
        .map_err(public_error)?
        .into_inner();
    print_policy(&policy, json)
}

async fn recover(
    run_id: &str,
    project: Option<uuid::Uuid>,
    key_path: &Path,
    json: bool,
) -> Result<()> {
    let key = load_key(key_path)?;
    let (client, _, config) = super::authenticated_context().await?;
    let project_id = resolve_project_id(&config, project)?.to_string();
    let recovery = client
        .api()
        .get_run_recovery(&project_id, run_id)
        .await
        .map_err(public_error)?
        .into_inner();
    if recovery.run_id != run_id
        || recovery.project_id != project_id
        || recovery.recipient_fingerprint != key.recipient.fingerprint
    {
        bail!("recovery metadata does not match the requested run, project, and key");
    }
    let envelope_bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(&recovery.envelope)
        .context("Server returned malformed recovery-envelope encoding")?;
    let envelope = decode_run_key_envelope(&envelope_bytes, MAX_ENVELOPE_BYTES)?;
    let run_key = PortableExecutionEncryption.open_run_key(
        &key.private_key,
        &envelope,
        &recovery.recipient_fingerprint,
        run_id,
        1,
    )?;
    if let Some(artifact_id) = recovery.result_artifact_id {
        let ciphertext = client
            .download_artifact(&project_id, &artifact_id, &recovery.recipient_fingerprint)
            .await?;
        let plaintext = super::crypto::open_run_object(
            &run_key,
            run_id,
            EncryptionPurpose::Result,
            &ciphertext,
        )?;
        let result: ProgramExecutionResponse =
            serde_json::from_slice(&plaintext).context("recovered result payload is malformed")?;
        if json {
            println!(
                "{}",
                serde_json::to_string(&serde_json::json!({
                    "runId": run_id,
                    "result": result,
                }))?
            );
        } else {
            match result {
                ProgramExecutionResponse::Success { value } => {
                    println!("{}", serde_json::to_string_pretty(&value)?);
                }
                ProgramExecutionResponse::ExternalizedSuccess { .. } => {
                    bail!("recovered result requires an artifact-aware consumer")
                }
                ProgramExecutionResponse::Failure { message } => bail!("{message}"),
            }
        }
        return Ok(());
    }
    if let Some(artifact_id) = recovery.diagnostic_artifact_id {
        let ciphertext = client
            .download_artifact(&project_id, &artifact_id, &recovery.recipient_fingerprint)
            .await?;
        let plaintext = super::crypto::open_run_object(
            &run_key,
            run_id,
            EncryptionPurpose::DetailedEvent,
            &ciphertext,
        )?;
        let message =
            String::from_utf8(plaintext).context("recovered diagnostic is not valid UTF-8")?;
        if json {
            println!(
                "{}",
                serde_json::to_string(&serde_json::json!({
                    "runId": run_id,
                    "diagnostic": message,
                }))?
            );
            return Ok(());
        }
        bail!("{message}");
    }
    bail!("run has no terminal result or diagnostic artifact to recover")
}

fn print_policy(value: &types::ExecutionOrganizationPolicyResponse, json: bool) -> Result<()> {
    if json {
        println!("{}", serde_json::to_string(value)?);
        return Ok(());
    }
    println!("organization\t{}", value.org_id);
    println!("version\t{}", value.version);
    println!("max_active_runs\t{}", value.max_active_runs);
    println!(
        "max_active_runs_per_project\t{}",
        value.max_active_runs_per_project
    );
    println!(
        "max_active_runs_per_principal\t{}",
        value.max_active_runs_per_principal
    );
    if let Some(recipient) = &value.recovery_recipient {
        println!("recovery_fingerprint\t{}", recipient.fingerprint);
        println!(
            "recovery_valid_until\t{}",
            recipient.valid_before_unix_millis
        );
    } else {
        println!("recovery_fingerprint\tdisabled");
    }
    Ok(())
}

impl RecoveryRecipientFile {
    fn from_recipient(value: &ExecutionRecipientKey) -> Self {
        Self {
            suite: RECOVERY_SUITE.into(),
            public_key: base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(&value.public_key),
            fingerprint: value.fingerprint.clone(),
            valid_after_unix_millis: value.valid_after_unix_millis,
            valid_before_unix_millis: value.valid_before_unix_millis,
            custodian_uri: value.custodian_uri.clone(),
        }
    }

    fn to_recipient(&self) -> Result<ExecutionRecipientKey> {
        if self.suite != RECOVERY_SUITE {
            bail!("recovery key uses an unsupported encryption suite");
        }
        let public_key = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(&self.public_key)
            .context("recovery public key is malformed")?;
        if recipient_fingerprint(&public_key) != self.fingerprint {
            bail!("recovery public key fingerprint does not match its key material");
        }
        let recipient = ExecutionRecipientKey {
            suite: ExecutionHpkeSuite::X25519HkdfSha256Aes128GcmV1,
            public_key,
            fingerprint: self.fingerprint.clone(),
            valid_after_unix_millis: self.valid_after_unix_millis,
            valid_before_unix_millis: self.valid_before_unix_millis,
            custodian_uri: self.custodian_uri.clone(),
        };
        recipient.validate()?;
        Ok(recipient)
    }

    fn to_api(&self) -> Result<types::ExecutionRecoveryRecipientBody> {
        self.to_recipient()?;
        Ok(types::ExecutionRecoveryRecipientBody {
            suite: self.suite.clone(),
            public_key: self.public_key.clone(),
            fingerprint: self.fingerprint.clone(),
            valid_after_unix_millis: i64::try_from(self.valid_after_unix_millis)?,
            valid_before_unix_millis: i64::try_from(self.valid_before_unix_millis)?,
            custodian_uri: self.custodian_uri.clone(),
        })
    }
}

fn load_key(path: &Path) -> Result<LoadedRecoveryKey> {
    require_private_file(path)?;
    let encoded = Zeroizing::new(
        std::fs::read(path).with_context(|| format!("read recovery key {}", path.display()))?,
    );
    let mut document: RecoveryKeyFile =
        serde_json::from_slice(encoded.as_slice()).context("recovery key file is malformed")?;
    if document.schema_version != 1 {
        bail!(
            "unsupported recovery key schema version {}",
            document.schema_version
        );
    }
    let recipient = document.recipient.to_recipient()?;
    let decoded =
        base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(document.private_key.as_bytes());
    document.private_key.zeroize();
    let secret = Zeroizing::new(decoded.context("recovery private key is malformed")?);
    let private_key = PortableExecutionPrivateKey::from_bytes(secret.as_slice())?;
    if private_key.public_key_bytes().as_slice() != recipient.public_key {
        bail!("recovery private key does not match the stored public recipient");
    }
    Ok(LoadedRecoveryKey {
        recipient: document.recipient,
        private_key,
    })
}

fn write_new_key(path: &Path, value: &RecoveryKeyFile) -> Result<()> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    if !parent.is_dir() {
        bail!(
            "recovery key output directory does not exist: {}",
            parent.display()
        );
    }
    let mut temporary =
        tempfile::NamedTempFile::new_in(parent).context("create temporary recovery key")?;
    set_private_file(temporary.path())?;
    let encoded = Zeroizing::new(serde_json::to_vec_pretty(value)?);
    temporary
        .write_all(encoded.as_slice())
        .context("write recovery key")?;
    temporary.flush().context("flush recovery key")?;
    temporary
        .as_file()
        .sync_all()
        .context("sync recovery key")?;
    temporary.persist_noclobber(path).map_err(|error| {
        anyhow::anyhow!(
            "refusing to overwrite recovery key {}: {}",
            path.display(),
            error.error
        )
    })?;
    Ok(())
}

fn random_entropy() -> Result<[u8; 32]> {
    use rand::RngCore as _;
    let mut entropy = [0_u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut entropy);
    if entropy.iter().all(|byte| *byte == 0) {
        bail!("operating system random source returned invalid entropy");
    }
    Ok(entropy)
}

fn now_unix_millis() -> Result<u64> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)?
        .as_millis()
        .try_into()
        .context("system clock exceeds the execution timestamp range")
}

fn nonzero(value: u32, name: &str) -> Result<NonZeroU32> {
    NonZeroU32::new(value).with_context(|| format!("{name} must be greater than zero"))
}

#[cfg(unix)]
fn set_private_file(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt as _;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
    Ok(())
}

#[cfg(not(unix))]
fn set_private_file(_path: &Path) -> Result<()> {
    Ok(())
}

#[cfg(unix)]
fn require_private_file(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt as _;
    let metadata = std::fs::symlink_metadata(path)
        .with_context(|| format!("inspect recovery key {}", path.display()))?;
    if !metadata.is_file() || metadata.file_type().is_symlink() {
        bail!("recovery key {} must be a regular file", path.display());
    }
    let mode = metadata.permissions().mode();
    if mode & 0o077 != 0 {
        bail!(
            "recovery key {} is accessible by group or other users; require mode 0600",
            path.display()
        );
    }
    Ok(())
}

#[cfg(not(unix))]
fn require_private_file(path: &Path) -> Result<()> {
    let metadata = std::fs::symlink_metadata(path)
        .with_context(|| format!("inspect recovery key {}", path.display()))?;
    if !metadata.is_file() || metadata.file_type().is_symlink() {
        bail!("recovery key {} must be a regular file", path.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovery_key_file_is_private_exact_and_never_overwritten() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("recovery-key.json");
        let valid_after = now_unix_millis().unwrap();
        let (recipient, private_key) = PortableExecutionEncryption
            .recipient_from_entropy_with_derived_fingerprint(
                [7; 32],
                valid_after,
                valid_after + MILLIS_PER_DAY,
            )
            .unwrap();
        let document = RecoveryKeyFile {
            schema_version: 1,
            recipient: RecoveryRecipientFile::from_recipient(&recipient),
            private_key: base64::engine::general_purpose::URL_SAFE_NO_PAD
                .encode(private_key.to_bytes()),
        };
        write_new_key(&path, &document).unwrap();
        let loaded = load_key(&path).unwrap();
        assert_eq!(
            loaded.private_key.public_key_bytes().as_slice(),
            recipient.public_key
        );
        assert!(write_new_key(&path, &document).is_err());
        assert_eq!(
            std::fs::read(&path).unwrap(),
            serde_json::to_vec_pretty(&document).unwrap()
        );
        #[cfg(unix)]
        {
            let link = directory.path().join("recovery-key-link.json");
            std::os::unix::fs::symlink(&path, &link).unwrap();
            assert!(load_key(&link).is_err());
        }
    }
}
