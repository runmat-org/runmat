use std::io::{Read as _, Write as _};
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
use runmat_execution_artifact::ProgramExecutionOutputs;
use runmat_server_client::auth::resolve_public_project_id;
use runmat_server_client::execution::public_error;
use runmat_server_client::public_api::types;
use serde::{Deserialize, Serialize};
use zeroize::{Zeroize as _, Zeroizing};

use crate::cli::JobRecoveryCommand;

#[cfg(windows)]
#[path = "recovery_windows.rs"]
mod windows;

const RECOVERY_SUITE: &str = "x25519-hkdf-sha256-aes128gcm-v1";
const MAX_ENVELOPE_BYTES: usize = 64 * 1024;
const MILLIS_PER_DAY: u64 = 24 * 60 * 60 * 1_000;

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct RecoveryKeyFile {
    schema_version: u16,
    recipient: RecoveryRecipientFile,
    #[serde(
        serialize_with = "serialize_secret",
        deserialize_with = "deserialize_secret"
    )]
    private_key: Zeroizing<String>,
}

fn serialize_secret<S: serde::Serializer>(
    value: &Zeroizing<String>,
    serializer: S,
) -> std::result::Result<S::Ok, S::Error> {
    serializer.serialize_str(value.as_str())
}

fn deserialize_secret<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<Zeroizing<String>, D::Error> {
    String::deserialize(deserializer).map(Zeroizing::new)
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
        JobRecoveryCommand::Decrypt {
            key,
            envelope,
            artifact,
            run_id,
            diagnostic,
            output,
        } => {
            let key = load_key(&key)?;
            let envelope = read_bounded(&envelope, MAX_ENVELOPE_BYTES)?;
            let artifact = read_bounded(&artifact, 64 * 1024 * 1024)?;
            let plaintext = Zeroizing::new(
                runmat_execution_artifact::encryption::open_recovery_artifact(
                    &key.private_key,
                    &envelope,
                    &artifact,
                    runmat_execution_artifact::encryption::RecoveryArtifactScope {
                        run_identity: &run_id,
                        key_epoch: 1,
                        purpose: if diagnostic {
                            EncryptionPurpose::DetailedEvent
                        } else {
                            EncryptionPurpose::Result
                        },
                    },
                    64 * 1024 * 1024,
                )?,
            );
            write_new_private_bytes(&output, &plaintext)?;
            println!("Recovered plaintext written to {}", output.display());
            Ok(())
        }
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
        private_key: Zeroizing::new(
            base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(private_bytes.as_slice()),
        ),
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
        println!("public_key\t{}", document.recipient.public_key);
        println!("suite\t{}", document.recipient.suite);
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
    let project_id = resolve_public_project_id(&config, project)?;
    let recovery = client
        .api()
        .get_run_recovery(project_id.as_str(), run_id)
        .await
        .map_err(public_error)?
        .into_inner();
    if recovery.run_id != run_id
        || runmat_server_client::ids::ProjectId::parse(&recovery.project_id)
            .context("recovery contains an invalid public project identity")?
            != project_id
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
            .download_artifact(
                project_id.as_str(),
                &artifact_id,
                &recovery.recipient_fingerprint,
            )
            .await?;
        let plaintext = super::crypto::open_run_object(
            &run_key,
            run_id,
            EncryptionPurpose::Result,
            &ciphertext,
        )?;
        let outputs: ProgramExecutionOutputs =
            serde_json::from_slice(&plaintext).context("recovered result payload is malformed")?;
        outputs
            .validate()
            .context("recovered result payload failed integrity validation")?;
        if !json {
            super::console::print_outputs(&outputs)?;
        }
        let result_object_count = outputs.objects.len();
        let result = outputs.response;
        if json {
            println!(
                "{}",
                serde_json::to_string(&serde_json::json!({
                    "runId": run_id,
                    "result": result,
                    "resultObjectCount": result_object_count,
                    "console": outputs.console,
                    "consoleScopes": outputs.console_scopes,
                    "consoleBudget": outputs.console_budget,
                    "delegatedConsoleRoot": outputs.delegated_console_root,
                }))?
            );
            super::result::ensure_terminal_success(result, super::result::ResultOrigin::Recovered)?;
        } else {
            println!(
                "{}",
                super::result::format_terminal_result(
                    result,
                    super::result::ResultOrigin::Recovered,
                )?
            );
        }
        return Ok(());
    }
    if let Some(artifact_id) = recovery.diagnostic_artifact_id {
        let ciphertext = client
            .download_artifact(
                project_id.as_str(),
                &artifact_id,
                &recovery.recipient_fingerprint,
            )
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
    #[cfg(windows)]
    let file = runmat_filesystem::NativeFsProvider
        .open_owner_private_file(path)
        .context("open private recovery key")?;
    #[cfg(not(windows))]
    let file = std::fs::OpenOptions::new()
        .read(true)
        .open(path)
        .context("open private recovery key")?;
    if !file.metadata()?.is_file() {
        bail!("recovery key must be a regular file");
    }
    let encoded = Zeroizing::new(read_file_bounded(file, MAX_ENVELOPE_BYTES)?);
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
    let encoded = Zeroizing::new(serde_json::to_vec_pretty(value)?);
    write_new_private_bytes(path, &encoded)
}

fn read_bounded(path: &Path, maximum: usize) -> Result<Vec<u8>> {
    read_file_bounded(
        std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?,
        maximum,
    )
}

fn read_file_bounded(file: std::fs::File, maximum: usize) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    file.take(maximum as u64 + 1).read_to_end(&mut bytes)?;
    if bytes.len() > maximum {
        bail!("input exceeds the {} byte limit", maximum);
    }
    Ok(bytes)
}

fn write_new_private_bytes(path: &Path, encoded: &[u8]) -> Result<()> {
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
    #[cfg(windows)]
    let mut temporary = windows::create_temporary(parent)?;
    #[cfg(not(windows))]
    let mut temporary = {
        let temporary =
            tempfile::NamedTempFile::new_in(parent).context("create temporary recovery key")?;
        set_private_file(temporary.path())?;
        temporary
    };
    temporary.write_all(encoded).context("write recovery key")?;
    temporary.flush().context("flush recovery key")?;
    temporary
        .as_file()
        .sync_all()
        .context("sync recovery key")?;
    temporary.persist_noclobber(path).map_err(|error| {
        let context = match error.error.kind() {
            std::io::ErrorKind::AlreadyExists => {
                format!("refusing to overwrite recovery key {path:?}")
            }
            _ => format!("persist private recovery key {path:?}"),
        };
        anyhow::Error::new(error.error).context(context)
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

#[cfg(not(any(unix, windows)))]
fn set_private_file(_path: &Path) -> Result<()> {
    bail!("private recovery-file creation requires an OS access-control adapter on this platform")
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
        return Err(private_file_permissions_error(path));
    }
    Ok(())
}

#[cfg(unix)]
fn private_file_permissions_error(path: &Path) -> anyhow::Error {
    anyhow::anyhow!(
        "recovery key {:?} is accessible by group or other users; require mode 0600; {}",
        path,
        private_mode_guidance(path)
    )
}

#[cfg(unix)]
fn private_mode_guidance(path: &Path) -> String {
    let Some(path) = path.to_str() else {
        return "the pathname contains non-UTF-8 bytes: use chmod -- 600 with the original filename selected by your shell's filename completion; no lossy copy-pastable pathname is shown".into();
    };
    if path.chars().any(char::is_control) {
        return "the pathname contains control characters: use chmod -- 600 with the original filename selected by your shell's filename completion; no copy-pastable pathname is shown".into();
    }
    // POSIX single-quoted arguments preserve spaces and shell metacharacters.
    // A literal apostrophe closes the quote, emits one quoted apostrophe,
    // then reopens it. `--` also protects relative names beginning with '-'.
    format!("run: chmod -- 600 '{}'", path.replace('\'', "'\"'\"'"))
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
            private_key: Zeroizing::new(
                base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(private_key.to_bytes()),
            ),
        };
        write_new_key(&path, &document).unwrap();
        let loaded = load_key(&path).unwrap();
        assert_eq!(
            loaded.private_key.public_key_bytes().as_slice(),
            recipient.public_key
        );
        let overwrite = write_new_key(&path, &document).unwrap_err();
        assert_eq!(
            overwrite.downcast_ref::<std::io::Error>().unwrap().kind(),
            std::io::ErrorKind::AlreadyExists
        );
        assert_eq!(
            std::fs::read(&path).unwrap(),
            serde_json::to_vec_pretty(&document).unwrap()
        );
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStringExt as _;
            use std::os::unix::fs::PermissionsExt as _;

            let downloaded = directory.path().join("browser owner's key.json");
            write_new_key(&downloaded, &document).unwrap();
            std::fs::set_permissions(&downloaded, std::fs::Permissions::from_mode(0o644)).unwrap();
            let refusal = load_key(&downloaded)
                .err()
                .expect("insecure import refused")
                .to_string();
            // Execute the exact diagnostic guidance, including POSIX quoting
            // of the filename's space and literal apostrophe.
            let guidance = private_mode_guidance(&downloaded);
            assert!(guidance.starts_with("run: chmod -- 600 '"));
            assert!(guidance.ends_with("/browser owner'\"'\"'s key.json'"));
            assert!(refusal.contains(&guidance));
            let emitted = guidance.strip_prefix("run: ").unwrap();
            assert!(std::process::Command::new("sh")
                .arg("-c")
                .arg(emitted)
                .status()
                .unwrap()
                .success());
            assert_eq!(
                std::fs::metadata(&downloaded).unwrap().permissions().mode() & 0o777,
                0o600
            );
            load_key(&downloaded).unwrap();

            let non_utf8 = directory
                .path()
                .join(std::ffi::OsString::from_vec(b"download-\xff.json".to_vec()));
            // Unix path values can contain non-UTF-8 bytes even where the
            // filesystem (including APFS) rejects those names. Exercise the
            // production diagnostic independently of filesystem encoding.
            let refusal = private_file_permissions_error(&non_utf8).to_string();
            assert!(refusal.contains("non-UTF-8"));
            assert!(refusal.contains("no lossy copy-pastable pathname is shown"));
            assert!(!refusal.contains('\u{fffd}'));
            assert!(!private_mode_guidance(&non_utf8).starts_with("run: "));

            let link = directory.path().join("recovery-key-link.json");
            std::os::unix::fs::symlink(&path, &link).unwrap();
            assert!(load_key(&link).is_err());
        }
    }
}
