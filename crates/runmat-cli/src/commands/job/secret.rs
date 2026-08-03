use std::path::PathBuf;

use anyhow::{Context, Result};
use base64::Engine as _;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SavedRemoteRun {
    pub run_id: String,
    pub server_url: String,
    pub project_id: String,
    pub endpoint_fingerprint: String,
    pub run_key: String,
}

impl SavedRemoteRun {
    pub fn new(
        run_id: String,
        server_url: String,
        project_id: String,
        endpoint_fingerprint: String,
        run_key: &runmat_execution_artifact::encryption::RunKeyMaterial,
    ) -> Self {
        Self {
            run_id,
            server_url,
            project_id,
            endpoint_fingerprint,
            run_key: base64::engine::general_purpose::URL_SAFE_NO_PAD
                .encode(run_key.expose_for_recipient_envelope()),
        }
    }

    pub fn key(&self) -> Result<runmat_execution_artifact::encryption::RunKeyMaterial> {
        let bytes: [u8; 32] = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(&self.run_key)
            .context("saved run secret is malformed")?
            .try_into()
            .map_err(|_| anyhow::anyhow!("saved run secret has an invalid length"))?;
        runmat_execution_artifact::encryption::RunKeyMaterial::from_entropy(bytes)
            .map_err(anyhow::Error::from)
    }
}

pub fn save(value: &SavedRemoteRun) -> Result<()> {
    let path = path(&value.run_id)?;
    let parent = path.parent().context("saved run path has no parent")?;
    std::fs::create_dir_all(parent).context("create saved run directory")?;
    set_private_directory(parent)?;
    std::fs::write(&path, serde_json::to_vec_pretty(value)?)
        .context("write saved remote run secret")?;
    set_private_file(&path)?;
    Ok(())
}

pub fn load(run_id: &str) -> Result<SavedRemoteRun> {
    let path = path(run_id)?;
    serde_json::from_slice(
        &std::fs::read(&path)
            .with_context(|| format!("no local attach secret is available for run {run_id}"))?,
    )
    .context("read saved remote run secret")
}

fn path(run_id: &str) -> Result<PathBuf> {
    if run_id.is_empty()
        || run_id.len() > 256
        || !run_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        anyhow::bail!("remote run id is malformed");
    }
    let root = if let Some(path) = std::env::var_os("RUNMAT_CLI_CONFIG_DIR") {
        PathBuf::from(path)
    } else {
        dirs::config_dir()
            .context("unable to locate the user configuration directory")?
            .join("runmat")
    };
    Ok(root.join("execution-runs").join(format!("{run_id}.json")))
}

#[cfg(unix)]
fn set_private_directory(path: &std::path::Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt as _;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))?;
    Ok(())
}

#[cfg(not(unix))]
fn set_private_directory(_path: &std::path::Path) -> Result<()> {
    Ok(())
}

#[cfg(unix)]
fn set_private_file(path: &std::path::Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt as _;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
    Ok(())
}

#[cfg(not(unix))]
fn set_private_file(_path: &std::path::Path) -> Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attach_secret_round_trips_without_exposing_it_in_job_identity() {
        let temp = tempfile::tempdir().unwrap();
        std::env::set_var("RUNMAT_CLI_CONFIG_DIR", temp.path());
        let key =
            runmat_execution_artifact::encryption::RunKeyMaterial::from_entropy([7; 32]).unwrap();
        let saved = SavedRemoteRun::new(
            "run_123".into(),
            "https://example.invalid".into(),
            "project_1".into(),
            "fingerprint".into(),
            &key,
        );
        save(&saved).unwrap();
        let loaded = load("run_123").unwrap();
        assert_eq!(
            loaded.key().unwrap().expose_for_recipient_envelope(),
            key.expose_for_recipient_envelope()
        );
        std::env::remove_var("RUNMAT_CLI_CONFIG_DIR");
    }
}
