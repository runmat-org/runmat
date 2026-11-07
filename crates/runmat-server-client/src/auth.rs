use anyhow::{Context, Result};
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use chrono::{DateTime, Utc};
use rand::RngCore;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::env;
use std::fs;
#[cfg(unix)]
use std::io::Write;
#[cfg(unix)]
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use tokio::net::TcpListener;
use tokio::time::{timeout, Duration};
use url::Url;
use uuid::Uuid;

mod organization;
pub use organization::configured_public_org_id;

use crate::endpoint::ServerEndpoint;
use crate::public_api;

pub const DEFAULT_SERVER_URL: &str = "https://api.runmat.com";

#[derive(Default, Serialize, Deserialize, Clone, Debug)]
pub struct RemoteConfig {
    pub server_url: Option<String>,
    pub org_id: Option<Uuid>,
    pub project_id: Option<Uuid>,
    pub credential_store: Option<CredentialStoreMode>,
    pub token_expires_at: Option<DateTime<Utc>>,
    pub token_endpoint: Option<String>,
    pub token_client_id: Option<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CredentialStoreMode {
    Auto,
    Secure,
    File,
    Memory,
}

impl Default for CredentialStoreMode {
    fn default() -> Self {
        Self::Auto
    }
}

impl std::str::FromStr for CredentialStoreMode {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "auto" => Ok(Self::Auto),
            "secure" => Ok(Self::Secure),
            "file" => Ok(Self::File),
            "memory" => Ok(Self::Memory),
            _ => anyhow::bail!("invalid credential store mode: {s}"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct AuthToken {
    pub access_token: String,
    pub refresh_token: Option<String>,
    pub expires_at: Option<DateTime<Utc>>,
    pub token_endpoint: Option<String>,
    pub client_id: Option<String>,
}

#[derive(Debug, thiserror::Error)]
#[error("No stored credentials. Run `runmat login` or set RUNMAT_API_KEY.")]
pub struct SignInRequired;

impl RemoteConfig {
    pub fn load() -> Result<Self> {
        let path = config_path()?;
        if !path.exists() {
            return Ok(Self::default());
        }
        let contents = fs::read_to_string(&path)
            .with_context(|| format!("Failed to read config {}", path.display()))?;
        serde_json::from_str(&contents).context("Failed to parse remote config")
    }

    pub fn save(&self) -> Result<()> {
        let path = config_path()?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create config dir {}", parent.display()))?;
        }
        let contents = serde_json::to_string_pretty(self).context("Failed to serialize config")?;
        fs::write(&path, contents)
            .with_context(|| format!("Failed to write config {}", path.display()))?;
        Ok(())
    }
}

pub fn config_path() -> Result<PathBuf> {
    if let Ok(dir) = env::var("RUNMAT_CLI_CONFIG_DIR") {
        return Ok(PathBuf::from(dir).join("remote.json"));
    }
    let base = dirs::config_dir().context("Unable to locate config directory")?;
    Ok(base.join("runmat").join("remote.json"))
}

pub async fn execute_login(
    server: Option<String>,
    api_key: Option<String>,
    email: Option<String>,
    org: Option<Uuid>,
    project: Option<Uuid>,
    credential_store: Option<CredentialStoreMode>,
) -> Result<()> {
    let mut config = RemoteConfig::load()?;
    let server_url = resolve_server_url(&config, server)?;
    let credential_store = credential_store
        .or(config.credential_store)
        .unwrap_or(CredentialStoreMode::Auto);
    let auth = if let Some(api_key) = api_key {
        AuthToken {
            access_token: api_key,
            refresh_token: None,
            expires_at: None,
            token_endpoint: None,
            client_id: None,
        }
    } else if let Ok(env_token) = env::var("RUNMAT_API_KEY") {
        AuthToken {
            access_token: env_token,
            refresh_token: None,
            expires_at: None,
            token_endpoint: None,
            client_id: None,
        }
    } else {
        interactive_login(&server_url, email).await?
    };
    store_token(&server_url, auth.access_token.trim(), credential_store)?;
    if let Some(refresh) = auth.refresh_token.as_deref() {
        store_refresh_token(&server_url, refresh, credential_store)?;
    } else {
        clear_refresh_token(&server_url, credential_store)?;
    }
    config.credential_store = Some(credential_store);
    config.token_expires_at = auth.expires_at;
    config.token_endpoint = auth.token_endpoint;
    config.token_client_id = auth.client_id;
    config.server_url = Some(server_url.clone());
    if let Some(org) = org {
        config.org_id = Some(org);
    }
    if let Some(project) = project {
        config.project_id = Some(project);
    }
    config.save()?;
    println!("Stored credentials for {server_url}");
    Ok(())
}

pub fn resolve_server_url(config: &RemoteConfig, override_value: Option<String>) -> Result<String> {
    let value = override_value
        .or_else(|| {
            env::var("RUNMAT_SERVER_URL")
                .ok()
                .filter(|value| !value.is_empty())
        })
        .or_else(|| config.server_url.clone())
        .unwrap_or_else(|| DEFAULT_SERVER_URL.to_owned());
    Ok(ServerEndpoint::parse(&value)?.into())
}

pub fn resolve_org_id(config: &RemoteConfig, override_value: Option<Uuid>) -> Result<Uuid> {
    organization::configured_org_uuid(config, override_value)?.context(
        "Organization id not configured. Use --org, set RUNMAT_ORG_ID, or run `runmat login --org <id>`."
    )
}

pub fn resolve_project_id(config: &RemoteConfig, override_value: Option<Uuid>) -> Result<Uuid> {
    if let Some(value) = override_value {
        return Ok(value);
    }
    if let Ok(value) = env::var("RUNMAT_PROJECT_ID") {
        if !value.is_empty() {
            return parse_project_id_input(&value).context("Invalid RUNMAT_PROJECT_ID");
        }
    }
    config.project_id.context(
        "Project id not configured. Use --project, set RUNMAT_PROJECT_ID, or run `runmat login --project <id>`."
    )
}

/// Accept canonical public IDs and the existing UUID CLI/config shorthand.
pub fn parse_org_id_input(value: &str) -> Result<Uuid> {
    if let Ok(value) = Uuid::parse_str(value) {
        return Ok(value);
    }
    crate::ids::OrgId::parse(value)?
        .as_uuid()
        .map_err(Into::into)
}

pub fn parse_project_id_input(value: &str) -> Result<Uuid> {
    if let Ok(value) = Uuid::parse_str(value) {
        return Ok(value);
    }
    crate::ids::ProjectId::parse(value)?
        .as_uuid()
        .map_err(Into::into)
}

pub fn parse_fs_version_id_input(value: &str) -> Result<Uuid> {
    if let Ok(value) = Uuid::parse_str(value) {
        return Ok(value);
    }
    crate::ids::FsVersionId::parse(value)?
        .as_uuid()
        .map_err(Into::into)
}

pub fn parse_snapshot_id_input(value: &str) -> Result<Uuid> {
    if let Ok(value) = Uuid::parse_str(value) {
        return Ok(value);
    }
    crate::ids::SnapshotId::parse(value)?
        .as_uuid()
        .map_err(Into::into)
}

pub fn resolve_public_org_id(
    config: &RemoteConfig,
    override_value: Option<Uuid>,
) -> Result<crate::ids::OrgId> {
    Ok(crate::ids::OrgId::from_uuid(resolve_org_id(
        config,
        override_value,
    )?))
}

pub fn resolve_public_project_id(
    config: &RemoteConfig,
    override_value: Option<Uuid>,
) -> Result<crate::ids::ProjectId> {
    Ok(crate::ids::ProjectId::from_uuid(resolve_project_id(
        config,
        override_value,
    )?))
}

pub async fn resolve_auth_token(config: &mut RemoteConfig, server_url: &str) -> Result<String> {
    if let Ok(value) = env::var("RUNMAT_API_KEY") {
        if !value.is_empty() {
            return Ok(value);
        }
    }
    let credential_store = config.credential_store.unwrap_or(CredentialStoreMode::Auto);
    let token = load_token(server_url, credential_store)?;
    let refresh = load_refresh_token(server_url, credential_store)?;
    let expired = config
        .token_expires_at
        .map(|expiry| expiry <= Utc::now() + chrono::Duration::seconds(30))
        .unwrap_or(false);
    if let Some(token) = token.as_ref() {
        if !expired {
            return Ok(token.clone());
        }
    }
    match refresh_stored_token(config, server_url, credential_store, refresh).await? {
        Some(refreshed) => Ok(refreshed),
        None => token.ok_or_else(|| SignInRequired.into()),
    }
}

/// A new access token from the stored refresh token, whatever the stored expiry says: for a
/// caller whose current token the server has just rejected. An API key (`RUNMAT_API_KEY`) or a
/// sign-in without a refresh token cannot be refreshed and returns [`SignInRequired`].
pub async fn refresh_auth_token(config: &mut RemoteConfig, server_url: &str) -> Result<String> {
    if env::var("RUNMAT_API_KEY").is_ok_and(|value| !value.is_empty()) {
        return Err(SignInRequired.into());
    }
    let credential_store = config.credential_store.unwrap_or(CredentialStoreMode::Auto);
    let refresh = load_refresh_token(server_url, credential_store)?;
    refresh_stored_token(config, server_url, credential_store, refresh)
        .await?
        .ok_or_else(|| SignInRequired.into())
}

/// Exchange `refresh` for a new access token and store both, with the new expiry. `None` when
/// there is no refresh token or the sign-in has no token endpoint.
async fn refresh_stored_token(
    config: &mut RemoteConfig,
    server_url: &str,
    credential_store: CredentialStoreMode,
    refresh: Option<String>,
) -> Result<Option<String>> {
    let (Some(refresh), Some(endpoint), Some(client_id)) = (
        refresh,
        config.token_endpoint.clone(),
        config.token_client_id.clone(),
    ) else {
        return Ok(None);
    };
    let refreshed = refresh_access_token(&endpoint, &client_id, &refresh).await?;
    store_token(server_url, &refreshed.access_token, credential_store)?;
    if let Some(new_refresh) = refreshed.refresh_token.as_deref() {
        store_refresh_token(server_url, new_refresh, credential_store)?;
    }
    config.token_expires_at = refreshed.expires_at;
    config.save()?;
    Ok(Some(refreshed.access_token))
}

pub fn build_public_client(server_url: &str, token: &str) -> Result<public_api::Client> {
    let endpoint = ServerEndpoint::parse(server_url)?;
    let mut headers = HeaderMap::new();
    let mut value = HeaderValue::from_str(&format!("Bearer {token}"))
        .context("Invalid authorization header")?;
    value.set_sensitive(true);
    headers.insert(AUTHORIZATION, value);
    let client = reqwest::Client::builder()
        .default_headers(headers)
        .build()
        .context("Failed to build HTTP client")?;
    Ok(public_api::Client::new_with_client(
        endpoint.as_str(),
        client,
    ))
}

pub fn map_public_error<T: std::fmt::Debug>(err: public_api::Error<T>) -> anyhow::Error {
    anyhow::anyhow!(err.to_string())
}

fn load_token(server_url: &str, mode: CredentialStoreMode) -> Result<Option<String>> {
    match mode {
        CredentialStoreMode::Auto => load_token_auto(server_url),
        CredentialStoreMode::Secure => load_token_keyring(server_url),
        CredentialStoreMode::File => load_token_file(server_url),
        CredentialStoreMode::Memory => Ok(env::var(memory_token_key(server_url)?).ok()),
    }
}

fn load_refresh_token(server_url: &str, mode: CredentialStoreMode) -> Result<Option<String>> {
    match mode {
        CredentialStoreMode::Auto => load_refresh_token_auto(server_url),
        CredentialStoreMode::Secure => load_refresh_token_keyring(server_url),
        CredentialStoreMode::File => load_refresh_token_file(server_url),
        CredentialStoreMode::Memory => Ok(env::var(memory_refresh_key(server_url)?).ok()),
    }
}

fn store_token(server_url: &str, token: &str, mode: CredentialStoreMode) -> Result<()> {
    match mode {
        CredentialStoreMode::Auto => store_token_auto(server_url, token),
        CredentialStoreMode::Secure => store_token_keyring(server_url, token),
        CredentialStoreMode::File => store_token_file(server_url, token),
        CredentialStoreMode::Memory => {
            unsafe { env::set_var(memory_token_key(server_url)?, token) };
            Ok(())
        }
    }
}

fn store_refresh_token(server_url: &str, token: &str, mode: CredentialStoreMode) -> Result<()> {
    match mode {
        CredentialStoreMode::Auto => store_refresh_token_auto(server_url, token),
        CredentialStoreMode::Secure => store_refresh_token_keyring(server_url, token),
        CredentialStoreMode::File => store_refresh_token_file(server_url, token),
        CredentialStoreMode::Memory => {
            unsafe { env::set_var(memory_refresh_key(server_url)?, token) };
            Ok(())
        }
    }
}

fn clear_refresh_token(server_url: &str, mode: CredentialStoreMode) -> Result<()> {
    match mode {
        CredentialStoreMode::Auto => clear_refresh_token_auto(server_url),
        CredentialStoreMode::Secure => clear_refresh_token_keyring(server_url),
        CredentialStoreMode::File => clear_refresh_token_file(server_url),
        CredentialStoreMode::Memory => {
            unsafe { env::remove_var(memory_refresh_key(server_url)?) };
            Ok(())
        }
    }
}

fn token_file_path(server_url: &str) -> Result<PathBuf> {
    let path = config_path()?;
    let parent = path.parent().context("Missing config directory")?;
    Ok(parent.join(format!("token-{}.json", server_credential_key(server_url)?)))
}

#[derive(Serialize, Deserialize)]
struct FileCredentialPayload {
    access_token: Option<String>,
    refresh_token: Option<String>,
}

fn load_token_file(server_url: &str) -> Result<Option<String>> {
    let path = token_file_path(server_url)?;
    if !path.exists() {
        return Ok(None);
    }
    let payload: FileCredentialPayload =
        serde_json::from_str(&fs::read_to_string(&path).context("Failed to read token file")?)
            .context("Failed to parse token file")?;
    Ok(payload.access_token)
}

fn load_refresh_token_file(server_url: &str) -> Result<Option<String>> {
    let path = token_file_path(server_url)?;
    if !path.exists() {
        return Ok(None);
    }
    let payload: FileCredentialPayload =
        serde_json::from_str(&fs::read_to_string(&path).context("Failed to read token file")?)
            .context("Failed to parse token file")?;
    Ok(payload.refresh_token)
}

fn store_token_file(server_url: &str, token: &str) -> Result<()> {
    let path = token_file_path(server_url)?;
    let refresh = load_refresh_token_file(server_url).ok().flatten();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).context("Failed to create token dir")?;
    }
    let payload = FileCredentialPayload {
        access_token: Some(token.to_string()),
        refresh_token: refresh,
    };
    write_token_payload_file(&path, &payload)?;
    Ok(())
}

fn store_refresh_token_file(server_url: &str, token: &str) -> Result<()> {
    let path = token_file_path(server_url)?;
    let access = load_token_file(server_url).ok().flatten();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).context("Failed to create token dir")?;
    }
    let payload = FileCredentialPayload {
        access_token: access,
        refresh_token: Some(token.to_string()),
    };
    write_token_payload_file(&path, &payload)?;
    Ok(())
}

fn clear_refresh_token_file(server_url: &str) -> Result<()> {
    let path = token_file_path(server_url)?;
    if !path.exists() {
        return Ok(());
    }
    let access = load_token_file(server_url).ok().flatten();
    let payload = FileCredentialPayload {
        access_token: access,
        refresh_token: None,
    };
    write_token_payload_file(&path, &payload)?;
    Ok(())
}

fn load_token_auto(server_url: &str) -> Result<Option<String>> {
    match load_token_keyring(server_url) {
        Ok(Some(value)) => Ok(Some(value)),
        Ok(None) | Err(_) => load_token_file(server_url),
    }
}

fn load_refresh_token_auto(server_url: &str) -> Result<Option<String>> {
    match load_refresh_token_keyring(server_url) {
        Ok(Some(value)) => Ok(Some(value)),
        Ok(None) | Err(_) => load_refresh_token_file(server_url),
    }
}

fn store_token_auto(server_url: &str, token: &str) -> Result<()> {
    match store_token_keyring(server_url, token) {
        Ok(()) => Ok(()),
        Err(_) => {
            // Evict any stale keyring entry so load_token_auto won't return it on
            // the next call (keyring is preferred over file in the load path).
            let _ = clear_token_keyring(server_url);
            store_token_file(server_url, token)
        }
    }
}

fn store_refresh_token_auto(server_url: &str, token: &str) -> Result<()> {
    match store_refresh_token_keyring(server_url, token) {
        Ok(()) => Ok(()),
        Err(_) => {
            let _ = clear_refresh_token_keyring(server_url);
            store_refresh_token_file(server_url, token)
        }
    }
}

fn clear_refresh_token_auto(server_url: &str) -> Result<()> {
    // Clear from both backends unconditionally: a previous store fallback may
    // have left data in the file while the keyring still holds a stale entry,
    // or vice-versa.  Treat keyring errors as non-fatal (keyring may be
    // unavailable) and return the file result as the authoritative outcome.
    let _ = clear_refresh_token_keyring(server_url);
    clear_refresh_token_file(server_url)
}

fn write_token_payload_file(path: &Path, payload: &FileCredentialPayload) -> Result<()> {
    let contents = serde_json::to_vec_pretty(payload)?;
    #[cfg(unix)]
    {
        let mut file = fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .mode(0o600)
            .open(path)
            .context("Failed to write token file")?;
        file.write_all(&contents)
            .context("Failed to write token file")?;
        fs::set_permissions(path, fs::Permissions::from_mode(0o600))
            .context("Failed to set token file permissions")?;
    }
    #[cfg(not(unix))]
    {
        fs::write(path, contents).context("Failed to write token file")?;
    }
    Ok(())
}

fn load_token_keyring(server_url: &str) -> Result<Option<String>> {
    let entry = keyring_entry(server_url)?;
    match entry.get_password() {
        Ok(value) => Ok(Some(value)),
        Err(keyring::Error::NoEntry) => Ok(None),
        Err(err) => Err(err).context("Failed to access keyring"),
    }
}

fn load_refresh_token_keyring(server_url: &str) -> Result<Option<String>> {
    let entry = keyring_refresh_entry(server_url)?;
    match entry.get_password() {
        Ok(value) => Ok(Some(value)),
        Err(keyring::Error::NoEntry) => Ok(None),
        Err(err) => Err(err).context("Failed to access keyring"),
    }
}

fn store_token_keyring(server_url: &str, token: &str) -> Result<()> {
    let entry = keyring_entry(server_url)?;
    entry
        .set_password(token)
        .context("Failed to store credentials")
}

fn store_refresh_token_keyring(server_url: &str, token: &str) -> Result<()> {
    let entry = keyring_refresh_entry(server_url)?;
    entry
        .set_password(token)
        .context("Failed to store refresh token")
}

fn clear_refresh_token_keyring(server_url: &str) -> Result<()> {
    let entry = keyring_refresh_entry(server_url)?;
    match entry.delete_password() {
        Ok(_) => Ok(()),
        Err(keyring::Error::NoEntry) => Ok(()),
        Err(err) => Err(err).context("Failed to clear refresh token"),
    }
}

fn clear_token_keyring(server_url: &str) -> Result<()> {
    let entry = keyring_entry(server_url)?;
    match entry.delete_password() {
        Ok(_) => Ok(()),
        Err(keyring::Error::NoEntry) => Ok(()),
        Err(err) => Err(err).context("Failed to clear token"),
    }
}

fn server_credential_key(server_url: &str) -> Result<String> {
    let endpoint = ServerEndpoint::parse(server_url)?;
    let mut digest = Sha256::new();
    digest.update(b"runmat.server-credential.v1\0");
    digest.update(endpoint.as_str().as_bytes());
    Ok(hex::encode(digest.finalize()))
}

fn memory_token_key(server_url: &str) -> Result<String> {
    Ok(format!(
        "RUNMAT_MEMORY_TOKEN_{}",
        server_credential_key(server_url)?
    ))
}

fn memory_refresh_key(server_url: &str) -> Result<String> {
    Ok(format!(
        "RUNMAT_MEMORY_REFRESH_{}",
        server_credential_key(server_url)?
    ))
}

fn keyring_entry(server_url: &str) -> Result<keyring::Entry> {
    let account = format!("server:{}:access", server_credential_key(server_url)?);
    keyring::Entry::new("runmat", &account).context("Failed to open keyring")
}

fn keyring_refresh_entry(server_url: &str) -> Result<keyring::Entry> {
    let account = format!("server:{}:refresh", server_credential_key(server_url)?);
    keyring::Entry::new("runmat", &account).context("Failed to open keyring")
}

async fn interactive_login(server_url: &str, email: Option<String>) -> Result<AuthToken> {
    let client = public_api::Client::new(server_url);
    let response = client
        .auth_resolve(&public_api::types::AuthResolveRequest {
            email,
            client_kind: Some("cli".to_string()),
        })
        .await
        .map_err(map_public_error)?
        .into_inner();
    if response.mode == "workos" {
        return workos_device_login(
            response
                .client_id
                .as_deref()
                .context("Missing WorkOS client ID")?,
        )
        .await;
    }
    let pkce = generate_pkce();
    let redirect_uri = default_loopback_redirect_uri();
    let client_id = response
        .client_id
        .clone()
        .context("Missing client_id in auth resolve response")?;
    let (authorize_url, expected_state) = build_pkce_authorize_url(
        &response.redirect_url,
        &client_id,
        redirect_uri.as_str(),
        response.audience.as_deref(),
        response.scope.as_deref(),
        &pkce,
    )?;
    let (host, port) = loopback_host_port(&redirect_uri)?;
    let listener = TcpListener::bind((host.as_str(), port))
        .await
        .context("Failed to bind local callback listener")?;
    let auth_url = authorize_url.as_str().to_string();
    webbrowser::open(&auth_url).context("Failed to open browser")?;
    let code = listen_for_auth_code(listener, &expected_state).await?;
    let token_endpoint = discover_token_endpoint(&authorize_url).await?;
    let token = exchange_code_for_token(
        &token_endpoint,
        &code,
        &client_id,
        redirect_uri.as_str(),
        &pkce.verifier,
    )
    .await?;
    let expires_at = token_expiry(&token);
    Ok(AuthToken {
        access_token: token.access_token,
        refresh_token: token.refresh_token,
        expires_at,
        token_endpoint: Some(token_endpoint),
        client_id: Some(client_id),
    })
}

/// WorkOS's public CLI flow never embeds a server API key. Refresh credentials
/// reuse the existing credential-store implementation and rotation handling.
/// https://workos.com/docs/authkit/cli-auth
async fn workos_device_login(client_id: &str) -> Result<AuthToken> {
    #[derive(Deserialize)]
    struct DeviceAuthorization {
        device_code: String,
        user_code: String,
        verification_uri: String,
        verification_uri_complete: Option<String>,
        expires_in: u64,
        interval: Option<u64>,
    }
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(15))
        .redirect(reqwest::redirect::Policy::none())
        .build()?;
    let response = client
        .post("https://api.workos.com/user_management/authorize/device")
        .form(&[("client_id", client_id)])
        .send()
        .await?
        .error_for_status()?;
    let device = response.json::<DeviceAuthorization>().await?;
    let uri = device
        .verification_uri_complete
        .as_deref()
        .unwrap_or(&device.verification_uri);
    let url = Url::parse(uri)?;
    anyhow::ensure!(url.scheme() == "https", "Invalid verification URL");
    println!(
        "Open {} and enter code {}",
        device.verification_uri, device.user_code
    );
    let _ = webbrowser::open(url.as_str());
    let deadline = tokio::time::Instant::now() + Duration::from_secs(device.expires_in.min(900));
    let mut interval = device.interval.unwrap_or(5).clamp(1, 30);
    let token_endpoint = "https://api.workos.com/user_management/authenticate";
    loop {
        anyhow::ensure!(
            tokio::time::Instant::now() < deadline,
            "Login code expired. Please run login again."
        );
        tokio::time::sleep(Duration::from_secs(interval)).await;
        let response = client
            .post(token_endpoint)
            .form(&[
                ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
                ("client_id", client_id),
                ("device_code", device.device_code.as_str()),
            ])
            .send()
            .await?;
        if response.status().is_success() {
            let token = response.json::<TokenResponse>().await?;
            let expires_at = token_expiry(&token);
            return Ok(AuthToken {
                access_token: token.access_token,
                refresh_token: token.refresh_token,
                expires_at,
                token_endpoint: Some(token_endpoint.to_owned()),
                client_id: Some(client_id.to_owned()),
            });
        }
        let error = response.json::<serde_json::Value>().await?;
        match error
            .get("error")
            .or_else(|| error.get("code"))
            .and_then(serde_json::Value::as_str)
        {
            Some("authorization_pending") => continue,
            Some("slow_down") => {
                interval = (interval + 5).min(30);
            }
            _ => anyhow::bail!("Login was declined or expired. Please run login again."),
        }
    }
}

// JWT payload is used only to schedule refresh, never to authorize a request.
// RunMat's backend still verifies signature, issuer and recipient independently.
fn token_expiry(token: &TokenResponse) -> Option<DateTime<Utc>> {
    if let Some(seconds) = token.expires_in {
        return Some(Utc::now() + chrono::Duration::seconds(seconds));
    }
    let payload = URL_SAFE_NO_PAD
        .decode(token.access_token.split('.').nth(1)?)
        .ok()?;
    let claims: serde_json::Value = serde_json::from_slice(&payload).ok()?;
    DateTime::from_timestamp(claims.get("exp")?.as_i64()?, 0)
}

#[derive(Debug)]
struct PkceState {
    verifier: String,
    challenge: String,
    state: String,
}

fn generate_pkce() -> PkceState {
    let mut verifier_bytes = [0u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut verifier_bytes);
    let verifier = URL_SAFE_NO_PAD.encode(verifier_bytes);
    let challenge = URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()));
    let mut state_bytes = [0u8; 16];
    rand::rngs::OsRng.fill_bytes(&mut state_bytes);
    let state = URL_SAFE_NO_PAD.encode(state_bytes);
    PkceState {
        verifier,
        challenge,
        state,
    }
}

fn build_pkce_authorize_url(
    authorize_base: &str,
    client_id: &str,
    redirect_uri: &str,
    audience: Option<&str>,
    scope: Option<&str>,
    pkce: &PkceState,
) -> Result<(Url, String)> {
    let mut url = Url::parse(authorize_base).context("Invalid redirect URL")?;
    let scope = scope.unwrap_or("openid profile email offline_access");
    let mut scopes = scope.split_whitespace().collect::<Vec<_>>();
    if !scopes.contains(&"offline_access") {
        scopes.push("offline_access");
    }
    url.set_query(None);
    {
        let mut pairs = url.query_pairs_mut();
        pairs.append_pair("client_id", client_id);
        pairs.append_pair("redirect_uri", redirect_uri);
        pairs.append_pair("response_type", "code");
        pairs.append_pair("scope", &scopes.join(" "));
        if let Some(audience) = audience.filter(|value| !value.is_empty()) {
            pairs.append_pair("audience", audience);
        }
        pairs.append_pair("code_challenge", &pkce.challenge);
        pairs.append_pair("code_challenge_method", "S256");
        pairs.append_pair("state", &pkce.state);
    }
    Ok((url, pkce.state.clone()))
}

fn default_loopback_redirect_uri() -> Url {
    Url::parse("http://127.0.0.1:7777/callback").expect("valid loopback url")
}

fn loopback_host_port(redirect_uri: &Url) -> Result<(String, u16)> {
    if redirect_uri.scheme() != "http" {
        anyhow::bail!(
            "Interactive login requires http loopback redirect URIs; use --api-key instead."
        );
    }
    let host = redirect_uri
        .host_str()
        .context("Missing redirect_uri host")?
        .to_string();
    if host != "127.0.0.1" && host != "localhost" {
        anyhow::bail!("Interactive login requires a loopback redirect URI; use --api-key instead.");
    }
    let port = redirect_uri.port().unwrap_or(80);
    Ok((host, port))
}

async fn listen_for_auth_code(listener: TcpListener, expected_state: &str) -> Result<String> {
    let (stream, _) = timeout(Duration::from_secs(180), listener.accept())
        .await
        .context("Timed out waiting for login")?
        .context("Failed to accept callback")?;
    let mut buf = vec![0u8; 4096];
    stream.readable().await.context("Failed to read callback")?;
    let n = stream
        .try_read(&mut buf)
        .context("Failed to read callback")?;
    let request = String::from_utf8_lossy(&buf[..n]);
    let path = request
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .unwrap_or("/");
    let callback_url =
        Url::parse(&format!("http://localhost{path}")).context("Invalid callback URL")?;
    let code = callback_url
        .query_pairs()
        .find(|(key, _)| key == "code")
        .map(|(_, value)| value.to_string())
        .context("Missing authorization code")?;
    if let Some(state) = callback_url
        .query_pairs()
        .find(|(key, _)| key == "state")
        .map(|(_, value)| value.to_string())
    {
        if state != expected_state {
            anyhow::bail!("Login state mismatch")
        }
    }
    let response = "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n<html><body><p>RunMat login complete. You can close this window.</p></body></html>";
    let _ = stream.try_write(response.as_bytes());
    Ok(code)
}

#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
    #[serde(default)]
    refresh_token: Option<String>,
    #[serde(default)]
    expires_in: Option<i64>,
}

#[derive(Deserialize)]
struct OidcConfig {
    token_endpoint: String,
}

async fn discover_token_endpoint(authorize_url: &Url) -> Result<String> {
    let mut origin = authorize_url.clone();
    origin.set_path("/");
    origin.set_query(None);
    origin.set_fragment(None);
    let base = origin.as_str().trim_end_matches('/');
    let discovery_url = format!("{base}/.well-known/openid-configuration");
    let client = reqwest::Client::new();
    if let Ok(response) = client.get(&discovery_url).send().await {
        if response.status().is_success() {
            if let Ok(config) = response.json::<OidcConfig>().await {
                return Ok(config.token_endpoint);
            }
        }
    }
    Ok(format!("{base}/oauth/token"))
}

async fn exchange_code_for_token(
    token_endpoint: &str,
    code: &str,
    client_id: &str,
    redirect_uri: &str,
    verifier: &str,
) -> Result<TokenResponse> {
    let client = reqwest::Client::new();
    let response = client
        .post(token_endpoint)
        .form(&[
            ("grant_type", "authorization_code"),
            ("code", code),
            ("client_id", client_id),
            ("redirect_uri", redirect_uri),
            ("code_verifier", verifier),
        ])
        .send()
        .await
        .context("Failed to request access token")?;
    if !response.status().is_success() {
        let status = response.status();
        anyhow::bail!("Token exchange failed ({status})")
    }
    let token = response
        .json::<TokenResponse>()
        .await
        .context("Failed to parse access token")?;
    Ok(token)
}

async fn refresh_access_token(
    token_endpoint: &str,
    client_id: &str,
    refresh_token: &str,
) -> Result<AuthToken> {
    let client = reqwest::Client::new();
    let response = client
        .post(token_endpoint)
        .form(&[
            ("grant_type", "refresh_token"),
            ("client_id", client_id),
            ("refresh_token", refresh_token),
        ])
        .send()
        .await
        .context("Failed to refresh access token")?;
    let response = response
        .error_for_status()
        .context("Token refresh failed")?;
    let token = response
        .json::<TokenResponse>()
        .await
        .context("Failed to parse refresh response")?;
    let expires_at = token_expiry(&token);
    Ok(AuthToken {
        access_token: token.access_token,
        refresh_token: token.refresh_token,
        expires_at,
        token_endpoint: Some(token_endpoint.to_string()),
        client_id: Some(client_id.to_string()),
    })
}

#[cfg(test)]
mod public_id_tests {
    use super::*;
    #[test]
    fn credential_namespaces_use_the_complete_canonical_endpoint() {
        let canonical = server_credential_key("https://example.com").unwrap();
        assert_eq!(
            canonical,
            server_credential_key("https://EXAMPLE.com:443/").unwrap()
        );
        let distinct = [
            "https://example.com:8443",
            "https://example.com/tenant-a",
            "https://example.com/tenant_a",
            "https://example-com",
            "http://localhost:3000",
            "https://localhost:3000",
        ]
        .map(|url| server_credential_key(url).unwrap());
        let keys: std::collections::HashSet<_> = distinct.iter().collect();
        assert_eq!(keys.len(), distinct.len());
        assert!(!keys.contains(&canonical));
        assert_ne!(
            memory_token_key("https://example.com").unwrap(),
            memory_refresh_key("https://example.com").unwrap()
        );
        // No real keychain or process environment is read or changed here.
        for invalid in [
            "http://example.com",
            "https://user:secret@example.com",
            "https://example.com?token=value",
            "not a URL",
        ] {
            assert!(server_credential_key(invalid).is_err());
            assert!(build_public_client(invalid, "test-key").is_err());
        }
    }

    #[test]
    fn canonical_namespace_and_legacy_config_have_one_wire_identity() {
        let uuid = Uuid::from_u128(42);
        let org = crate::ids::OrgId::from_uuid(uuid);
        let project = crate::ids::ProjectId::from_uuid(uuid);
        assert_eq!(parse_org_id_input(org.as_str()).unwrap(), uuid);
        assert_eq!(parse_project_id_input(project.as_str()).unwrap(), uuid);
        assert_eq!(parse_project_id_input(&uuid.to_string()).unwrap(), uuid);
        assert!(parse_project_id_input(org.as_str()).is_err());
        assert!(parse_org_id_input(project.as_str()).is_err());
        let version = crate::ids::FsVersionId::from_uuid(uuid);
        let snapshot = crate::ids::SnapshotId::from_uuid(uuid);
        assert_eq!(parse_fs_version_id_input(version.as_str()).unwrap(), uuid);
        assert_eq!(parse_snapshot_id_input(snapshot.as_str()).unwrap(), uuid);
        assert!(parse_snapshot_id_input(version.as_str()).is_err());
        assert!(parse_fs_version_id_input(snapshot.as_str()).is_err());
        let config = RemoteConfig::default();
        assert_eq!(resolve_public_org_id(&config, Some(uuid)).unwrap(), org);
        assert_eq!(
            resolve_public_project_id(&config, Some(uuid)).unwrap(),
            project
        );
    }
}
