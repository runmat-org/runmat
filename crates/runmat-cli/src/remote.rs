use anyhow::{Context, Result};
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use rand::RngCore;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION};
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::Command;
use uuid::Uuid;
use chrono::{DateTime, Utc};
use url::Url;
use sha2::{Digest, Sha256};
use tokio::net::TcpListener;
use tokio::time::{timeout, Duration};

use runmat_filesystem::{FsFileType, FsProvider, RemoteFsConfig, RemoteFsProvider};
use runmat_config::RunMatConfig;

const DEFAULT_SERVER_URL: &str = "https://api.runmat.com";

use crate::{
    FsCommand, OrgCommand, ProjectCommand, ProjectMembersCommand, ProjectRetentionCommand,
    RemoteCommand,
};
use crate::public_api;

#[derive(Default, Serialize, Deserialize)]
struct RemoteConfig {
    server_url: Option<String>,
    org_id: Option<Uuid>,
    project_id: Option<Uuid>,
    token_expires_at: Option<DateTime<Utc>>,
    token_endpoint: Option<String>,
    token_client_id: Option<String>,
}

struct AuthToken {
    access_token: String,
    refresh_token: Option<String>,
    expires_at: Option<DateTime<Utc>>,
    token_endpoint: Option<String>,
    client_id: Option<String>,
}

impl RemoteConfig {
    fn load() -> Result<Self> {
        let path = config_path()?;
        if !path.exists() {
            return Ok(Self::default());
        }
        let contents = fs::read_to_string(&path)
            .with_context(|| format!("Failed to read config {}", path.display()))?;
        serde_json::from_str(&contents).context("Failed to parse remote config")
    }

    fn save(&self) -> Result<()> {
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

fn config_path() -> Result<PathBuf> {
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
) -> Result<()> {
    let mut config = RemoteConfig::load()?;
    let server_url = resolve_server_url(&config, server)?;
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
    store_token(&server_url, auth.access_token.trim())?;
    if let Some(refresh) = auth.refresh_token.as_deref() {
        store_refresh_token(&server_url, refresh)?;
    } else {
        clear_refresh_token(&server_url)?;
    }
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

pub async fn execute_org_command(command: OrgCommand) -> Result<()> {
    match command {
        OrgCommand::List { limit, cursor } => list_orgs(limit, cursor).await,
    }
}

pub async fn execute_project_command(command: ProjectCommand) -> Result<()> {
    match command {
        ProjectCommand::List { org, limit, cursor } => list_projects(org, limit, cursor).await,
        ProjectCommand::Create { org, name } => create_project(org, name).await,
        ProjectCommand::Members { members_command } => execute_project_members_command(members_command).await,
        ProjectCommand::Retention { retention_command } => {
            execute_project_retention_command(retention_command).await
        }
        ProjectCommand::Select { project } => select_project(project).await,
    }
}

pub async fn execute_fs_command(command: FsCommand) -> Result<()> {
    let mut config = RemoteConfig::load()?;
    let project_id = match &command {
        FsCommand::Read { project, .. }
        | FsCommand::Write { project, .. }
        | FsCommand::Ls { project, .. }
        | FsCommand::Mkdir { project, .. }
        | FsCommand::Rm { project, .. }
        | FsCommand::History { project, .. }
        | FsCommand::Restore { project, .. }
        | FsCommand::HistoryDelete { project, .. }
        | FsCommand::SnapshotList { project }
        | FsCommand::SnapshotCreate { project, .. }
        | FsCommand::SnapshotRestore { project, .. }
        | FsCommand::SnapshotDelete { project, .. }
        | FsCommand::SnapshotTagList { project }
        | FsCommand::SnapshotTagSet { project, .. }
        | FsCommand::SnapshotTagDelete { project, .. }
        | FsCommand::GitClone { project, .. }
        | FsCommand::GitPull { project, .. }
        | FsCommand::GitPush { project, .. }
        | FsCommand::ManifestHistory { project, .. }
        | FsCommand::ManifestRestore { project, .. }
        | FsCommand::ManifestUpdate { project, .. } => resolve_project_id(&config, *project)?,
    };
    let server_url = resolve_server_url(&config, None)?;
    let token = resolve_auth_token(&mut config, &server_url).await?;
    let base_url = format!("{}/v1/projects/{}", server_url.trim_end_matches('/'), project_id);

    if matches!(
        command,
        FsCommand::ManifestHistory { .. }
            | FsCommand::ManifestRestore { .. }
            | FsCommand::ManifestUpdate { .. }
            | FsCommand::History { .. }
            | FsCommand::Restore { .. }
            | FsCommand::HistoryDelete { .. }
            | FsCommand::SnapshotList { .. }
            | FsCommand::SnapshotCreate { .. }
            | FsCommand::SnapshotRestore { .. }
            | FsCommand::SnapshotDelete { .. }
            | FsCommand::SnapshotTagList { .. }
            | FsCommand::SnapshotTagSet { .. }
            | FsCommand::SnapshotTagDelete { .. }
            | FsCommand::GitClone { .. }
            | FsCommand::GitPull { .. }
            | FsCommand::GitPush { .. }
    ) {
        match command {
            FsCommand::History { path, .. } => {
                list_history(&server_url, &token, project_id, &path).await?
            }
            FsCommand::Restore { version, .. } => {
                restore_version(&server_url, &token, project_id, version).await?
            }
            FsCommand::HistoryDelete { version, .. } => {
                delete_version(&server_url, &token, project_id, version).await?
            }
            FsCommand::ManifestHistory { path, .. } => {
                list_manifest_history(&server_url, &token, project_id, &path).await?
            }
            FsCommand::ManifestRestore { version, .. } => {
                restore_manifest_version(&server_url, &token, project_id, version).await?
            }
            FsCommand::ManifestUpdate {
                path,
                base_version,
                manifest,
                ..
            } => {
                update_manifest(&server_url, &token, project_id, &path, base_version, &manifest)
                    .await?
            }
            FsCommand::SnapshotList { .. } => {
                list_snapshots(&server_url, &token, project_id).await?
            }
            FsCommand::SnapshotCreate {
                message,
                parent,
                tag,
                ..
            } => {
                create_snapshot(&server_url, &token, project_id, message, parent, tag).await?
            }
            FsCommand::SnapshotRestore { snapshot, .. } => {
                restore_snapshot(&server_url, &token, project_id, snapshot).await?
            }
            FsCommand::SnapshotDelete { snapshot, .. } => {
                delete_snapshot(&server_url, &token, project_id, snapshot).await?
            }
            FsCommand::SnapshotTagList { .. } => {
                list_snapshot_tags(&server_url, &token, project_id).await?
            }
            FsCommand::SnapshotTagSet { snapshot, tag, .. } => {
                set_snapshot_tag(&server_url, &token, project_id, snapshot, &tag).await?
            }
            FsCommand::SnapshotTagDelete { tag, .. } => {
                delete_snapshot_tag(&server_url, &token, project_id, &tag).await?
            }
            FsCommand::GitClone {
                directory,
                server,
                ..
            } => {
                let url = resolve_server_url(&config, server.clone())?;
                git_clone(&url, &token, project_id, &directory).await?
            }
            FsCommand::GitPull {
                directory,
                server,
                ..
            } => {
                let url = resolve_server_url(&config, server.clone())?;
                git_pull(&url, &token, project_id, &directory).await?
            }
            FsCommand::GitPush {
                directory,
                server,
                ..
            } => {
                let url = resolve_server_url(&config, server.clone())?;
                git_push(&url, &token, project_id, &directory).await?
            }
            _ => {}
        }
        return Ok(());
    }

    let task = tokio::task::spawn_blocking(move || -> Result<()> {
        let shard_threshold_bytes = read_u64_env("RUNMAT_FS_SHARD_THRESHOLD_BYTES");
        let shard_size_bytes = read_u64_env("RUNMAT_FS_SHARD_SIZE_BYTES");
        let provider = RemoteFsProvider::new(RemoteFsConfig {
            base_url,
            auth_token: Some(token),
            shard_threshold_bytes: shard_threshold_bytes.unwrap_or(
                RemoteFsConfig::default().shard_threshold_bytes,
            ),
            shard_size_bytes: shard_size_bytes.unwrap_or(RemoteFsConfig::default().shard_size_bytes),
            ..RemoteFsConfig::default()
        })
        .context("Failed to initialize remote filesystem")?;

        match command {
            FsCommand::Read { path, output, .. } => {
                let bytes = provider
                    .read(std::path::Path::new(&path))
                    .context("Failed to read remote file")?;
                if let Some(output) = output {
                    fs::write(&output, bytes)
                        .with_context(|| format!("Failed to write {}", output.display()))?;
                } else {
                    let mut stdout = io::stdout();
                    stdout.write_all(&bytes).context("Failed to write output")?;
                }
                Ok(())
            }
            FsCommand::Write { path, input, .. } => {
                let bytes = fs::read(&input)
                    .with_context(|| format!("Failed to read {}", input.display()))?;
                provider
                    .write(std::path::Path::new(&path), &bytes)
                    .context("Failed to write remote file")?;
                Ok(())
            }
            FsCommand::Ls { path, .. } => {
                let entries = provider
                    .read_dir(std::path::Path::new(&path))
                    .context("Failed to list directory")?;
                for entry in entries {
                    let kind = match entry.file_type() {
                        FsFileType::Directory => "dir",
                        FsFileType::File => "file",
                        FsFileType::Symlink => "symlink",
                        FsFileType::Other => "other",
                        FsFileType::Unknown => "unknown",
                    };
                    println!("{kind}\t{}", entry.path().display());
                }
                Ok(())
            }
            FsCommand::Mkdir {
                path,
                recursive,
                ..
            } => {
                if recursive {
                    provider
                        .create_dir_all(std::path::Path::new(&path))
                        .context("Failed to create directory")?;
                } else {
                    provider
                        .create_dir(std::path::Path::new(&path))
                        .context("Failed to create directory")?;
                }
                Ok(())
            }
            FsCommand::Rm {
                path,
                dir,
                recursive,
                ..
            } => {
                if recursive {
                    provider
                        .remove_dir_all(std::path::Path::new(&path))
                        .context("Failed to remove directory")?;
                } else if dir {
                    provider
                        .remove_dir(std::path::Path::new(&path))
                        .context("Failed to remove directory")?;
                } else {
                    provider
                        .remove_file(std::path::Path::new(&path))
                        .context("Failed to remove file")?;
                }
                Ok(())
            }
            FsCommand::History { .. }
            | FsCommand::Restore { .. }
            | FsCommand::HistoryDelete { .. }
            | FsCommand::ManifestHistory { .. }
            | FsCommand::ManifestRestore { .. }
            | FsCommand::ManifestUpdate { .. }
            | FsCommand::SnapshotList { .. }
            | FsCommand::SnapshotCreate { .. }
            | FsCommand::SnapshotRestore { .. }
            | FsCommand::SnapshotDelete { .. }
            | FsCommand::SnapshotTagList { .. }
            | FsCommand::SnapshotTagSet { .. }
            | FsCommand::SnapshotTagDelete { .. }
            | FsCommand::GitClone { .. }
            | FsCommand::GitPull { .. }
            | FsCommand::GitPush { .. } => Ok(()),
        }
    });

    task.await.context("Filesystem task join failed")??;
    Ok(())
}

pub async fn execute_remote_command(command: RemoteCommand) -> Result<()> {
    match command {
        RemoteCommand::Run {
            script,
            project,
            server,
        } => run_with_remote_fs(script, project, server).await,
    }
}

async fn execute_project_members_command(command: ProjectMembersCommand) -> Result<()> {
    match command {
        ProjectMembersCommand::List {
            project,
            limit,
            cursor,
        } => list_project_members(project, limit, cursor).await,
    }
}

async fn execute_project_retention_command(command: ProjectRetentionCommand) -> Result<()> {
    match command {
        ProjectRetentionCommand::Get { project } => get_project_retention(project).await,
        ProjectRetentionCommand::Set {
            max_versions,
            project,
        } => set_project_retention(project, max_versions).await,
    }
}

async fn list_orgs(limit: Option<u32>, cursor: Option<String>) -> Result<()> {
    let mut config = RemoteConfig::load()?;
    let server_url = resolve_server_url(&config, None)?;
    let token = resolve_auth_token(&mut config, &server_url).await?;
    let client = build_public_client(&server_url, &token)?;
    let response = client
        .list_orgs(cursor.as_deref(), limit.map(|value| value as i32))
        .await
        .map_err(map_public_error)?
        .into_inner();
    for org in response.items {
        println!("{}\t{}", org.id, org.name);
    }
    if let Some(cursor) = response.next_cursor {
        println!("next_cursor\t{cursor}");
    }
    Ok(())
}

async fn list_projects(
    org: Option<Uuid>,
    limit: Option<u32>,
    cursor: Option<String>,
) -> Result<()> {
    let mut config = RemoteConfig::load()?;
    let org_id = resolve_org_id(&config, org)?;
    let server_url = resolve_server_url(&config, None)?;
    let token = resolve_auth_token(&mut config, &server_url).await?;
    let client = build_public_client(&server_url, &token)?;
    let response = client
        .list_projects(&org_id, cursor.as_deref(), limit.map(|value| value as i32))
        .await
        .map_err(map_public_error)?
        .into_inner();
    for project in response.items {
        println!("{}\t{}", project.id, project.name);
    }
    if let Some(cursor) = response.next_cursor {
        println!("next_cursor\t{cursor}");
    }
    Ok(())
}

async fn create_project(org: Option<Uuid>, name: String) -> Result<()> {
    let mut config = RemoteConfig::load()?;
    let org_id = resolve_org_id(&config, org)?;
    let server_url = resolve_server_url(&config, None)?;
    let token = resolve_auth_token(&mut config, &server_url).await?;
    let client = build_public_client(&server_url, &token)?;
    let response = client
        .create_project(
            &org_id,
            &public_api::types::ProjectCreateRequest { name },
        )
        .await
        .map_err(map_public_error)?
        .into_inner();
    println!("{}\t{}", response.id, response.name);
    Ok(())
}

async fn list_project_members(
    project: Option<Uuid>,
    limit: Option<u32>,
    cursor: Option<String>,
) -> Result<()> {
    let mut config = RemoteConfig::load()?;
    let project_id = resolve_project_id(&config, project)?;
    let server_url = resolve_server_url(&config, None)?;
    let token = resolve_auth_token(&mut config, &server_url).await?;
    let client = build_public_client(&server_url, &token)?;
    let response = client
        .list_project_memberships(&project_id, cursor.as_deref(), limit.map(|value| value as i32))
        .await
        .map_err(map_public_error)?
        .into_inner();
    for member in response.items {
        println!("{}\t{}\t{}", member.id, member.user_id, member.role);
    }
    if let Some(cursor) = response.next_cursor {
        println!("next_cursor\t{cursor}");
    }
    Ok(())
}

async fn select_project(project_id: Uuid) -> Result<()> {
    let mut config = RemoteConfig::load()?;
    config.project_id = Some(project_id);
    config.save()?;
    println!("Default project set to {project_id}");
    Ok(())
}

fn build_public_client(server_url: &str, token: &str) -> Result<public_api::Client> {
    let mut headers = HeaderMap::new();
    let auth_value = HeaderValue::from_str(&format!("Bearer {token}"))
        .context("Invalid auth token")?;
    headers.insert(AUTHORIZATION, auth_value);
    let client = reqwest::Client::builder()
        .default_headers(headers)
        .build()
        .context("Failed to build HTTP client")?;
    Ok(public_api::Client::new_with_client(server_url, client))
}

fn build_http_client(server_url: &str, token: &str) -> Result<(reqwest::Client, String)> {
    let mut headers = HeaderMap::new();
    let auth_value = HeaderValue::from_str(&format!("Bearer {token}"))
        .context("Invalid auth token")?;
    headers.insert(AUTHORIZATION, auth_value);
    let client = reqwest::Client::builder()
        .default_headers(headers)
        .build()
        .context("Failed to build HTTP client")?;
    Ok((client, server_url.trim_end_matches('/').to_string()))
}

async fn run_with_remote_fs(
    script: PathBuf,
    project: Option<Uuid>,
    server: Option<String>,
) -> Result<()> {
    let mut config = RemoteConfig::load()?;
    let project_id = resolve_project_id(&config, project)?;
    let server_url = resolve_server_url(&config, server)?;
    let token = resolve_auth_token(&mut config, &server_url).await?;
    let shard_threshold_bytes = read_u64_env("RUNMAT_FS_SHARD_THRESHOLD_BYTES")
        .unwrap_or(RemoteFsConfig::default().shard_threshold_bytes);
    let shard_size_bytes = read_u64_env("RUNMAT_FS_SHARD_SIZE_BYTES")
        .unwrap_or(RemoteFsConfig::default().shard_size_bytes);
    let base_url = format!("{}/v1/projects/{}", server_url.trim_end_matches('/'), project_id);
    let provider = RemoteFsProvider::new(RemoteFsConfig {
        base_url,
        auth_token: Some(token),
        shard_threshold_bytes,
        shard_size_bytes,
        ..RemoteFsConfig::default()
    })
    .context("Failed to initialize remote filesystem")?;
    runmat_filesystem::set_provider(std::sync::Arc::new(provider));
    let config = RunMatConfig::default();
    crate::execute_script_with_remote_provider(script, &config).await
}

fn resolve_server_url(config: &RemoteConfig, override_value: Option<String>) -> Result<String> {
    if let Some(value) = override_value {
        return Ok(value);
    }
    if let Ok(value) = env::var("RUNMAT_SERVER_URL") {
        if !value.is_empty() {
            return Ok(value);
        }
    }
    if let Some(value) = &config.server_url {
        return Ok(value.clone());
    }
    Ok(DEFAULT_SERVER_URL.to_string())
}

fn resolve_org_id(config: &RemoteConfig, override_value: Option<Uuid>) -> Result<Uuid> {
    if let Some(value) = override_value {
        return Ok(value);
    }
    if let Ok(value) = env::var("RUNMAT_ORG_ID") {
        if !value.is_empty() {
            return Ok(Uuid::parse_str(&value)
                .context("RUNMAT_ORG_ID must be a UUID")?);
        }
    }
    config
        .org_id
        .context(
            "Organization id not configured. Use --org, set RUNMAT_ORG_ID, or run `runmat login --org <id>`."
        )
}

fn resolve_project_id(config: &RemoteConfig, override_value: Option<Uuid>) -> Result<Uuid> {
    if let Some(value) = override_value {
        return Ok(value);
    }
    if let Ok(value) = env::var("RUNMAT_PROJECT_ID") {
        if !value.is_empty() {
            return Ok(Uuid::parse_str(&value)
                .context("RUNMAT_PROJECT_ID must be a UUID")?);
        }
    }
    config
        .project_id
        .context(
            "Project id not configured. Use --project, set RUNMAT_PROJECT_ID, or run `runmat login --project <id>`."
        )
}

async fn resolve_auth_token(config: &mut RemoteConfig, server_url: &str) -> Result<String> {
    if let Ok(value) = env::var("RUNMAT_API_KEY") {
        if !value.is_empty() {
            return Ok(value);
        }
    }
    let token = load_token(server_url)?;
    let refresh = load_refresh_token(server_url)?;
    let expired = config
        .token_expires_at
        .map(|expiry| expiry <= Utc::now() + chrono::Duration::seconds(30))
        .unwrap_or(false);
    if let Some(token) = token.as_ref() {
        if !expired {
            return Ok(token.clone());
        }
    }
    if let (Some(refresh), Some(endpoint), Some(client_id)) = (
        refresh,
        config.token_endpoint.clone(),
        config.token_client_id.clone(),
    ) {
        let refreshed = refresh_access_token(&endpoint, &client_id, &refresh).await?;
        store_token(server_url, &refreshed.access_token)?;
        if let Some(new_refresh) = refreshed.refresh_token.as_deref() {
            store_refresh_token(server_url, new_refresh)?;
        }
        config.token_expires_at = refreshed.expires_at;
        config.save()?;
        return Ok(refreshed.access_token);
    }
    token.context(
        "No stored credentials. Run `runmat login --server <url> --project <id>` or set RUNMAT_API_KEY."
    )
}

fn load_token(server_url: &str) -> Result<Option<String>> {
    let entry = keyring_entry(server_url)?;
    match entry.get_password() {
        Ok(value) => Ok(Some(value)),
        Err(keyring::Error::NoEntry) => Ok(None),
        Err(err) => Err(err).context("Failed to access keyring"),
    }
}

fn load_refresh_token(server_url: &str) -> Result<Option<String>> {
    let entry = keyring_refresh_entry(server_url)?;
    match entry.get_password() {
        Ok(value) => Ok(Some(value)),
        Err(keyring::Error::NoEntry) => Ok(None),
        Err(err) => Err(err).context("Failed to access keyring"),
    }
}

fn store_token(server_url: &str, token: &str) -> Result<()> {
    let entry = keyring_entry(server_url)?;
    entry
        .set_password(token)
        .context("Failed to store credentials")
}

fn store_refresh_token(server_url: &str, token: &str) -> Result<()> {
    let entry = keyring_refresh_entry(server_url)?;
    entry
        .set_password(token)
        .context("Failed to store refresh token")
}

fn clear_refresh_token(server_url: &str) -> Result<()> {
    let entry = keyring_refresh_entry(server_url)?;
    match entry.delete_password() {
        Ok(_) => Ok(()),
        Err(keyring::Error::NoEntry) => Ok(()),
        Err(err) => Err(err).context("Failed to clear refresh token"),
    }
}

fn keyring_entry(server_url: &str) -> Result<keyring::Entry> {
    let account = Url::parse(server_url)
        .ok()
        .and_then(|url| url.host_str().map(|host| host.to_string()))
        .unwrap_or_else(|| server_url.to_string());
    Ok(keyring::Entry::new("runmat", &account).context("Failed to open keyring")?)
}

fn keyring_refresh_entry(server_url: &str) -> Result<keyring::Entry> {
    let account = Url::parse(server_url)
        .ok()
        .and_then(|url| url.host_str().map(|host| format!("{host}:refresh")))
        .unwrap_or_else(|| format!("{}:refresh", server_url));
    Ok(keyring::Entry::new("runmat", &account).context("Failed to open keyring")?)
}

async fn interactive_login(server_url: &str, email: Option<String>) -> Result<AuthToken> {
    let email = match email {
        Some(value) => value,
        None => prompt_for_email()?,
    };
    let client = public_api::Client::new(server_url);
    let response = client
        .auth_resolve(&public_api::types::AuthResolveRequest { email })
        .await
        .map_err(map_public_error)?
        .into_inner();
    let pkce = generate_pkce();
    let (authorize_url, expected_state) = apply_pkce_to_authorize_url(&response.redirect_url, &pkce)?;
    let redirect_uri = extract_redirect_uri(&authorize_url)?;
    let client_id = extract_client_id(&authorize_url)?;
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
    Ok(AuthToken {
        access_token: token.access_token,
        refresh_token: token.refresh_token,
        expires_at: token
            .expires_in
            .map(|value| Utc::now() + chrono::Duration::seconds(value)),
        token_endpoint: Some(token_endpoint),
        client_id: Some(client_id),
    })
}

fn prompt_for_email() -> Result<String> {
    let mut stdout = io::stdout();
    stdout
        .write_all(b"Email: ")
        .context("Failed to prompt")?;
    stdout.flush().context("Failed to flush prompt")?;
    let mut input = String::new();
    io::stdin()
        .read_line(&mut input)
        .context("Failed to read input")?;
    let email = input.trim().to_string();
    if email.is_empty() {
        anyhow::bail!("Email cannot be empty")
    }
    Ok(email)
}

#[derive(Deserialize)]
struct ManifestHistoryResponse {
    items: Vec<ManifestHistoryEntry>,
}

#[derive(Deserialize)]
struct ManifestHistoryEntry {
    #[serde(rename = "versionId")]
    version_id: Uuid,
    #[serde(rename = "createdAt")]
    created_at: chrono::DateTime<chrono::Utc>,
    #[serde(rename = "sizeBytes")]
    size_bytes: i64,
    #[serde(rename = "totalSize")]
    total_size: u64,
    #[serde(rename = "shardCount")]
    shard_count: usize,
}

#[derive(Deserialize)]
struct HistoryResponse {
    items: Vec<HistoryEntry>,
}

#[derive(Deserialize)]
struct HistoryEntry {
    id: Uuid,
    path: String,
    size: i64,
    #[serde(rename = "createdAt")]
    created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Deserialize)]
struct SnapshotListResponse {
    items: Vec<SnapshotResponse>,
}

#[derive(Deserialize)]
struct SnapshotResponse {
    id: Uuid,
    #[serde(rename = "projectId")]
    _project_id: Uuid,
    #[serde(rename = "parentId")]
    parent_id: Option<Uuid>,
    message: Option<String>,
    #[serde(rename = "createdAt")]
    created_at: chrono::DateTime<chrono::Utc>,
    #[serde(rename = "createdBy")]
    created_by: Option<Uuid>,
    tags: Vec<String>,
}

#[derive(Deserialize)]
struct SnapshotTagListResponse {
    items: Vec<SnapshotTagResponse>,
}

#[derive(Deserialize)]
struct SnapshotTagResponse {
    tag: String,
    #[serde(rename = "snapshotId")]
    snapshot_id: Uuid,
    #[serde(rename = "createdAt")]
    created_at: chrono::DateTime<chrono::Utc>,
    #[serde(rename = "createdBy")]
    created_by: Option<Uuid>,
}

#[derive(Deserialize)]
struct RetentionResponse {
    #[serde(rename = "projectId")]
    project_id: Uuid,
    #[serde(rename = "maxVersions")]
    max_versions: usize,
    #[serde(rename = "updatedAt")]
    updated_at: Option<DateTime<Utc>>,
    #[serde(rename = "isDefault")]
    is_default: bool,
}

async fn list_manifest_history(
    server_url: &str,
    token: &str,
    project_id: Uuid,
    path: &str,
) -> Result<()> {
    let (client, base) = build_http_client(server_url, token)?;
    let url = format!("{}/v1/projects/{}/fs/manifest/history", base, project_id);
    let response = client
        .get(&url)
        .query(&[("path", path)])
        .send()
        .await
        .context("Failed to fetch manifest history")?;
    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        anyhow::bail!("Manifest history failed ({status}): {text}")
    }
    let payload = response
        .json::<ManifestHistoryResponse>()
        .await
        .context("Failed to parse manifest history")?;
    for item in payload.items {
        println!(
            "{}\t{}\t{}\t{}\t{}",
            item.version_id, item.created_at, item.size_bytes, item.total_size, item.shard_count
        );
    }
    Ok(())
}

async fn restore_manifest_version(
    server_url: &str,
    token: &str,
    project_id: Uuid,
    version_id: Uuid,
) -> Result<()> {
    let (client, base) = build_http_client(server_url, token)?;
    let url = format!("{}/v1/projects/{}/history/restore", base, project_id);
    let response = client
        .post(&url)
        .json(&serde_json::json!({ "versionId": version_id }))
        .send()
        .await
        .context("Failed to restore manifest")?;
    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        anyhow::bail!("Manifest restore failed ({status}): {text}")
    }
    println!("Restored manifest version {version_id}");
    Ok(())
}

async fn list_history(
    server_url: &str,
    token: &str,
    project_id: Uuid,
    path: &str,
) -> Result<()> {
    let (client, base) = build_http_client(server_url, token)?;
    let url = format!("{}/v1/projects/{}/history", base, project_id);
    let response = client
        .get(&url)
        .query(&[("path", path)])
        .send()
        .await
        .context("Failed to fetch history")?;
    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        anyhow::bail!("History failed ({status}): {text}")
    }
    let payload = response
        .json::<HistoryResponse>()
        .await
        .context("Failed to parse history")?;
    for item in payload.items {
        println!("{}\t{}\t{}\t{}", item.id, item.created_at, item.size, item.path);
    }
    Ok(())
}

async fn restore_version(
    server_url: &str,
    token: &str,
    project_id: Uuid,
    version_id: Uuid,
) -> Result<()> {
    let (client, base) = build_http_client(server_url, token)?;
    let url = format!("{}/v1/projects/{}/history/restore", base, project_id);
    let response = client
        .post(&url)
        .json(&serde_json::json!({ "versionId": version_id }))
        .send()
        .await
        .context("Failed to restore version")?;
    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        anyhow::bail!("Restore failed ({status}): {text}")
    }
    println!("Restored version {version_id}");
    Ok(())
}

async fn delete_version(
    server_url: &str,
    token: &str,
    project_id: Uuid,
    version_id: Uuid,
) -> Result<()> {
    let (client, base) = build_http_client(server_url, token)?;
    let url = format!("{}/v1/projects/{}/fs/history/{}", base, project_id, version_id);
    let response = client
        .delete(&url)
        .send()
        .await
        .context("Failed to delete version")?;
    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        anyhow::bail!("Delete failed ({status}): {text}")
    }
    println!("Deleted version {version_id}");
    Ok(())
}

async fn get_project_retention(project: Option<Uuid>) -> Result<()> {
    let mut config = RemoteConfig::load()?;
    let project_id = resolve_project_id(&config, project)?;
    let server_url = resolve_server_url(&config, None)?;
    let token = resolve_auth_token(&mut config, &server_url).await?;
    let (client, base) = build_http_client(&server_url, &token)?;
    let url = format!("{}/v1/projects/{}/fs/retention", base, project_id);
    let response = client
        .get(&url)
        .send()
        .await
        .context("Failed to fetch retention")?;
    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        anyhow::bail!("Retention fetch failed ({status}): {text}")
    }
    let payload = response
        .json::<RetentionResponse>()
        .await
        .context("Failed to parse retention response")?;
    let updated_at = payload
        .updated_at
        .map(|value| value.to_string())
        .unwrap_or_else(|| "-".to_string());
    println!(
        "{}\t{}\t{}\t{}",
        payload.project_id, payload.max_versions, updated_at, payload.is_default
    );
    Ok(())
}

async fn set_project_retention(project: Option<Uuid>, max_versions: usize) -> Result<()> {
    let mut config = RemoteConfig::load()?;
    let project_id = resolve_project_id(&config, project)?;
    let server_url = resolve_server_url(&config, None)?;
    let token = resolve_auth_token(&mut config, &server_url).await?;
    let (client, base) = build_http_client(&server_url, &token)?;
    let url = format!("{}/v1/projects/{}/fs/retention", base, project_id);
    let response = client
        .post(&url)
        .json(&serde_json::json!({ "maxVersions": max_versions }))
        .send()
        .await
        .context("Failed to update retention")?;
    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        anyhow::bail!("Retention update failed ({status}): {text}")
    }
    let payload = response
        .json::<RetentionResponse>()
        .await
        .context("Failed to parse retention response")?;
    let updated_at = payload
        .updated_at
        .map(|value| value.to_string())
        .unwrap_or_else(|| "-".to_string());
    println!(
        "{}\t{}\t{}\t{}",
        payload.project_id, payload.max_versions, updated_at, payload.is_default
    );
    Ok(())
}

async fn update_manifest(
    server_url: &str,
    token: &str,
    project_id: Uuid,
    path: &str,
    base_version: Uuid,
    manifest_path: &PathBuf,
) -> Result<()> {
    let contents = fs::read_to_string(manifest_path)
        .with_context(|| format!("Failed to read {}", manifest_path.display()))?;
    let manifest: serde_json::Value = serde_json::from_str(&contents)
        .context("Manifest must be valid JSON")?;
    let (client, base) = build_http_client(server_url, token)?;
    let url = format!("{}/v1/projects/{}/fs/manifest/update", base, project_id);
    let response = client
        .post(&url)
        .json(&serde_json::json!({
            "path": path,
            "baseVersionId": base_version,
            "manifest": manifest,
        }))
        .send()
        .await
        .context("Failed to update manifest")?;
    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        anyhow::bail!("Manifest update failed ({status}): {text}")
    }
    println!("Updated manifest for {path}");
    Ok(())
}

async fn list_snapshots(server_url: &str, token: &str, project_id: Uuid) -> Result<()> {
    let (client, base) = build_http_client(server_url, token)?;
    let url = format!("{}/v1/projects/{}/fs/snapshots", base, project_id);
    let response = client
        .get(&url)
        .send()
        .await
        .context("Failed to fetch snapshots")?;
    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        anyhow::bail!("Snapshot list failed ({status}): {text}")
    }
    let payload = response
        .json::<SnapshotListResponse>()
        .await
        .context("Failed to parse snapshot list")?;
    for item in payload.items {
        let parent = item
            .parent_id
            .map(|value| value.to_string())
            .unwrap_or_else(|| "-".to_string());
        let message = item.message.unwrap_or_else(|| "".to_string());
        let tags = if item.tags.is_empty() {
            "-".to_string()
        } else {
            item.tags.join(",")
        };
        let created_by = item
            .created_by
            .map(|value| value.to_string())
            .unwrap_or_else(|| "-".to_string());
        println!(
            "{}\t{}\t{}\t{}\t{}\t{}",
            item.id, item.created_at, parent, created_by, tags, message
        );
    }
    Ok(())
}

async fn create_snapshot(
    server_url: &str,
    token: &str,
    project_id: Uuid,
    message: Option<String>,
    parent: Option<Uuid>,
    tag: Option<String>,
) -> Result<()> {
    let (client, base) = build_http_client(server_url, token)?;
    let url = format!("{}/v1/projects/{}/fs/snapshots", base, project_id);
    let response = client
        .post(&url)
        .json(&serde_json::json!({
            "message": message,
            "parentId": parent,
            "tag": tag,
        }))
        .send()
        .await
        .context("Failed to create snapshot")?;
    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        anyhow::bail!("Snapshot create failed ({status}): {text}")
    }
    let payload = response
        .json::<SnapshotResponse>()
        .await
        .context("Failed to parse snapshot response")?;
    let parent = payload
        .parent_id
        .map(|value| value.to_string())
        .unwrap_or_else(|| "-".to_string());
    let message = payload.message.unwrap_or_else(|| "".to_string());
    let tags = if payload.tags.is_empty() {
        "-".to_string()
    } else {
        payload.tags.join(",")
    };
    let created_by = payload
        .created_by
        .map(|value| value.to_string())
        .unwrap_or_else(|| "-".to_string());
    println!(
        "{}\t{}\t{}\t{}\t{}\t{}",
        payload.id, payload.created_at, parent, created_by, tags, message
    );
    Ok(())
}

async fn restore_snapshot(
    server_url: &str,
    token: &str,
    project_id: Uuid,
    snapshot_id: Uuid,
) -> Result<()> {
    let (client, base) = build_http_client(server_url, token)?;
    let url = format!(
        "{}/v1/projects/{}/fs/snapshots/{}/restore",
        base, project_id, snapshot_id
    );
    let response = client
        .post(&url)
        .send()
        .await
        .context("Failed to restore snapshot")?;
    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        anyhow::bail!("Snapshot restore failed ({status}): {text}")
    }
    println!("Restored snapshot {snapshot_id}");
    Ok(())
}

async fn delete_snapshot(
    server_url: &str,
    token: &str,
    project_id: Uuid,
    snapshot_id: Uuid,
) -> Result<()> {
    let (client, base) = build_http_client(server_url, token)?;
    let url = format!("{}/v1/projects/{}/fs/snapshots/{}", base, project_id, snapshot_id);
    let response = client
        .delete(&url)
        .send()
        .await
        .context("Failed to delete snapshot")?;
    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        anyhow::bail!("Snapshot delete failed ({status}): {text}")
    }
    println!("Deleted snapshot {snapshot_id}");
    Ok(())
}

async fn list_snapshot_tags(server_url: &str, token: &str, project_id: Uuid) -> Result<()> {
    let (client, base) = build_http_client(server_url, token)?;
    let url = format!("{}/v1/projects/{}/fs/snapshots/tags", base, project_id);
    let response = client
        .get(&url)
        .send()
        .await
        .context("Failed to fetch snapshot tags")?;
    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        anyhow::bail!("Snapshot tag list failed ({status}): {text}")
    }
    let payload = response
        .json::<SnapshotTagListResponse>()
        .await
        .context("Failed to parse snapshot tags")?;
    for item in payload.items {
        let created_by = item
            .created_by
            .map(|value| value.to_string())
            .unwrap_or_else(|| "-".to_string());
        println!(
            "{}\t{}\t{}\t{}",
            item.tag, item.snapshot_id, item.created_at, created_by
        );
    }
    Ok(())
}

async fn set_snapshot_tag(
    server_url: &str,
    token: &str,
    project_id: Uuid,
    snapshot_id: Uuid,
    tag: &str,
) -> Result<()> {
    let (client, base) = build_http_client(server_url, token)?;
    let url = format!(
        "{}/v1/projects/{}/fs/snapshots/{}/tags",
        base, project_id, snapshot_id
    );
    let response = client
        .post(&url)
        .json(&serde_json::json!({ "tag": tag }))
        .send()
        .await
        .context("Failed to set snapshot tag")?;
    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        anyhow::bail!("Snapshot tag set failed ({status}): {text}")
    }
    let payload = response
        .json::<SnapshotTagResponse>()
        .await
        .context("Failed to parse snapshot tag")?;
    let created_by = payload
        .created_by
        .map(|value| value.to_string())
        .unwrap_or_else(|| "-".to_string());
    println!(
        "{}\t{}\t{}\t{}",
        payload.tag, payload.snapshot_id, payload.created_at, created_by
    );
    Ok(())
}

async fn delete_snapshot_tag(
    server_url: &str,
    token: &str,
    project_id: Uuid,
    tag: &str,
) -> Result<()> {
    let (client, base) = build_http_client(server_url, token)?;
    let url = format!("{}/v1/projects/{}/fs/snapshots/tags/{}", base, project_id, tag);
    let response = client
        .delete(&url)
        .send()
        .await
        .context("Failed to delete snapshot tag")?;
    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        anyhow::bail!("Snapshot tag delete failed ({status}): {text}")
    }
    println!("Deleted snapshot tag {tag}");
    Ok(())
}

async fn git_clone(
    server_url: &str,
    token: &str,
    project_id: Uuid,
    directory: &PathBuf,
) -> Result<()> {
    if directory.exists() && fs::read_dir(directory)?.next().is_some() {
        anyhow::bail!("Destination directory is not empty")
    }
    fs::create_dir_all(directory)?;
    run_git(directory, &["init"]).context("Failed to init git repo")?;
    let stream = fetch_git_stream(server_url, token, project_id).await?;
    run_git_with_input(directory, &["fast-import"], &stream)
        .context("Failed to import snapshot history")?;
    run_git(directory, &["checkout", "-f", "main"]).context("Failed to checkout")?;
    run_git(
        directory,
        &["config", "runmat.server", server_url.trim_end_matches('/')],
    )
    .ok();
    run_git(
        directory,
        &["config", "runmat.project", &project_id.to_string()],
    )
    .ok();
    println!("Cloned snapshots into {}", directory.display());
    Ok(())
}

async fn git_pull(
    server_url: &str,
    token: &str,
    project_id: Uuid,
    directory: &PathBuf,
) -> Result<()> {
    let stream = fetch_git_stream(server_url, token, project_id).await?;
    run_git_with_input(directory, &["fast-import"], &stream)
        .context("Failed to import snapshot history")?;
    run_git(directory, &["checkout", "-f", "main"]).context("Failed to checkout")?;
    println!("Pulled snapshots into {}", directory.display());
    Ok(())
}

async fn git_push(
    server_url: &str,
    token: &str,
    project_id: Uuid,
    directory: &PathBuf,
) -> Result<()> {
    let branches = run_git_capture(directory, &["for-each-ref", "refs/heads", "--format=%(refname)"])
        .context("Failed to list branches")?;
    let branch_list = branches
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect::<Vec<_>>();
    if branch_list.len() != 1 || branch_list[0] != "refs/heads/main" {
        anyhow::bail!("Only refs/heads/main is supported for git push")
    }
    let export = run_git_capture(
        directory,
        &["fast-export", "--full-tree", "--all", "--show-original-ids"],
    )
        .context("Failed to export git history")?;
    let client = reqwest::Client::new();
    let url = format!("{}/v1/projects/{}/fs/git/receive", server_url.trim_end_matches('/'), project_id);
    let response = client
        .post(url)
        .header("authorization", format!("Bearer {}", token))
        .header("content-type", "application/octet-stream")
        .body(export)
        .send()
        .await
        .context("Failed to push git history")?;
    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        anyhow::bail!("Git push failed ({status}): {text}")
    }
    println!("Pushed git history to project {project_id}");
    Ok(())
}

async fn fetch_git_stream(server_url: &str, token: &str, project_id: Uuid) -> Result<Vec<u8>> {
    let client = reqwest::Client::new();
    let url = format!("{}/v1/projects/{}/fs/git/upload", server_url.trim_end_matches('/'), project_id);
    let response = client
        .get(url)
        .header("authorization", format!("Bearer {}", token))
        .send()
        .await
        .context("Failed to download git export")?;
    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        anyhow::bail!("Git export failed ({status}): {text}")
    }
    Ok(response.bytes().await?.to_vec())
}

fn run_git(directory: &PathBuf, args: &[&str]) -> Result<()> {
    let status = Command::new("git")
        .current_dir(directory)
        .args(args)
        .status()
        .context("Failed to run git")?;
    if !status.success() {
        anyhow::bail!("Git command failed")
    }
    Ok(())
}

fn run_git_with_input(directory: &PathBuf, args: &[&str], input: &[u8]) -> Result<()> {
    let mut child = Command::new("git")
        .current_dir(directory)
        .args(args)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::inherit())
        .spawn()
        .context("Failed to run git")?;
    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(input)?;
    }
    let status = child.wait()?;
    if !status.success() {
        anyhow::bail!("Git command failed")
    }
    Ok(())
}

fn run_git_capture(directory: &PathBuf, args: &[&str]) -> Result<String> {
    let output = Command::new("git")
        .current_dir(directory)
        .args(args)
        .output()
        .context("Failed to run git")?;
    if !output.status.success() {
        anyhow::bail!("Git command failed")
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn read_u64_env(key: &str) -> Option<u64> {
    env::var(key).ok().and_then(|value| value.parse().ok())
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

fn apply_pkce_to_authorize_url(redirect_url: &str, pkce: &PkceState) -> Result<(Url, String)> {
    let mut url = Url::parse(redirect_url).context("Invalid redirect URL")?;
    let mut has_challenge = false;
    let mut has_method = false;
    let mut state_value = None;
    let mut scope_value = None;
    let pairs = url
        .query_pairs()
        .map(|(key, value)| (key.to_string(), value.to_string()))
        .collect::<Vec<_>>();
    for (key, value) in &pairs {
        match key.as_str() {
            "code_challenge" => has_challenge = true,
            "code_challenge_method" => has_method = true,
            "state" => state_value = Some(value.to_string()),
            "scope" => scope_value = Some(value.to_string()),
            _ => {}
        }
    }
    let mut updated_pairs = Vec::new();
    for (key, value) in pairs {
        if key == "scope" {
            let mut scopes = value.split_whitespace().collect::<Vec<_>>();
            if !scopes.contains(&"offline_access") {
                scopes.push("offline_access");
            }
            updated_pairs.push((key, scopes.join(" ")));
        } else {
            updated_pairs.push((key, value));
        }
    }
    if scope_value.is_none() {
        updated_pairs.push(("scope".to_string(), "offline_access".to_string()));
    }
    url.set_query(None);
    {
        let mut pairs = url.query_pairs_mut();
        for (key, value) in updated_pairs {
            pairs.append_pair(&key, &value);
        }
        if !has_challenge {
            pairs.append_pair("code_challenge", &pkce.challenge);
        }
        if !has_method {
            pairs.append_pair("code_challenge_method", "S256");
        }
        if state_value.is_none() {
            pairs.append_pair("state", &pkce.state);
        }
    }
    let expected_state = state_value.unwrap_or_else(|| pkce.state.clone());
    Ok((url, expected_state))
}

fn extract_redirect_uri(authorize_url: &Url) -> Result<Url> {
    let redirect_uri = authorize_url
        .query_pairs()
        .find(|(key, _)| key == "redirect_uri")
        .map(|(_, value)| value.to_string())
        .context("Missing redirect_uri in auth URL")?;
    Url::parse(&redirect_uri).context("Invalid redirect_uri")
}

fn extract_client_id(authorize_url: &Url) -> Result<String> {
    authorize_url
        .query_pairs()
        .find(|(key, _)| key == "client_id")
        .map(|(_, value)| value.to_string())
        .context("Missing client_id in auth URL")
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
        anyhow::bail!(
            "Interactive login requires a loopback redirect URI; use --api-key instead."
        );
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
    let n = stream.try_read(&mut buf).context("Failed to read callback")?;
    let request = String::from_utf8_lossy(&buf[..n]);
    let path = request
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .unwrap_or("/");
    let callback_url = Url::parse(&format!("http://localhost{path}"))
        .context("Invalid callback URL")?;
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
        let text = response.text().await.unwrap_or_default();
        anyhow::bail!("Token exchange failed ({status}): {text}")
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
    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        anyhow::bail!("Token refresh failed ({status}): {text}")
    }
    let token = response
        .json::<TokenResponse>()
        .await
        .context("Failed to parse refresh response")?;
    Ok(AuthToken {
        access_token: token.access_token,
        refresh_token: token.refresh_token,
        expires_at: token
            .expires_in
            .map(|value| Utc::now() + chrono::Duration::seconds(value)),
        token_endpoint: Some(token_endpoint.to_string()),
        client_id: Some(client_id.to_string()),
    })
}

fn map_public_error<T: std::fmt::Debug>(err: public_api::Error<T>) -> anyhow::Error {
    anyhow::anyhow!(err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn remote_config_roundtrip() {
        let dir = tempdir().expect("tempdir");
        env::set_var("RUNMAT_CLI_CONFIG_DIR", dir.path());
        let mut config = RemoteConfig::default();
        config.server_url = Some("http://localhost".to_string());
        config.org_id = Some(Uuid::new_v4());
        config.project_id = Some(Uuid::new_v4());
        config.save().expect("save");

        let loaded = RemoteConfig::load().expect("load");
        assert_eq!(loaded.server_url, config.server_url);
        assert_eq!(loaded.org_id, config.org_id);
        assert_eq!(loaded.project_id, config.project_id);
        env::remove_var("RUNMAT_CLI_CONFIG_DIR");
    }

    #[test]
    fn resolve_project_id_from_env() {
        let project_id = Uuid::new_v4();
        env::set_var("RUNMAT_PROJECT_ID", project_id.to_string());
        let config = RemoteConfig::default();
        let resolved = resolve_project_id(&config, None).expect("resolve");
        assert_eq!(resolved, project_id);
        env::remove_var("RUNMAT_PROJECT_ID");
    }

    #[test]
    fn pkce_is_added_to_authorize_url() {
        let pkce = generate_pkce();
        let url = "https://auth.example/authorize?client_id=abc&redirect_uri=http%3A%2F%2F127.0.0.1%3A7777%2Fcallback&response_type=code";
        let (updated, state) = apply_pkce_to_authorize_url(url, &pkce).expect("apply pkce");
        let has_challenge = updated
            .query_pairs()
            .any(|(key, _)| key == "code_challenge");
        let has_method = updated
            .query_pairs()
            .any(|(key, _)| key == "code_challenge_method");
        let scope = updated
            .query_pairs()
            .find(|(key, _)| key == "scope")
            .map(|(_, value)| value.to_string())
            .unwrap_or_default();
        assert!(has_challenge);
        assert!(has_method);
        assert!(scope.contains("offline_access"));
        assert!(!state.is_empty());
    }
}
