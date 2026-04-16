use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION};
use serde::Deserialize;
use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::Command;
use uuid::Uuid;

use runmat_config::RunMatConfig;
use runmat_filesystem::{FsFileType, FsProvider, RemoteFsConfig, RemoteFsProvider};

use crate::{
    FsCommand, OrgCommand, ProjectCommand, ProjectMembersCommand, ProjectRetentionCommand,
    RemoteCommand,
};
use runmat_server_client::auth::{
    build_public_client, execute_login, map_public_error, resolve_auth_token, resolve_org_id,
    resolve_project_id, resolve_server_url, CredentialStoreMode, RemoteConfig,
};
use runmat_server_client::public_api;

pub async fn execute_login_command(
    server: Option<String>,
    api_key: Option<String>,
    email: Option<String>,
    credential_store: CredentialStoreMode,
    org: Option<Uuid>,
    project: Option<Uuid>,
) -> Result<()> {
    execute_login(server, api_key, email, org, project, Some(credential_store)).await
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
        ProjectCommand::Members { members_command } => {
            execute_project_members_command(members_command).await
        }
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
    let base_url = format!(
        "{}/v1/projects/{}",
        server_url.trim_end_matches('/'),
        project_id
    );

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
                update_manifest(
                    &server_url,
                    &token,
                    project_id,
                    &path,
                    base_version,
                    &manifest,
                )
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
            } => create_snapshot(&server_url, &token, project_id, message, parent, tag).await?,
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
                directory, server, ..
            } => {
                let url = resolve_server_url(&config, server.clone())?;
                git_clone(&url, &token, project_id, &directory).await?
            }
            FsCommand::GitPull {
                directory, server, ..
            } => {
                let url = resolve_server_url(&config, server.clone())?;
                git_pull(&url, &token, project_id, &directory).await?
            }
            FsCommand::GitPush {
                directory, server, ..
            } => {
                let url = resolve_server_url(&config, server.clone())?;
                git_push(&url, &token, project_id, &directory).await?
            }
            _ => {}
        }
        return Ok(());
    }

    let runtime = tokio::runtime::Handle::current();
    let task = tokio::task::spawn_blocking(move || -> Result<()> {
        let shard_threshold_bytes = read_u64_env("RUNMAT_FS_SHARD_THRESHOLD_BYTES");
        let shard_size_bytes = read_u64_env("RUNMAT_FS_SHARD_SIZE_BYTES");
        let provider = RemoteFsProvider::new(RemoteFsConfig {
            base_url,
            auth_token: Some(token),
            shard_threshold_bytes: shard_threshold_bytes
                .unwrap_or(RemoteFsConfig::default().shard_threshold_bytes),
            shard_size_bytes: shard_size_bytes
                .unwrap_or(RemoteFsConfig::default().shard_size_bytes),
            ..RemoteFsConfig::default()
        })
        .context("Failed to initialize remote filesystem")?;

        match command {
            FsCommand::Read { path, output, .. } => {
                let bytes = runtime
                    .block_on(provider.read(std::path::Path::new(&path)))
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
                runtime
                    .block_on(provider.write(std::path::Path::new(&path), &bytes))
                    .context("Failed to write remote file")?;
                Ok(())
            }
            FsCommand::Ls { path, .. } => {
                let entries = runtime
                    .block_on(provider.read_dir(std::path::Path::new(&path)))
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
                path, recursive, ..
            } => {
                if recursive {
                    runtime
                        .block_on(provider.create_dir_all(std::path::Path::new(&path)))
                        .context("Failed to create directory")?;
                } else {
                    runtime
                        .block_on(provider.create_dir(std::path::Path::new(&path)))
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
                    runtime
                        .block_on(provider.remove_dir_all(std::path::Path::new(&path)))
                        .context("Failed to remove directory")?;
                } else if dir {
                    runtime
                        .block_on(provider.remove_dir(std::path::Path::new(&path)))
                        .context("Failed to remove directory")?;
                } else {
                    runtime
                        .block_on(provider.remove_file(std::path::Path::new(&path)))
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
        .list_orgs(cursor.as_deref(), limit.map(|value| value as u64))
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
    let org_id = org_id.to_string();
    let response = client
        .list_projects(&org_id, cursor.as_deref(), limit.map(|value| value as u64))
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
    let org_id = org_id.to_string();
    let response = client
        .create_project(&org_id, &public_api::types::ProjectCreateRequest { name })
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
    let project_id = project_id.to_string();
    let response = client
        .list_project_memberships(
            &project_id,
            cursor.as_deref(),
            limit.map(|value| value as u64),
        )
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

fn build_http_client(server_url: &str, token: &str) -> Result<(reqwest::Client, String)> {
    let mut headers = HeaderMap::new();
    let auth_value =
        HeaderValue::from_str(&format!("Bearer {token}")).context("Invalid auth token")?;
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
    let base_url = format!(
        "{}/v1/projects/{}",
        server_url.trim_end_matches('/'),
        project_id
    );
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

async fn list_history(server_url: &str, token: &str, project_id: Uuid, path: &str) -> Result<()> {
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
        println!(
            "{}\t{}\t{}\t{}",
            item.id, item.created_at, item.size, item.path
        );
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
    let url = format!(
        "{}/v1/projects/{}/fs/history/{}",
        base, project_id, version_id
    );
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
    let manifest: serde_json::Value =
        serde_json::from_str(&contents).context("Manifest must be valid JSON")?;
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
    let url = format!(
        "{}/v1/projects/{}/fs/snapshots/{}",
        base, project_id, snapshot_id
    );
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
    let url = format!(
        "{}/v1/projects/{}/fs/snapshots/tags/{}",
        base, project_id, tag
    );
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
    let branches = run_git_capture(
        directory,
        &["for-each-ref", "refs/heads", "--format=%(refname)"],
    )
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
    let url = format!(
        "{}/v1/projects/{}/fs/git/receive",
        server_url.trim_end_matches('/'),
        project_id
    );
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
    let url = format!(
        "{}/v1/projects/{}/fs/git/upload",
        server_url.trim_end_matches('/'),
        project_id
    );
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

#[cfg(test)]
mod tests {
    use runmat_server_client::auth::{resolve_project_id, RemoteConfig};
    use std::env;
    use tempfile::tempdir;
    use uuid::Uuid;

    #[test]
    fn remote_config_roundtrip() {
        let dir = tempdir().expect("tempdir");
        env::set_var("RUNMAT_CLI_CONFIG_DIR", dir.path());
        let config = RemoteConfig {
            server_url: Some("http://localhost".to_string()),
            org_id: Some(Uuid::new_v4()),
            project_id: Some(Uuid::new_v4()),
            ..RemoteConfig::default()
        };
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
}
