use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::Deserialize;
use std::fs;
use std::path::PathBuf;
use uuid::Uuid;

use crate::remote::shared::build_http_client;

#[derive(Deserialize)]
struct ManifestHistoryResponse {
    items: Vec<ManifestHistoryEntry>,
}

#[derive(Deserialize)]
struct ManifestHistoryEntry {
    #[serde(rename = "versionId")]
    version_id: Uuid,
    #[serde(rename = "createdAt")]
    created_at: DateTime<Utc>,
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
    created_at: DateTime<Utc>,
}

pub async fn list_manifest_history(
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

pub async fn restore_manifest_version(
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

pub async fn list_history(
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
        println!(
            "{}\t{}\t{}\t{}",
            item.id, item.created_at, item.size, item.path
        );
    }
    Ok(())
}

pub async fn restore_version(
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

pub async fn delete_version(
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

pub async fn update_manifest(
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
