use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::Deserialize;
use uuid::Uuid;

use crate::remote::shared::build_http_client;

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
    created_at: DateTime<Utc>,
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
    created_at: DateTime<Utc>,
    #[serde(rename = "createdBy")]
    created_by: Option<Uuid>,
}

pub async fn list_snapshots(server_url: &str, token: &str, project_id: Uuid) -> Result<()> {
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
        let message = item.message.unwrap_or_default();
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

pub async fn create_snapshot(
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
    let message = payload.message.unwrap_or_default();
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

pub async fn restore_snapshot(
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

pub async fn delete_snapshot(
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

pub async fn list_snapshot_tags(server_url: &str, token: &str, project_id: Uuid) -> Result<()> {
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

pub async fn set_snapshot_tag(
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

pub async fn delete_snapshot_tag(
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
