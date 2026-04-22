use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use runmat_server_client::auth::{
    resolve_auth_token, resolve_project_id, resolve_server_url, RemoteConfig,
};
use serde::Deserialize;
use uuid::Uuid;

use crate::remote::shared::build_http_client;

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

pub async fn get_project_retention(project: Option<Uuid>) -> Result<()> {
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

pub async fn set_project_retention(project: Option<Uuid>, max_versions: usize) -> Result<()> {
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
