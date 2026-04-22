use anyhow::Result;
use runmat_server_client::auth::{
    build_public_client, map_public_error, resolve_auth_token, resolve_org_id, resolve_project_id,
    resolve_server_url, RemoteConfig,
};
use runmat_server_client::public_api;
use uuid::Uuid;

pub async fn list_projects(
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

pub async fn create_project(org: Option<Uuid>, name: String) -> Result<()> {
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

pub async fn list_project_members(
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

pub async fn select_project(project_id: Uuid) -> Result<()> {
    let mut config = RemoteConfig::load()?;
    config.project_id = Some(project_id);
    config.save()?;
    println!("Default project set to {project_id}");
    Ok(())
}
