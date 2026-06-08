use anyhow::Result;
use runmat_server_client::auth::{
    build_public_client, map_public_error, resolve_auth_token, resolve_server_url, RemoteConfig,
};

pub async fn list_orgs(limit: Option<u32>, cursor: Option<String>) -> Result<()> {
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
