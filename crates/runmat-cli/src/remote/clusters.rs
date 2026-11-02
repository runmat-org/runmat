use std::num::NonZeroU64;

use anyhow::{Context, Result};
use runmat_server_client::auth::{
    build_public_client, map_public_error, resolve_auth_token, resolve_server_url, RemoteConfig,
};
use runmat_server_client::public_api;
use uuid::Uuid;

use crate::cli::{ClusterStateArg, NodeStateArg};

async fn client(
    org: Option<Uuid>,
    machine_output: bool,
) -> Result<(public_api::Client, runmat_server_client::ids::OrgId)> {
    let mut config = RemoteConfig::load()?;
    let server_url = resolve_server_url(&config, None)?;
    let token = resolve_auth_token(&mut config, &server_url).await?;
    let client = build_public_client(&server_url, &token)?;
    let org_id = super::organization::resolve(&client, &config, org, machine_output).await?;
    Ok((client, org_id))
}

pub async fn list(
    org: Option<Uuid>,
    limit: Option<u32>,
    cursor: Option<String>,
    json: bool,
) -> Result<()> {
    let (client, org_id) = client(org, json).await?;
    let response = client
        .list_clusters(org_id.as_str(), cursor.as_deref(), nonzero_limit(limit)?)
        .await
        .map_err(map_public_error)?
        .into_inner();
    if json {
        println!("{}", serde_json::to_string(&response)?);
    } else {
        for cluster in response.clusters {
            println!(
                "{}\t{}\t{}\t{}",
                cluster.id,
                cluster.name,
                cluster.state,
                cluster.queues.join(",")
            );
        }
        print_cursor(response.next_cursor);
    }
    Ok(())
}

pub async fn create(
    org: Option<Uuid>,
    name: String,
    project_id: Option<Uuid>,
    queues: Vec<String>,
    json: bool,
) -> Result<()> {
    let (client, org_id) = client(org, json).await?;
    let cluster = client
        .create_cluster(
            org_id.as_str(),
            &public_api::types::CreateClusterRequest {
                name: name.try_into().context("invalid cluster name")?,
                project_id: project_id
                    .map(|id| runmat_server_client::ids::ProjectId::from_uuid(id).to_string()),
                queues,
            },
        )
        .await
        .map_err(map_public_error)?
        .into_inner();
    if json {
        println!("{}", serde_json::to_string(&cluster)?);
    } else {
        println!("{}\t{}\t{}", cluster.id, cluster.name, cluster.state);
    }
    Ok(())
}

pub async fn set_state(
    org: Option<Uuid>,
    cluster_id: String,
    state: ClusterStateArg,
    json: bool,
) -> Result<()> {
    let (client, org_id) = client(org, json).await?;
    let cluster_id =
        runmat_server_client::execution::org_cluster_by_name(&client, &org_id, &cluster_id).await?;
    let state = match state {
        ClusterStateArg::Active => public_api::types::ClusterStateRequest::Active,
        ClusterStateArg::Draining => public_api::types::ClusterStateRequest::Draining,
        ClusterStateArg::Disabled => public_api::types::ClusterStateRequest::Disabled,
    };
    let cluster = client
        .set_cluster_state(
            org_id.as_str(),
            cluster_id.as_str(),
            &public_api::types::SetClusterStateRequest { state },
        )
        .await
        .map_err(map_public_error)?
        .into_inner();
    if json {
        println!("{}", serde_json::to_string(&cluster)?);
    } else {
        println!("{}\t{}\t{}", cluster.id, cluster.name, cluster.state);
    }
    Ok(())
}

pub async fn enroll(
    org: Option<Uuid>,
    cluster_id: String,
    ttl_seconds: i64,
    requested_identity_fingerprint: Option<String>,
    json: bool,
) -> Result<()> {
    let (client, org_id) = client(org, json).await?;
    let cluster_id =
        runmat_server_client::execution::org_cluster_by_name(&client, &org_id, &cluster_id).await?;
    let enrollment = client
        .create_node_enrollment(
            org_id.as_str(),
            cluster_id.as_str(),
            &public_api::types::CreateEnrollmentRequest {
                requested_identity_fingerprint,
                ttl_seconds,
            },
        )
        .await
        .map_err(map_public_error)?
        .into_inner();
    if json {
        println!("{}", serde_json::to_string(&enrollment)?);
    } else {
        println!(
            "{}\t{}\t{}\t{}",
            enrollment.id, enrollment.cluster_id, enrollment.expires_at, enrollment.token
        );
    }
    Ok(())
}

pub async fn list_nodes(
    org: Option<Uuid>,
    cluster_id: String,
    limit: Option<u32>,
    cursor: Option<String>,
    json: bool,
) -> Result<()> {
    let (client, org_id) = client(org, json).await?;
    let cluster_id =
        runmat_server_client::execution::org_cluster_by_name(&client, &org_id, &cluster_id).await?;
    let response = client
        .list_nodes(
            org_id.as_str(),
            cluster_id.as_str(),
            cursor.as_deref(),
            nonzero_limit(limit)?,
        )
        .await
        .map_err(map_public_error)?
        .into_inner();
    if json {
        println!("{}", serde_json::to_string(&response)?);
    } else {
        for node in response.nodes {
            println!(
                "{}\t{}\t{}\t{}",
                node.id, node.state, node.identity_fingerprint, node.heartbeat_expires_at
            );
        }
        print_cursor(response.next_cursor);
    }
    Ok(())
}

pub async fn set_node_state(
    org: Option<Uuid>,
    cluster_id: String,
    node_id: String,
    state: NodeStateArg,
    json: bool,
) -> Result<()> {
    let (client, org_id) = client(org, json).await?;
    let cluster_id =
        runmat_server_client::execution::org_cluster_by_name(&client, &org_id, &cluster_id).await?;
    let state = match state {
        NodeStateArg::Active => public_api::types::NodeStateRequest::Active,
        NodeStateArg::Draining => public_api::types::NodeStateRequest::Draining,
        NodeStateArg::Offline => public_api::types::NodeStateRequest::Offline,
        NodeStateArg::Revoked => public_api::types::NodeStateRequest::Revoked,
    };
    let node = client
        .set_node_state(
            org_id.as_str(),
            cluster_id.as_str(),
            &node_id,
            &public_api::types::SetNodeStateRequest { state },
        )
        .await
        .map_err(map_public_error)?
        .into_inner();
    if json {
        println!("{}", serde_json::to_string(&node)?);
    } else {
        println!("{}\t{}\t{}", node.id, node.state, node.heartbeat_expires_at);
    }
    Ok(())
}

fn nonzero_limit(value: Option<u32>) -> Result<Option<NonZeroU64>> {
    value
        .map(|value| NonZeroU64::new(u64::from(value)).context("limit must be greater than zero"))
        .transpose()
}

fn print_cursor(cursor: Option<String>) {
    if let Some(cursor) = cursor {
        println!("next_cursor\t{cursor}");
    }
}
