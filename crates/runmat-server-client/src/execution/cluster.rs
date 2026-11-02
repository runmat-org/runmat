//! Resolve user-facing names once; execution requests retain canonical identities.
use anyhow::{Context as _, Result};

use crate::ids::{ClusterId, OrgId, ProjectId};
use crate::public_api;

pub async fn org_cluster_by_name(
    client: &public_api::Client,
    org: &OrgId,
    name: &str,
) -> Result<ClusterId> {
    let cluster = client
        .get_cluster_by_name(org.as_str(), name)
        .await
        .map_err(super::public_error)?
        .into_inner();
    ClusterId::try_from(cluster.id).context("Server returned an invalid cluster identity")
}

pub async fn project_cluster_by_name(
    client: &public_api::Client,
    project: &ProjectId,
    name: &str,
) -> Result<ClusterId> {
    let cluster = client
        .get_project_cluster_by_name(project.as_str(), name)
        .await
        .map_err(super::public_error)?
        .into_inner();
    ClusterId::try_from(cluster.id).context("Server returned an invalid cluster identity")
}
