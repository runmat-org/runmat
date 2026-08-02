use std::str::FromStr;
use std::sync::Arc;

use runmat_package::{ContentDigest, ServerProjectAcquisitionPlan};
use runmat_package_cache::{
    ServerProjectTreeInventory, TreeInventoryEntry, TreeInventoryEntryKind,
};
use runmat_server_client::package_snapshot::{
    ProjectSnapshotClient, ProjectSnapshotClientError, ProjectSnapshotEntryKind,
    ProjectSnapshotOutcome,
};

use crate::server::ServerSnapshotTransport;

use super::AccessTokenProvider;

pub struct HttpServerSnapshotTransport {
    credentials: Arc<dyn AccessTokenProvider>,
}

impl HttpServerSnapshotTransport {
    pub fn new(credentials: Arc<dyn AccessTokenProvider>) -> Self {
        Self { credentials }
    }
}

impl ServerSnapshotTransport for HttpServerSnapshotTransport {
    fn fetch<'a>(
        &'a self,
        plan: &'a ServerProjectAcquisitionPlan,
    ) -> futures::future::LocalBoxFuture<'a, Result<ServerProjectTreeInventory, String>> {
        Box::pin(async move {
            let mut credentials = self.credentials.snapshot(&plan.service).await?;
            let outcome = match fetch(plan, credentials.token()).await {
                Err(
                    ProjectSnapshotClientError::Unauthorized
                    | ProjectSnapshotClientError::Forbidden,
                ) => {
                    credentials = self
                        .credentials
                        .refresh_after_rejection(&plan.service, credentials.generation)
                        .await?;
                    fetch(plan, credentials.token()).await
                }
                outcome => outcome,
            }
            .map_err(|error| error.to_string())?;
            let response = match outcome {
                ProjectSnapshotOutcome::Snapshot(response) => response,
                ProjectSnapshotOutcome::NotModified => {
                    return Err("unexpected not-modified response without an ETag".into())
                }
            };
            let inventory = response.inventory;
            Ok(ServerProjectTreeInventory {
                project: inventory.project,
                snapshot: inventory.snapshot,
                tree_digest: ContentDigest::from_str(&inventory.tree_digest)
                    .map_err(|error| error.to_string())?,
                entries: inventory
                    .entries
                    .into_iter()
                    .map(|entry| TreeInventoryEntry {
                        path: entry.path,
                        kind: match entry.kind {
                            ProjectSnapshotEntryKind::File => TreeInventoryEntryKind::File,
                            ProjectSnapshotEntryKind::Directory => {
                                TreeInventoryEntryKind::Directory
                            }
                        },
                        bytes: entry.bytes,
                        executable: entry.executable,
                        link_target: None,
                    })
                    .collect(),
            })
        })
    }
}

async fn fetch(
    plan: &ServerProjectAcquisitionPlan,
    token: Option<&str>,
) -> Result<ProjectSnapshotOutcome, ProjectSnapshotClientError> {
    ProjectSnapshotClient::new(&plan.service, token.map(str::to_owned))?
        .fetch(&plan.project, plan.selector.value(), None)
        .await
}
