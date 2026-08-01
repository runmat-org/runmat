use runmat_package::{ContentDigest, ServerProjectAcquisitionPlan, ServerProjectSourceId};
use runmat_package_cache::{
    ServerProjectTreeInventory, TreeInventoryEntry, TreeInventoryEntryKind,
};
use runmat_package_cache_native::server::ServerSnapshotTransport;
use runmat_server_client::auth::{
    resolve_auth_token, resolve_server_url, RemoteConfig, DEFAULT_SERVER_URL,
};
use runmat_server_client::package_snapshot::{
    ProjectSnapshotClient, ProjectSnapshotEntryKind, ProjectSnapshotOutcome,
};
use std::str::FromStr;

#[derive(Debug, Default)]
pub(super) struct RunMatServerSnapshotTransport;

impl RunMatServerSnapshotTransport {
    pub(super) fn default_origin(&self) -> String {
        let config = RemoteConfig::load().unwrap_or_default();
        resolve_server_url(&config, None).unwrap_or_else(|_| DEFAULT_SERVER_URL.to_string())
    }
}

impl ServerSnapshotTransport for RunMatServerSnapshotTransport {
    fn fetch<'a>(
        &'a self,
        plan: &'a ServerProjectAcquisitionPlan,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<ServerProjectTreeInventory, String>> + 'a>,
    > {
        Box::pin(async move {
            let mut config = RemoteConfig::load().map_err(|error| error.to_string())?;
            let configured =
                resolve_server_url(&config, None).map_err(|error| error.to_string())?;
            let configured_origin = ServerProjectSourceId::normalize_service(&configured)
                .map_err(|error| error.to_string())?;
            let token = if configured_origin == plan.service {
                resolve_auth_token(&mut config, &configured).await.ok()
            } else {
                None
            };
            let client = ProjectSnapshotClient::new(&plan.service, token)
                .map_err(|error| error.to_string())?;
            let outcome = client
                .fetch(&plan.project, plan.selector.value(), None)
                .await
                .map_err(|error| error.to_string())?;
            let response = match outcome {
                ProjectSnapshotOutcome::Snapshot(response) => response,
                ProjectSnapshotOutcome::NotModified => {
                    return Err("unexpected not-modified response without an ETag".to_string());
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
