use futures::executor::block_on;
use runmat_package::{
    resolve_project_async, DependencyGroup, ProjectResolveOptions, SourceAcquisitionIntent,
    SourceAcquisitionPolicy,
};
use runmat_package_cache::{
    CacheBackend, ServerProjectTreeInventory, SnapshotBlob, TreeEntry, TreeInventoryEntry,
    TreeManifest,
};
use runmat_package_cache_native::server::ServerSnapshotTransport;
use runmat_package_cache_native::{
    git::NativeGitClient, NativeCacheConfig, NativePackageSourceProvider, SqliteCacheBackend,
};
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use tempfile::TempDir;

struct FixtureTransport {
    inventory: ServerProjectTreeInventory,
    unavailable: AtomicBool,
    calls: AtomicUsize,
}

impl ServerSnapshotTransport for FixtureTransport {
    fn fetch<'a>(
        &'a self,
        _plan: &'a runmat_package::ServerProjectAcquisitionPlan,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<ServerProjectTreeInventory, String>> + 'a>,
    > {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if self.unavailable.load(Ordering::SeqCst) {
                Err("snapshot permission was revoked or snapshot was deleted".to_string())
            } else {
                Ok(self.inventory.clone())
            }
        })
    }
}

#[test]
fn locked_server_project_replays_from_cache_after_access_or_snapshot_loss() {
    let temp = TempDir::new().unwrap();
    let project = temp.path().join("project");
    std::fs::create_dir_all(project.join("src")).unwrap();
    std::fs::write(project.join("src/main.m"), "answer = helper();\n").unwrap();
    std::fs::write(
        project.join("runmat.toml"),
        r#"
[package]
name = "application"
version = "1.0.0"
[sources]
roots = ["src"]
[dependencies]
helper = { project = "proj_0123456789abcdef0123456789abcdef", service = "https://api.runmat.com", snapshot = "main", version = "2.0.0" }
"#,
    )
    .unwrap();

    let cache_config = NativeCacheConfig {
        root: temp.path().join("cache"),
        quota_bytes: None,
    };
    let layout = cache_config.layout();
    let backend = Arc::new(SqliteCacheBackend::open(&cache_config).unwrap());
    let transport = Arc::new(FixtureTransport {
        inventory: server_inventory(),
        unavailable: AtomicBool::new(false),
        calls: AtomicUsize::new(0),
    });
    let provider = NativePackageSourceProvider::new(
        NativeGitClient::new(layout.clone()),
        backend.clone(),
        layout,
    )
    .with_server_transport(transport.clone());
    let manifest = project.join("runmat.toml");
    transport.unavailable.store(true, Ordering::SeqCst);
    assert!(block_on(resolve_project_async(
        &manifest,
        None,
        options(SourceAcquisitionPolicy::default()),
        &provider,
    ))
    .is_err());
    assert!(block_on(backend.snapshot())
        .unwrap()
        .state
        .objects
        .is_empty());

    transport.unavailable.store(false, Ordering::SeqCst);
    let first = block_on(resolve_project_async(
        &manifest,
        None,
        options(SourceAcquisitionPolicy::default()),
        &provider,
    ))
    .unwrap();
    assert_eq!(transport.calls.load(Ordering::SeqCst), 2);
    assert_eq!(first.acquired_server_sources.len(), 1);
    assert_eq!(
        first.acquired_server_sources[0].snapshot,
        "snap_0123456789abcdef0123456789abcdef"
    );

    transport.unavailable.store(true, Ordering::SeqCst);
    let replay = block_on(resolve_project_async(
        &manifest,
        Some(&first.lock),
        options(SourceAcquisitionPolicy {
            locked: true,
            frozen: false,
            offline: true,
        }),
        &provider,
    ))
    .unwrap();
    assert_eq!(transport.calls.load(Ordering::SeqCst), 2);
    assert_eq!(replay.lock, first.lock);
    assert_eq!(
        replay.frozen.graph.graph_digest,
        first.frozen.graph.graph_digest
    );
}

fn options(policy: SourceAcquisitionPolicy) -> ProjectResolveOptions {
    ProjectResolveOptions {
        target: "x86_64-unknown-linux-gnu".to_string(),
        default_server_origin: "https://api.runmat.com".to_string(),
        groups: [DependencyGroup::Runtime].into_iter().collect(),
        root_features: BTreeSet::new(),
        host_capabilities: BTreeSet::new(),
        source_intent: SourceAcquisitionIntent::Execute,
        source_policy: policy,
    }
}

fn server_inventory() -> ServerProjectTreeInventory {
    let manifest = br#"
[package]
name = "helper"
version = "2.0.0"
[sources]
roots = ["src"]
"#
    .to_vec();
    let source = b"function value = helper(); value = 42; end\n".to_vec();
    let manifest_blob = SnapshotBlob::new(manifest.clone());
    let source_blob = SnapshotBlob::new(source.clone());
    let tree = TreeManifest::new(vec![
        TreeEntry::file(
            "runmat.toml".parse().unwrap(),
            manifest_blob.digest,
            manifest.len() as u64,
            false,
        ),
        TreeEntry::directory("src".parse().unwrap()),
        TreeEntry::file(
            "src/helper.m".parse().unwrap(),
            source_blob.digest,
            source.len() as u64,
            false,
        ),
    ])
    .unwrap();
    ServerProjectTreeInventory {
        project: "proj_0123456789abcdef0123456789abcdef".to_string(),
        snapshot: "snap_0123456789abcdef0123456789abcdef".to_string(),
        tree_digest: tree.digest,
        entries: vec![
            TreeInventoryEntry::file("runmat.toml", manifest, false),
            TreeInventoryEntry::directory("src"),
            TreeInventoryEntry::file("src/helper.m", source, false),
        ],
    }
}
