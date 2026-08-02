use futures::executor::block_on;
use runmat_package::{
    decode_lock, DependencyGroup, ProjectResolveOptions, SourceAcquisitionIntent,
    SourceAcquisitionPolicy,
};
use runmat_package_cache_native::{
    git::NativeGitClient, resolve_native_project, NativeCacheConfig, NativePackageSourceProvider,
    NativeProjectResolveRequest, SqliteCacheBackend,
};
use std::collections::BTreeSet;
use std::sync::Arc;
use tempfile::TempDir;

#[test]
fn native_project_service_owns_lock_cache_and_handoff_lifecycle() {
    let temporary = TempDir::new().unwrap();
    let project = temporary.path().join("project");
    std::fs::create_dir_all(project.join("src")).unwrap();
    std::fs::write(project.join("src/main.m"), "answer = 42;\n").unwrap();
    std::fs::write(
        project.join("runmat.toml"),
        r#"
[package]
name = "application"
version = "1.0.0"

[sources]
roots = ["src"]
"#,
    )
    .unwrap();

    let cache_config = NativeCacheConfig {
        root: temporary.path().join("cache"),
        quota_bytes: None,
    };
    let layout = cache_config.layout();
    let backend = Arc::new(SqliteCacheBackend::open(&cache_config).unwrap());
    let provider = NativePackageSourceProvider::new(
        NativeGitClient::new(layout.clone()),
        backend.clone(),
        layout,
    );
    let project_result = block_on(resolve_native_project(
        NativeProjectResolveRequest {
            manifest_path: project.join("runmat.toml"),
            options: ProjectResolveOptions {
                target: "x86_64-unknown-linux-gnu".to_string(),
                default_server_origin: "https://api.runmat.com".to_string(),
                default_registry_index: "https://api.runmat.com".to_string(),
                groups: [DependencyGroup::Runtime].into_iter().collect(),
                root_features: BTreeSet::new(),
                host_capabilities: BTreeSet::new(),
                source_intent: SourceAcquisitionIntent::Execute,
                source_policy: SourceAcquisitionPolicy::default(),
            },
        },
        backend,
        cache_config.clone(),
        provider,
    ))
    .unwrap();

    let lock = decode_lock(&std::fs::read_to_string(project.join("runmat.lock")).unwrap()).unwrap();
    assert_eq!(lock, project_result.resolved.lock);
    assert_eq!(project_result.cache_config, cache_config);
    assert_eq!(
        project_result.handoff().project.graph_digest(),
        project_result.resolved.frozen.graph_digest()
    );
}
