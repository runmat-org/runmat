use futures::executor::block_on;
use runmat_package::{
    resolve_project_async, CanonicalPackageId, ContentDigest, DependencyGroup, PackageVersion,
    ProjectResolveOptions, RegistryCandidateRecord, RegistryOrigin, RegistryReleaseId,
    RegistryReleaseMetadata, RegistrySourceId, SourceAcquisitionIntent, SourceAcquisitionPolicy,
};
use runmat_package_cache::{
    CacheBackend, GitTreeInventory, RegistryArtifactInventory, TreeInventoryEntry,
    REGISTRY_ARTIFACT_SCHEMA_VERSION,
};
use runmat_package_cache_native::registry::{
    encrypt_private_artifact, InMemoryRecipientKeyRing, RecipientKeyPair,
};
use runmat_package_cache_native::registry::{RegistryArtifactTransfer, RegistryTransport};
use runmat_package_cache_native::{
    git::NativeGitClient, NativeCacheConfig, NativePackageSourceProvider, SqliteCacheBackend,
};
use std::collections::BTreeSet;
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use tempfile::TempDir;

struct FixtureRegistryTransport {
    record: RegistryCandidateRecord,
    artifact: Vec<u8>,
    key_envelopes: Vec<runmat_package::PackageKeyEnvelope>,
    corrupt: AtomicBool,
    unavailable: AtomicBool,
    candidate_calls: AtomicUsize,
    fetch_calls: AtomicUsize,
}

impl RegistryTransport for FixtureRegistryTransport {
    fn candidates<'a>(
        &'a self,
        _plan: &'a runmat_package::RegistryCandidatePlan,
    ) -> futures::future::LocalBoxFuture<'a, Result<Vec<RegistryCandidateRecord>, String>> {
        Box::pin(async move {
            self.candidate_calls.fetch_add(1, Ordering::SeqCst);
            if self.unavailable.load(Ordering::SeqCst) {
                Err("registry is unavailable".to_string())
            } else {
                Ok(vec![self.record.clone()])
            }
        })
    }

    fn fetch<'a>(
        &'a self,
        _plan: &'a runmat_package::RegistryAcquisitionPlan,
    ) -> futures::future::LocalBoxFuture<'a, Result<RegistryArtifactTransfer, String>> {
        Box::pin(async move {
            self.fetch_calls.fetch_add(1, Ordering::SeqCst);
            if self.unavailable.load(Ordering::SeqCst) {
                return Err("registry is unavailable".to_string());
            }
            let mut artifact_bytes = self.artifact.clone();
            if self.corrupt.load(Ordering::SeqCst) {
                artifact_bytes.push(b' ');
            }
            Ok(RegistryArtifactTransfer {
                package_id: "pkg_0123456789abcdef0123456789abcdef".to_string(),
                source: self.record.source.clone(),
                metadata: self.record.metadata.clone(),
                artifact_bytes,
                key_envelopes: self.key_envelopes.clone(),
            })
        })
    }
}

#[test]
fn registry_resolution_is_transactional_and_locked_offline_replay_uses_cache() {
    let temp = TempDir::new().unwrap();
    let project = temp.path().join("project");
    std::fs::create_dir_all(project.join("src")).unwrap();
    std::fs::write(project.join("src/main.m"), "answer = tools();\n").unwrap();
    std::fs::write(
        project.join("runmat.toml"),
        r#"
[package]
name = "application"
version = "1.0.0"
[sources]
roots = ["src"]
[dependencies]
tools = { package = "acme/tools", version = "^1" }
"#,
    )
    .unwrap();

    let (record, artifact) = release();
    let transport = Arc::new(FixtureRegistryTransport {
        record,
        artifact,
        key_envelopes: Vec::new(),
        corrupt: AtomicBool::new(true),
        unavailable: AtomicBool::new(false),
        candidate_calls: AtomicUsize::new(0),
        fetch_calls: AtomicUsize::new(0),
    });
    let cache_config = NativeCacheConfig {
        root: temp.path().join("cache"),
        quota_bytes: None,
    };
    let layout = cache_config.layout();
    let backend = Arc::new(SqliteCacheBackend::open(&cache_config).unwrap());
    let provider = NativePackageSourceProvider::new(
        NativeGitClient::new(layout.clone()),
        backend.clone(),
        layout,
    )
    .with_registry_transport(transport.clone());
    let manifest = project.join("runmat.toml");

    let error = block_on(resolve_project_async(
        &manifest,
        None,
        options(SourceAcquisitionPolicy::default()),
        &provider,
    ))
    .unwrap_err()
    .to_string();
    assert!(error.contains("digest"), "unexpected error: {error}");
    assert!(block_on(backend.snapshot())
        .unwrap()
        .state
        .objects
        .is_empty());

    transport.corrupt.store(false, Ordering::SeqCst);
    let first = block_on(resolve_project_async(
        &manifest,
        None,
        options(SourceAcquisitionPolicy::default()),
        &provider,
    ))
    .unwrap();
    assert_eq!(first.acquired_registry_sources.len(), 1);
    assert_eq!(transport.candidate_calls.load(Ordering::SeqCst), 2);
    assert_eq!(transport.fetch_calls.load(Ordering::SeqCst), 2);

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
    assert_eq!(replay.lock, first.lock);
    assert_eq!(
        replay.frozen.graph.graph_digest,
        first.frozen.graph.graph_digest
    );
    assert_eq!(transport.candidate_calls.load(Ordering::SeqCst), 2);
    assert_eq!(transport.fetch_calls.load(Ordering::SeqCst), 2);
}

#[test]
fn encrypted_registry_resolution_decrypts_without_persisting_plaintext_in_the_cache() {
    let temp = TempDir::new().unwrap();
    let project = temp.path().join("project");
    std::fs::create_dir_all(project.join("src")).unwrap();
    std::fs::write(project.join("src/main.m"), "answer = tools();\n").unwrap();
    std::fs::write(
        project.join("runmat.toml"),
        r#"
[package]
name = "application"
version = "1.0.0"
[sources]
roots = ["src"]
[dependencies]
tools = { package = "acme/tools", version = "^1" }
"#,
    )
    .unwrap();

    let (mut record, plaintext) = release();
    let recipient = RecipientKeyPair::from_secret_bytes("pkr_test", [17; 32]).unwrap();
    let bundle = encrypt_private_artifact(
        &record.source,
        &plaintext,
        1,
        &[recipient.public_key().unwrap()],
    )
    .unwrap();
    record.source.artifact_digest = ContentDigest::sha256(&bundle.ciphertext);
    record.metadata.encryption = Some(bundle.metadata);
    record.source.release_digest = record.metadata.compute_digest(&record.source).unwrap();
    let transport = Arc::new(FixtureRegistryTransport {
        record,
        artifact: bundle.ciphertext,
        key_envelopes: bundle.envelopes,
        corrupt: AtomicBool::new(false),
        unavailable: AtomicBool::new(false),
        candidate_calls: AtomicUsize::new(0),
        fetch_calls: AtomicUsize::new(0),
    });
    let cache_config = NativeCacheConfig {
        root: temp.path().join("cache"),
        quota_bytes: None,
    };
    let layout = cache_config.layout();
    let backend = Arc::new(SqliteCacheBackend::open(&cache_config).unwrap());
    let provider = NativePackageSourceProvider::new(
        NativeGitClient::new(layout.clone()),
        backend.clone(),
        layout,
    )
    .with_registry_transport(transport)
    .with_private_artifact_decryptor(Arc::new(InMemoryRecipientKeyRing::new(vec![recipient])));

    let resolved = block_on(resolve_project_async(
        &project.join("runmat.toml"),
        None,
        options(SourceAcquisitionPolicy::default()),
        &provider,
    ))
    .unwrap();
    assert_eq!(resolved.acquired_registry_sources.len(), 1);
    let state = block_on(backend.snapshot()).unwrap().state;
    assert!(
        state.objects.is_empty() && state.materializations.is_empty(),
        "decrypted private package data must not enter the durable cache"
    );
    let private_source_path = resolved
        .frozen
        .access_paths
        .values()
        .find(|path| path.to_string_lossy().contains("runmat-private-package-"))
        .expect("private source path is mounted in the ephemeral tree")
        .clone();
    assert!(private_source_path.exists());
    drop(provider);
    assert!(
        !private_source_path.exists(),
        "dropping the provider must remove ephemeral plaintext mounts"
    );
}

fn options(policy: SourceAcquisitionPolicy) -> ProjectResolveOptions {
    ProjectResolveOptions {
        target: "x86_64-unknown-linux-gnu".to_string(),
        default_server_origin: "https://api.runmat.test".to_string(),
        default_registry_index: "https://packages.runmat.test".to_string(),
        groups: [DependencyGroup::Runtime].into_iter().collect(),
        root_features: BTreeSet::new(),
        host_capabilities: BTreeSet::new(),
        source_intent: SourceAcquisitionIntent::Execute,
        source_policy: policy,
    }
}

fn release() -> (RegistryCandidateRecord, Vec<u8>) {
    let manifest = br#"
[package]
name = "tools"
organization = "acme"
version = "1.2.3"
[sources]
roots = ["src"]
"#
    .to_vec();
    let source = b"function value = tools(); value = 42; end\n".to_vec();
    let inventory = RegistryArtifactInventory {
        schema_version: REGISTRY_ARTIFACT_SCHEMA_VERSION,
        entries: vec![
            TreeInventoryEntry::file("runmat.toml", manifest, false),
            TreeInventoryEntry::directory("src"),
            TreeInventoryEntry::file("src/tools.m", source, false),
        ],
    };
    let artifact = inventory.canonical_bytes().unwrap();
    let tree = GitTreeInventory {
        commit: "a".repeat(40),
        entries: inventory.entries,
    }
    .into_snapshot(
        "https://example.test/tools.git",
        ".",
        runmat_package_cache::ArchiveLimits::default(),
    )
    .unwrap()
    .tree;
    let metadata = RegistryReleaseMetadata {
        singleton: false,
        runmat_requirement: None,
        dependencies: Vec::new(),
        features: Default::default(),
        required_capabilities: Vec::new(),
        optional_capabilities: Vec::new(),
        readme_digest: None,
        license: Some("MIT".to_string()),
        encryption: None,
        supply_chain: None,
    };
    let mut source = RegistrySourceId {
        registry_origin: RegistryOrigin::new("https://packages.runmat.test").unwrap(),
        package: CanonicalPackageId::from_str("default:acme/tools").unwrap(),
        release: RegistryReleaseId::new("rel_0123456789abcdef0123456789abcdef").unwrap(),
        version: PackageVersion::from_str("1.2.3").unwrap(),
        release_digest: ContentDigest::sha256("placeholder"),
        artifact_digest: ContentDigest::sha256(&artifact),
        tree_digest: tree.digest,
    };
    source.release_digest = metadata.compute_digest(&source).unwrap();
    (
        RegistryCandidateRecord {
            source,
            metadata,
            yanked: false,
        },
        artifact,
    )
}
