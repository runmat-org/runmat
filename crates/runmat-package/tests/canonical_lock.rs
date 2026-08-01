use runmat_package::{
    decode_lock, diff_locks, encode_lock, CanonicalPackageId, ContentDigest, DependencyGroup,
    HostCapability, LockCompatibility, LockSelection, LockedEdge, LockedPackage, PackageAlias,
    PackageInstanceId, PackageLock, PathSourceId, RootLock, ServerProjectSourceId, SourceId,
};
use semver::Version;
use std::collections::BTreeSet;

fn set<T: Ord>(values: impl IntoIterator<Item = T>) -> BTreeSet<T> {
    values.into_iter().collect()
}

fn instance(package: &str, path: &str, tree: &str) -> PackageInstanceId {
    let tree_digest = ContentDigest::sha256(tree);
    PackageInstanceId::new(
        package.parse::<CanonicalPackageId>().unwrap(),
        SourceId::Path(PathSourceId {
            workspace_path: path.parse().unwrap(),
            manifest_digest: ContentDigest::sha256(format!("{package} manifest")),
            tree_digest: tree_digest.clone(),
        }),
        Some("1.2.3".parse().unwrap()),
        tree_digest,
    )
}

fn fixture_lock() -> PackageLock {
    let matrix = instance("default:runmat/matrix", "deps/matrix", "matrix tree");
    let helper = instance("default:runmat/helper", "deps/helper", "helper tree");
    PackageLock::new(
        RootLock {
            manifest_digest: ContentDigest::sha256("root manifest"),
            package: Some("default:acme/application".parse().unwrap()),
        },
        LockSelection {
            target: "wasm32-unknown-unknown".to_string(),
            groups: set([DependencyGroup::Runtime, DependencyGroup::Test]),
            root_features: set(["default".to_string(), "web".to_string()]),
            host_capabilities: set([
                HostCapability::BrowserFilesystem,
                HostCapability::Network,
                HostCapability::Worker,
            ]),
        },
        vec![
            LockedPackage {
                instance: matrix.clone(),
                features: set(["sparse".to_string()]),
                required_capabilities: set([HostCapability::Network]),
                runmat_version: Some("^0.6.1".to_string()),
                singleton: false,
            },
            LockedPackage {
                instance: helper.clone(),
                features: BTreeSet::new(),
                required_capabilities: BTreeSet::new(),
                runmat_version: None,
                singleton: false,
            },
        ],
        vec![
            LockedEdge {
                from: Some(matrix.identity_digest.clone()),
                alias: "helper".parse::<PackageAlias>().unwrap(),
                to: helper.identity_digest.clone(),
                group: DependencyGroup::Runtime,
                optional: false,
                target: None,
            },
            LockedEdge {
                from: None,
                alias: "matrix".parse().unwrap(),
                to: matrix.identity_digest,
                group: DependencyGroup::Runtime,
                optional: false,
                target: None,
            },
        ],
    )
    .unwrap()
}

#[test]
fn canonical_toml_is_byte_stable_and_round_trips() {
    let lock = fixture_lock();
    let encoded = encode_lock(&lock).unwrap();
    assert_eq!(encoded, include_str!("fixtures/runmat.lock"));
    assert_eq!(decode_lock(&encoded).unwrap(), lock);

    let mut reordered = lock.clone();
    reordered.packages.reverse();
    reordered.edges.reverse();
    assert!(encode_lock(&reordered).is_ok());
    assert_eq!(
        encode_lock(&reordered).unwrap(),
        include_str!("fixtures/runmat.lock")
    );
}

#[test]
fn validation_rejects_tampered_redundant_identity_and_graph_digest() {
    let mut lock = fixture_lock();
    lock.packages[0].instance.identity_digest = ContentDigest::sha256("forged identity");
    assert!(lock.validate().is_err());

    let mut lock = fixture_lock();
    lock.graph_digest = ContentDigest::sha256("forged graph");
    assert!(lock.validate().is_err());
}

#[test]
fn compatibility_and_diff_are_semantic() {
    let lock = fixture_lock();
    LockCompatibility {
        runmat_version: Version::new(0, 6, 2),
    }
    .validate(&lock)
    .unwrap();
    assert!(LockCompatibility {
        runmat_version: Version::new(1, 0, 0),
    }
    .validate(&lock)
    .is_err());

    assert!(diff_locks(&lock, &lock).is_empty());
    let mut changed = lock.clone();
    changed.packages.pop();
    assert_eq!(diff_locks(&lock, &changed).removed.len(), 1);
}

#[test]
fn lock_construction_rejects_secret_bearing_sources_and_unavailable_capabilities() {
    let tree_digest = ContentDigest::sha256("server tree");
    let forged = PackageInstanceId::new(
        "default:runmat/private".parse().unwrap(),
        SourceId::ServerProject(ServerProjectSourceId {
            service: "https://token@example.com".to_string(),
            project: "project_1".to_string(),
            snapshot: "snapshot_1".to_string(),
            tree_digest: tree_digest.clone(),
        }),
        None,
        tree_digest,
    );
    let selection = LockSelection {
        target: "wasm32-unknown-unknown".to_string(),
        groups: set([DependencyGroup::Runtime]),
        root_features: BTreeSet::new(),
        host_capabilities: set([HostCapability::Worker]),
    };
    let root = RootLock {
        manifest_digest: ContentDigest::sha256("root"),
        package: None,
    };
    let result = PackageLock::new(
        root.clone(),
        selection.clone(),
        vec![LockedPackage {
            instance: forged,
            features: BTreeSet::new(),
            required_capabilities: BTreeSet::new(),
            runmat_version: None,
            singleton: false,
        }],
        Vec::new(),
    );
    assert!(result.is_err());

    let result = PackageLock::new(
        root,
        selection,
        vec![LockedPackage {
            instance: instance("default:runmat/native", "deps/native", "native tree"),
            features: BTreeSet::new(),
            required_capabilities: set([HostCapability::NativeLibrary]),
            runmat_version: None,
            singleton: false,
        }],
        Vec::new(),
    );
    assert!(result.is_err());
}
