use futures::executor::block_on;
use runmat_package::{ContentDigest, NormalizedRelativePath};
use runmat_package_cache::{
    BlobMetadata, CacheBackend, CacheObject, CacheTransaction, ObjectWrite, TreeEntry, TreeManifest,
};
use runmat_package_cache_native::filesystem::CacheLayout;
use runmat_package_cache_native::materialize::{materialize_tree, verify_materialized_tree};
use runmat_package_cache_native::SqliteCacheBackend;

fn path(value: &str) -> NormalizedRelativePath {
    NormalizedRelativePath::new(value).unwrap()
}

async fn seed(
    backend: &SqliteCacheBackend,
    bytes: &[u8],
    entries: impl FnOnce(ContentDigest) -> Vec<TreeEntry>,
) -> TreeManifest {
    let blob = BlobMetadata::from_bytes(bytes);
    let tree = TreeManifest::new(entries(blob.digest.clone())).unwrap();
    let mut state = backend.snapshot().await.unwrap().state;
    state
        .objects
        .insert(blob.digest.clone(), CacheObject::Blob(blob.clone()));
    state
        .objects
        .insert(tree.digest.clone(), CacheObject::Tree(tree.clone()));
    let mut transaction = CacheTransaction::metadata_only(0, state);
    transaction.writes.insert(
        blob.digest.clone(),
        ObjectWrite::new(CacheObject::Blob(blob), Some(bytes.to_vec())).unwrap(),
    );
    backend.commit(transaction).await.unwrap();
    tree
}

#[test]
fn verified_tree_is_atomically_materialized_and_reused() {
    block_on(async {
        let directory = tempfile::tempdir().unwrap();
        let layout = CacheLayout::new(directory.path().join("cache"));
        layout.create().unwrap();
        let backend = SqliteCacheBackend::open_path(&layout.database, None).unwrap();
        let tree = seed(&backend, b"disp('ok')", |digest| {
            vec![
                TreeEntry::directory(path("src")),
                TreeEntry::file(path("src/main.m"), digest, 10, false),
            ]
        })
        .await;

        let mounted = materialize_tree(&backend, &layout, &tree).await.unwrap();
        assert_eq!(
            std::fs::read(mounted.join("src/main.m")).unwrap(),
            b"disp('ok')"
        );
        assert!(std::fs::metadata(mounted.join("src/main.m"))
            .unwrap()
            .permissions()
            .readonly());
        assert_eq!(
            materialize_tree(&backend, &layout, &tree).await.unwrap(),
            mounted
        );
        verify_materialized_tree(&mounted, &tree).unwrap();
    });
}

#[test]
fn existing_tree_is_verified_before_reuse() {
    block_on(async {
        let directory = tempfile::tempdir().unwrap();
        let layout = CacheLayout::new(directory.path().join("cache"));
        layout.create().unwrap();
        let backend = SqliteCacheBackend::open_path(&layout.database, None).unwrap();
        let tree = seed(&backend, b"original", |digest| {
            vec![TreeEntry::file(path("main.m"), digest, 8, false)]
        })
        .await;
        let mounted = materialize_tree(&backend, &layout, &tree).await.unwrap();
        let file = mounted.join("main.m");
        make_test_file_writable(&file);
        std::fs::write(&file, b"tampered").unwrap();

        assert!(materialize_tree(&backend, &layout, &tree).await.is_err());
    });
}

#[cfg(unix)]
fn make_test_file_writable(path: &std::path::Path) {
    use std::os::unix::fs::PermissionsExt;
    let mut permissions = std::fs::metadata(path).unwrap().permissions();
    permissions.set_mode(permissions.mode() | 0o200);
    std::fs::set_permissions(path, permissions).unwrap();
}

#[cfg(windows)]
#[allow(clippy::permissions_set_readonly_false)] // Clears the Windows read-only attribute; it does not broaden ACLs.
fn make_test_file_writable(path: &std::path::Path) {
    let mut permissions = std::fs::metadata(path).unwrap().permissions();
    permissions.set_readonly(false);
    std::fs::set_permissions(path, permissions).unwrap();
}

#[cfg(unix)]
#[test]
fn internal_symlink_remains_inside_promoted_tree() {
    block_on(async {
        let directory = tempfile::tempdir().unwrap();
        let layout = CacheLayout::new(directory.path().join("cache"));
        layout.create().unwrap();
        let backend = SqliteCacheBackend::open_path(&layout.database, None).unwrap();
        let tree = seed(&backend, b"target", |digest| {
            vec![
                TreeEntry::directory(path("src")),
                TreeEntry::file(path("src/target.m"), digest, 6, false),
                TreeEntry::symlink(path("main.m"), path("src/target.m")),
            ]
        })
        .await;
        let mounted = materialize_tree(&backend, &layout, &tree).await.unwrap();
        assert_eq!(std::fs::read(mounted.join("main.m")).unwrap(), b"target");
        verify_materialized_tree(&mounted, &tree).unwrap();
    });
}
