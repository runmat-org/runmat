use super::*;
use std::io::Read as _;

pub(super) async fn assert_native_cache_access(pinned: &PinnedFsProvider) {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("cache");
    std::fs::create_dir(&root).unwrap();
    let source = root.join("helper.m");
    std::fs::write(&source, b"immutable source").unwrap();
    std::fs::write(
        root.join("manifest.json"),
        serde_json::to_vec(&descriptor()).unwrap(),
    )
    .unwrap();
    let native: Arc<dyn FsProvider> = Arc::new(crate::NativeFsProvider);
    assert!(pinned
        .mount_readonly_dependency(root.ancestors().last().unwrap(), native.clone())
        .is_err());
    pinned.mount_readonly_dependency(&root, native).unwrap();
    assert_eq!(pinned.read(&source).await.unwrap(), b"immutable source");
    assert!(pinned.metadata(&source).await.unwrap().is_readonly());
    assert_eq!(pinned.canonicalize(&root).await.unwrap(), root);
    assert_eq!(pinned.read_dir(&root).await.unwrap().len(), 2);
    let mut file = pinned
        .open_async(
            &source,
            &OpenFlags {
                read: true,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).unwrap();
    assert_eq!(bytes, b"immutable source");
    let request = crate::DataManifestRequest {
        path: root.to_string_lossy().into_owned(),
        version: None,
    };
    assert_eq!(
        pinned.data_manifest_descriptor(&request).await.unwrap(),
        descriptor()
    );
    assert!(pinned
        .data_manifest_descriptor(&crate::DataManifestRequest {
            version: Some("uncaptured-version".into()),
            ..request
        })
        .await
        .is_err());
    assert!(pinned.write(&source, b"overwrite").await.is_err());
    assert!(pinned.read(&root.join("../outside.m")).await.is_err());
    let mut bad = manifest("v1");
    bad.entries[0].path = source.to_string_lossy().into_owned();
    assert!(PinnedFsProvider::new(Arc::new(MemoryFsProvider::new()), bad).is_err());
}
