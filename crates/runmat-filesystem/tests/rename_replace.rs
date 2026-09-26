#![cfg(not(target_arch = "wasm32"))]

use runmat_filesystem::{FsProvider, MemoryFsProvider, NativeFsProvider, SandboxFsProvider};
use std::path::Path;

async fn conformance(provider: &dyn FsProvider, root: &Path) {
    let source = root.join("source");
    let destination = root.join("destination");
    provider.write(&source, b"new").await.unwrap();
    provider.write(&destination, b"old").await.unwrap();
    provider
        .rename_replace(&source, &destination)
        .await
        .unwrap();
    assert_eq!(provider.read(&destination).await.unwrap(), b"new");
    assert!(provider.metadata(&source).await.is_err());
    provider
        .rename_replace(&destination, &destination)
        .await
        .unwrap();
    assert_eq!(provider.read(&destination).await.unwrap(), b"new");
    assert!(provider
        .rename_replace(&root.join("missing"), &destination)
        .await
        .is_err());
    assert_eq!(provider.read(&destination).await.unwrap(), b"new");
    assert!(provider
        .rename_replace(&destination, &root.join("missing-parent/file"))
        .await
        .is_err());
    assert_eq!(provider.read(&destination).await.unwrap(), b"new");
    assert!(provider
        .metadata(&root.join("missing-parent"))
        .await
        .is_err());

    let directory = root.join("directory");
    let occupied = root.join("occupied");
    provider.create_dir(&directory).await.unwrap();
    provider.create_dir(&occupied).await.unwrap();
    provider
        .write(&occupied.join("keep"), b"keep")
        .await
        .unwrap();
    assert!(provider
        .rename_replace(&directory, &occupied)
        .await
        .is_err());
    assert!(provider.metadata(&directory).await.unwrap().is_dir());
    assert_eq!(
        provider.read(&occupied.join("keep")).await.unwrap(),
        b"keep"
    );
    assert!(provider
        .rename_replace(&destination, &occupied)
        .await
        .is_err());
    assert_eq!(provider.read(&destination).await.unwrap(), b"new");
    assert!(provider
        .rename_replace(&directory, &destination)
        .await
        .is_err());
    assert_eq!(provider.read(&destination).await.unwrap(), b"new");
    let child = directory.join("child");
    provider.create_dir(&child).await.unwrap();
    assert!(provider
        .rename_replace(&directory, &child.join("target"))
        .await
        .is_err());
    assert!(provider.metadata(&child).await.unwrap().is_dir());
}

#[tokio::test]
async fn compatible_replacement_preserves_failed_targets_across_providers() {
    let native_root = tempfile::tempdir().unwrap();
    conformance(&NativeFsProvider, native_root.path()).await;
    let sandbox_root = tempfile::tempdir().unwrap();
    let sandbox = SandboxFsProvider::new(sandbox_root.path().to_owned()).unwrap();
    conformance(&sandbox, Path::new("/")).await;
    conformance(&MemoryFsProvider::new(), Path::new("/")).await;
}

#[tokio::test]
async fn memory_rename_preflights_self_readonly_and_directory_collisions() {
    let provider = MemoryFsProvider::new();
    provider
        .write(Path::new("/file"), b"original")
        .await
        .unwrap();
    provider
        .rename(Path::new("/file"), Path::new("/file"))
        .await
        .unwrap();
    assert_eq!(
        provider.read(Path::new("/file")).await.unwrap(),
        b"original"
    );
    provider
        .write(Path::new("/readonly"), b"keep")
        .await
        .unwrap();
    provider
        .set_readonly(Path::new("/readonly"), true)
        .await
        .unwrap();
    assert!(provider
        .rename_replace(Path::new("/file"), Path::new("/readonly"))
        .await
        .is_err());
    assert_eq!(
        provider.read(Path::new("/file")).await.unwrap(),
        b"original"
    );
    assert_eq!(
        provider.read(Path::new("/readonly")).await.unwrap(),
        b"keep"
    );
    provider.create_dir(Path::new("/empty")).await.unwrap();
    provider.create_dir(Path::new("/source-dir")).await.unwrap();
    provider
        .write(Path::new("/source-dir/child"), b"child")
        .await
        .unwrap();
    provider
        .rename_replace(Path::new("/source-dir"), Path::new("/empty"))
        .await
        .unwrap();
    assert_eq!(
        provider.read(Path::new("/empty/child")).await.unwrap(),
        b"child"
    );
    assert!(provider.metadata(Path::new("/source-dir")).await.is_err());
}
