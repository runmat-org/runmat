use super::*;
use crate::{
    AtomicCreateResult, DirEntry, FileHandle, FsMetadata, FsProvider, MemoryFsProvider, OpenFlags,
};
use async_trait::async_trait;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

type OutputVersions = (BTreeMap<PathBuf, String>, BTreeMap<String, Vec<u8>>);

#[cfg(windows)]
#[path = "tests/native_mounts.rs"]
mod native_mounts;

struct Versions {
    live: MemoryFsProvider,
    outputs: Mutex<OutputVersions>,
}

#[tokio::test]
async fn output_versions_survive_peer_edits_and_reject_stale_writes_and_deletes() {
    let base = Arc::new(Versions {
        live: MemoryFsProvider::new(),
        outputs: Mutex::default(),
    });
    let pinned = PinnedFsProvider::new(base.clone(), manifest("v1")).unwrap();
    let path = Path::new("/result.txt");
    pinned.write(path, b"first").await.unwrap();
    pinned.write(path, b"second").await.unwrap();
    base.write(path, b"teammate").await.unwrap();
    assert_eq!(pinned.read(path).await.unwrap(), b"second");
    assert_eq!(pinned.metadata(path).await.unwrap().len(), 6);
    assert!(pinned.write(path, b"overwrite").await.is_err());
    assert!(pinned.remove_file(path).await.is_err());
    assert_eq!(base.read(path).await.unwrap(), b"teammate");
}

#[tokio::test]
async fn buffered_output_handles_use_exact_versions_for_every_flush() {
    use std::io::{Read, Seek, SeekFrom, Write};
    let base = Arc::new(Versions {
        live: MemoryFsProvider::new(),
        outputs: Mutex::default(),
    });
    let pinned = PinnedFsProvider::new(base.clone(), manifest("v1")).unwrap();
    let path = Path::new("/result.txt");
    let flags = OpenFlags {
        read: true,
        write: true,
        create: true,
        ..Default::default()
    };
    let mut a = pinned.open_async(path, &flags).await.unwrap();
    a.write_all(b"first").unwrap();
    a.flush_async().await.unwrap();
    let mut b = pinned.open_async(path, &flags).await.unwrap();
    a.seek(SeekFrom::End(0)).unwrap();
    a.write_all(b" next").unwrap();
    a.flush_async().await.unwrap();
    b.write_all(b"stale").unwrap();
    assert!(b.flush_async().await.is_err());
    assert_eq!(pinned.read(path).await.unwrap(), b"first next");
    a.seek(SeekFrom::Start(0)).unwrap();
    let mut bytes = vec![];
    a.read_to_end(&mut bytes).unwrap();
    assert_eq!(bytes, b"first next");
    base.write(path, b"peer").await.unwrap();
    a.write_all(b" more").unwrap();
    assert!(a.flush_async().await.is_err());
    assert_eq!(base.read(path).await.unwrap(), b"peer");
}

#[tokio::test]
async fn owned_outputs_can_be_removed_and_recreated_but_cannot_steal_new_paths() {
    let base = Arc::new(Versions {
        live: MemoryFsProvider::new(),
        outputs: Mutex::default(),
    });
    let pinned = PinnedFsProvider::new(base.clone(), manifest("v1")).unwrap();
    let path = Path::new("/result.txt");
    pinned.write(path, b"first").await.unwrap();
    pinned.remove_file(path).await.unwrap();
    assert!(pinned.read(path).await.is_err());
    pinned.write(path, b"recreated").await.unwrap();
    pinned.remove_file(path).await.unwrap();
    base.write(path, b"peer").await.unwrap();
    assert!(pinned.write(path, b"overwrite").await.is_err());
    assert_eq!(base.read(path).await.unwrap(), b"peer");
}
#[async_trait(?Send)]
impl FsProvider for Versions {
    fn open(&self, path: &Path, flags: &OpenFlags) -> io::Result<Box<dyn FileHandle>> {
        self.live.open(path, flags)
    }
    async fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
        {
            let outputs = self.outputs.lock().unwrap();
            if let Some(version) = outputs.0.get(path) {
                return Ok(outputs.1[version].clone());
            }
        }
        self.live.read(path).await
    }
    async fn read_version(
        &self,
        path: &Path,
        version: &str,
        _: u64,
        _: &str,
    ) -> io::Result<Vec<u8>> {
        if let Some(bytes) = self.outputs.lock().unwrap().1.get(version).cloned() {
            return Ok(bytes);
        }
        if path == Path::new("/data/manifest.json") && version == "dataset-v1" {
            return Ok(serde_json::to_vec(&descriptor()).unwrap());
        }
        assert_eq!(path, Path::new("/source.m"));
        if version == "v1" {
            Ok(b"original".to_vec())
        } else {
            Err(io::ErrorKind::NotFound.into())
        }
    }
    async fn write(&self, path: &Path, data: &[u8]) -> io::Result<()> {
        self.live.write(path, data).await?;
        let mut outputs = self.outputs.lock().unwrap();
        let version = uuid::Uuid::new_v4().to_string();
        outputs.0.insert(path.to_owned(), version.clone());
        outputs.1.insert(version, data.to_vec());
        Ok(())
    }
    async fn publish_output(
        &self,
        path: &Path,
        data: &[u8],
        expected: Option<&str>,
    ) -> io::Result<String> {
        let mut outputs = self.outputs.lock().unwrap();
        if outputs.0.get(path).map(String::as_str) != expected {
            return Err(io::ErrorKind::PermissionDenied.into());
        }
        let version = uuid::Uuid::new_v4().to_string();
        outputs.0.insert(path.to_owned(), version.clone());
        outputs.1.insert(version.clone(), data.to_vec());
        Ok(version)
    }
    async fn remove_output(&self, path: &Path, expected: &str) -> io::Result<()> {
        let mut outputs = self.outputs.lock().unwrap();
        if outputs.0.get(path).map(String::as_str) != Some(expected) {
            return Err(io::ErrorKind::PermissionDenied.into());
        }
        outputs.0.remove(path);
        Ok(())
    }
    async fn atomic_create_new(&self, path: &Path, data: &[u8]) -> io::Result<AtomicCreateResult> {
        self.live.atomic_create_new(path, data).await
    }
    async fn remove_file(&self, path: &Path) -> io::Result<()> {
        self.live.remove_file(path).await
    }
    async fn metadata(&self, path: &Path) -> io::Result<FsMetadata> {
        self.live.metadata(path).await
    }
    async fn symlink_metadata(&self, path: &Path) -> io::Result<FsMetadata> {
        self.live.symlink_metadata(path).await
    }
    async fn read_dir(&self, path: &Path) -> io::Result<Vec<DirEntry>> {
        self.live.read_dir(path).await
    }
    async fn canonicalize(&self, path: &Path) -> io::Result<PathBuf> {
        self.live.canonicalize(path).await
    }
    async fn create_dir(&self, path: &Path) -> io::Result<()> {
        self.live.create_dir(path).await
    }
    async fn create_dir_all(&self, path: &Path) -> io::Result<()> {
        self.live.create_dir_all(path).await
    }
    async fn remove_dir(&self, path: &Path) -> io::Result<()> {
        self.live.remove_dir(path).await
    }
    async fn remove_dir_all(&self, path: &Path) -> io::Result<()> {
        self.live.remove_dir_all(path).await
    }
    async fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        self.live.rename(from, to).await
    }
    async fn set_readonly(&self, path: &Path, readonly: bool) -> io::Result<()> {
        self.live.set_readonly(path, readonly).await
    }
}

fn descriptor() -> crate::DataManifestDescriptor {
    crate::DataManifestDescriptor {
        schema_version: 1,
        format: "runmat-data".into(),
        dataset_id: "captured-data".into(),
        updated_at: "2026-09-26T00:00:00Z".into(),
        txn_sequence: 3,
    }
}

#[tokio::test]
async fn dataset_metadata_uses_the_captured_version_and_never_the_live_descriptor() {
    let base = Arc::new(Versions {
        live: MemoryFsProvider::new(),
        outputs: Mutex::default(),
    });
    base.write(Path::new("/data/manifest.json"), b"new live metadata")
        .await
        .unwrap();
    let pinned = PinnedFsProvider::new(
        base,
        InputManifest {
            snapshot_id: "capture".into(),
            entries: vec![PinnedEntry {
                path: "/data/manifest.json".into(),
                content: PinnedContent::File {
                    version_id: "dataset-v1".into(),
                    size: serde_json::to_vec(&descriptor()).unwrap().len() as u64,
                },
            }],
        },
    )
    .unwrap();
    let request = crate::DataManifestRequest {
        path: "/data".into(),
        version: None,
    };
    assert_eq!(
        pinned.data_manifest_descriptor(&request).await.unwrap(),
        descriptor()
    );
    assert!(pinned
        .data_manifest_descriptor(&crate::DataManifestRequest {
            version: Some("newer-version".into()),
            ..request
        })
        .await
        .is_err());
    assert!(pinned
        .data_manifest_descriptor(&crate::DataManifestRequest {
            path: "/unknown".into(),
            version: None
        })
        .await
        .is_err());
}
fn manifest(version: &str) -> InputManifest {
    InputManifest {
        snapshot_id: "capture".into(),
        entries: vec![PinnedEntry {
            path: "/source.m".into(),
            content: PinnedContent::File {
                version_id: version.into(),
                size: 8,
            },
        }],
    }
}

#[tokio::test]
async fn pins_names_and_bytes_while_other_users_edit_and_create_files() {
    let base = Arc::new(Versions {
        live: MemoryFsProvider::new(),
        outputs: Mutex::default(),
    });
    base.write(Path::new("/source.m"), b"new live bytes")
        .await
        .unwrap();
    base.write(Path::new("/later.m"), b"later").await.unwrap();
    let pinned = PinnedFsProvider::new(base.clone(), manifest("v1")).unwrap();
    assert_eq!(
        pinned.read(Path::new("/source.m")).await.unwrap(),
        b"original"
    );
    assert_eq!(pinned.read_dir(Path::new("/")).await.unwrap().len(), 1);
    assert!(pinned.read(Path::new("/later.m")).await.is_err());
    assert!(pinned
        .write(Path::new("/source.m"), b"overwrite")
        .await
        .is_err());
    assert!(pinned
        .write(Path::new("/later.m"), b"overwrite")
        .await
        .is_err());
    pinned
        .write(Path::new("/result.txt"), b"result")
        .await
        .unwrap();
    assert_eq!(
        pinned.read(Path::new("/result.txt")).await.unwrap(),
        b"result"
    );
    assert_eq!(
        base.read(Path::new("/source.m")).await.unwrap(),
        b"new live bytes"
    );
}
#[tokio::test]
async fn unavailable_version_never_falls_back_to_live_content() {
    let base = Arc::new(Versions {
        live: MemoryFsProvider::new(),
        outputs: Mutex::default(),
    });
    base.write(Path::new("/source.m"), b"new live bytes")
        .await
        .unwrap();
    let pinned = PinnedFsProvider::new(base, manifest("deleted")).unwrap();
    assert_eq!(
        pinned
            .read(Path::new("/source.m"))
            .await
            .unwrap_err()
            .kind(),
        io::ErrorKind::NotFound
    );
}
#[test]
fn rejects_ambiguous_paths_and_missing_versions() {
    let base = Arc::new(Versions {
        live: MemoryFsProvider::new(),
        outputs: Mutex::default(),
    });
    let mut bad = manifest("v1");
    bad.entries.push(PinnedEntry {
        path: "source.m".into(),
        ..bad.entries[0].clone()
    });
    assert!(PinnedFsProvider::new(base.clone(), bad).is_err());
    let mut bad = manifest("v1");
    bad.entries[0].path = "../../outside".into();
    assert!(PinnedFsProvider::new(base.clone(), bad).is_err());
    let mut bad = manifest("v1");
    bad.entries[0].content = PinnedContent::File {
        version_id: String::new(),
        size: 8,
    };
    assert!(PinnedFsProvider::new(base, bad).is_err());
}

#[test]
fn wire_contract_requires_file_version_and_rejects_directory_metadata() {
    for invalid in [
        r#"{"path":"/x","content":{"kind":"file","size":8}}"#,
        r#"{"path":"/x","content":{"kind":"directory","versionId":"v1"}}"#,
        r#"{"path":"/x","content":{"kind":"directory","size":8}}"#,
    ] {
        assert!(serde_json::from_str::<PinnedEntry>(invalid).is_err());
    }
    let value = serde_json::to_value(&manifest("v1").entries[0]).unwrap();
    assert_eq!(
        value["content"],
        serde_json::json!({
            "kind": "file", "versionId": "v1", "size": 8,
        })
    );
    let decoded: PinnedEntry = serde_json::from_value(value).unwrap();
    assert!(matches!(
        decoded.content,
        PinnedContent::File { size: 8, .. }
    ));
}

#[tokio::test]
async fn dependency_mounts_are_readonly_and_cannot_shadow_workspace_authority() {
    let base = Arc::new(MemoryFsProvider::new());
    let pinned = PinnedFsProvider::new(base.clone(), manifest("v1")).unwrap();
    let cache = Arc::new(MemoryFsProvider::new());
    cache
        .create_dir_all(Path::new("/packages/lib"))
        .await
        .unwrap();
    cache
        .write(Path::new("/packages/lib/function.m"), b"immutable package")
        .await
        .unwrap();
    pinned
        .mount_readonly_dependency(Path::new("/packages/lib"), cache.clone())
        .unwrap();
    assert_eq!(
        pinned
            .read(Path::new("/packages/lib/function.m"))
            .await
            .unwrap(),
        b"immutable package"
    );
    assert_eq!(
        pinned
            .read_dir(Path::new("/packages/lib"))
            .await
            .unwrap()
            .len(),
        1
    );
    assert!(pinned
        .metadata(Path::new("/packages/lib/function.m"))
        .await
        .unwrap()
        .is_readonly());
    assert!(pinned
        .write(Path::new("/packages/lib/function.m"), b"overwrite")
        .await
        .is_err());
    assert!(pinned
        .atomic_create_new(Path::new("/packages/lib/new.m"), b"new")
        .await
        .is_err());
    assert!(pinned
        .atomic_create_new(Path::new("/packages"), b"shadow directory")
        .await
        .is_err());
    assert!(pinned
        .remove_file(Path::new("/packages/lib/function.m"))
        .await
        .is_err());
    assert!(pinned
        .mount_readonly_dependency(Path::new("/packages"), Arc::new(MemoryFsProvider::new()))
        .is_err());
    pinned
        .mount_readonly_dependency(Path::new("/packages/lib/nested"), cache.clone())
        .unwrap();
    assert!(pinned
        .mount_readonly_dependency(Path::new("/"), cache.clone())
        .is_err());
    for entry in manifest("v1").entries {
        assert!(pinned
            .mount_readonly_dependency(Path::new(&entry.path), cache.clone())
            .is_err());
    }
    pinned.write(Path::new("/owned"), b"output").await.unwrap();
    assert!(pinned
        .mount_readonly_dependency(Path::new("/owned"), cache)
        .is_err());
    assert_eq!(base.read(Path::new("/owned")).await.unwrap(), b"output");
    #[cfg(windows)]
    native_mounts::assert_native_cache_access(&pinned).await;
}
