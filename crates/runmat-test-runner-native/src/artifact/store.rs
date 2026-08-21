use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use runmat_test::identity::RunId;
use runmat_test_runner::artifact::{
    safe_artifact_name, ArtifactFuture, ArtifactStore, StoredArtifact,
};
use runmat_test_runner::RunnerError;
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;

use super::paths::run_directory;

#[derive(Debug)]
pub struct FilesystemArtifactStore {
    root: PathBuf,
    nonce: AtomicU64,
}

impl FilesystemArtifactStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            nonce: AtomicU64::new(0),
        }
    }

    pub fn root(&self) -> &std::path::Path {
        &self.root
    }
}

impl ArtifactStore for FilesystemArtifactStore {
    fn put<'a>(
        &'a self,
        run_id: &'a RunId,
        name: &'a str,
        media_type: &'a str,
        bytes: &'a [u8],
    ) -> ArtifactFuture<'a, StoredArtifact> {
        Box::pin(async move {
            let name = safe_artifact_name(name)?;
            let directory = run_directory(&self.root, run_id);
            let target = directory.join(&name);
            let parent = target.parent().ok_or_else(|| {
                RunnerError::Artifact("artifact target has no parent directory".into())
            })?;
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(artifact_error)?;
            let nonce = self.nonce.fetch_add(1, Ordering::Relaxed);
            let temporary = parent.join(format!(".runmat-artifact-{nonce}.tmp"));
            let mut file = tokio::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temporary)
                .await
                .map_err(artifact_error)?;
            if let Err(error) = file.write_all(bytes).await {
                let _ = tokio::fs::remove_file(&temporary).await;
                return Err(artifact_error(error));
            }
            file.flush().await.map_err(artifact_error)?;
            drop(file);
            if let Err(error) = tokio::fs::rename(&temporary, &target).await {
                let _ = tokio::fs::remove_file(&temporary).await;
                return Err(artifact_error(error));
            }
            Ok(StoredArtifact {
                name: name.clone(),
                media_type: media_type.into(),
                byte_len: bytes.len() as u64,
                content_digest: format!("sha256:{:x}", Sha256::digest(bytes)),
                store_key: target
                    .strip_prefix(&self.root)
                    .unwrap_or(&target)
                    .to_string_lossy()
                    .replace('\\', "/"),
            })
        })
    }

    fn remove_run<'a>(&'a self, run_id: &'a RunId) -> ArtifactFuture<'a, ()> {
        Box::pin(async move {
            super::cleanup::remove_artifact_run(&self.root, run_id)
                .await
                .map_err(|error| RunnerError::Artifact(error.to_string()))
        })
    }
}

fn artifact_error(error: std::io::Error) -> RunnerError {
    RunnerError::Artifact(error.to_string())
}
