use std::path::PathBuf;

use runmat_test::identity::RunId;
use runmat_test_runner::artifact::{
    safe_artifact_name, ArtifactFuture, ArtifactStore, StoredArtifact,
};
use runmat_test_runner::RunnerError;
use sha2::{Digest, Sha256};

use super::atomic_write::write_atomic;
use super::paths::run_directory;

#[derive(Debug)]
pub struct FilesystemArtifactStore {
    root: PathBuf,
}

impl FilesystemArtifactStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
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
            write_atomic(&target, bytes).await.map_err(artifact_error)?;
            Ok(StoredArtifact {
                name: name.clone(),
                media_type: media_type.into(),
                byte_len: bytes.len() as u64,
                content_digest: format!("sha256:{}", hex::encode(Sha256::digest(bytes))),
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
