//! Read immutable checkpoint inputs; all diagnostic writes use a separate store.
use runmat_execution::Digest;
use runmat_execution_artifact::{
    cache::{CacheExport, CacheImport},
    ArtifactResult, LogicalObject,
};
use runmat_execution_runner_native::FilesystemObjectStore;

pub struct ReplayStore {
    pub source: FilesystemObjectStore,
    pub output: FilesystemObjectStore,
}
impl ReplayStore {
    fn reader(&self, digest: Digest) -> &FilesystemObjectStore {
        let name = digest.to_string();
        if self
            .output
            .root()
            .join(name.trim_start_matches("sha256:"))
            .is_file()
        {
            &self.output
        } else {
            &self.source
        }
    }
}
impl CacheImport for ReplayStore {
    fn read_verified(&self, digest: Digest) -> ArtifactResult<Option<Vec<u8>>> {
        self.reader(digest).read_verified(digest)
    }
    fn read_verified_bounded(
        &self,
        digest: Digest,
        maximum: u64,
    ) -> ArtifactResult<Option<Vec<u8>>> {
        self.reader(digest).read_verified_bounded(digest, maximum)
    }
    fn read_verified_bounded_with_admission(
        &self,
        digest: Digest,
        maximum: u64,
        admit: &mut dyn FnMut(u64) -> ArtifactResult<()>,
    ) -> ArtifactResult<Option<Vec<u8>>> {
        self.reader(digest)
            .read_verified_bounded_with_admission(digest, maximum, admit)
    }
}
impl CacheExport for ReplayStore {
    fn write_verified(&mut self, object: &LogicalObject) -> ArtifactResult<()> {
        self.output.write_verified(object)
    }
}
