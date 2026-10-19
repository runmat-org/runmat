//! A memory stage is private bytes until one locked map replacement.

use super::*;
use crate::{FsPublicationFailure, FsStagedOutput};
use std::sync::atomic::AtomicU64;

pub(super) fn begin(
    provider: MemoryFsProvider,
    destination: &Path,
) -> io::Result<Box<dyn FsStagedOutput>> {
    if !crate::FsPathNamespace::VirtualRooted.is_absolute(destination) {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "staged replacement requires an absolute destination",
        ));
    }
    let destination = normalize_path(destination)?;
    Ok(Box::new(MemoryStagedOutput {
        provider,
        destination,
        bytes: Vec::new(),
    }))
}

struct MemoryStagedOutput {
    provider: MemoryFsProvider,
    destination: PathBuf,
    bytes: Vec<u8>,
}

#[async_trait(?Send)]
impl FsStagedOutput for MemoryStagedOutput {
    async fn write_chunk(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.bytes
            .try_reserve(bytes.len())
            .map_err(io::Error::other)?;
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    async fn flush_and_sync(&mut self) -> io::Result<()> {
        Ok(())
    }

    async fn publish(mut self: Box<Self>) -> Result<(), FsPublicationFailure> {
        let mut tree = self.provider.inner.lock().map_err(|_| {
            FsPublicationFailure::not_published(io::Error::other("memory filesystem lock poisoned"))
        })?;
        ensure_parent_dir(&tree, &self.destination).map_err(FsPublicationFailure::not_published)?;
        match tree.entries.get(&self.destination) {
            Some(MemoryEntry::Directory) => {
                return Err(FsPublicationFailure::not_published(io::Error::new(
                    ErrorKind::InvalidInput,
                    "cannot replace a directory with staged file bytes",
                )));
            }
            Some(MemoryEntry::File { readonly: true, .. }) => {
                return Err(FsPublicationFailure::not_published(io::Error::new(
                    ErrorKind::PermissionDenied,
                    "destination file is readonly",
                )));
            }
            Some(MemoryEntry::File { .. }) | None => {}
        }
        tree.entries.insert(
            self.destination.clone(),
            MemoryEntry::File {
                bytes: Arc::new(std::mem::take(&mut self.bytes)),
                readonly: false,
                revision: uuid::Uuid::new_v4().to_string(),
                identity: Arc::new(AtomicU64::new(0)),
            },
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dropped_stage_does_not_touch_existing_or_neighbor_bytes() {
        let provider = MemoryFsProvider::with_current_dir("/reports");
        provider
            .write_project_path("/reports/data.csv", b"original")
            .expect("destination");
        provider
            .write_project_path("/reports/neighbor.csv", b"neighbor")
            .expect("neighbor");
        let mut stage = begin(provider.clone(), Path::new("/reports/data.csv")).unwrap();
        block_on(stage.write_chunk(b"replacement")).expect("stage bytes");
        drop(stage);
        assert_eq!(
            provider.read_project_path("/reports/data.csv").unwrap(),
            b"original"
        );
        assert_eq!(
            provider.read_project_path("/reports/neighbor.csv").unwrap(),
            b"neighbor"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn publication_is_one_memory_tree_replacement() {
        let provider = MemoryFsProvider::with_current_dir("/reports");
        provider
            .write_project_path("/reports/data.csv", b"original")
            .expect("destination");
        let mut stage = begin(provider.clone(), Path::new("/reports/data.csv")).unwrap();
        assert_eq!(block_on(stage.write_chunk(b"new ")).unwrap(), 4);
        assert_eq!(block_on(stage.write_chunk(b"bytes")).unwrap(), 5);
        assert_eq!(
            provider.read_project_path("/reports/data.csv").unwrap(),
            b"original"
        );
        block_on(stage.flush_and_sync()).unwrap();
        block_on(stage.publish()).unwrap();
        assert_eq!(
            provider.read_project_path("/reports/data.csv").unwrap(),
            b"new bytes"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn failed_publication_leaves_readonly_target_unchanged() {
        let provider = MemoryFsProvider::with_current_dir("/reports");
        provider
            .write_project_path("/reports/data.csv", b"original")
            .expect("destination");
        block_on(provider.set_readonly(Path::new("/reports/data.csv"), true)).unwrap();
        let mut stage = begin(provider.clone(), Path::new("/reports/data.csv")).unwrap();
        block_on(stage.write_chunk(b"replacement")).unwrap();
        let failure = block_on(stage.publish()).unwrap_err();
        assert_eq!(
            failure.disposition(),
            crate::FsPublicationDisposition::NotPublished
        );
        assert_eq!(failure.io_error().kind(), ErrorKind::PermissionDenied);
        assert_eq!(
            provider.read_project_path("/reports/data.csv").unwrap(),
            b"original"
        );
    }
}
