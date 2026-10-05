//! Public status forms must not imply rollback after uncertain publication.

use super::*;
use async_trait::async_trait;
use runmat_filesystem::{
    DirEntry, FileHandle, FsMetadata, FsProvider, FsPublicationDisposition, FsPublicationFailure,
    FsStagedOutput, MemoryFsProvider, OpenFlags,
};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

struct PublicationProvider {
    memory: MemoryFsProvider,
    disposition: FsPublicationDisposition,
}

struct PublicationStage {
    memory: MemoryFsProvider,
    destination: PathBuf,
    disposition: FsPublicationDisposition,
    bytes: Vec<u8>,
}

#[async_trait(?Send)]
impl FsStagedOutput for PublicationStage {
    async fn write_chunk(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    async fn flush_and_sync(&mut self) -> io::Result<()> {
        Ok(())
    }

    async fn publish(self: Box<Self>) -> Result<(), FsPublicationFailure> {
        if self.disposition != FsPublicationDisposition::NotPublished {
            self.memory
                .write_project_path(&self.destination, &self.bytes)
                .map_err(FsPublicationFailure::not_published)?;
        }
        Err(FsPublicationFailure::new(
            self.disposition,
            io::Error::other("injected publication failure"),
        ))
    }
}

#[async_trait(?Send)]
impl FsProvider for PublicationProvider {
    fn current_dir_override(&self) -> Option<PathBuf> {
        self.memory.current_dir_override()
    }

    fn begin_staged_replace(
        &self,
        destination: &Path,
    ) -> io::Result<Option<Box<dyn FsStagedOutput>>> {
        Ok(Some(Box::new(PublicationStage {
            memory: self.memory.clone(),
            destination: destination.to_path_buf(),
            disposition: self.disposition,
            bytes: Vec::new(),
        })))
    }

    fn open(&self, path: &Path, flags: &OpenFlags) -> io::Result<Box<dyn FileHandle>> {
        self.memory.open(path, flags)
    }
    async fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
        self.memory.read(path).await
    }
    async fn write(&self, path: &Path, data: &[u8]) -> io::Result<()> {
        self.memory.write(path, data).await
    }
    async fn remove_file(&self, path: &Path) -> io::Result<()> {
        self.memory.remove_file(path).await
    }
    async fn metadata(&self, path: &Path) -> io::Result<FsMetadata> {
        self.memory.metadata(path).await
    }
    async fn symlink_metadata(&self, path: &Path) -> io::Result<FsMetadata> {
        self.memory.symlink_metadata(path).await
    }
    async fn read_dir(&self, path: &Path) -> io::Result<Vec<DirEntry>> {
        self.memory.read_dir(path).await
    }
    async fn canonicalize(&self, path: &Path) -> io::Result<PathBuf> {
        self.memory.canonicalize(path).await
    }
    async fn create_dir(&self, path: &Path) -> io::Result<()> {
        self.memory.create_dir(path).await
    }
    async fn create_dir_all(&self, path: &Path) -> io::Result<()> {
        self.memory.create_dir_all(path).await
    }
    async fn remove_dir(&self, path: &Path) -> io::Result<()> {
        self.memory.remove_dir(path).await
    }
    async fn remove_dir_all(&self, path: &Path) -> io::Result<()> {
        self.memory.remove_dir_all(path).await
    }
    async fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        self.memory.rename(from, to).await
    }
    async fn set_readonly(&self, path: &Path, readonly: bool) -> io::Result<()> {
        self.memory.set_readonly(path, readonly).await
    }
}

fn fixture_root() -> &'static str {
    if cfg!(windows) {
        r"C:\publication"
    } else {
        "/publication"
    }
}

fn public_call(
    disposition: FsPublicationDisposition,
    output_count: usize,
) -> (BuiltinResult<Value>, MemoryFsProvider) {
    let memory = MemoryFsProvider::with_current_dir(fixture_root());
    block_on(memory.create_dir_all(Path::new(fixture_root()))).expect("fixture directory");
    let provider = Arc::new(PublicationProvider {
        memory: memory.clone(),
        disposition,
    });
    let _restore = runmat_filesystem::replace_provider(provider);
    runmat_filesystem::set_current_dir(fixture_root()).expect("fixture cwd");
    let _outputs = crate::output_count::push_output_count(Some(output_count));
    let result = block_on(xlswrite_call(
        Value::from("out.xlsx"),
        Value::Num(7.0),
        Vec::new(),
    ));
    (result, memory)
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn public_status_forms_only_capture_known_unpublished_failure() {
    let _lock = runmat_filesystem::provider_override_lock();
    for outputs in [1, 2] {
        let (result, memory) = public_call(FsPublicationDisposition::NotPublished, outputs);
        let values = output_list(result.expect("known unpublished failure becomes status"));
        assert_eq!(values[0], Value::Bool(false));
        assert_eq!(values.len(), outputs);
        if outputs == 2 {
            let Value::Struct(message) = &values[1] else {
                panic!("message struct")
            };
            assert!(
                matches!(message.fields.get("identifier"), Some(Value::String(id)) if id == "RunMat:xlswrite:Io")
            );
        }
        assert!(memory
            .read_project_path(Path::new(fixture_root()).join("out.xlsx"))
            .is_err());
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn public_status_forms_propagate_published_and_indeterminate_failures() {
    let _lock = runmat_filesystem::provider_override_lock();
    for disposition in [
        FsPublicationDisposition::Published,
        FsPublicationDisposition::Indeterminate,
    ] {
        for outputs in [1, 2] {
            let (result, memory) = public_call(disposition, outputs);
            let error = result.expect_err("uncertain or committed publication must throw");
            assert_eq!(
                write_owner::publication_error_disposition(&error),
                Some(disposition)
            );
            assert!(memory
                .read_project_path(Path::new(fixture_root()).join("out.xlsx"))
                .is_ok());
        }
    }
}
