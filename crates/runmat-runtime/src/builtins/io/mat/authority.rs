//! Invocation-owned MAT filesystem authority. A selected path and provider
//! stay together across asynchronous opens, reads, and publication.

use std::io;
use std::path::PathBuf;
use std::sync::Arc;

use runmat_filesystem::{self as fs, CapturedFsResourceLease, FsProvider};

mod handle;
pub(crate) use handle::{issue_matfile, resolve_matfile};

pub(crate) struct MatFsPath {
    provider: Arc<dyn FsProvider>,
    path: PathBuf,
    resource: Option<CapturedFsResourceLease>,
}

impl MatFsPath {
    pub(crate) fn capture(path: PathBuf) -> io::Result<Self> {
        let (provider, path) = fs::capture_provider_path(&path)?;
        let resource = fs::capture_resource_lease(provider.clone());
        Ok(Self {
            provider,
            path,
            resource,
        })
    }

    pub(crate) fn path(&self) -> &std::path::Path {
        &self.path
    }

    pub(crate) fn provider(&self) -> &Arc<dyn FsProvider> {
        &self.provider
    }

    pub(crate) fn check_current(&self) -> io::Result<()> {
        if self
            .resource
            .as_ref()
            .is_some_and(|lease| !lease.is_valid())
        {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "captured MAT filesystem authority was revoked",
            ));
        }
        Ok(())
    }

    pub(crate) async fn open(&self) -> io::Result<fs::File> {
        self.check_current()?;
        let file = fs::File::open_async_on(self.provider.as_ref(), &self.path).await?;
        self.check_current()?;
        Ok(file)
    }

    pub(crate) async fn metadata(&self) -> io::Result<fs::FsMetadata> {
        self.check_current()?;
        let metadata = self.provider.metadata(&self.path).await?;
        self.check_current()?;
        Ok(metadata)
    }

    pub(crate) async fn write(&self, bytes: &[u8]) -> io::Result<()> {
        self.check_current()?;
        self.provider.write(&self.path, bytes).await?;
        self.check_current()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    #[cfg(not(windows))]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rooted_mat_path_keeps_exact_selected_provider_across_swap() {
        let _state = crate::builtins::io::mat::tests::state_guard();
        let selected = Arc::new(fs::MemoryFsProvider::with_current_dir("/selected"));
        let replacement = Arc::new(fs::MemoryFsProvider::with_current_dir("/replacement"));
        selected
            .write_project_path("/project/rooted.mat", b"selected")
            .expect("selected source");
        replacement
            .write_project_path("/project/rooted.mat", b"replacement")
            .expect("replacement source");
        let _selected = fs::replace_provider(selected);
        fs::set_current_dir("/selected").expect("selected cwd");
        let captured =
            MatFsPath::capture(PathBuf::from("/project/rooted.mat")).expect("rooted capture");
        assert_eq!(captured.path(), std::path::Path::new("/project/rooted.mat"));
        let _replacement = fs::replace_provider(replacement);
        fs::set_current_dir("/replacement").expect("replacement cwd");
        let mut opened = block_on(captured.open()).expect("original owner open");
        let mut bytes = Vec::new();
        use std::io::Read;
        opened.read_to_end(&mut bytes).expect("selected bytes");
        assert_eq!(bytes, b"selected");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn selected_provider_and_relative_path_survive_ambient_swap() {
        let _state = crate::builtins::io::mat::tests::state_guard();
        let selected = Arc::new(fs::MemoryFsProvider::with_current_dir("/selected"));
        let replacement = Arc::new(fs::MemoryFsProvider::with_current_dir("/replacement"));
        selected
            .write_project_path("/selected/fixture.mat", b"selected")
            .expect("selected fixture");
        replacement
            .write_project_path("/replacement/fixture.mat", b"replacement")
            .expect("replacement fixture");
        let _selected = fs::replace_provider(selected.clone());
        fs::set_current_dir("/selected").expect("selected cwd");
        let captured = MatFsPath::capture(PathBuf::from("fixture.mat")).expect("capture");
        let _replacement = fs::replace_provider(replacement.clone());
        fs::set_current_dir("/replacement").expect("replacement cwd");
        assert_eq!(
            captured.path(),
            std::path::Path::new("/selected/fixture.mat")
        );
        let mut opened = block_on(captured.open()).expect("original owner open");
        let mut bytes = Vec::new();
        use std::io::Read;
        opened.read_to_end(&mut bytes).expect("read");
        assert_eq!(bytes, b"selected");
        block_on(captured.write(b"published")).expect("original owner write");
        assert_eq!(
            block_on(selected.read(captured.path())).expect("selected"),
            b"published"
        );
        assert_eq!(
            block_on(replacement.read(std::path::Path::new("/replacement/fixture.mat")))
                .expect("replacement"),
            b"replacement"
        );
    }
}
