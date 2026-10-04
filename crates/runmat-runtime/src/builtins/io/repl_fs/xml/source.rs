//! One invocation's XML path, provider, and optional provider resource lease.
use std::io;
use std::path::PathBuf;
use std::sync::Arc;

use runmat_filesystem::{self as fs, CapturedFsResourceLease, FsProvider};

pub(super) struct XmlFsPath {
    provider: Arc<dyn FsProvider>,
    path: PathBuf,
    lease: Option<CapturedFsResourceLease>,
}

impl XmlFsPath {
    pub(super) fn capture(path: PathBuf) -> io::Result<Self> {
        let (provider, path) = if path.is_absolute() {
            (fs::current_provider(), path)
        } else {
            let (provider, cwd) = fs::current_provider_with_current_dir()?;
            (provider, cwd.join(path))
        };
        let lease = fs::capture_resource_lease(provider.clone());
        Ok(Self {
            provider,
            path,
            lease,
        })
    }

    pub(super) fn check_current(&self) -> io::Result<()> {
        if self.lease.as_ref().is_some_and(|lease| !lease.is_valid()) {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "captured XML filesystem authority was revoked",
            ));
        }
        Ok(())
    }

    pub(super) async fn read_to_string(&self) -> io::Result<String> {
        self.check_current()?;
        let bytes = self.provider.read(&self.path).await?;
        self.check_current()?;
        let text = String::from_utf8(bytes)
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
        self.check_current()?;
        Ok(text)
    }

    pub(super) async fn write(&self, bytes: &[u8]) -> io::Result<()> {
        self.check_current()?;
        self.provider.write(&self.path, bytes).await?;
        self.check_current()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn captured_provider_and_relative_path_survive_ambient_swap() {
        let _lock = fs::provider_override_lock();
        let selected = Arc::new(fs::MemoryFsProvider::with_current_dir("/selected"));
        let replacement = Arc::new(fs::MemoryFsProvider::with_current_dir("/replacement"));
        selected
            .write_project_path("/selected/test.xml", b"<selected/>")
            .expect("selected fixture");
        replacement
            .write_project_path("/replacement/test.xml", b"<replacement/>")
            .expect("replacement fixture");
        let _selected = fs::replace_provider(selected.clone());
        fs::set_current_dir("/selected").expect("selected cwd");
        let captured = XmlFsPath::capture(PathBuf::from("test.xml")).expect("capture");
        let _replacement = fs::replace_provider(replacement.clone());
        fs::set_current_dir("/replacement").expect("replacement cwd");
        assert_eq!(
            futures::executor::block_on(captured.read_to_string()).unwrap(),
            "<selected/>"
        );
        futures::executor::block_on(captured.write(b"<changed/>")).expect("selected write");
        assert_eq!(
            futures::executor::block_on(selected.read(std::path::Path::new("/selected/test.xml")))
                .unwrap(),
            b"<changed/>"
        );
        assert_eq!(
            futures::executor::block_on(
                replacement.read(std::path::Path::new("/replacement/test.xml"))
            )
            .unwrap(),
            b"<replacement/>"
        );
    }
}
