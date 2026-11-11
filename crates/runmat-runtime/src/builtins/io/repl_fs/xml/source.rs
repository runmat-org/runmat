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

/// A filename that must be materialized can still retain the provider and
/// working directory selected before that suspension.
pub(super) struct XmlPathOwner {
    provider: Arc<dyn FsProvider>,
    cwd: PathBuf,
    lease: Option<CapturedFsResourceLease>,
}

impl XmlPathOwner {
    pub(super) fn capture() -> io::Result<Self> {
        let (provider, cwd) = fs::current_provider_with_current_dir()?;
        let lease = capture_lease(&provider)?;
        Ok(Self {
            provider,
            cwd,
            lease,
        })
    }

    pub(super) fn resolve(self, path: PathBuf) -> XmlFsPath {
        let path = if self.provider.path_namespace().is_absolute(&path) {
            path
        } else {
            self.cwd.join(path)
        };
        XmlFsPath {
            provider: self.provider,
            path,
            lease: self.lease,
        }
    }
}

impl XmlFsPath {
    pub(super) fn capture(path: PathBuf) -> io::Result<Self> {
        let (provider, path) = fs::capture_provider_path(&path)?;
        let lease = capture_lease(&provider)?;
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

    pub(super) async fn read_bytes(&self) -> io::Result<Vec<u8>> {
        self.check_current()?;
        let bytes = self.provider.read(&self.path).await?;
        self.check_current()?;
        Ok(bytes)
    }

    pub(super) async fn write(&self, bytes: &[u8]) -> io::Result<()> {
        self.check_current()?;
        self.provider.write(&self.path, bytes).await?;
        self.check_current()
    }
}

fn capture_lease(provider: &Arc<dyn FsProvider>) -> io::Result<Option<CapturedFsResourceLease>> {
    let advertised = provider.resource_lease().is_some();
    let captured = fs::capture_resource_lease(provider.clone());
    if advertised && captured.is_none() {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "captured XML filesystem authority was revoked",
        ));
    }
    Ok(captured)
}

#[cfg(test)]
mod tests {
    use super::super::{XmlCooperation, XmlOperation};
    use super::*;

    fn test_dir(name: &str) -> PathBuf {
        PathBuf::from(format!("/{name}"))
    }

    #[cfg(not(windows))]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rooted_xml_path_uses_selected_provider_without_cwd_prefix() {
        let _lock = fs::provider_override_lock();
        let selected = Arc::new(fs::MemoryFsProvider::with_current_dir(test_dir("selected")));
        let replacement = Arc::new(fs::MemoryFsProvider::with_current_dir(test_dir(
            "replacement",
        )));
        let absolute = test_dir("project").join("rooted.xml");
        selected
            .write_project_path(&absolute, b"<selected/>")
            .expect("selected source");
        replacement
            .write_project_path(&absolute, b"<replacement/>")
            .expect("replacement source");
        let _selected = fs::replace_provider(selected);
        fs::set_current_dir(test_dir("selected")).expect("selected cwd");
        let captured = XmlFsPath::capture(absolute.clone()).expect("absolute capture");
        assert_eq!(captured.path, absolute);
        let _replacement = fs::replace_provider(replacement);
        fs::set_current_dir(test_dir("replacement")).expect("replacement cwd");
        assert_eq!(
            futures::executor::block_on(captured.read_bytes()).unwrap(),
            b"<selected/>"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn captured_provider_and_relative_path_survive_ambient_swap() {
        let _lock = fs::provider_override_lock();
        let selected_dir = test_dir("selected");
        let replacement_dir = test_dir("replacement");
        let selected_path = selected_dir.join("test.xml");
        let replacement_path = replacement_dir.join("test.xml");
        let selected = Arc::new(fs::MemoryFsProvider::with_current_dir(selected_dir.clone()));
        let replacement = Arc::new(fs::MemoryFsProvider::with_current_dir(
            replacement_dir.clone(),
        ));
        selected
            .write_project_path(&selected_path, b"<selected/>")
            .expect("selected fixture");
        replacement
            .write_project_path(&replacement_path, b"<replacement/>")
            .expect("replacement fixture");
        let _selected = fs::replace_provider(selected.clone());
        fs::set_current_dir(&selected_dir).expect("selected cwd");
        let captured = XmlFsPath::capture(PathBuf::from("test.xml")).expect("capture");
        let _replacement = fs::replace_provider(replacement.clone());
        fs::set_current_dir(&replacement_dir).expect("replacement cwd");
        assert_eq!(
            futures::executor::block_on(captured.read_bytes()).unwrap(),
            b"<selected/>"
        );
        futures::executor::block_on(captured.write(b"<changed/>")).expect("selected write");
        assert_eq!(
            futures::executor::block_on(selected.read(&selected_path)).unwrap(),
            b"<changed/>"
        );
        assert_eq!(
            futures::executor::block_on(replacement.read(&replacement_path)).unwrap(),
            b"<replacement/>"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn encoded_xml_uses_the_captured_provider_after_ambient_swap() {
        let _lock = fs::provider_override_lock();
        let selected_dir = test_dir("selected");
        let replacement_dir = test_dir("replacement");
        let selected_path = selected_dir.join("encoded.xml");
        let replacement_path = replacement_dir.join("encoded.xml");
        let selected = Arc::new(fs::MemoryFsProvider::with_current_dir(selected_dir.clone()));
        let replacement = Arc::new(fs::MemoryFsProvider::with_current_dir(
            replacement_dir.clone(),
        ));
        selected
            .write_project_path(
                &selected_path,
                b"<?xml version='1.0' encoding='windows-1252'?><root>caf\xE9</root>",
            )
            .expect("encoded fixture");
        replacement
            .write_project_path(&replacement_path, b"<wrong/>")
            .expect("replacement fixture");
        let _selected = fs::replace_provider(selected);
        fs::set_current_dir(&selected_dir).expect("selected cwd");
        let captured = XmlFsPath::capture(PathBuf::from("encoded.xml")).expect("capture");
        let _replacement = fs::replace_provider(replacement);
        fs::set_current_dir(&replacement_dir).expect("replacement cwd");
        let bytes = futures::executor::block_on(captured.read_bytes()).expect("captured read");
        let mut cooperation = XmlCooperation::new(None, XmlOperation::XmlRead);
        let text = futures::executor::block_on(super::super::parser::decode_document_bytes(
            &bytes,
            &mut cooperation,
        ))
        .expect("declared source decoding");
        assert!(text.contains("café"));
        assert!(!text.contains("wrong"));
    }
}
