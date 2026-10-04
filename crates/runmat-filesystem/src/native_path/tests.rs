use super::*;
use crate::{MemoryFsProvider, NativeFsProvider, SandboxFsProvider};
use std::sync::atomic::{AtomicBool, Ordering};

#[test]
fn native_relative_path_and_intent_remain_with_selected_provider_after_swap() {
    let _serial = crate::provider_override_lock();
    let selected: Arc<dyn FsProvider> = Arc::new(NativeFsProvider);
    let _selected = crate::replace_provider(selected.clone());
    let cwd = std::env::current_dir().expect("native working directory");
    let read = capture_native_path("selected.h5", NativePathIntent::Read)
        .expect("capture")
        .expect("native route");
    let write = capture_native_path("selected.h5", NativePathIntent::ReadWriteCreate)
        .expect("capture")
        .expect("native route");
    assert_eq!(read.path(), cwd.join("selected.h5"));
    assert_eq!(write.path(), read.path());
    assert_eq!(read.intent(), NativePathIntent::Read);
    assert_eq!(write.intent(), NativePathIntent::ReadWriteCreate);

    let _replacement =
        crate::replace_provider(Arc::new(MemoryFsProvider::with_current_dir("/virtual")));
    assert!(Arc::ptr_eq(read.provider(), &selected));
    assert_eq!(read.path(), cwd.join("selected.h5"));
    assert!(read.check_current().is_ok());
    assert!(write.check_current().is_ok());
    assert_eq!(crate::current_dir().unwrap(), PathBuf::from("/virtual"));
}

#[test]
fn virtual_providers_do_not_export_an_ambient_native_path() {
    let _serial = crate::provider_override_lock();
    let _memory = crate::replace_provider(Arc::new(MemoryFsProvider::with_current_dir("/")));
    assert!(
        capture_native_path("/not-an-os-path.h5", NativePathIntent::Read)
            .unwrap()
            .is_none()
    );
    drop(_memory);

    let root = tempfile::tempdir().expect("sandbox root");
    let _sandbox = crate::replace_provider(Arc::new(
        SandboxFsProvider::new(root.path().to_path_buf()).expect("sandbox"),
    ));
    assert!(
        capture_native_path("/virtual.h5", NativePathIntent::ReadWriteCreate)
            .unwrap()
            .is_none()
    );
}

struct Revocable(Arc<AtomicBool>);

impl NativePathAuthority for Revocable {
    fn is_valid(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

#[test]
fn issuer_revocation_is_visible_to_consumer_checkpoint() {
    let valid = Arc::new(AtomicBool::new(true));
    let path = std::env::current_dir().unwrap().join("revocable.h5");
    let grant = NativePathGrant::issued_by_provider(
        path,
        NativePathIntent::Read,
        Box::new(Revocable(valid.clone())),
    )
    .expect("provider issue");
    let selected = CapturedNativePath {
        provider: Arc::new(NativeFsProvider),
        grant,
    };
    assert!(selected.check_current().is_ok());
    valid.store(false, Ordering::SeqCst);
    assert_eq!(
        selected.check_current().unwrap_err().kind(),
        io::ErrorKind::PermissionDenied
    );
}

#[test]
fn provider_cannot_issue_relative_or_revoked_native_path() {
    let valid = Arc::new(AtomicBool::new(true));
    let error = NativePathGrant::issued_by_provider(
        PathBuf::from("relative.h5"),
        NativePathIntent::Read,
        Box::new(Revocable(valid.clone())),
    )
    .err()
    .expect("relative path rejected");
    assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    valid.store(false, Ordering::SeqCst);
    let error = NativePathGrant::issued_by_provider(
        std::env::current_dir().unwrap().join("revoked.h5"),
        NativePathIntent::Read,
        Box::new(Revocable(valid)),
    )
    .err()
    .expect("revoked issue rejected");
    assert_eq!(error.kind(), io::ErrorKind::PermissionDenied);
}
