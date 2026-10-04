use super::*;
use crate::{
    DirEntry, FileHandle, FsMetadata, MemoryFsProvider, NativeFsProvider, OpenFlags,
    SandboxFsProvider,
};
use std::sync::atomic::{AtomicBool, Ordering};

#[test]
fn native_relative_path_and_intent_remain_with_selected_provider_after_swap() {
    let _serial = crate::provider_override_lock();
    let selected: Arc<dyn FsProvider> = Arc::new(NativeFsProvider);
    let _selected = crate::replace_provider(selected.clone());
    let cwd = std::env::current_dir().expect("native working directory");
    let temporary = tempfile::Builder::new()
        .prefix("runmat-native-path-")
        .suffix(".h5")
        .tempfile_in(&cwd)
        .expect("temporary native file");
    let file_name = temporary.path().file_name().expect("file name");
    let read = capture_native_path(file_name, NativePathIntent::Read)
        .expect("capture")
        .expect("native route");
    let write = capture_native_path(file_name, NativePathIntent::ReadWriteCreate)
        .expect("capture")
        .expect("native route");
    assert_eq!(read.path(), temporary.path());
    assert_eq!(write.path(), read.path());
    assert_eq!(read.intent(), NativePathIntent::Read);
    assert_eq!(write.intent(), NativePathIntent::ReadWriteCreate);

    let replacement = Arc::new(MemoryFsProvider::with_current_dir("/virtual"));
    replacement
        .write_project_path(Path::new("/virtual").join(file_name), b"virtual")
        .expect("virtual fixture");
    let _replacement = crate::replace_provider(replacement.clone());
    assert!(Arc::ptr_eq(read.provider(), &selected));
    assert_eq!(read.path(), temporary.path());
    assert!(read.check_current().is_ok());
    assert!(write.check_current().is_ok());
    assert_eq!(crate::current_dir().unwrap(), PathBuf::from("/virtual"));
    std::fs::write(write.path(), b"selected native bytes").expect("native write");
    assert_eq!(
        std::fs::read(read.path()).unwrap(),
        b"selected native bytes"
    );
    assert_eq!(
        futures::executor::block_on(replacement.read(&Path::new("/virtual").join(file_name)))
            .unwrap(),
        b"virtual"
    );
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

#[test]
fn process_working_directory_is_issued_only_by_a_real_directory_owner() {
    let _serial = crate::provider_override_lock();
    let _native = crate::replace_provider(Arc::new(NativeFsProvider));
    let directory = tempfile::tempdir().expect("directory");
    let issued = capture_native_path(directory.path(), NativePathIntent::ProcessWorkingDirectory)
        .expect("native capture")
        .expect("native directory route");
    assert_eq!(issued.path(), directory.path());
    assert_eq!(issued.intent(), NativePathIntent::ProcessWorkingDirectory);
    assert!(issued.check_current().is_ok());
    assert!(matches!(
        issued.process_revocation(),
        NativeProcessRevocation::Stable
    ));
    let missing = directory.path().join("missing");
    let error = capture_native_path(&missing, NativePathIntent::ProcessWorkingDirectory)
        .err()
        .expect("missing directory cannot be issued");
    assert_eq!(error.kind(), io::ErrorKind::NotFound);
    let _virtual =
        crate::replace_provider(Arc::new(MemoryFsProvider::with_current_dir("/virtual")));
    assert!(
        capture_native_path("/virtual", NativePathIntent::ProcessWorkingDirectory)
            .expect("virtual provider route")
            .is_none()
    );
    assert!(issued.check_current().is_ok());
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

#[derive(Clone, Copy)]
enum FixtureIssue {
    WrongIntent,
    RevokedAfterIssue,
    Failure,
}

struct FixtureProvider(FixtureIssue);

#[derive(Debug)]
struct FixtureCause;

impl std::fmt::Display for FixtureCause {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("fixture provider cause")
    }
}

impl std::error::Error for FixtureCause {}

fn unused_operation() -> io::Error {
    io::Error::new(
        io::ErrorKind::Unsupported,
        "fixture does not perform filesystem I/O",
    )
}

#[async_trait::async_trait(?Send)]
impl FsProvider for FixtureProvider {
    fn acquire_native_path(
        &self,
        path: &Path,
        intent: NativePathIntent,
    ) -> io::Result<Option<NativePathGrant>> {
        if matches!(self.0, FixtureIssue::Failure) {
            return Err(io::Error::other(FixtureCause));
        }
        let valid = Arc::new(AtomicBool::new(true));
        let issued_intent = if matches!(self.0, FixtureIssue::WrongIntent) {
            NativePathIntent::ReadWriteCreate
        } else {
            intent
        };
        let grant = NativePathGrant::issued_by_provider(
            path.to_path_buf(),
            issued_intent,
            Box::new(Revocable(valid.clone())),
        )?;
        if matches!(self.0, FixtureIssue::RevokedAfterIssue) {
            valid.store(false, Ordering::SeqCst);
        }
        Ok(Some(grant))
    }

    fn open(&self, _: &Path, _: &OpenFlags) -> io::Result<Box<dyn FileHandle>> {
        Err(unused_operation())
    }
    async fn read(&self, _: &Path) -> io::Result<Vec<u8>> {
        Err(unused_operation())
    }
    async fn write(&self, _: &Path, _: &[u8]) -> io::Result<()> {
        Err(unused_operation())
    }
    async fn remove_file(&self, _: &Path) -> io::Result<()> {
        Err(unused_operation())
    }
    async fn metadata(&self, _: &Path) -> io::Result<FsMetadata> {
        Err(unused_operation())
    }
    async fn symlink_metadata(&self, _: &Path) -> io::Result<FsMetadata> {
        Err(unused_operation())
    }
    async fn read_dir(&self, _: &Path) -> io::Result<Vec<DirEntry>> {
        Err(unused_operation())
    }
    async fn canonicalize(&self, _: &Path) -> io::Result<PathBuf> {
        Err(unused_operation())
    }
    async fn create_dir(&self, _: &Path) -> io::Result<()> {
        Err(unused_operation())
    }
    async fn create_dir_all(&self, _: &Path) -> io::Result<()> {
        Err(unused_operation())
    }
    async fn remove_dir(&self, _: &Path) -> io::Result<()> {
        Err(unused_operation())
    }
    async fn remove_dir_all(&self, _: &Path) -> io::Result<()> {
        Err(unused_operation())
    }
    async fn rename(&self, _: &Path, _: &Path) -> io::Result<()> {
        Err(unused_operation())
    }
    async fn set_readonly(&self, _: &Path, _: bool) -> io::Result<()> {
        Err(unused_operation())
    }
}

#[test]
fn capture_rejects_wrong_intent_and_revoked_returned_grant_without_losing_io_cause() {
    let _serial = crate::provider_override_lock();
    for (issue, expected_kind) in [
        (FixtureIssue::WrongIntent, io::ErrorKind::PermissionDenied),
        (
            FixtureIssue::RevokedAfterIssue,
            io::ErrorKind::PermissionDenied,
        ),
    ] {
        let _selected = crate::replace_provider(Arc::new(FixtureProvider(issue)));
        let error = capture_native_path("fixture.h5", NativePathIntent::Read)
            .err()
            .expect("invalid grant rejected");
        assert_eq!(error.kind(), expected_kind);
    }
    let _selected = crate::replace_provider(Arc::new(FixtureProvider(FixtureIssue::Failure)));
    let error = capture_native_path("fixture.h5", NativePathIntent::Read)
        .err()
        .expect("provider error retained");
    assert!(error
        .get_ref()
        .is_some_and(|source| source.is::<FixtureCause>()));
}
