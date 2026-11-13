use super::*;
use futures::executor::block_on;
use runmat_filesystem::{
    DirEntry, FileHandle, FsMetadata, FsProvider, MemoryFsProvider, OpenFlags,
};
use std::{
    io,
    sync::{Arc, Mutex},
};

#[derive(Debug, thiserror::Error)]
#[error("selected filesystem denied this operation")]
struct ProbeFailure;

struct Probe {
    inner: MemoryFsProvider,
    metadata_failure: Option<io::ErrorKind>,
    read_failure: bool,
    inspected: Mutex<Vec<PathBuf>>,
}

impl Probe {
    fn new(metadata_failure: Option<io::ErrorKind>, read_failure: bool) -> Arc<Self> {
        let inner = MemoryFsProvider::with_current_dir("/workspace");
        inner
            .write_project_path("/workspace/entry.m", b"classdef entry\nend")
            .unwrap();
        inner
            .write_project_path("/workspace/entry.p", b"not selected")
            .unwrap();
        Arc::new(Self {
            inner,
            metadata_failure,
            read_failure,
            inspected: Mutex::new(Vec::new()),
        })
    }
}

#[async_trait::async_trait(?Send)]
impl FsProvider for Probe {
    fn path_namespace(&self) -> runmat_filesystem::FsPathNamespace {
        self.inner.path_namespace()
    }
    fn current_dir_override(&self) -> Option<PathBuf> {
        Some(PathBuf::from("/workspace"))
    }
    fn open(&self, path: &Path, flags: &OpenFlags) -> io::Result<Box<dyn FileHandle>> {
        self.inner.open(path, flags)
    }
    async fn metadata(&self, path: &Path) -> io::Result<FsMetadata> {
        self.inspected.lock().unwrap().push(path.to_path_buf());
        if let Some(kind) = self.metadata_failure {
            return Err(io::Error::new(kind, ProbeFailure));
        }
        self.inner.metadata(path).await
    }
    async fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
        if self.read_failure {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                ProbeFailure,
            ));
        }
        self.inner.read(path).await
    }
    async fn write(&self, path: &Path, bytes: &[u8]) -> io::Result<()> {
        self.inner.write(path, bytes).await
    }
    async fn remove_file(&self, path: &Path) -> io::Result<()> {
        self.inner.remove_file(path).await
    }
    async fn symlink_metadata(&self, path: &Path) -> io::Result<FsMetadata> {
        self.inner.symlink_metadata(path).await
    }
    async fn read_dir(&self, path: &Path) -> io::Result<Vec<DirEntry>> {
        self.inner.read_dir(path).await
    }
    async fn canonicalize(&self, path: &Path) -> io::Result<PathBuf> {
        self.inner.canonicalize(path).await
    }
    async fn create_dir(&self, path: &Path) -> io::Result<()> {
        self.inner.create_dir(path).await
    }
    async fn create_dir_all(&self, path: &Path) -> io::Result<()> {
        self.inner.create_dir_all(path).await
    }
    async fn remove_dir(&self, path: &Path) -> io::Result<()> {
        self.inner.remove_dir(path).await
    }
    async fn remove_dir_all(&self, path: &Path) -> io::Result<()> {
        self.inner.remove_dir_all(path).await
    }
    async fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        self.inner.rename(from, to).await
    }
    async fn set_readonly(&self, path: &Path, value: bool) -> io::Result<()> {
        self.inner.set_readonly(path, value).await
    }
}

fn assert_cause(error: PathSearchError, kind: io::ErrorKind) {
    let runtime = error.into_runtime_error("dbtype");
    assert_eq!(runtime.gpu_gather_retry, crate::GpuGatherRetry::Never);
    let path = runtime
        .source
        .as_ref()
        .unwrap()
        .downcast_ref::<PathSearchError>()
        .unwrap();
    let PathSearchError::Io { source, .. } = path else {
        panic!("typed IO failure expected")
    };
    assert_eq!(source.kind(), kind);
    assert!(source.get_ref().unwrap().is::<ProbeFailure>());
}

#[test]
fn only_missing_candidates_are_absence() {
    let _serial = vfs::provider_override_lock();
    for kind in [io::ErrorKind::NotFound, io::ErrorKind::NotADirectory] {
        let probe = Probe::new(Some(kind), false);
        vfs::with_provider_override(probe, || {
            assert!(!block_on(path_is_file(Path::new("./entry.m"))).unwrap());
            assert!(!block_on(path_is_directory(Path::new("./entry"))).unwrap());
            assert!(
                block_on(find_file_with_extensions("./entry", &[".m", ".p"], "test"))
                    .unwrap()
                    .is_none()
            );
        });
    }
}

#[test]
fn operational_metadata_failure_is_terminal_before_later_candidate() {
    let _serial = vfs::provider_override_lock();
    for kind in [
        io::ErrorKind::PermissionDenied,
        io::ErrorKind::Interrupted,
        io::ErrorKind::Other,
    ] {
        let probe = Probe::new(Some(kind), false);
        vfs::with_provider_override(probe.clone(), || {
            assert_cause(
                block_on(find_file_with_extensions("./entry", &[".m", ".p"], "test")).unwrap_err(),
                kind,
            );
            assert_eq!(
                *probe.inspected.lock().unwrap(),
                vec![PathBuf::from("/workspace/./entry.m")]
            );
        });
    }
}

#[test]
fn class_source_read_failure_is_not_a_nonclass_result() {
    let _serial = vfs::provider_override_lock();
    let probe = Probe::new(None, true);
    vfs::with_provider_override(probe, || {
        assert_cause(
            block_on(class_file_exists("./entry.m", &[".m"], "classdef", "exist")).unwrap_err(),
            io::ErrorKind::PermissionDenied,
        );
    });
}

#[test]
fn pcode_preference_does_not_fall_back_on_probe_failure() {
    let _serial = vfs::provider_override_lock();
    let probe = Probe::new(Some(io::ErrorKind::PermissionDenied), false);
    vfs::with_provider_override(probe, || {
        assert_cause(
            block_on(
                crate::builtins::io::repl_fs::pcode::prefer_pcode_source_path(Path::new(
                    "./entry.m",
                )),
            )
            .unwrap_err(),
            io::ErrorKind::PermissionDenied,
        );
    });
    let probe = Probe::new(None, false);
    vfs::with_provider_override(probe, || {
        assert_eq!(
            block_on(
                crate::builtins::io::repl_fs::pcode::prefer_pcode_source_path(Path::new(
                    "./entry.m"
                ))
            )
            .unwrap(),
            PathBuf::from("./entry.p")
        );
        assert_eq!(
            block_on(
                crate::builtins::io::repl_fs::pcode::prefer_pcode_source_path(Path::new(
                    "./missing.m"
                ))
            )
            .unwrap(),
            PathBuf::from("./missing.m")
        );
    });
}

#[test]
fn selected_control_error_keeps_identifier_category_and_terminal_retry() {
    let original = crate::build_runtime_error("selected scope cancelled")
        .with_identifier("SelectedScope:Cancelled")
        .with_source(crate::execution::ExecutionServiceError::Cancelled)
        .build();
    let error = PathSearchError::Control(Box::new(original))
        .into_identified_runtime_error("which", "RunMat:FunctionResolution");
    assert_eq!(error.identifier(), Some("SelectedScope:Cancelled"));
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert!(matches!(
        error
            .source
            .as_ref()
            .unwrap()
            .downcast_ref::<crate::execution::ExecutionServiceError>(),
        Some(crate::execution::ExecutionServiceError::Cancelled)
    ));
}

#[test]
fn public_observers_do_not_report_permission_failure_as_missing() {
    let _serial = vfs::provider_override_lock();
    let probe = Probe::new(Some(io::ErrorKind::PermissionDenied), false);
    vfs::with_provider_override(probe, || {
        for builtin in ["exist", "which", "dbtype"] {
            let error = crate::dispatcher::call_builtin(
                builtin,
                &[runmat_value::Value::string("./entry.m")],
            )
            .unwrap_err();
            assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
            let mut source: &(dyn std::error::Error + 'static) = &error;
            loop {
                if let Some(io) = source.downcast_ref::<io::Error>() {
                    assert_eq!(io.kind(), io::ErrorKind::PermissionDenied);
                    assert!(io.get_ref().unwrap().is::<ProbeFailure>());
                    break;
                }
                source = source
                    .source()
                    .expect("original typed IO cause must survive builtin projection");
            }
        }
    });
}
