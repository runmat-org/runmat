use super::*;
use async_trait::async_trait;
use runmat_filesystem::{
    DirEntry, FileHandle, FsMetadata, FsProvider, MemoryFsProvider, OpenFlags,
};
use std::future::Future;
use std::io;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};

struct DelayedOwner {
    memory: MemoryFsProvider,
    first_open_polled: AtomicBool,
    opens: AtomicUsize,
}

#[async_trait(?Send)]
impl FsProvider for DelayedOwner {
    fn path_namespace(&self) -> runmat_filesystem::FsPathNamespace {
        self.memory.path_namespace()
    }

    fn current_dir_override(&self) -> Option<PathBuf> {
        Some(PathBuf::from(selected_dir()))
    }

    fn open(&self, path: &Path, flags: &OpenFlags) -> io::Result<Box<dyn FileHandle>> {
        self.opens.fetch_add(1, Ordering::SeqCst);
        self.memory.open(path, flags)
    }

    async fn open_async(&self, path: &Path, flags: &OpenFlags) -> io::Result<Box<dyn FileHandle>> {
        futures::future::poll_fn(|cx| {
            if self.first_open_polled.swap(true, Ordering::SeqCst) {
                Poll::Ready(())
            } else {
                cx.waker().wake_by_ref();
                Poll::Pending
            }
        })
        .await;
        self.open(path, flags)
    }

    async fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
        self.memory.read(path).await
    }
    async fn write(&self, path: &Path, bytes: &[u8]) -> io::Result<()> {
        self.memory.write(path, bytes).await
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

fn selected() -> Arc<DelayedOwner> {
    let memory = MemoryFsProvider::with_current_dir(selected_dir());
    memory
        .write_project_path(selected_file(), &in_memory_xlsx("Data"))
        .expect("fixture");
    Arc::new(DelayedOwner {
        memory,
        first_open_polled: AtomicBool::new(false),
        opens: AtomicUsize::new(0),
    })
}

fn selected_dir() -> &'static str {
    "/selected"
}

fn replacement_dir() -> &'static str {
    "/replacement"
}

fn selected_file() -> PathBuf {
    Path::new(selected_dir()).join("input.xlsx")
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn invocation_retains_selected_provider_and_cwd_across_pending_open() {
    let _lock = runmat_filesystem::provider_override_lock();
    let selected = selected();
    let replacement = Arc::new(MemoryFsProvider::with_current_dir(replacement_dir()));
    replacement
        .write_project_path(selected_file(), &in_memory_xlsx("Wrong"))
        .expect("conflicting workbook at same path");
    let _restore = runmat_filesystem::replace_provider(selected.clone());
    runmat_filesystem::set_current_dir(selected_dir()).expect("selected cwd");
    let mut call = Box::pin(xlsread_builtin(
        None,
        Value::from("input.xlsx"),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("Data"),
            Value::from("A2:B4"),
        ]),
    ));
    let waker = futures::task::noop_waker();
    assert!(matches!(
        call.as_mut().poll(&mut Context::from_waker(&waker)),
        Poll::Pending
    ));
    assert!(selected.first_open_polled.load(Ordering::SeqCst));
    runmat_filesystem::set_provider(replacement);
    runmat_filesystem::set_current_dir(replacement_dir()).expect("replacement cwd");
    let Value::Tensor(numeric) = block_on(call).expect("selected workbook result") else {
        panic!("numeric worksheet result")
    };
    assert_eq!(numeric.shape(), &[3, 2]);
    assert_eq!(
        numeric.materialize_f64(),
        vec![1.5, 2.5, 3.5, 10.0, 20.0, 30.0]
    );
    assert_eq!(selected.opens.load(Ordering::SeqCst), 1);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn invalid_host_range_and_output_count_reject_without_open() {
    let _lock = runmat_filesystem::provider_override_lock();
    let selected = selected();
    let _restore = runmat_filesystem::replace_provider(selected.clone());
    runmat_filesystem::set_current_dir(selected_dir()).expect("selected cwd");
    let bad_range = block_on(xlsread_builtin(
        None,
        Value::from("input.xlsx"),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("Data"),
            Value::from("A0:B2"),
        ]),
    ));
    assert!(bad_range.is_err());
    let _outputs = crate::output_count::push_output_count(Some(4)).expect("output-count scope");
    let bad_outputs = block_on(xlsread_builtin(
        None,
        Value::from("input.xlsx"),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ));
    assert_eq!(
        bad_outputs.expect_err("four outputs").identifier(),
        XLSREAD_ERROR_OUTPUT_COUNT.identifier
    );
    assert_eq!(selected.opens.load(Ordering::SeqCst), 0);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn cancelling_pending_open_releases_selected_work_without_replay() {
    use crate::context::RuntimeContext;
    use crate::execution::RuntimeExecutionService;
    use runmat_execution::cancellation::CancellationSource;

    let _lock = runmat_filesystem::provider_override_lock();
    let selected = selected();
    let _restore = runmat_filesystem::replace_provider(selected.clone());
    runmat_filesystem::set_current_dir(selected_dir()).expect("selected cwd");
    let source = CancellationSource::new();
    let runtime =
        RuntimeContext::with_cancellation(Rc::new(RuntimeExecutionService::new()), source.token());
    let mut call = Box::pin(xlsread_builtin(
        Some(runtime),
        Value::from("input.xlsx"),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ));
    let waker = futures::task::noop_waker();
    assert!(matches!(
        call.as_mut().poll(&mut Context::from_waker(&waker)),
        Poll::Pending
    ));
    assert!(selected.first_open_polled.load(Ordering::SeqCst));
    source.cancel(runmat_execution::CancellationReason::User);
    let error = block_on(call).expect_err("cancelled open cannot publish values");
    assert!(error
        .identifier()
        .is_some_and(|id| id.ends_with(":Cancelled")));
    assert_eq!(selected.opens.load(Ordering::SeqCst), 0);
}
