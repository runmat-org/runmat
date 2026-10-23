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
        .write_project_path(selected_file(), b"1,2\n3,4\n")
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
    Path::new(selected_dir()).join("input.csv")
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn invocation_keeps_opened_provider_and_cwd_when_ambient_owner_changes_while_pending() {
    let _lock = runmat_filesystem::provider_override_lock();
    let selected = selected();
    let replacement = Arc::new(MemoryFsProvider::with_current_dir(replacement_dir()));
    replacement
        .write_project_path(selected_file(), b"9,8\n7,6\n")
        .expect("conflicting same path");
    let _restore = runmat_filesystem::replace_provider(selected.clone());
    runmat_filesystem::set_current_dir(selected_dir()).expect("selected cwd");
    let mut call = Box::pin(readmatrix_builtin(
        None,
        Value::from("input.csv"),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ));
    let waker = futures::task::noop_waker();
    assert!(matches!(
        call.as_mut().poll(&mut Context::from_waker(&waker)),
        Poll::Pending
    ));
    assert!(selected.first_open_polled.load(Ordering::SeqCst));
    runmat_filesystem::set_provider(replacement);
    runmat_filesystem::set_current_dir(replacement_dir()).expect("replacement cwd");
    let Value::Tensor(matrix) = block_on(call).expect("original captured owner") else {
        panic!("numeric matrix")
    };
    assert_eq!(matrix.shape(), vec![2, 2]);
    assert_eq!(matrix.materialize_f64(), vec![1.0, 3.0, 2.0, 4.0]);
    assert_eq!(selected.opens.load(Ordering::SeqCst), 1);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn known_invalid_late_control_rejects_without_opening_selected_file() {
    let _lock = runmat_filesystem::provider_override_lock();
    let selected = selected();
    let _restore = runmat_filesystem::replace_provider(selected.clone());
    runmat_filesystem::set_current_dir(selected_dir()).expect("selected cwd");
    let error = block_on(readmatrix_builtin(
        None,
        Value::from("input.csv"),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("Delimiter"),
            Value::from(","),
            Value::from("Range"),
            Value::from("not-a-range"),
        ]),
    ))
    .expect_err("known invalid range");
    assert!(error.message().contains("Range"));
    assert_eq!(selected.opens.load(Ordering::SeqCst), 0);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn paged_text_filename_and_option_reject_before_file_open() {
    let _lock = runmat_filesystem::provider_override_lock();
    let selected = selected();
    let _restore = runmat_filesystem::replace_provider(selected.clone());
    runmat_filesystem::set_current_dir(selected_dir()).expect("selected cwd");
    let paged = Value::CharArray(
        runmat_value::CharArray::from_unicode_scalars_with_shape(vec!['x', 'y'], vec![1, 1, 2])
            .expect("paged character array"),
    );
    assert!(block_on(readmatrix_builtin(
        None,
        paged.clone(),
        crate::builtins::common::test_support::builtin_arguments(Vec::new())
    ))
    .is_err());
    assert!(block_on(readmatrix_builtin(
        None,
        Value::from("input.csv"),
        crate::builtins::common::test_support::builtin_arguments(vec![paged, Value::from(",")]),
    ))
    .is_err());
    assert_eq!(selected.opens.load(Ordering::SeqCst), 0);

    let trailing = Value::CharArray(
        runmat_value::CharArray::from_unicode_scalars_with_shape(
            "input.csv".chars().collect(),
            vec![1, "input.csv".len(), 1],
        )
        .expect("trailing singleton character vector"),
    );
    let Value::Tensor(result) = block_on(readmatrix_builtin(
        None,
        trailing,
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("effective character row accepted") else {
        panic!("numeric matrix")
    };
    assert_eq!(result.shape(), vec![2, 2]);
    assert_eq!(selected.opens.load(Ordering::SeqCst), 1);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn explicit_double_output_type_and_like_conflict_before_file_open() {
    let _lock = runmat_filesystem::provider_override_lock();
    let selected = selected();
    let _restore = runmat_filesystem::replace_provider(selected.clone());
    runmat_filesystem::set_current_dir(selected_dir()).expect("selected cwd");
    // Like is a RunMat-only form; exercise its admitted controls rather than
    // the earlier compatibility gate for the default MATLAB profile.
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let error = block_on(readmatrix_builtin(
        None,
        Value::from("input.csv"),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("OutputType"),
            Value::from("double"),
            Value::from("Like"),
            Value::Num(0.0),
        ]),
    ))
    .expect_err("controls conflict");
    assert!(error.message().contains("cannot combine"));
    assert_eq!(selected.opens.load(Ordering::SeqCst), 0);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn like_word_as_option_value_does_not_enable_the_like_extension() {
    let _lock = runmat_filesystem::provider_override_lock();
    let selected = selected();
    let _restore = runmat_filesystem::replace_provider(selected);
    runmat_filesystem::set_current_dir(selected_dir()).expect("selected cwd");
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let value = block_on(readmatrix_builtin(
        None,
        Value::from("input.csv"),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("TreatAsMissing"),
            Value::from("Like"),
        ]),
    ))
    .expect("ordinary TreatAsMissing value");
    assert!(matches!(value, Value::Tensor(_)));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn cancelling_pending_open_returns_no_value_and_does_not_reopen() {
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
    let mut call = Box::pin(readmatrix_builtin(
        Some(runtime),
        Value::from("input.csv"),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ));
    let waker = futures::task::noop_waker();
    assert!(matches!(
        call.as_mut().poll(&mut Context::from_waker(&waker)),
        Poll::Pending
    ));
    assert!(selected.first_open_polled.load(Ordering::SeqCst));
    source.cancel(runmat_execution::CancellationReason::User);
    let error = block_on(call).expect_err("cancelled open cannot publish a matrix");
    assert!(error
        .identifier()
        .is_some_and(|id| id.ends_with(":Cancelled")));
    assert_eq!(selected.opens.load(Ordering::SeqCst), 0);
}
