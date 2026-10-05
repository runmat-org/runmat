use super::*;
use crate::context::RuntimeContext;
use crate::execution::{
    CpuCooperationBudget, CpuCooperationPolicy, ExecutionServiceError, RuntimeExecutionService,
};
use futures::executor::block_on;
use runmat_execution::cancellation::CancellationSource;
use runmat_filesystem::MemoryFsProvider;
use runmat_value::IntegerStorage;
use std::cell::Cell;
use std::future::Future;
use std::num::NonZeroU64;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::rc::Rc;
use std::sync::Arc;
use std::task::{Context, Poll};

fn fixture_path() -> PathBuf {
    if cfg!(windows) {
        PathBuf::from(r"C:\readmatrix-row\input.csv")
    } else {
        PathBuf::from("/readmatrix-row/input.csv")
    }
}

fn selected_path(bytes: &[u8]) -> (runmat_filesystem::ProviderGuard, TabularFsPath) {
    let path = fixture_path();
    let provider = Arc::new(MemoryFsProvider::with_current_dir(
        path.parent().expect("fixture parent"),
    ));
    provider.write_project_path(&path, bytes).expect("fixture");
    let restore = runmat_filesystem::replace_provider(provider);
    let selected = TabularFsOwner::capture(Some(Path::new(&path)))
        .expect("captured provider")
        .path(path)
        .expect("selected path");
    (restore, selected)
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn borrowed_field_scanner_preserves_quoted_and_multichar_forms() {
    let _lock = runmat_filesystem::provider_override_lock();
    let (_restore, path) = selected_path(b"1\n");
    let mut cooperation = TabularCooperation::new(None, BUILTIN_NAME);
    let quoted = block_on(scan::split_fields(
        "\"1,2\",3,\"4\"\"5\"",
        &Delimiter::Char(','),
        &mut cooperation,
        &path,
    ))
    .expect("quoted cells");
    assert_eq!(
        quoted
            .iter()
            .map(|field| field.as_ref())
            .collect::<Vec<_>>(),
        vec!["1,2", "3", "4\"5"]
    );
    let multi = block_on(scan::split_fields(
        "a::b::::c",
        &Delimiter::String("::".into()),
        &mut cooperation,
        &path,
    ))
    .expect("multi-character delimiter");
    assert_eq!(
        multi.iter().map(|field| field.as_ref()).collect::<Vec<_>>(),
        vec!["a", "b", "", "c"]
    );
    let whitespace = block_on(scan::split_fields(
        "  1\u{2003}2\t3  ",
        &Delimiter::Whitespace,
        &mut cooperation,
        &path,
    ))
    .expect("Unicode whitespace");
    assert_eq!(
        whitespace
            .iter()
            .map(|field| field.as_ref())
            .collect::<Vec<_>>(),
        vec!["1", "2", "3"]
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn borrowed_rows_retain_exact_wide_integer_range_and_row_diagnostics() {
    let _lock = runmat_filesystem::provider_override_lock();
    let (_restore, path) = selected_path(b"0,0\n9007199254740993,18446744073709551615\n");
    let mut cooperation = TabularCooperation::new(None, BUILTIN_NAME);
    let mut options = ReadMatrixOptions::default();
    options
        .set_output_type("uint64")
        .expect("typed output control");
    options.range = Some(super::super::range::parse_range(&Value::from("A2:B2")).expect("range"));
    let matrix = block_on(read_numeric_matrix(None, &mut cooperation, &path, &options))
        .expect("selected exact integers");
    assert_eq!(matrix.shape, vec![1, 2]);
    assert_eq!(
        matrix.integer_storage(),
        Some(&IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]))
    );

    let (_restore, bad_path) = selected_path(b"1,2\n3,not-a-number\n");
    let error = block_on(read_numeric_matrix(
        None,
        &mut cooperation,
        &bad_path,
        &ReadMatrixOptions::default(),
    ))
    .expect_err("invalid second-row field");
    assert!(error.message().contains("line 2 column 2"));
}

struct PendingTurn(Rc<Cell<bool>>);

impl Future for PendingTurn {
    type Output = Result<(), ExecutionServiceError>;

    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
        Poll::Pending
    }
}

impl Drop for PendingTurn {
    fn drop(&mut self) {
        self.0.set(true);
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn cancelling_pending_single_wide_row_scan_releases_work_without_output() {
    let _lock = runmat_filesystem::provider_override_lock();
    let mut input = vec![b'1'; 32_767];
    for index in (1..input.len()).step_by(2) {
        input[index] = b',';
    }
    input.push(b'\n');
    let (_restore, path) = selected_path(&input);
    let source = CancellationSource::new();
    let runtime =
        RuntimeContext::with_cancellation(Rc::new(RuntimeExecutionService::new()), source.token());
    let turns = Rc::new(Cell::new(0usize));
    let dropped = Rc::new(Cell::new(false));
    let turn_count = turns.clone();
    let turn_dropped = dropped.clone();
    let budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(4_096).expect("nonzero")),
        source.token(),
        move || -> crate::context::RuntimeServiceFuture<Result<(), ExecutionServiceError>> {
            let count = turn_count.get() + 1;
            turn_count.set(count);
            if count < 11 {
                Box::pin(async { Ok(()) })
            } else {
                Box::pin(PendingTurn(turn_dropped.clone()))
            }
        },
    );
    let mut cooperation = TabularCooperation::with_test_budget(&runtime, BUILTIN_NAME, budget);
    let mut options = ReadMatrixOptions::default();
    options.delimiter = Some(Delimiter::Char(','));
    let mut call = Box::pin(read_numeric_matrix(
        Some(&runtime),
        &mut cooperation,
        &path,
        &options,
    ));
    let waker = futures::task::noop_waker();
    assert!(matches!(
        call.as_mut().poll(&mut Context::from_waker(&waker)),
        Poll::Pending
    ));
    assert_eq!(
        turns.get(),
        11,
        "one wide row reached incremental parsing work"
    );
    source.cancel(runmat_execution::CancellationReason::User);
    let error = block_on(call).expect_err("cancelled row cannot publish a tensor");
    assert!(error
        .identifier()
        .is_some_and(|id| id.ends_with(":Cancelled")));
    assert!(dropped.get(), "pending service turn was released");
}
