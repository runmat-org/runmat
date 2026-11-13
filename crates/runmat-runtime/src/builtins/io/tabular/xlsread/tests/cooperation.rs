//! Selected output conversion retains the invocation's cancellation budget.

use super::*;
use crate::context::RuntimeContext;
use crate::execution::{
    CpuCooperationBudget, CpuCooperationPolicy, ExecutionServiceError, RuntimeExecutionService,
};
use calamine::Data as SpreadsheetData;
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::CancellationReason;
use runmat_filesystem::MemoryFsProvider;
use std::cell::Cell;
use std::future::Future;
use std::num::NonZeroU64;
use std::pin::Pin;
use std::rc::Rc;
use std::sync::Arc;
use std::task::{Context, Poll};

struct PendingTurn(Rc<Cell<bool>>);

fn pending_budget(
    source: &CancellationSource,
    turns: &Rc<Cell<usize>>,
    dropped: &Rc<Cell<bool>>,
    allowed_turns: usize,
) -> CpuCooperationBudget {
    let turn_count = turns.clone();
    let turn_dropped = dropped.clone();
    CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).expect("nonzero")),
        source.token(),
        move || -> crate::context::RuntimeServiceFuture<Result<(), ExecutionServiceError>> {
            let count = turn_count.get() + 1;
            turn_count.set(count);
            if count <= allowed_turns {
                Box::pin(async { Ok(()) })
            } else {
                Box::pin(PendingTurn(turn_dropped.clone()))
            }
        },
    )
}

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
fn cancellation_during_numeric_output_conversion_drops_pending_turn_without_output() {
    let _lock = runmat_filesystem::provider_override_lock();
    let provider = Arc::new(MemoryFsProvider::with_current_dir("/xlsread-output"));
    let _restore = runmat_filesystem::replace_provider(provider);
    let selected = TabularFsOwner::capture(Some(Path::new("/xlsread-output/input.xlsx")))
        .expect("captured owner")
        .path(PathBuf::from("/xlsread-output/input.xlsx"))
        .expect("selected path");
    let source = CancellationSource::new();
    let runtime =
        RuntimeContext::with_cancellation(Rc::new(RuntimeExecutionService::new()), source.token());
    let turns = Rc::new(Cell::new(0usize));
    let dropped = Rc::new(Cell::new(false));
    let budget = pending_budget(&source, &turns, &dropped, 16);
    let mut cooperation = TabularCooperation::with_test_budget(&runtime, BUILTIN_NAME, budget);
    let rows = vec![vec![SpreadsheetData::Float(1.0); 64]; 128];
    let _outputs = crate::output_count::push_output_count(Some(1)).expect("output-count scope");
    let requested = XlsReadOutputs::requested().expect("one output");
    let mut call = Box::pin(XlsReadResult::from_cells(
        rows,
        requested,
        &mut cooperation,
        &selected,
    ));
    assert!(matches!(
        call.as_mut()
            .poll(&mut Context::from_waker(futures::task::noop_waker_ref())),
        Poll::Pending
    ));
    assert_eq!(
        turns.get(),
        17,
        "conversion entered numeric cell projection"
    );
    source.cancel(CancellationReason::User);
    let Poll::Ready(Err(error)) = call
        .as_mut()
        .poll(&mut Context::from_waker(futures::task::noop_waker_ref()))
    else {
        panic!("cancelled conversion must not publish an output")
    };
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert!(dropped.get(), "pending execution turn must be retired");
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn numeric_scan_yields_inside_one_wide_row() {
    let _lock = runmat_filesystem::provider_override_lock();
    let provider = Arc::new(MemoryFsProvider::with_current_dir("/xlsread-output"));
    let _restore = runmat_filesystem::replace_provider(provider);
    let selected = TabularFsOwner::capture(Some(Path::new("/xlsread-output/input.xlsx")))
        .expect("captured owner")
        .path(PathBuf::from("/xlsread-output/input.xlsx"))
        .expect("selected path");
    let source = CancellationSource::new();
    let runtime =
        RuntimeContext::with_cancellation(Rc::new(RuntimeExecutionService::new()), source.token());
    let turns = Rc::new(Cell::new(0usize));
    let dropped = Rc::new(Cell::new(false));
    let budget = pending_budget(&source, &turns, &dropped, 0);
    let mut cooperation = TabularCooperation::with_test_budget(&runtime, BUILTIN_NAME, budget);
    let rows = vec![vec![SpreadsheetData::Float(1.0); 4096]];
    let _outputs = crate::output_count::push_output_count(Some(1)).expect("output-count scope");
    let requested = XlsReadOutputs::requested().expect("one output");
    let mut scan = Box::pin(XlsReadResult::from_cells(
        rows,
        requested,
        &mut cooperation,
        &selected,
    ));
    assert!(matches!(
        scan.as_mut()
            .poll(&mut Context::from_waker(futures::task::noop_waker_ref())),
        Poll::Pending
    ));
    assert_eq!(
        turns.get(),
        1,
        "wide row yielded after a bounded cell batch"
    );
    source.cancel(CancellationReason::User);
    let Poll::Ready(Err(error)) = scan
        .as_mut()
        .poll(&mut Context::from_waker(futures::task::noop_waker_ref()))
    else {
        panic!("cancelled scan must not finish the row")
    };
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert!(dropped.get());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn selected_numeric_cells_yield_inside_one_wide_row() {
    let _lock = runmat_filesystem::provider_override_lock();
    let provider = Arc::new(MemoryFsProvider::with_current_dir("/xlsread-output"));
    let _restore = runmat_filesystem::replace_provider(provider);
    let selected = TabularFsOwner::capture(Some(Path::new("/xlsread-output/input.xlsx")))
        .expect("captured owner")
        .path(PathBuf::from("/xlsread-output/input.xlsx"))
        .expect("selected path");
    let source = CancellationSource::new();
    let runtime =
        RuntimeContext::with_cancellation(Rc::new(RuntimeExecutionService::new()), source.token());
    let turns = Rc::new(Cell::new(0usize));
    let dropped = Rc::new(Cell::new(false));
    let budget = pending_budget(&source, &turns, &dropped, 0);
    let mut cooperation = TabularCooperation::with_test_budget(&runtime, BUILTIN_NAME, budget);
    let mut range = calamine::Range::<SpreadsheetData>::new((0, 0), (0, 16_383));
    range.set_value((0, 0), SpreadsheetData::Float(1.0));
    let selection = super::range::RangeSpec {
        start_row: 0,
        start_col: 0,
        end_row: Some(0),
        end_col: Some(16_383),
    };
    let mut call = Box::pin(super::range::selected_cells(
        &range,
        Some(selection),
        &mut cooperation,
        &selected,
    ));
    assert!(matches!(
        call.as_mut()
            .poll(&mut Context::from_waker(futures::task::noop_waker_ref())),
        Poll::Pending
    ));
    assert_eq!(turns.get(), 1, "selection yielded after 1024 cells");
    source.cancel(CancellationReason::User);
    let Poll::Ready(Err(error)) = call
        .as_mut()
        .poll(&mut Context::from_waker(futures::task::noop_waker_ref()))
    else {
        panic!("cancelled selection must not complete the wide row")
    };
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert!(dropped.get());
}
