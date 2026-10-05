use super::*;
use crate::context::RuntimeContext;
use crate::execution::{
    CpuCooperationBudget, CpuCooperationPolicy, ExecutionServiceError, RuntimeExecutionService,
};
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

fn selected() -> TabularFsPath {
    TabularFsOwner::capture(Some(Path::new("/xlswrite-input/input.xlsx")))
        .expect("captured owner")
        .path("/xlswrite-input/input.xlsx".into())
        .expect("selected path")
}

fn pending_budget(
    source: &CancellationSource,
    dropped: Rc<Cell<bool>>,
    allowed_turns: usize,
    turns: Rc<Cell<usize>>,
) -> CpuCooperationBudget {
    CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).expect("nonzero")),
        source.token(),
        move || -> crate::context::RuntimeServiceFuture<Result<(), ExecutionServiceError>> {
            let next = turns.get() + 1;
            turns.set(next);
            if next <= allowed_turns {
                Box::pin(async { Ok(()) })
            } else {
                Box::pin(PendingTurn(dropped.clone()))
            }
        },
    )
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn wide_numeric_input_yields_and_cancels_before_table_publication() {
    let _lock = runmat_filesystem::provider_override_lock();
    let _restore = runmat_filesystem::replace_provider(Arc::new(
        MemoryFsProvider::with_current_dir("/xlswrite-input"),
    ));
    let selected = selected();
    let source = CancellationSource::new();
    let runtime =
        RuntimeContext::with_cancellation(Rc::new(RuntimeExecutionService::new()), source.token());
    let dropped = Rc::new(Cell::new(false));
    let turns = Rc::new(Cell::new(0));
    let budget = pending_budget(&source, dropped.clone(), 0, turns.clone());
    let mut cooperation = TabularCooperation::with_test_budget(&runtime, BUILTIN_NAME, budget);
    let tensor = Tensor::new(vec![7.0; 4096], vec![1, 4096]).expect("wide tensor");
    let mut call = Box::pin(XlsTable::from_value_cooperative(
        Value::Tensor(tensor),
        Some(&runtime),
        &mut cooperation,
        &selected,
    ));
    assert!(matches!(
        call.as_mut()
            .poll(&mut Context::from_waker(futures::task::noop_waker_ref())),
        Poll::Pending
    ));
    assert_eq!(turns.get(), 1, "wide tensor yielded after bounded cells");
    source.cancel(CancellationReason::User);
    let Poll::Ready(Err(error)) = call
        .as_mut()
        .poll(&mut Context::from_waker(futures::task::noop_waker_ref()))
    else {
        panic!("cancelled numeric projection must not return a table")
    };
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    drop(call);
    assert!(dropped.get());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn wide_existing_sheet_projection_yields_before_table_publication() {
    let _lock = runmat_filesystem::provider_override_lock();
    let _restore = runmat_filesystem::replace_provider(Arc::new(
        MemoryFsProvider::with_current_dir("/xlswrite-input"),
    ));
    let selected = selected();
    let source = CancellationSource::new();
    let runtime =
        RuntimeContext::with_cancellation(Rc::new(RuntimeExecutionService::new()), source.token());
    let dropped = Rc::new(Cell::new(false));
    let turns = Rc::new(Cell::new(0));
    let budget = pending_budget(&source, dropped.clone(), 16, turns.clone());
    let mut cooperation = TabularCooperation::with_test_budget(&runtime, BUILTIN_NAME, budget);
    let mut range = calamine::Range::<SpreadsheetData>::new((0, 0), (0, 16_383));
    range.set_value((0, 0), SpreadsheetData::String("Ω".repeat(8192)));
    let mut call = Box::pin(XlsTable::from_calamine_range_cooperative(
        &range,
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
        "borrowed text copy yielded after cell fill"
    );
    drop(call);
    assert!(dropped.get());
}
