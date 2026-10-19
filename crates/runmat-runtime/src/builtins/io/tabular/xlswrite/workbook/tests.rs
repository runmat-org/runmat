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
use std::path::Path;
use std::pin::Pin;
use std::rc::Rc;
use std::sync::Arc;
use std::task::{Context, Poll};

#[path = "tests/revocation.rs"]
mod revocation;

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

fn pending_budget(source: &CancellationSource, dropped: Rc<Cell<bool>>) -> CpuCooperationBudget {
    CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).expect("nonzero")),
        source.token(),
        move || -> crate::context::RuntimeServiceFuture<Result<(), ExecutionServiceError>> {
            Box::pin(PendingTurn(dropped.clone()))
        },
    )
}

async fn table(text: &str, selected: &TabularFsPath) -> XlsTable {
    let mut cooperation = TabularCooperation::new(None, BUILTIN_NAME);
    XlsTable::from_value_cooperative(Value::from(text), None, &mut cooperation, selected)
        .await
        .expect("table")
}

async fn seed_workbook(workbook: &mut XlsWorkbook, selected: &TabularFsPath) -> BuiltinResult<()> {
    let mut cooperation = TabularCooperation::new(None, BUILTIN_NAME);
    workbook
        .overlay(
            &XlsWriteRequest::default(),
            table("old", selected).await,
            &mut cooperation,
            selected,
        )
        .await
}

async fn recovered_workbook(
    workbook: &mut XlsWorkbook,
    selected: &TabularFsPath,
) -> BuiltinResult<()> {
    let mut cooperation = TabularCooperation::new(None, BUILTIN_NAME);
    workbook
        .overlay(
            &XlsWriteRequest::default(),
            table("recovered", selected).await,
            &mut cooperation,
            selected,
        )
        .await?;
    let previous = std::mem::replace(&mut workbook.sheets[0].table, XlsTable::empty());
    let cells = previous.into_cell_table()?;
    assert_eq!((cells.rows, cells.cols), (1, 1));
    assert_eq!(cells.get(0, 0), &CellValue::Text("recovered".into()));
    Ok(())
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn dropped_pending_resize_keeps_retained_workbook_valid() {
    let _lock = runmat_filesystem::provider_override_lock();
    let provider = Arc::new(MemoryFsProvider::with_current_dir("/xlswrite-overlay"));
    let _restore = runmat_filesystem::replace_provider(provider);
    let selected = TabularFsOwner::capture(Some(Path::new("/xlswrite-overlay/out.xlsx")))
        .expect("captured owner")
        .path("/xlswrite-overlay/out.xlsx".into())
        .expect("selected path");
    let mut workbook = XlsWorkbook::new();
    futures::executor::block_on(seed_workbook(&mut workbook, &selected))
        .expect("initial valid sheet");
    let source = CancellationSource::new();
    let runtime =
        RuntimeContext::with_cancellation(Rc::new(RuntimeExecutionService::new()), source.token());
    let dropped = Rc::new(Cell::new(false));
    let budget = pending_budget(&source, dropped.clone());
    let mut cooperation = TabularCooperation::with_test_budget(&runtime, BUILTIN_NAME, budget);
    let request = XlsWriteRequest {
        sheet: SheetSelector::Default,
        range: RangeStart { row: 1, col: 1 },
    };
    let incoming = futures::executor::block_on(table("new", &selected));
    let mut call = Box::pin(workbook.overlay(&request, incoming, &mut cooperation, &selected));
    assert!(matches!(
        call.as_mut()
            .poll(&mut Context::from_waker(futures::task::noop_waker_ref())),
        Poll::Pending
    ));
    drop(call);
    assert!(dropped.get());
    futures::executor::block_on(recovered_workbook(&mut workbook, &selected))
        .expect("retained workbook accepts a later overlay");
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn cancelled_resize_keeps_retained_workbook_valid() {
    let _lock = runmat_filesystem::provider_override_lock();
    let provider = Arc::new(MemoryFsProvider::with_current_dir("/xlswrite-overlay"));
    let _restore = runmat_filesystem::replace_provider(provider);
    let selected = TabularFsOwner::capture(Some(Path::new("/xlswrite-overlay/out.xlsx")))
        .expect("captured owner")
        .path("/xlswrite-overlay/out.xlsx".into())
        .expect("selected path");
    let mut workbook = XlsWorkbook::new();
    futures::executor::block_on(seed_workbook(&mut workbook, &selected))
        .expect("initial valid sheet");
    let source = CancellationSource::new();
    let runtime =
        RuntimeContext::with_cancellation(Rc::new(RuntimeExecutionService::new()), source.token());
    let dropped = Rc::new(Cell::new(false));
    let budget = pending_budget(&source, dropped.clone());
    let mut cooperation = TabularCooperation::with_test_budget(&runtime, BUILTIN_NAME, budget);
    let request = XlsWriteRequest {
        sheet: SheetSelector::Default,
        range: RangeStart { row: 1, col: 1 },
    };
    let incoming = futures::executor::block_on(table("new", &selected));
    let mut call = Box::pin(workbook.overlay(&request, incoming, &mut cooperation, &selected));
    assert!(matches!(
        call.as_mut()
            .poll(&mut Context::from_waker(futures::task::noop_waker_ref())),
        Poll::Pending
    ));
    source.cancel(CancellationReason::User);
    let Poll::Ready(Err(error)) = call
        .as_mut()
        .poll(&mut Context::from_waker(futures::task::noop_waker_ref()))
    else {
        panic!("cancelled overlay must not return a workbook")
    };
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    drop(call);
    assert!(dropped.get());
    futures::executor::block_on(recovered_workbook(&mut workbook, &selected))
        .expect("retained workbook remains valid after cancellation");
}
