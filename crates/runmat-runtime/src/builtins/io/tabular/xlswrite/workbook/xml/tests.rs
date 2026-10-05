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
fn cooperative_xml_preserves_the_shared_renderer_bytes() {
    let _lock = runmat_filesystem::provider_override_lock();
    let provider = Arc::new(MemoryFsProvider::with_current_dir("/xlswrite-xml"));
    let _restore = runmat_filesystem::replace_provider(provider);
    let selected = TabularFsOwner::capture(Some(Path::new("/xlswrite-xml/out.xlsx")))
        .expect("captured owner")
        .path("/xlswrite-xml/out.xlsx".into())
        .expect("selected path");
    let table = CellTable::from_cells(
        1,
        3,
        vec![
            CellValue::Text(format!("<{}&>", "a".repeat(8192))),
            CellValue::Integer(IntValue::U64(u64::MAX)),
            CellValue::Boolean(true),
        ],
    )
    .expect("cell table");
    let expected = writecell::build_sheet_xml(&table, RangeStart::default());
    let mut cooperation = TabularCooperation::new(None, BUILTIN_NAME);
    let actual = futures::executor::block_on(build_sheet_xml_cooperative(
        &table,
        &mut cooperation,
        &selected,
    ))
    .expect("cooperative XML");
    assert_eq!(actual, expected);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn long_single_cell_xml_yields_and_cancel_drops_pending_turn() {
    let _lock = runmat_filesystem::provider_override_lock();
    let provider = Arc::new(MemoryFsProvider::with_current_dir("/xlswrite-xml"));
    let _restore = runmat_filesystem::replace_provider(provider);
    let selected = TabularFsOwner::capture(Some(Path::new("/xlswrite-xml/out.xlsx")))
        .expect("captured owner")
        .path("/xlswrite-xml/out.xlsx".into())
        .expect("selected path");
    let table =
        CellTable::from_cells(1, 1, vec![CellValue::Text("x".repeat(8192))]).expect("cell table");
    let source = CancellationSource::new();
    let runtime =
        RuntimeContext::with_cancellation(Rc::new(RuntimeExecutionService::new()), source.token());
    let turns = Rc::new(Cell::new(0usize));
    let dropped = Rc::new(Cell::new(false));
    let turn_count = turns.clone();
    let turn_dropped = dropped.clone();
    let budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).expect("nonzero")),
        source.token(),
        move || -> crate::context::RuntimeServiceFuture<Result<(), ExecutionServiceError>> {
            turn_count.set(turn_count.get() + 1);
            Box::pin(PendingTurn(turn_dropped.clone()))
        },
    );
    let mut cooperation = TabularCooperation::with_test_budget(&runtime, BUILTIN_NAME, budget);
    let mut call = Box::pin(build_sheet_xml_cooperative(
        &table,
        &mut cooperation,
        &selected,
    ));
    assert!(matches!(
        call.as_mut()
            .poll(&mut Context::from_waker(futures::task::noop_waker_ref())),
        Poll::Pending
    ));
    assert_eq!(turns.get(), 1, "XML yielded within a single text cell");
    source.cancel(CancellationReason::User);
    let Poll::Ready(Err(error)) = call
        .as_mut()
        .poll(&mut Context::from_waker(futures::task::noop_waker_ref()))
    else {
        panic!("cancelled XML projection must not return bytes")
    };
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert!(dropped.get());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn tall_empty_sheet_yields_before_rendering_all_rows() {
    let _lock = runmat_filesystem::provider_override_lock();
    let provider = Arc::new(MemoryFsProvider::with_current_dir("/xlswrite-xml"));
    let _restore = runmat_filesystem::replace_provider(provider);
    let selected = TabularFsOwner::capture(Some(Path::new("/xlswrite-xml/out.xlsx")))
        .expect("captured owner")
        .path("/xlswrite-xml/out.xlsx".into())
        .expect("selected path");
    let table = CellTable::from_cells(1_000_000, 0, Vec::new()).expect("empty-column table");
    let source = CancellationSource::new();
    let runtime =
        RuntimeContext::with_cancellation(Rc::new(RuntimeExecutionService::new()), source.token());
    let turns = Rc::new(Cell::new(0usize));
    let dropped = Rc::new(Cell::new(false));
    let turn_count = turns.clone();
    let turn_dropped = dropped.clone();
    let budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).expect("nonzero")),
        source.token(),
        move || -> crate::context::RuntimeServiceFuture<Result<(), ExecutionServiceError>> {
            turn_count.set(turn_count.get() + 1);
            Box::pin(PendingTurn(turn_dropped.clone()))
        },
    );
    let mut cooperation = TabularCooperation::with_test_budget(&runtime, BUILTIN_NAME, budget);
    let mut call = Box::pin(build_sheet_xml_cooperative(
        &table,
        &mut cooperation,
        &selected,
    ));
    assert!(matches!(
        call.as_mut()
            .poll(&mut Context::from_waker(futures::task::noop_waker_ref())),
        Poll::Pending
    ));
    assert_eq!(turns.get(), 1, "empty rows yielded after 1024 tags");
    source.cancel(CancellationReason::User);
    let Poll::Ready(Err(error)) = call
        .as_mut()
        .poll(&mut Context::from_waker(futures::task::noop_waker_ref()))
    else {
        panic!("cancelled row projection must not return XML")
    };
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert!(dropped.get());
}
