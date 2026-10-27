use crate::context::RuntimeContext;
use crate::execution::{ExecutionServiceError, RuntimeExecutionService};
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::CancellationReason;
use std::cell::Cell;
use std::rc::Rc;
use wasm_bindgen_test::wasm_bindgen_test;

struct Retained(Rc<Cell<usize>>);
impl Drop for Retained {
    fn drop(&mut self) {
        self.0.set(self.0.get() + 1);
    }
}

#[wasm_bindgen_test(async)]
async fn owned_cpu_browser_work_keeps_local_payload_and_retained_owner() {
    let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let owner = Rc::new(Cell::new(0));
    let drops = Rc::new(Cell::new(0));
    let input = Rc::clone(&owner);
    let during_work = Rc::clone(&drops);
    let output = runtime
        .cpu_cooperation()
        .run_owned_blocking_in_context(&runtime, None, Retained(Rc::clone(&drops)), move || {
            assert_eq!(during_work.get(), 0);
            input.set(42);
            input
        })
        .await
        .unwrap();
    assert!(Rc::ptr_eq(&owner, &output));
    assert_eq!(owner.get(), 42);
    assert_eq!(drops.get(), 1);
}

#[wasm_bindgen_test(async)]
async fn owned_cpu_browser_cancellation_retires_retained_owner_after_kernel_exit() {
    let cancellation = CancellationSource::new();
    let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
        .with_cancellation_token(cancellation.token());
    let drops = Rc::new(Cell::new(0));
    let during_work = Rc::clone(&drops);
    let result = runtime
        .cpu_cooperation()
        .run_owned_blocking_in_context(&runtime, None, Retained(Rc::clone(&drops)), move || {
            assert_eq!(during_work.get(), 0);
            cancellation.cancel(CancellationReason::User);
            Rc::new(())
        })
        .await;
    assert!(matches!(result, Err(ExecutionServiceError::Cancelled)));
    assert_eq!(drops.get(), 1);
}
