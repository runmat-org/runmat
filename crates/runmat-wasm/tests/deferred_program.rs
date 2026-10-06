#![cfg(target_arch = "wasm32")]
use runmat_execution::{CancellationReason, ExecutionScopeId, FutureHandle, TaskHandle};
use runmat_runtime::execution::{
    AwaitAction, DeferredCall, ExecutionServiceError, RuntimeExecutionService,
    RuntimeExecutionServices,
};
use runmat_value::Value;
use std::{cell::RefCell, rc::Rc};
use wasm_bindgen_test::wasm_bindgen_test;
#[derive(Default)]
struct CapturingService {
    serial: RuntimeExecutionService,
    original: RefCell<Option<DeferredCall>>,
}

impl RuntimeExecutionServices for CapturingService {
    fn scope_id(&self) -> ExecutionScopeId {
        self.serial.scope_id()
    }
    fn requires_program_capture(&self) -> bool {
        true
    }
    fn create_future(&self, call: DeferredCall) -> Result<FutureHandle, ExecutionServiceError> {
        *self.original.borrow_mut() = Some(call.clone());
        self.serial.create_future(call)
    }
    fn spawn(&self, future: &FutureHandle) -> Result<TaskHandle, ExecutionServiceError> {
        self.serial.spawn(future)
    }
    fn begin_await(&self, value: Value) -> Result<AwaitAction, ExecutionServiceError> {
        self.serial.begin_await(value)
    }
    fn complete_future(
        &self,
        future: &FutureHandle,
        result: Result<Value, ExecutionServiceError>,
    ) -> Result<(), ExecutionServiceError> {
        self.serial.complete_future(future, result)
    }
    fn cancel(
        &self,
        value: &Value,
        reason: CancellationReason,
    ) -> Result<(), ExecutionServiceError> {
        self.serial.cancel(value, reason)
    }
    fn drain_scope(&self, reason: CancellationReason) {
        self.serial.drain_scope(reason)
    }
}

#[wasm_bindgen_test]
async fn raw_wasm_script_captures_complete_original_program() {
    runmat_runtime::builtins::wasm_registry::register_all();
    let mut session = runmat_core::RunMatSession::with_options(false, false).unwrap();
    let service = Rc::new(CapturingService::default());
    session.install_execution_services(service.clone());
    let source = "f = rawAsync(5);\nasync function y = rawAsync(x); y = x + 7; end\n";
    let request = runmat_core::abi::ExecutionRequest::for_source(
        runmat_core::abi::SourceInput::Text {
            name: "raw_original.m".into(),
            text: source.into(),
        },
        Default::default(),
        runmat_core::abi::HostExecutionPolicy::default(),
        runmat_core::abi::WorkspaceHandle(uuid::Uuid::nil()),
    );
    session
        .execute_request(request)
        .await
        .result
        .expect("raw WASM capture");
    let call = service
        .original
        .borrow_mut()
        .take()
        .expect("original raw call");
    assert_eq!(call.arguments, vec![Value::Num(5.0)]);
    assert_eq!(call.requested_outputs, 1);
    let envelope = runmat_execution::ExecutableUnitEnvelope::from_canonical_bytes(
        call.program.as_deref().expect("complete original program"),
    )
    .unwrap();
    assert_eq!(
        call.program_revision.as_ref(),
        Some(&envelope.manifest.identity.program)
    );
    let restored = runmat_core::ExecutableUnit::from_portable_envelope(&envelope).unwrap();
    assert_eq!(restored.source().text, source);
    let function = runmat_types::ProgramFunctionId(u32::try_from(call.function).unwrap());
    assert!(restored.function_analysis(function).is_some());
    let (_, artifact, _) = runmat_vm::materialize_deferred_call(
        &call,
        runmat_execution::OutputContract {
            requested_outputs: 1,
        },
        runmat_execution_artifact::ProgramTarget::portable("original-raw-wasm"),
    )
    .unwrap();
    assert_eq!(artifact.executable_unit().unwrap().unwrap(), envelope);
}
