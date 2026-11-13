#![cfg(target_arch = "wasm32")]
use runmat_execution::{
    CancellationReason, ExecutionHandleSnapshot, ExecutionScopeId, FutureHandle, PoolHandle,
    PoolRequest, PoolSnapshot, TaskHandle, TaskResultClaim,
};
use runmat_runtime::execution::{
    AwaitAction, DeferredCall, ExecutionServiceError, RootedValueSequence, RuntimeExecutionService,
    RuntimeExecutionServices,
};
use runmat_value::Value;
use std::{cell::RefCell, rc::Rc};
use wasm_bindgen_test::wasm_bindgen_test;
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

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
    fn visit_retained_values(&self, visitor: &mut dyn FnMut(&Value) -> Option<()>) -> Option<()> {
        self.serial.visit_retained_values(visitor)
    }
    fn current_pool(&self) -> Result<Option<PoolSnapshot>, ExecutionServiceError> {
        self.serial.current_pool()
    }
    fn ensure_pool(&self, request: PoolRequest) -> Result<PoolSnapshot, ExecutionServiceError> {
        self.serial.ensure_pool(request)
    }
    fn close_pool(&self, pool: &PoolHandle) -> Result<(), ExecutionServiceError> {
        self.serial.close_pool(pool)
    }
    fn create_future(&self, call: DeferredCall) -> Result<FutureHandle, ExecutionServiceError> {
        *self.original.borrow_mut() = Some(call.clone());
        self.serial.create_future(call)
    }
    fn spawn(&self, future: &FutureHandle) -> Result<TaskHandle, ExecutionServiceError> {
        self.serial.spawn(future)
    }
    fn inspect_future(
        &self,
        future: &FutureHandle,
    ) -> Result<ExecutionHandleSnapshot, ExecutionServiceError> {
        self.serial.inspect_future(future)
    }
    fn inspect_task(
        &self,
        task: &TaskHandle,
    ) -> Result<ExecutionHandleSnapshot, ExecutionServiceError> {
        self.serial.inspect_task(task)
    }
    fn claim_next_task_result(
        &self,
        tasks: &[runmat_value::SharedTaskHandle],
    ) -> Result<TaskResultClaim, ExecutionServiceError> {
        self.serial.claim_next_task_result(tasks)
    }
    fn mark_task_result_read(&self, task: &TaskHandle) -> Result<(), ExecutionServiceError> {
        self.serial.mark_task_result_read(task)
    }
    fn begin_await(&self, value: Value) -> Result<AwaitAction, ExecutionServiceError> {
        self.serial.begin_await(value)
    }
    fn complete_future(
        &self,
        future: runmat_runtime::execution::FutureCompletionRef<'_>,
        result: Result<RootedValueSequence, ExecutionServiceError>,
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

fn captured_request(
    call: &DeferredCall,
) -> (
    runmat_execution::ProgramCallable,
    runmat_execution_artifact::GraphExecutionRequest,
) {
    let (callable, request, _) = runmat_vm::materialize_deferred_graph_call(
        call,
        runmat_execution::OutputContract {
            requested_outputs: u16::try_from(call.invocation.requested_outputs()).unwrap(),
            owned_work: None,
        },
        runmat_execution_artifact::ProgramTarget::portable("original-wasm-capture"),
        runmat_execution::value::ValueLimits::default(),
    )
    .expect("materialize the original selected graph");
    (callable, request)
}

fn selected_callable_product(
    request: &runmat_execution_artifact::GraphExecutionRequest,
) -> (runmat_execution::FunctionBindingOrigin, &[u8]) {
    use runmat_execution::value::{GraphCallableOwner, GraphInvocationRoots, GraphNode};
    let GraphInvocationRoots::Callable { target, .. } = &request.roots else {
        panic!("expected the selected callable root");
    };
    let GraphNode::Callable(callable) = &request.graph.nodes[target.0 as usize] else {
        panic!("expected the selected callable node");
    };
    let GraphCallableOwner::Bound(owner) = callable.owner else {
        panic!("expected an immutable selected definition");
    };
    let bytes = match usize::from(owner.unit.0) {
        0 => request.artifact.executable_bytes.as_slice(),
        index => request.retained_units[index - 1].as_slice(),
    };
    (owner.origin, bytes)
}

#[wasm_bindgen_test]
async fn raw_wasm_script_captures_complete_original_program() {
    let _provider = runmat_filesystem::replace_provider(std::sync::Arc::new(
        runmat_filesystem::MemoryFsProvider::new(),
    ));
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
    assert_eq!(call.invocation.arguments(), [Value::Num(5.0)]);
    assert_eq!(call.invocation.requested_outputs(), 1);
    let (callable, request) = captured_request(&call);
    let envelope = request.artifact.executable_unit().unwrap().unwrap();
    let restored = runmat_core::ExecutableUnit::from_portable_envelope(&envelope).unwrap();
    assert_eq!(restored.source().text, source);
    let runmat_execution::value::GraphInvocationRoots::Callable { target, .. } = &request.roots
    else {
        panic!("expected the original callable root");
    };
    let runmat_execution::value::GraphNode::Callable(target) =
        &request.graph.nodes[target.0 as usize]
    else {
        panic!("expected selected callable metadata");
    };
    let runmat_execution::value::GraphCallableOwner::Bound(owner) = target.owner else {
        panic!("expected the original immutable function owner");
    };
    assert_eq!(
        owner.origin.program,
        envelope
            .manifest
            .identity
            .program
            .identity_digest()
            .unwrap()
    );
    assert!(restored.function_analysis(owner.origin.function).is_some());

    // A later caller must capture the earlier declaration's complete product,
    // even when its own local function ordinal collides with the retained ID.
    let later = runmat_core::abi::ExecutionRequest::for_source(
        runmat_core::abi::SourceInput::Text {
            name: "later_caller.m".into(),
            text: "f = rawAsync(5);\nfunction y = unrelated(x); y = x + 90; end".into(),
        },
        Default::default(),
        runmat_core::abi::HostExecutionPolicy::default(),
        runmat_core::abi::WorkspaceHandle(uuid::Uuid::nil()),
    );
    session
        .execute_request(later)
        .await
        .result
        .expect("retained WASM capture");
    let retained = service
        .original
        .borrow_mut()
        .take()
        .expect("original retained call");
    let (retained_callable, retained_request) = captured_request(&retained);
    assert_eq!(retained_callable, callable);
    // The caller product changes; the unit-qualified callable must still
    // select the exact original definition and complete canonical product.
    assert_eq!(
        selected_callable_product(&retained_request),
        selected_callable_product(&request)
    );
    assert_eq!(retained.invocation.arguments(), call.invocation.arguments());
    assert_eq!(
        retained.invocation.requested_outputs(),
        call.invocation.requested_outputs()
    );
    use serde::Serialize as _;
    let wire = retained_request
        .serialize(
            &serde_wasm_bindgen::Serializer::new().serialize_large_number_types_as_bigints(true),
        )
        .unwrap();
    let response = runmat_wasm::execute_worker_program(wire).await.unwrap();
    let response: runmat_execution_artifact::ProgramExecutionResponse =
        serde_wasm_bindgen::from_value(response).unwrap();
    let runmat_execution_artifact::ProgramExecutionResponse::GraphSuccess { graph } = response
    else {
        panic!("retained original callable failed: {response:?}");
    };
    assert_eq!(graph.roots.len(), 1);
    assert_eq!(
        graph.nodes[graph.roots[0].0 as usize],
        runmat_execution::value::GraphNode::Leaf {
            value: runmat_execution::value::ValuePayload::Inline(Box::new(
                runmat_execution::value::InlineValue::F64Bits(12.0_f64.to_bits()),
            )),
        }
    );
}

#[wasm_bindgen_test]
async fn original_transitive_products_survive_wasm_producer_teardown() {
    let _provider = runmat_filesystem::replace_provider(std::sync::Arc::new(
        runmat_filesystem::MemoryFsProvider::new(),
    ));
    runmat_runtime::builtins::wasm_registry::register_all();
    let mut producer = runmat_core::RunMatSession::with_options(false, false).unwrap();
    let service = Rc::new(CapturingService::default());
    producer.install_execution_services(service.clone());
    for (name, source) in [
        ("leaf.m", "function y = originalLeaf(x); y = x + 7; end"),
        (
            "middle.m",
            "function y = originalMiddle(x); y = originalLeaf(x) * 2; end",
        ),
        (
            "root.m",
            "async function y = retainedAsync(x); y = originalMiddle(x) + 3; end",
        ),
        ("caller.m", "f = retainedAsync(5);"),
    ] {
        let request = runmat_core::abi::ExecutionRequest::for_source(
            runmat_core::abi::SourceInput::Text {
                name: name.into(),
                text: source.into(),
            },
            Default::default(),
            runmat_core::abi::HostExecutionPolicy::default(),
            runmat_core::abi::WorkspaceHandle(uuid::Uuid::nil()),
        );
        producer
            .execute_request(request)
            .await
            .result
            .expect("original declaration/capture");
    }
    let call = service.original.borrow_mut().take().unwrap();
    // Selection happens at task acceptance while the issuing session exists;
    // the accepted request, not a live session, then owns the worker products.
    let (_, request) = captured_request(&call);
    drop(call);
    producer
        .execute_request(runmat_core::abi::ExecutionRequest::for_source(
            runmat_core::abi::SourceInput::Text {
                name: "replacement.m".into(),
                text: "function y = originalLeaf(x); y = x + 100; end".into(),
            },
            Default::default(),
            runmat_core::abi::HostExecutionPolicy::default(),
            runmat_core::abi::WorkspaceHandle(uuid::Uuid::nil()),
        ))
        .await
        .result
        .expect("later publication must not replace the accepted target");
    drop(producer);
    drop(service);
    let retained_entries: Vec<_> = request
        .retained_units
        .iter()
        .map(|bytes| {
            runmat_execution::ExecutableUnitEnvelope::from_canonical_bytes(bytes)
                .expect("original retained executable product")
                .manifest
                .identity
                .entrypoint
        })
        .collect();
    assert!(request.static_dependencies.len() >= 2);
    assert!(
        request.static_dependencies.iter().any(|edge| {
            let caller = match edge.caller {
                runmat_execution::value::GraphStaticCaller::Script(unit) => unit,
                runmat_execution::value::GraphStaticCaller::Function(owner) => owner.unit,
            };
            caller != edge.target.unit
        }),
        "the cold control must cross separately admitted wire units"
    );
    for original in ["originalLeaf", "originalMiddle"] {
        assert_eq!(
            retained_entries
                .iter()
                .filter(|entry| entry.as_str() == original)
                .count(),
            1,
            "retain each original transitive function product exactly once"
        );
    }
    // The public worker boundary admits the retained products after producer
    // teardown. This invocation does not claim physical worker isolation.
    use serde::Serialize as _;
    let mut incomplete = request.clone();
    incomplete
        .static_dependencies
        .pop()
        .expect("transitive static site");
    let wire = incomplete
        .serialize(
            &serde_wasm_bindgen::Serializer::new().serialize_large_number_types_as_bigints(true),
        )
        .unwrap();
    if let Ok(response) = runmat_wasm::execute_worker_program(wire).await {
        let response: runmat_execution_artifact::ProgramExecutionResponse =
            serde_wasm_bindgen::from_value(response).unwrap();
        assert!(
            matches!(
                &response,
                runmat_execution_artifact::ProgramExecutionResponse::Failure { .. }
                    | runmat_execution_artifact::ProgramExecutionResponse::RuntimeFailure { .. }
            ),
            "missing transitive site must be rejected before execution: {response:?}"
        );
    }
    let wire = request
        .serialize(
            &serde_wasm_bindgen::Serializer::new().serialize_large_number_types_as_bigints(true),
        )
        .unwrap();
    let response = runmat_wasm::execute_worker_program(wire).await.unwrap();
    let response: runmat_execution_artifact::ProgramExecutionResponse =
        serde_wasm_bindgen::from_value(response).unwrap();
    let runmat_execution_artifact::ProgramExecutionResponse::GraphSuccess { graph } = response
    else {
        panic!("cold graph execution failed: {response:?}");
    };
    assert_eq!(graph.roots.len(), 1);
    assert_eq!(
        graph.nodes[graph.roots[0].0 as usize],
        runmat_execution::value::GraphNode::Leaf {
            value: runmat_execution::value::ValuePayload::Inline(Box::new(
                runmat_execution::value::InlineValue::F64Bits(27.0_f64.to_bits())
            )),
        }
    );
}

#[wasm_bindgen_test]
async fn cold_wasm_program_uses_session_environment_without_cross_session_mutation() {
    let _provider = runmat_filesystem::replace_provider(std::sync::Arc::new(
        runmat_filesystem::MemoryFsProvider::new(),
    ));
    runmat_runtime::builtins::wasm_registry::register_all();
    let mut producer = runmat_core::RunMatSession::with_options(false, false).unwrap();
    let unit = producer
        .compile_executable_unit(
            runmat_core::ExecutableSource::frozen(
                "environment-wasm",
                "environment.m",
                "environment.m",
                "setenv('RUNMAT_WASM_MODE', 'changed'); ans = getenv('RUNMAT_WASM_MODE');",
            ),
            None,
        )
        .await
        .unwrap();
    let bytes = unit.portable_envelope().unwrap().canonical_bytes().unwrap();
    drop(producer);
    drop(unit);
    let envelope = runmat_execution::ExecutableUnitEnvelope::from_canonical_bytes(&bytes).unwrap();
    let mut first = runmat_core::RunMatSession::with_options(false, false).unwrap();
    let second = runmat_core::RunMatSession::with_options(false, false).unwrap();
    first
        .install_environment_snapshot(vec![("RUNMAT_WASM_MODE".into(), "first".into())])
        .unwrap();
    let mut second = second;
    second
        .install_environment_snapshot(vec![("RUNMAT_WASM_MODE".into(), "second".into())])
        .unwrap();
    let result = first
        .invoke_portable_unit(&envelope, Vec::new(), 1, &Default::default())
        .await
        .unwrap();
    assert_eq!(
        result.as_slice(),
        [Value::CharArray(runmat_value::CharArray::new_row(
            "changed"
        ))]
    );
    assert_eq!(
        second.environment_snapshot(),
        [("RUNMAT_WASM_MODE".into(), "second".into())]
    );
}
