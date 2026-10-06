use super::*;
use runmat_execution::{CancellationReason, ExecutionScopeId, FutureHandle, TaskHandle};
use runmat_runtime::execution::{
    AwaitAction, DeferredCall, ExecutionServiceError, RuntimeExecutionService,
    RuntimeExecutionServices,
};
use runmat_value::Value;

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

#[tokio::test]
async fn deferred_capture_preserves_original_complete_product_in_both_executors() {
    for native in [false, true] {
        let mut session = RunMatSession::with_options(false, false).unwrap();
        let service = Rc::new(CapturingService::default());
        session.install_execution_services(service.clone());
        let unit = session.compile_executable_unit(ExecutableSource::new(
            "original-deferred-owner", "entry.m",
            "function f = entry(x); f = asyncPass(x); end\nasync function y = asyncPass(x); y = x + 7; end\n"), None).await.unwrap();
        let mut control = crate::InvocationControl::default();
        if native {
            control = control.force_generic_native();
        }
        let result = session
            .invoke_executable(
                &unit,
                crate::ProcedureInvocation {
                    target: crate::ProcedureTarget::Function("entry".into()),
                    arguments: vec![Value::Num(5.0)],
                    requested_outputs: 1,
                },
                &control,
            )
            .await
            .unwrap();
        assert!(matches!(result, Value::Future(_)));
        let call = service
            .original
            .borrow_mut()
            .take()
            .expect("original deferred call");
        assert_eq!(call.arguments, vec![Value::Num(5.0)]);
        assert_eq!(call.requested_outputs, 1);
        assert_eq!(
            call.program_revision.as_ref(),
            Some(&unit.revision().program_revision)
        );
        let captured = runmat_execution::ExecutableUnitEnvelope::from_canonical_bytes(
            call.program.as_deref().expect("original captured program"),
        )
        .unwrap_or_else(|error| {
            panic!("executor native={native} lost complete original product: {error}")
        });
        let function = ProgramFunctionId(u32::try_from(call.function).unwrap());
        assert_eq!(
            captured,
            unit.portable_envelope_for_function(function).unwrap()
        );
        let outputs = runmat_execution::OutputContract {
            requested_outputs: 1,
        };
        let target = runmat_execution_artifact::ProgramTarget::portable("original-deferred-v4");
        let (recipe, artifact, arguments) =
            runmat_vm::materialize_deferred_call(&call, outputs.clone(), target.clone()).unwrap();
        assert_eq!(recipe.program_revision, unit.revision().program_revision);
        assert_eq!(
            artifact.form,
            runmat_execution_artifact::ExecutableForm::ExecutableUnitV4
        );
        assert_eq!(artifact.executable_unit().unwrap().unwrap(), captured);
        assert_eq!(arguments.len(), 1);
        for mutation in 0..3 {
            let mut invalid = call.clone();
            match mutation {
                0 => invalid.program_revision = None,
                1 => invalid.function = usize::MAX,
                _ => invalid.requested_outputs = 2,
            }
            assert!(runmat_vm::materialize_deferred_call(
                &invalid,
                outputs.clone(),
                target.clone()
            )
            .is_err());
        }
        let restored = ExecutableUnit::from_portable_envelope(&captured).unwrap();
        assert_eq!(restored.source().text, unit.source().text);
        assert_eq!(
            restored.source_map().entries().len(),
            unit.source_map().entries().len()
        );
        for (cold, original) in restored
            .source_map()
            .entries()
            .iter()
            .zip(unit.source_map().entries())
        {
            assert_eq!(cold.source_id, original.source_id);
            assert_eq!(cold.owner_identity, original.owner_identity);
            assert_eq!(cold.relative_path, original.relative_path);
            assert_eq!(cold.display_name, original.display_name);
            assert_eq!(cold.text, original.text);
            assert!(
                cold.full_path.is_none(),
                "portable code must not authorize producer filesystem paths"
            );
        }
        assert_eq!(
            restored.function_analysis(function),
            unit.function_analysis(function)
        );
    }
}

#[tokio::test]
async fn raw_script_capture_has_original_product_and_restores_session_context() {
    let mut session = RunMatSession::with_options(false, false).unwrap();
    let service = Rc::new(CapturingService::default());
    session.install_execution_services(service.clone());
    let original_revision = session.runtime_context.program_revision().cloned();
    let source = "f = rawAsync(5);\nasync function y = rawAsync(x); y = x + 7; end\n";
    let request = crate::abi::ExecutionRequest::for_source(
        crate::abi::SourceInput::Text {
            name: "raw_original.m".into(),
            text: source.into(),
        },
        runmat_parser::CompatMode::RunMat,
        crate::abi::HostExecutionPolicy::default(),
        session.abi_workspace_handle,
    );
    let response = session.execute_request(request).await;
    response
        .result
        .expect("raw script must capture through its original program owner");
    assert_eq!(
        session.runtime_context.program_revision().cloned(),
        original_revision
    );
    assert!(session.program_service.active.borrow().is_empty());
    let call = service
        .original
        .borrow_mut()
        .take()
        .expect("raw deferred call");
    assert_eq!(call.arguments, vec![Value::Num(5.0)]);
    assert_eq!(call.requested_outputs, 1);
    let envelope = runmat_execution::ExecutableUnitEnvelope::from_canonical_bytes(
        call.program.as_deref().expect("complete raw product"),
    )
    .unwrap();
    assert_eq!(
        call.program_revision.as_ref(),
        Some(&envelope.manifest.identity.program)
    );
    let restored = ExecutableUnit::from_portable_envelope(&envelope).unwrap();
    assert_eq!(restored.source().text, source);
    let function = ProgramFunctionId(u32::try_from(call.function).unwrap());
    assert!(restored.function_analysis(function).is_some());
    let (_, artifact, _) = runmat_vm::materialize_deferred_call(
        &call,
        runmat_execution::OutputContract {
            requested_outputs: 1,
        },
        runmat_execution_artifact::ProgramTarget::portable("raw-original"),
    )
    .unwrap();
    assert_eq!(artifact.executable_unit().unwrap().unwrap(), envelope);
}

mod persistent;
