use super::*;
use runmat_execution::{
    CancellationReason, ExecutionHandleSnapshot, ExecutionScopeId, FutureHandle, PoolHandle,
    PoolRequest, PoolSnapshot, TaskHandle, TaskResultClaim,
};
use runmat_runtime::execution::{
    AwaitAction, DeferredCall, ExecutionServiceError, RuntimeExecutionService,
    RuntimeExecutionServices,
};
use runmat_value::Value;

#[derive(Default)]
struct CapturingService {
    serial: RuntimeExecutionService,
    original: RefCell<Option<(FutureHandle, DeferredCall)>>,
    original_source: RefCell<Option<runmat_types::SourceId>>,
    program: std::rc::Weak<SessionProgramService>,
}

/// A real reserved serial task and the graph materialized from its selected
/// compiler authority. The request is transport evidence, never a fabricated
/// selected snapshot or an independently issued execution grant.
struct CapturedCall {
    call: DeferredCall,
    request: runmat_execution_artifact::GraphExecutionRequest,
    original: Option<Rc<ExecutableUnit>>,
}

impl CapturedCall {
    fn function_in(&self, unit: &ExecutableUnit) -> ProgramFunctionId {
        let portable = self.origin().function;
        let body = unit
            .functions()
            .functions
            .values()
            .find(|body| body.binding_origin == Some(self.origin()))
            .expect("selected portable function has an original compiler body");
        assert_eq!(body.portable_function, portable);
        portable
    }
    fn origin(&self) -> runmat_execution::FunctionBindingOrigin {
        use runmat_execution::value::{GraphCallableOwner, GraphInvocationRoots, GraphNode};
        let GraphInvocationRoots::Callable { target, .. } = &self.request.roots else {
            panic!("expected a selected callable graph");
        };
        let GraphNode::Callable(callable) = &self.request.graph.nodes[target.0 as usize] else {
            panic!("callable root has no callable node");
        };
        let GraphCallableOwner::Bound(owner) = callable.owner else {
            panic!("deferred procedure must retain its exact bound origin");
        };
        owner.origin
    }

    fn envelope(&self) -> runmat_execution::ExecutableUnitEnvelope {
        // A Graph unit with sealed static links is not a standalone Core
        // product. Keep that transport oracle separate from the original
        // complete compiler product used by cold-import/tamper assertions.
        if let Some(original) = &self.original {
            return original
                .portable_envelope_for_function(self.origin().function)
                .unwrap();
        }
        use runmat_execution::value::{GraphCallableOwner, GraphInvocationRoots, GraphNode};
        let GraphInvocationRoots::Callable { target, .. } = &self.request.roots else {
            unreachable!()
        };
        let GraphNode::Callable(callable) = &self.request.graph.nodes[target.0 as usize] else {
            unreachable!()
        };
        let GraphCallableOwner::Bound(owner) = callable.owner else {
            unreachable!()
        };
        let envelope = if owner.unit.0 == 0 {
            self.request.artifact.executable_unit().unwrap().unwrap()
        } else {
            runmat_execution::ExecutableUnitEnvelope::from_canonical_bytes(
                &self.request.retained_units[usize::from(owner.unit.0) - 1],
            )
            .unwrap()
        };
        // Retarget only through the original compiler product's checked
        // projection; the caller's current ordinal or display name cannot
        // select this frozen procedure.
        let unit = ExecutableUnit::from_portable_envelope(&envelope).unwrap();
        unit.portable_envelope_for_function(self.function_in(&unit))
            .unwrap()
    }
}

impl CapturingService {
    fn new(session: &RunMatSession) -> Self {
        Self {
            program: Rc::downgrade(&session.program_service),
            ..Self::default()
        }
    }

    fn take_original(&self) -> CapturedCall {
        let (future, call) = self
            .original
            .borrow_mut()
            .take()
            .expect("original deferred call");
        self.serial
            .spawn(&future)
            .expect("reserve the actual task before graph capture");
        let (_, request, _) = runmat_vm::materialize_deferred_graph_call(
            &call,
            runmat_execution::OutputContract {
                requested_outputs: u16::try_from(call.invocation.requested_outputs()).unwrap(),
                owned_work: None,
            },
            runmat_execution_artifact::ProgramTarget::portable("original-selected-deferred"),
            runmat_execution::value::ValueLimits::default(),
        )
        .expect("capture through the actual selected compiler owner");
        request
            .validate()
            .expect("actual selected Graph request must validate");
        let mut captured = CapturedCall {
            call,
            request,
            original: None,
        };
        captured.original = self
            .program
            .upgrade()
            .and_then(|program| program.original_for_test(captured.origin()));
        captured
    }
}

impl RuntimeExecutionServices for CapturingService {
    fn scope_id(&self) -> ExecutionScopeId {
        self.serial.scope_id()
    }
    fn requires_program_capture(&self) -> bool {
        true
    }
    fn create_future(&self, call: DeferredCall) -> Result<FutureHandle, ExecutionServiceError> {
        *self.original_source.borrow_mut() = runmat_runtime::source_context::current_source_info()
            .and_then(|source| source.source_id);
        let original = call.clone();
        let future = self.serial.create_future(call)?;
        *self.original.borrow_mut() = Some((future.clone(), original));
        Ok(future)
    }
    fn spawn(&self, future: &FutureHandle) -> Result<TaskHandle, ExecutionServiceError> {
        self.serial.spawn(future)
    }
    fn begin_await(&self, value: Value) -> Result<AwaitAction, ExecutionServiceError> {
        self.serial.begin_await(value)
    }
    fn complete_future(
        &self,
        future: runmat_runtime::execution::FutureCompletionRef<'_>,
        result: Result<runmat_runtime::execution::RootedValueSequence, ExecutionServiceError>,
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
    fn current_pool(&self) -> Result<Option<PoolSnapshot>, ExecutionServiceError> {
        self.serial.current_pool()
    }
    fn ensure_pool(&self, request: PoolRequest) -> Result<PoolSnapshot, ExecutionServiceError> {
        self.serial.ensure_pool(request)
    }
    fn close_pool(&self, pool: &PoolHandle) -> Result<(), ExecutionServiceError> {
        self.serial.close_pool(pool)
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
}

#[tokio::test]
async fn deferred_capture_preserves_original_complete_product_in_both_executors() {
    for native in [false, true] {
        let mut session = RunMatSession::with_options(false, false).unwrap();
        let service = Rc::new(CapturingService::new(&session));
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
                    arguments: crate::ProcedureArguments::Source(vec![Value::Num(5.0)]),
                    requested_outputs: 1,
                },
                &control,
            )
            .await
            .unwrap();
        assert!(matches!(result.as_slice(), [Value::Future(_)]));
        let call = service.take_original();
        let source = service
            .original_source
            .borrow()
            .expect("issued cold source");
        let interop = runmat_types::InteropManifest::empty();
        let code = unit
            .cached_direct_code(&interop)
            .expect("cold admitted code");
        let warm = session
            .invoke_executable(
                &unit,
                crate::ProcedureInvocation {
                    target: crate::ProcedureTarget::Function("entry".into()),
                    arguments: crate::ProcedureArguments::Source(vec![Value::Num(5.0)]),
                    requested_outputs: 1,
                },
                &control,
            )
            .await
            .unwrap();
        assert!(matches!(warm.as_slice(), [Value::Future(_)]));
        assert!(
            Rc::ptr_eq(&code, &unit.cached_direct_code(&interop).unwrap()),
            "warm invocation must reuse the original admitted code facts"
        );
        assert_ne!(
            Some(source),
            *service.original_source.borrow(),
            "code reuse must not reuse the cold invocation's source issuance"
        );
        assert_eq!(service.take_original().envelope(), call.envelope());
        assert_eq!(call.call.invocation.arguments(), &[Value::Num(5.0)]);
        assert_eq!(call.call.invocation.requested_outputs(), 1);
        assert_eq!(
            call.call.program_revision.as_ref(),
            Some(&unit.revision().program_revision),
            "the observation belongs to the selected primary, not the ambient caller"
        );
        assert_eq!(
            call.origin().program,
            runmat_execution::FunctionBindingOrigin::from_revision(
                &unit.revision().program_revision,
                call.origin().function
            )
            .unwrap()
            .program
        );
        let captured = call.envelope();
        let function = call.function_in(&unit);
        assert_eq!(
            captured,
            unit.portable_envelope_for_function(function).unwrap()
        );
        let outputs = runmat_execution::OutputContract {
            requested_outputs: 1,
            owned_work: None,
        };
        let target = runmat_execution_artifact::ProgramTarget::portable("original-deferred-v4");
        let request = &call.request;
        assert_eq!(
            request.recipe.program_revision,
            unit.revision().program_revision
        );
        assert_eq!(
            call.call.environment.entries(),
            session.environment_snapshot()
        );
        assert_eq!(
            request.artifact.form,
            runmat_execution_artifact::ExecutableForm::ExecutableUnitV4
        );
        assert_eq!(
            request.artifact.executable_unit().unwrap().unwrap(),
            unit.portable_envelope().unwrap()
        );
        assert!(
            matches!(&request.roots, runmat_execution::value::GraphInvocationRoots::Callable { arguments, .. } if arguments.len() == 1)
        );
        for mutation in 0..3 {
            let mut invalid = call.call.clone();
            match mutation {
                0 => invalid.program = runmat_runtime::execution::DeferredProgram::Local,
                1 => {
                    let runmat_runtime::execution::DeferredInvocation::Callable(descriptor) =
                        &mut invalid.invocation
                    else {
                        panic!("selected callable")
                    };
                    let runmat_runtime::call::descriptor::CallableTarget::Bound(binding) =
                        &mut descriptor.target
                    else {
                        panic!("selected bound procedure")
                    };
                    binding.function = usize::MAX;
                }
                _ => {
                    let runmat_runtime::execution::DeferredInvocation::Callable(descriptor) =
                        &mut invalid.invocation
                    else {
                        panic!("selected callable")
                    };
                    descriptor.requested_outputs = 2;
                }
            }
            assert!(runmat_vm::materialize_deferred_graph_call(
                &invalid,
                outputs.clone(),
                target.clone(),
                runmat_execution::value::ValueLimits::default(),
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
            restored.function_analysis(call.function_in(&restored)),
            unit.function_analysis(function)
        );
    }
}

#[tokio::test]
async fn raw_script_capture_has_original_product_and_restores_session_context() {
    let mut session = RunMatSession::with_options(false, false).unwrap();
    let service = Rc::new(CapturingService::new(&session));
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
    let call = service.take_original();
    assert_eq!(call.call.invocation.arguments(), &[Value::Num(5.0)]);
    assert_eq!(call.call.invocation.requested_outputs(), 1);
    let envelope = call.envelope();
    assert_eq!(
        call.call.program_revision.as_ref(),
        Some(&envelope.manifest.identity.program)
    );
    assert_eq!(
        call.request.recipe.program_revision,
        envelope.manifest.identity.program
    );
    let restored = ExecutableUnit::from_portable_envelope(&envelope).unwrap();
    assert_eq!(restored.source().text, source);
    let function = call.function_in(&restored);
    assert!(restored.function_analysis(function).is_some());
    let request = &call.request;
    assert_eq!(
        ExecutableUnit::from_portable_envelope(
            &request.artifact.executable_unit().unwrap().unwrap()
        )
        .unwrap()
        .source()
        .text,
        source
    );
}

mod persistent;

mod environment;
