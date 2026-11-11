//! Generated-code continuation ownership, not terminal-only API rejection.
use super::*;
use runmat_execution::{cancellation::CancellationSource, CancellationReason};
use runmat_native_executor::execute::NativeInvocationStep;
use runmat_runtime::context::{RuntimeServicePorts, RuntimeWorkspaceService};

#[test]
fn valid_generic_deopt_preserves_published_workspace_but_releases_execution_roots() {
    for cancel in [false, true] {
        runmat_gc::gc_test_context(|| deopt_ownership(cancel));
    }
}

fn deopt_ownership(cancel: bool) {
    let source = CancellationSource::new();
    let base_workspace = runmat_runtime::workspace::BaseWorkspaceOwner::new();
    let published_service: Rc<dyn RuntimeWorkspaceService> = Rc::new(
        runmat_runtime::workspace::BaseWorkspaceService::new(base_workspace.clone()),
    );
    let runtime = runtime_context()
        .with_cancellation_token(source.token())
        .with_service_ports(
            RuntimeServicePorts::default().with_workspace(published_service.clone()),
        );
    let owned = runmat_gc::gc_allocate_rooted(Value::String("execution-only".into())).unwrap();
    let owned_handle = owned.handle();
    let published =
        runmat_gc::gc_allocate_rooted(Value::String("published-workspace".into())).unwrap();
    let published_handle = published.handle();
    let make_value = |target| {
        Value::HandleObject(runmat_value::HandleRef {
            class_name: "CancellationOwnershipWitness".into(),
            target,
            valid: true,
        })
    };
    let owned_value = make_value(owned_handle);
    let published_value = make_value(published_handle);
    let calls = Arc::new(AtomicUsize::new(0));
    let observed_calls = calls.clone();
    let activation = runtime.enter();
    // Native callbacks and GC handles are thread-affine; the runtime's callback
    // interface uses Arc without requiring this closure to cross threads.
    #[allow(clippy::arc_with_non_send_sync)]
    let invoker = runmat_runtime::user_functions::install_semantic_function_invoker(Some(
        Arc::new(move |_, _, _| {
            let ordinal = observed_calls.fetch_add(1, Ordering::SeqCst);
            if ordinal == 0 {
                let context = runmat_runtime::context::legacy::active().unwrap();
                let service = context.service_ports().workspace().unwrap().clone();
                service
                    .assign_with_context(&context, "published", published_value.clone())
                    .unwrap();
            }
            let result = owned_value.clone();
            Box::pin(async move {
                rooted_sequence(ValueSequence::single(result).unwrap()).map_err(|error| *error)
            })
        }),
    ))
    .expect("call-port scope");
    drop(activation);
    let assembly = two_semantic_calls_returning_first_fixture();
    let safepoint = assembly.functions[0].blocks[0]
        .instructions
        .iter()
        .filter_map(|instruction| instruction.safepoint)
        .nth(1)
        .unwrap();
    let executor = compile_executor(assembly).unwrap();
    let workspace = NativeWorkspaceInput::empty_for_context(&runtime, BTreeMap::new()).unwrap();
    let mut invocation = executor
        .begin_request(runmat_native_executor::NativeInvocationRequest {
            function: FunctionId(0),
            captures: vec![],
            arguments: vec![],
            requested_outputs: 1,
            runtime,
            deoptimization: DeoptimizationPolicy::default()
                .inject(FaultInjection::Safepoint(safepoint)),
            osr_target: None,
            workspace: Some(workspace),
        })
        .unwrap();
    let NativeInvocationStep::Deoptimized { target, frame, .. } =
        advance_after_suspended_work(&mut invocation).unwrap()
    else {
        panic!("real second-call safepoint must deoptimize");
    };
    assert_eq!(
        target,
        runmat_runtime::native::NativeResumeKind::GENERIC_NATIVE
    );
    assert!(
        matches!(&frame.locals[0].value, Some(Value::HandleObject(value)) if value.target == owned_handle)
    );
    drop(frame);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    owned.unroot().unwrap();
    published.unroot().unwrap();
    runmat_gc::gc_collect_major().unwrap();
    assert!(runmat_gc::gc_clone_value(&owned_handle).is_ok());
    assert!(runmat_gc::gc_clone_value(&published_handle).is_ok());
    if cancel {
        source.cancel(CancellationReason::User);
        assert!(matches!(
            invocation.resume_deoptimization(),
            Err(NativeExecutorError::Cancelled)
        ));
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "cancelled continuation must not replay or advance calls"
        );
    } else {
        invocation.resume_deoptimization().unwrap();
        let NativeInvocationStep::Completed(execution) =
            advance_after_suspended_work(&mut invocation).unwrap()
        else {
            panic!("uncancelled exact continuation must complete");
        };
        assert!(
            matches!(execution.outputs.as_slice(), [Value::HandleObject(value)] if value.target == owned_handle)
        );
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        drop(execution);
    }
    // Keep the terminal NativeInvocation object alive throughout collection.
    runmat_gc::gc_collect_major().unwrap();
    assert!(runmat_gc::gc_clone_value(&owned_handle).is_err());
    assert!(runmat_gc::gc_clone_value(&published_handle).is_ok());
    assert!(
        matches!(published_service.lookup("published"), Some(Value::HandleObject(value)) if value.target == published_handle)
    );
    assert!(invocation.advance().is_err());
    drop(invocation);
    base_workspace.close().unwrap();
    drop(published_service);
    runmat_gc::gc_collect_major().unwrap();
    assert!(runmat_gc::gc_clone_value(&published_handle).is_err());
    drop(invoker);
}

#[test]
fn valid_optimized_osr_cancellation_releases_retained_code_without_entering_target() {
    for cancel in [false, true] {
        let assembly = for_fixture();
        let point = assembly.functions[0]
            .blocks
            .iter()
            .find_map(|block| {
                matches!(
                    block.terminator.kind,
                    runmat_native_codegen::NativeTerminatorKind::For { .. }
                )
                .then_some(block.terminator.site.point)
            })
            .unwrap();
        let generic = compile_executor(assembly.clone()).unwrap();
        let profile =
            runmat_jit::tiering::RepresentationProfile::from_facts(vec![], usize::MAX).unwrap();
        let optimized = Rc::new(
            GenericCompiler::compile_specialized_executor_with_resume_points(
                assembly,
                None,
                BTreeMap::new(),
                profile,
                runmat_native_codegen::NativeLivenessAdmission::AuthorizedUnbounded,
            )
            .unwrap(),
        );
        let weak = Rc::downgrade(&optimized);
        let target = OsrTarget::new(optimized.clone(), FunctionId(0), point).unwrap();
        let source = CancellationSource::new();
        let runtime = runtime_context().with_cancellation_token(source.token());
        let mut invocation = generic
            .begin_request(runmat_native_executor::NativeInvocationRequest {
                function: FunctionId(0),
                captures: vec![],
                arguments: vec![],
                requested_outputs: 1,
                runtime,
                deoptimization: DeoptimizationPolicy::default(),
                osr_target: Some(target),
                workspace: None,
            })
            .unwrap();
        drop(optimized);
        assert!(
            weak.upgrade().is_some(),
            "invocation owns admitted optimized code"
        );
        let NativeInvocationStep::Deoptimized { target, .. } =
            advance_after_suspended_work(&mut invocation).unwrap()
        else {
            panic!("generated loop must reach valid OSR boundary");
        };
        assert_eq!(
            target,
            runmat_runtime::native::NativeResumeKind::OPTIMIZED_NATIVE
        );
        if cancel {
            source.cancel(CancellationReason::User);
            assert!(matches!(
                invocation.resume_optimization(),
                Err(NativeExecutorError::Cancelled)
            ));
        } else {
            invocation.resume_optimization().unwrap();
            assert!(
                weak.upgrade().is_some(),
                "active optimized frame retains its code"
            );
            let NativeInvocationStep::Completed(execution) =
                advance_after_suspended_work(&mut invocation).unwrap()
            else {
                panic!("uncancelled optimized continuation must complete");
            };
            assert_eq!(execution.outputs, vec![Value::Num(6.0)]);
            assert_eq!(execution.loop_backedges.values().sum::<u64>(), 3);
            assert_eq!(execution.osr_entry, Some(point));
        }
        assert!(
            weak.upgrade().is_none(),
            "terminal retained invocation must release optimized code"
        );
        assert!(invocation.advance().is_err());
        assert!(invocation.resume_optimization().is_err());
    }
}
