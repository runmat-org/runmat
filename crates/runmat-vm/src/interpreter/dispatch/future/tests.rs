use super::*;

struct ManagedControl(runmat_runtime::context::host_effect::ManagedHostEffectLedger);

impl runmat_runtime::context::RuntimeParallelService for ManagedControl {
    fn supports(&self, _: runmat_runtime::context::ParallelCapability) -> bool {
        false
    }

    fn placement_resources(&self) -> runmat_runtime::context::RuntimeParallelResources {
        let snapshot = self.0.snapshot();
        runmat_runtime::context::RuntimeParallelResources {
            memory_available_bytes: Some(snapshot.free_bytes()),
            epoch: snapshot.epoch,
            ..Default::default()
        }
    }

    fn managed_host_effect_ledger(
        &self,
    ) -> Option<runmat_runtime::context::host_effect::ManagedHostEffectLedger> {
        Some(self.0.clone())
    }
}

#[test]
fn immediate_instructions_run_only_when_polled_without_boxing() {
    let instruction = Instr::LoadConst(7.0);
    let bytecode = crate::bytecode::Bytecode::empty();
    let registry = crate::bytecode::FunctionRegistry::default();
    let mut stack = crate::interpreter::stack::InterpreterStack::unbounded();
    let mut vars = crate::interpreter::vars::InterpreterVars::from_owned(Vec::new());
    let mut sequence = SequenceState::default();
    let mut execution = crate::bytecode::program::ExecutionContext::default();
    let mut try_stack = Vec::new();
    let mut last_exception = None;
    let mut imports = Vec::new();
    let global_aliases = runmat_runtime::workspace::bindings::FrameGlobalBindings::default();
    let mut persistent_aliases = HashMap::new();
    let mut missing_input_slots = HashSet::new();
    let mut pc = 0;
    let mut before_var = |_index: usize| {};
    let mut after_var = |_index: usize| {};
    let mut before_local = |_current: &Value, _incoming: &Value| {};
    let mut before_local_var = |_index: usize| {};
    let mut after_local = |_index: usize, _value: &Value| {};
    let mut after_fallback = |_name: &str, _index: usize| {};

    macro_rules! immediate_future {
        ($instruction:expr) => {
            dispatch_instruction(
                DispatchMeta {
                    instr: $instruction,
                    bytecode: &bytecode,
                    #[cfg(feature = "native-accel")]
                    private_temp_load: None,
                    var_names: &bytecode.var_names,
                    function_registry: &registry,
                    source_id: None,
                    call_arg_spans: None,
                    invocation_arity: runmat_runtime::context::RuntimeInvocationArity::Script,
                    current_function_name: "ready_fixture",
                },
                DispatchState {
                    stack: &mut stack,
                    vars: &mut vars,
                    sequence_register: &mut sequence,
                    context: &mut execution,
                    try_stack: &mut try_stack,
                    last_exception: &mut last_exception,
                    imports: &mut imports,
                    global_aliases: &global_aliases,
                    persistent_aliases: &mut persistent_aliases,
                    missing_input_slots: &mut missing_input_slots,
                    pc: &mut pc,
                },
                DispatchHooks {
                    store_var_before_overwrite: &mut before_var,
                    store_var_after_store: &mut after_var,
                    store_local_before_local_overwrite: &mut before_local,
                    store_local_before_var_overwrite: &mut before_local_var,
                    store_local_after_store: &mut after_local,
                    store_local_after_fallback_store: &mut after_fallback,
                },
            )
        };
    }

    let mut future = immediate_future!(&instruction);

    let InstructionDispatchFutureState::Immediate(deferred) = &future.state else {
        panic!("LoadConst must use the unboxed immediate state");
    };
    let deferred = deferred.as_ref().expect("deferred instruction");
    assert!(deferred.state.stack.is_empty());
    let mut context = Context::from_waker(std::task::Waker::noop());
    assert!(matches!(
        Pin::new(&mut future).poll(&mut context),
        Poll::Ready(Ok(Some(DispatchHandled::Generic(
            DispatchDecision::FallThrough
        ))))
    ));
    drop(future);
    assert_eq!(stack.as_slice(), &[Value::Num(7.0)]);

    let dropped_instruction = Instr::LoadConst(9.0);
    let unpolled = immediate_future!(&dropped_instruction);
    drop(unpolled);
    assert_eq!(stack.as_slice(), &[Value::Num(7.0)]);

    let enter_try = Instr::EnterTry {
        scope: 3,
        catch_pc: 11,
        catch_var: Some(2),
    };
    let unpolled = immediate_future!(&enter_try);
    drop(unpolled);
    assert!(
        try_stack.is_empty(),
        "dropping must not enter the try scope"
    );

    let mut future = immediate_future!(&enter_try);
    let mut context = Context::from_waker(std::task::Waker::noop());
    assert!(matches!(
        Pin::new(&mut future).poll(&mut context),
        Poll::Ready(Ok(Some(DispatchHandled::Generic(
            DispatchDecision::FallThrough
        ))))
    ));
    drop(future);
    assert_eq!(
        try_stack,
        vec![crate::interpreter::state::ActiveTryHandler {
            scope: 3,
            catch_pc: 11,
            catch_var: Some(2),
            stack_depth: stack.len(),
        }]
    );

    // The actual selected prepared handler admits its concrete future before
    // touching the target state, and retains that claim until Box retirement.
    for capacity in [0, 1 << 20] {
        let ledger = runmat_runtime::context::host_effect::ManagedHostEffectLedger::new(capacity);
        execution.runtime = execution.runtime.clone().with_service_ports(
            runmat_runtime::context::RuntimeServicePorts::default()
                .with_parallel(std::rc::Rc::new(ManagedControl(ledger.clone()))),
        );
        let instruction = Instr::BeginOutputAssignment { target_count: 1 };
        let mut future = immediate_future!(&instruction);
        let mut context = Context::from_waker(std::task::Waker::noop());
        if capacity == 0 {
            assert!(matches!(
                future.state,
                InstructionDispatchFutureState::Ready(_)
            ));
            let Poll::Ready(Err(error)) = Pin::new(&mut future).poll(&mut context) else {
                panic!("zero-capacity selected future must be rejected");
            };
            let mut cause: &(dyn std::error::Error + 'static) = &error;
            loop {
                if let Some(allocation) = cause.downcast_ref::<
                    runmat_runtime::context::host_allocation::HostAllocationError,
                >() {
                    assert!(allocation.is_capacity_denial());
                    break;
                }
                cause = cause.source().expect("original future admission cause");
            }
            drop(future);
            assert_eq!(sequence.expected_output_targets, None);
        } else {
            assert!(matches!(
                future.state,
                InstructionDispatchFutureState::StackMutation(_)
            ));
            assert!(ledger.snapshot().live_reserved_bytes > 0);
            drop(future);
            assert_eq!(sequence.expected_output_targets, None);
            assert_eq!(ledger.snapshot().live_reserved_bytes, 0);

            let mut future = immediate_future!(&instruction);
            assert!(matches!(
                Pin::new(&mut future).poll(&mut context),
                Poll::Ready(Ok(Some(DispatchHandled::Generic(
                    DispatchDecision::FallThrough
                ))))
            ));
            assert!(ledger.snapshot().live_reserved_bytes > 0);
            drop(future);
            assert_eq!(sequence.expected_output_targets, Some(1));
        }
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    }

    // Each additional always-fall-through family chooses the exact admitted
    // owner before even invalid/absent operands or display/name effects run.
    for instruction in [
        Instr::CreateFunctionHandle("unpolled_handle".into()),
        Instr::Index(1),
        Instr::EmitStackTop {
            label: crate::bytecode::EmitLabel::Ans,
        },
        Instr::CreateRange(false),
    ] {
        for capacity in [0, 1 << 20] {
            let ledger =
                runmat_runtime::context::host_effect::ManagedHostEffectLedger::new(capacity);
            execution.runtime = execution.runtime.clone().with_service_ports(
                runmat_runtime::context::RuntimeServicePorts::default()
                    .with_parallel(std::rc::Rc::new(ManagedControl(ledger.clone()))),
            );
            let mut future = immediate_future!(&instruction);
            if capacity == 0 {
                assert!(matches!(
                    future.state,
                    InstructionDispatchFutureState::Ready(_)
                ));
                let mut context = Context::from_waker(std::task::Waker::noop());
                assert!(matches!(
                    Pin::new(&mut future).poll(&mut context),
                    Poll::Ready(Err(_))
                ));
            } else {
                assert!(matches!(
                    future.state,
                    InstructionDispatchFutureState::StackMutation(_)
                ));
                assert!(ledger.snapshot().live_reserved_bytes > 0);
            }
            drop(future);
            assert_eq!(stack.as_slice(), &[Value::Num(7.0)]);
            assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
            assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
        }
    }

    // Logical, call and parallel selectors admit their original concrete
    // control future before operands or callee/task effects can be entered.
    for instruction in [
        Instr::AndAnd(17),
        Instr::CallBuiltinMulti("disp".into(), 1, 0),
        Instr::Await,
    ] {
        for capacity in [0, 1 << 20] {
            let ledger =
                runmat_runtime::context::host_effect::ManagedHostEffectLedger::new(capacity);
            execution.runtime = execution.runtime.clone().with_service_ports(
                runmat_runtime::context::RuntimeServicePorts::default()
                    .with_parallel(std::rc::Rc::new(ManagedControl(ledger.clone()))),
            );
            let mut future = immediate_future!(&instruction);
            if capacity == 0 {
                assert!(matches!(
                    future.state,
                    InstructionDispatchFutureState::Ready(_)
                ));
                let mut context = Context::from_waker(std::task::Waker::noop());
                assert!(matches!(
                    Pin::new(&mut future).poll(&mut context),
                    Poll::Ready(Err(_))
                ));
            } else {
                assert!(matches!(
                    future.state,
                    InstructionDispatchFutureState::Control(_)
                ));
                assert!(ledger.snapshot().live_reserved_bytes > 0);
            }
            drop(future);
            assert_eq!(stack.as_slice(), &[Value::Num(7.0)]);
            assert_eq!(pc, 0);
            assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
            assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
        }
    }
}

#[test]
fn control_completions_preserve_all_outcomes_without_root_backing_in_finite_context() {
    use runmat_runtime::context::{RuntimeContext, RuntimeServicePorts};
    use runmat_runtime::execution::RuntimeExecutionService;
    use std::rc::Rc;
    fn key(value: Option<&DispatchHandled>) -> (u8, u8) {
        let (kind, decision) = match value {
            None => return (0, 0),
            Some(DispatchHandled::Generic(decision)) => (1, decision),
            Some(DispatchHandled::ReturnValue(decision)) => (2, decision),
            Some(DispatchHandled::Return(decision)) => (3, decision),
        };
        (
            kind,
            match decision {
                DispatchDecision::ContinueLoop => 1,
                DispatchDecision::FallThrough => 2,
                DispatchDecision::Return => 3,
            },
        )
    }
    let ledger = runmat_runtime::context::host_effect::ManagedHostEffectLedger::new(0);
    let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new())).with_service_ports(
        RuntimeServicePorts::default().with_parallel(Rc::new(ManagedControl(ledger.clone()))),
    );
    let _active = runtime.enter();
    for value in [
        None,
        Some(DispatchHandled::Generic(DispatchDecision::ContinueLoop)),
        Some(DispatchHandled::Generic(DispatchDecision::FallThrough)),
        Some(DispatchHandled::Generic(DispatchDecision::Return)),
        Some(DispatchHandled::ReturnValue(DispatchDecision::ContinueLoop)),
        Some(DispatchHandled::ReturnValue(DispatchDecision::FallThrough)),
        Some(DispatchHandled::ReturnValue(DispatchDecision::Return)),
        Some(DispatchHandled::Return(DispatchDecision::ContinueLoop)),
        Some(DispatchHandled::Return(DispatchDecision::FallThrough)),
        Some(DispatchHandled::Return(DispatchDecision::Return)),
    ] {
        let expected = key(value.as_ref());
        let rooted = control::root_control(Ok(value)).unwrap();
        let actual = rooted.map_or((0, 0), |value| value.with_owned(|value| key(Some(&value))));
        assert_eq!(actual, expected);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    }
}
