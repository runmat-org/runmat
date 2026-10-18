use super::*;

#[test]
fn immediate_instructions_run_only_when_polled_without_boxing() {
    let instruction = Instr::LoadConst(7.0);
    let bytecode = crate::bytecode::Bytecode::empty();
    let registry = crate::bytecode::FunctionRegistry::default();
    let mut stack = Vec::new();
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
    assert_eq!(stack, vec![Value::Num(7.0)]);

    let dropped_instruction = Instr::LoadConst(9.0);
    let unpolled = immediate_future!(&dropped_instruction);
    drop(unpolled);
    assert_eq!(stack, vec![Value::Num(7.0)]);

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
        }]
    );
}
