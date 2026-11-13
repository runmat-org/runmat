#[cfg(feature = "native-accel")]
use crate::accel::fusion as accel_fusion;
use crate::bytecode::{Bytecode, FunctionRegistry, Instr};
use crate::interpreter::api::{InterpreterOutcome, InterpreterResumeState, InterpreterState};
use crate::interpreter::dispatch::{self as interp_dispatch, DispatchDecision};
use crate::interpreter::dynamic_admission::DynamicEvalAdmissionPort;
use crate::interpreter::engine as interp_engine;
use crate::interpreter::errors::{attach_span_from_pc, mex, scope_vm_pc, set_vm_pc};
use crate::interpreter::process_driver::{ProcessContinuationDriver, ProcessDriverHandle};
use crate::interpreter::timing::InterpreterTiming;
use crate::interpreter::vars::InterpreterVars;
use crate::runtime::call_stack::attach_call_frames;
use crate::runtime::globals as runtime_globals;
use crate::runtime::workspace::{
    workspace_assign, workspace_clear, workspace_lookup, workspace_remove, workspace_snapshot,
};
use runmat_runtime::call::function_abi::{collect_function_outputs, FunctionInputSpec};
use runmat_runtime::context::RuntimeInvocationArity;
use runmat_runtime::{
    user_functions,
    workspace::{self as runtime_workspace, WorkspaceResolver},
    RuntimeError,
};
use runmat_value::{CellArray, Value};
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::Once;
use tracing::{debug, info_span, Instrument};

#[path = "runner/bound_invocation.rs"]
mod bound_invocation;
pub(crate) use bound_invocation::Invocation as BoundInvocation;
#[path = "runner/instruction_loop.rs"]
mod instruction_loop;
#[path = "runner/super_constructor.rs"]
mod super_constructor;
#[path = "runner/workspace_execution.rs"]
mod workspace_execution;

#[cfg(not(target_arch = "wasm32"))]
use std::future::Future;

#[cfg(feature = "native-accel")]
use runmat_accelerate::placement::{
    begin_trace, fusion_operation_token, PlacementEventKind, PlacementVariant,
};
#[cfg(feature = "native-accel")]
use runmat_accelerate::set_current_pc;
#[cfg(feature = "native-accel")]
use runmat_time::{duration_ns_saturating, Instant};

type VmResult<T> = Result<T, RuntimeError>;

fn ensure_workspace_resolver_registered() {
    static REGISTER: Once = Once::new();
    REGISTER.call_once(|| {
        runtime_workspace::register_workspace_resolver(WorkspaceResolver {
            lookup: workspace_lookup,
            snapshot: workspace_snapshot,
            globals: runtime_globals::workspace_global_names,
            assign: Some(workspace_assign),
            clear: Some(workspace_clear),
            remove: Some(workspace_remove),
        });
    });
}

fn ensure_wasm_builtins_registered() {
    #[cfg(target_arch = "wasm32")]
    {
        static REGISTER: Once = Once::new();
        REGISTER.call_once(|| {
            runmat_runtime::builtins::wasm_registry::register_all();
        });
    }
}

fn active_or_standalone_runtime_context() -> runmat_runtime::context::RuntimeContext {
    runmat_runtime::context::legacy::active().unwrap_or_else(|| {
        let execution = std::rc::Rc::new(runmat_runtime::execution::RuntimeExecutionService::new());
        let runtime = match runmat_runtime::interrupt::current_interrupt() {
            Some(cancellation) => {
                runmat_runtime::context::RuntimeContext::with_cancellation(execution, cancellation)
            }
            None => runmat_runtime::context::RuntimeContext::new(execution),
        };
        runmat_runtime::compatibility::inherit_legacy_extension_policy(&runtime);
        runmat_runtime::user_functions::inherit_legacy_call_environment(&runtime);
        runmat_runtime::source_context::inherit_legacy_source_context(&runtime);
        runtime
    })
}

pub async fn invoke_semantic_function_value(
    function: usize,
    args: &[Value],
    requested_outputs: usize,
    function_registry: &FunctionRegistry,
) -> Result<runmat_runtime::execution::RootedValueSequence, RuntimeError> {
    let runtime = active_or_standalone_runtime_context();
    let (value, _) = invoke_semantic_function_with_captures(
        function,
        args,
        requested_outputs,
        function_registry,
        runtime,
        None,
        None,
    )
    .await?;
    Ok(value)
}

pub async fn invoke_semantic_function_value_in_context(
    function: usize,
    args: &[Value],
    requested_outputs: usize,
    function_registry: &FunctionRegistry,
    runtime: runmat_runtime::context::RuntimeContext,
) -> Result<runmat_runtime::execution::RootedValueSequence, RuntimeError> {
    invoke_semantic_function_value_with_environment_in_context(
        function,
        args,
        requested_outputs,
        function_registry,
        runtime,
        None,
    )
    .await
}

pub async fn invoke_semantic_function_value_with_environment_in_context(
    function: usize,
    args: &[Value],
    requested_outputs: usize,
    function_registry: &FunctionRegistry,
    runtime: runmat_runtime::context::RuntimeContext,
    lexical_environment: Option<runmat_value::LexicalEnvironment>,
) -> Result<runmat_runtime::execution::RootedValueSequence, RuntimeError> {
    let (value, _) = invoke_semantic_function_with_captures(
        function,
        args,
        requested_outputs,
        function_registry,
        runtime,
        None,
        lexical_environment,
    )
    .await?;
    Ok(value)
}

/// Worker-selected bound invocation uses the same constructor, superclass,
/// access, and output policy as a local bound handle. The graph dispatcher
/// supplies its exact unit registry and issued runtime; this function never
/// resolves an origin or name from ambient state.
#[inline(always)]
pub(crate) fn invoke_selected_graph_bound_in_context(
    runtime: runmat_runtime::context::RuntimeContext,
    binding: runmat_value::BoundFunctionBinding,
    arguments: runmat_runtime::call::RootedFunctionArguments,
    requested_outputs: usize,
    registry: FunctionRegistry,
    lexical_environment: Option<runmat_value::LexicalEnvironment>,
    driver: Option<ProcessDriverHandle>,
) -> impl std::future::Future<Output = VmResult<runmat_runtime::execution::RootedValueSequence>> {
    super_constructor::invoke_selected_bound(
        runtime,
        binding,
        arguments,
        requested_outputs,
        registry,
        driver,
        lexical_environment,
    )
}

/// Host-selected source admission remains invocation-local across queued
/// semantic calls. Imported pure selected products may have no local source
/// capability; that absence must remain explicit until source is demanded.
pub async fn invoke_semantic_function_value_with_dynamic_admission_in_context(
    function: usize,
    args: &[Value],
    requested_outputs: usize,
    function_registry: &FunctionRegistry,
    runtime: runmat_runtime::context::RuntimeContext,
    lexical_environment: Option<runmat_value::LexicalEnvironment>,
    dynamic_admission: impl Into<Option<DynamicEvalAdmissionPort>>,
) -> Result<runmat_runtime::execution::RootedValueSequence, RuntimeError> {
    let driver_runtime = runtime.clone();
    let invocation = |driver| {
        invoke_semantic_function_with_captures(
            function,
            args,
            requested_outputs,
            function_registry,
            runtime,
            driver,
            lexical_environment,
        )
    };
    let (values, _) = if let Some(port) = dynamic_admission.into() {
        let owner = ProcessContinuationDriver::new_with_dynamic_admission_in_context(
            &driver_runtime,
            Some(port),
        )?;
        let driver = owner.handle();
        owner
            .drive_in_context(&driver_runtime, || invocation(Some(driver)))?
            .await?
    } else {
        invocation(None).await?
    };
    Ok(values)
}

/// Invoke an already rooted, possibly segmented callable argument carrier
/// while preserving invocation-local dynamic source admission.
pub async fn invoke_semantic_function_with_rooted_arguments_and_dynamic_admission_in_context(
    function: usize,
    arguments: runmat_runtime::call::RootedFunctionArguments,
    requested_outputs: usize,
    function_registry: &FunctionRegistry,
    runtime: runmat_runtime::context::RuntimeContext,
    lexical_environment: Option<runmat_value::LexicalEnvironment>,
    dynamic_admission: impl Into<Option<DynamicEvalAdmissionPort>>,
) -> Result<runmat_runtime::execution::RootedValueSequence, RuntimeError> {
    let driver_runtime = runtime.clone();
    let invocation = |driver| {
        invoke_semantic_function_with_rooted_captures(
            function,
            arguments,
            requested_outputs,
            function_registry.clone(),
            runtime,
            driver,
            lexical_environment,
        )
    };
    let (values, _) = if let Some(port) = dynamic_admission.into() {
        let owner = ProcessContinuationDriver::new_with_dynamic_admission_in_context(
            &driver_runtime,
            Some(port),
        )?;
        let driver = owner.handle();
        owner
            .drive_in_context(&driver_runtime, || invocation(Some(driver)))?
            .await?
    } else {
        invocation(None).await?
    };
    Ok(values)
}

pub(crate) async fn invoke_semantic_function_value_with_capture_updates(
    function: usize,
    args: &[Value],
    requested_outputs: usize,
    function_registry: &FunctionRegistry,
    runtime: runmat_runtime::context::RuntimeContext,
    driver: Option<ProcessDriverHandle>,
    lexical_environment: Option<runmat_value::LexicalEnvironment>,
) -> Result<
    (
        runmat_runtime::execution::RootedValueSequence,
        runmat_runtime::execution::RootedValues,
    ),
    RuntimeError,
> {
    invoke_semantic_function_with_captures(
        function,
        args,
        requested_outputs,
        function_registry,
        runtime,
        driver,
        lexical_environment,
    )
    .await
}

async fn invoke_semantic_function_with_captures(
    function: usize,
    args: &[Value],
    requested_outputs: usize,
    function_registry: &FunctionRegistry,
    runtime: runmat_runtime::context::RuntimeContext,
    driver: Option<ProcessDriverHandle>,
    lexical_environment: Option<runmat_value::LexicalEnvironment>,
) -> Result<
    (
        runmat_runtime::execution::RootedValueSequence,
        runmat_runtime::execution::RootedValues,
    ),
    RuntimeError,
> {
    let arguments = runmat_runtime::call::BuiltinArguments::from_borrowed(args, Some(&runtime))?
        .into_function_arguments();
    invoke_semantic_function_with_rooted_captures(
        function,
        arguments,
        requested_outputs,
        function_registry.clone(),
        runtime,
        driver,
        lexical_environment,
    )
    .await
}

async fn invoke_semantic_function_with_rooted_captures(
    function: usize,
    arguments: runmat_runtime::call::RootedFunctionArguments,
    requested_outputs: usize,
    function_registry: FunctionRegistry,
    runtime: runmat_runtime::context::RuntimeContext,
    driver: Option<ProcessDriverHandle>,
    lexical_environment: Option<runmat_value::LexicalEnvironment>,
) -> Result<
    (
        runmat_runtime::execution::RootedValueSequence,
        runmat_runtime::execution::RootedValues,
    ),
    RuntimeError,
> {
    let runtime = runtime.inherit_same_session_environment();
    let runtime = runtime.with_constructor_receiver_for_method(
        function_registry
            .get(runmat_hir::FunctionId(function))
            .and_then(|function| function.class_method_owner.as_ref()),
    )?;
    let lexical_roots = lexical_environment
        .as_ref()
        .map(|environment| {
            runmat_runtime::execution::RootedValueLease::for_lexical_environment_in_context(
                environment,
                &runtime,
            )
        })
        .transpose()?;
    let registry = function_registry;
    let runtime = with_default_acceleration_service(runtime);
    let owner = if driver.is_none() {
        Some(ProcessContinuationDriver::new_with_dynamic_admission_in_context(&runtime, None)?)
    } else {
        None
    };
    let handle = driver.unwrap_or_else(|| owner.as_ref().expect("new VM driver").handle());
    let selected_handle = handle.clone();
    let scoped_runtime = runtime.clone();
    let queue_runtime = runtime.clone();
    let queued_scope = scoped_runtime.prepare_poll_scope()?;
    let make_child = move || async move {
        let cleanup = runtime.function_cleanup_scope()?;
        let (result, cleanup) = cleanup
            .run(&scoped_runtime, async move {
                // Construct the scoped owner only after the queue has admitted its
                // future. The lexical guard travels with an unpolled child rather
                // than remaining in a suspended caller that may be canceled.
                let _lexical_roots = lexical_roots;
                invoke_semantic_function_with_captures_inner(
                    function,
                    arguments,
                    requested_outputs,
                    &registry,
                    runtime,
                    selected_handle,
                    lexical_environment,
                )
                .await
            })
            .await?;
        runmat_runtime::runtime_error::complete_with_cleanup(result, cleanup)
    };
    if let Some(owner) = owner {
        let drive_runtime = queue_runtime.clone();
        owner
            .drive_in_context(&drive_runtime, move || async move {
                handle
                    .enqueue_scoped_in_context(&queue_runtime, queued_scope, make_child)?
                    .await
            })?
            .await
    } else {
        handle
            .enqueue_scoped_in_context(&queue_runtime, queued_scope, make_child)?
            .await
    }
}

async fn invoke_semantic_function_with_captures_inner(
    function: usize,
    arguments: runmat_runtime::call::RootedFunctionArguments,
    requested_outputs: usize,
    function_registry: &FunctionRegistry,
    runtime: runmat_runtime::context::RuntimeContext,
    driver: ProcessDriverHandle,
    lexical_environment: Option<runmat_value::LexicalEnvironment>,
) -> Result<
    (
        runmat_runtime::execution::RootedValueSequence,
        runmat_runtime::execution::RootedValues,
    ),
    RuntimeError,
> {
    let _inherited_environment_roots = lexical_environment
        .as_ref()
        .map(runmat_runtime::call::lexical::root_environment)
        .transpose()?;
    let function_id = runmat_hir::FunctionId(function);
    let func = function_registry.get(function_id).ok_or_else(|| {
        let message = format!("Undefined semantic function: {function}");
        mex("UndefinedSemanticFunction", &message)
    })?;
    // The entered body, not a numeric caller ID or ambient name, selects the
    // instruction-site namespace for this invocation's static imports.
    let runtime = if func.selected_executable_product.is_some() {
        let binding = function_registry
            .exact_bound_binding(function_id)
            .ok_or_else(|| {
                mex(
                    "StaticImportUnavailable",
                    "selected frame has no exact definition graph",
                )
            })?;
        let runtime = function_registry.bind_selected_body_sites(runtime, &binding, func)?;
        if let Some(owner) = func
            .class_method_owner
            .as_ref()
            .filter(|owner| owner.role == runmat_types::ClassMethodRole::Constructor)
        {
            runtime.with_admitted_constructor_frame(owner, binding)?
        } else {
            runtime
        }
    } else {
        runtime
    };
    let runtime = runtime.with_selected_static_caller(func.binding_origin);
    lexical_storage::validate_inherited_storage(func, function_registry)?;
    let has_callable = arguments.callable().is_some();
    if !has_callable && arguments.len() < func.capture_slots.len() {
        let message = format!(
            "semantic function {} received too few arguments",
            func.display_name
        );
        return Err(mex("SemanticFunctionArity", &message));
    }
    if arguments.lexical_capture_count() != 0
        && arguments.lexical_capture_count() != func.capture_slots.len()
    {
        return Err(mex(
            "SemanticFunctionArity",
            "selected callable capture count does not match its function frame",
        ));
    }
    if has_callable
        && arguments.lexical_capture_count() == 0
        && !func.captures_owned_without_copies(lexical_environment.is_some())
    {
        return Err(mex(
            "SemanticFunctionArity",
            "selected callable omits captures without an admitted alternative owner",
        ));
    }
    let mut vars = if arguments.lexical_capture_count() == 0 {
        InterpreterVars::from_owned(vec![Value::Num(0.0); func.var_count])
    } else {
        InterpreterVars::prepare_captured_frame(&runtime, func.var_count).map_err(|error| {
            crate::runtime::workspace::frame_allocation_error_in_context(Some(&runtime), error)
        })?
    };
    for (index, slot) in func.capture_slots.iter().copied().enumerate() {
        let Some(capture) = arguments.capture_lease(index) else {
            break;
        };
        if vars.replace_captured(slot, capture).is_none() {
            return Err(mex(
                "SemanticFunctionArity",
                "selected callable capture slot is outside its function frame",
            ));
        }
    }
    let mut supplied = arguments.into_function_inputs();
    if !has_callable {
        for slot in func.capture_slots.iter().copied() {
            let value = supplied.pop_front().expect("validated capture prefix");
            if vars.replace_owned(slot, value).is_none() {
                return Err(mex(
                    "SemanticFunctionArity",
                    "semantic capture slot is outside its function frame",
                ));
            }
        }
    }
    invoke_semantic_function_with_owned_inputs(
        OwnedFunctionEntry {
            function_id,
            function: func,
            registry: function_registry,
            requested_outputs,
            runtime,
            driver,
            lexical_environment,
        },
        supplied,
        vars,
    )
    .await
}

struct OwnedFunctionEntry<'a> {
    function_id: runmat_hir::FunctionId,
    function: &'a crate::FunctionBytecode,
    registry: &'a FunctionRegistry,
    requested_outputs: usize,
    runtime: runmat_runtime::context::RuntimeContext,
    driver: ProcessDriverHandle,
    lexical_environment: Option<runmat_value::LexicalEnvironment>,
}

async fn invoke_semantic_function_with_owned_inputs(
    entry: OwnedFunctionEntry<'_>,
    supplied: runmat_runtime::call::BuiltinArguments,
    mut vars: InterpreterVars,
) -> Result<
    (
        runmat_runtime::execution::RootedValueSequence,
        runmat_runtime::execution::RootedValues,
    ),
    RuntimeError,
> {
    let OwnedFunctionEntry {
        function_id,
        function: func,
        registry: function_registry,
        requested_outputs,
        runtime,
        driver,
        lexical_environment,
    } = entry;
    let mut constructor_instance_slot = None;
    if let Some(receiver) = runtime.captured_constructor_receiver().filter(|_| {
        func.class_method_owner
            .as_ref()
            .is_some_and(|owner| owner.role == runmat_types::ClassMethodRole::Constructor)
    }) {
        let slot = *func.output_slots.first().ok_or_else(|| {
            mex(
                "ConstructorOutputInvalid",
                "constructor has no fixed instance output",
            )
        })?;
        let retired = vars.replace_owned(slot, receiver.clone()).ok_or_else(|| {
            mex(
                "ConstructorOutputInvalid",
                "constructor instance output is out of bounds",
            )
        })?;
        drop(retired);
        constructor_instance_slot = Some(slot);
    }
    let input_specs = func
        .argument_validations
        .iter()
        .map(|validation| {
            let input_index = func
                .input_slots
                .iter()
                .position(|slot| *slot == validation.input_slot)
                .ok_or_else(|| mex("InvalidInputSlot", "function argument slot out of bounds"))?;
            Ok(FunctionInputSpec {
                input_index,
                size: validation.size.as_ref(),
                class_name: validation.class_name.as_deref(),
                validators: &validation.validators,
                default_value: validation.default_value.as_ref(),
            })
        })
        .collect::<Result<Vec<_>, RuntimeError>>()?;
    let prepared = runmat_runtime::call::function_abi::prepare_function_inputs_rooted(
        &func.display_name,
        supplied,
        func.input_slots.len(),
        func.varargin_slot.is_some(),
        &input_specs,
        Some(&runtime),
    )
    .await?;
    let mut missing_input_slots = HashSet::new();
    for (slot, value) in func
        .input_slots
        .iter()
        .copied()
        .zip(prepared.fixed.iter().cloned())
    {
        if let Some(value) = value {
            let retired = vars.replace_owned(slot, value).ok_or_else(|| {
                mex(
                    "InvalidInputSlot",
                    "function argument slot is outside its frame",
                )
            })?;
            drop(retired);
        } else {
            missing_input_slots.insert(slot);
        }
    }
    if let (Some(slot), Some(cell)) = (func.varargin_slot, prepared.varargin) {
        if slot < vars.len() {
            let retired = vars.replace_owned(slot, Value::Cell(cell));
            drop(retired);
        }
    }
    if let Some(slot) = func.varargout_slot {
        if slot < vars.len() {
            let cell = CellArray::new(Vec::new(), 1, 0)
                .map_err(|err| mex("VarargoutPack", &format!("varargout: {err}")))?;
            let retired = vars.replace_owned(slot, Value::Cell(cell));
            drop(retired);
        }
    }
    if let Some(slot) = func.implicit_nargin_slot {
        if slot < vars.len() {
            let retired = vars.replace_owned(slot, Value::Num(prepared.nargin as f64));
            drop(retired);
        }
    }
    if let Some(slot) = func.implicit_nargout_slot {
        if slot < vars.len() {
            let retired = vars.replace_owned(slot, Value::Num(requested_outputs as f64));
            drop(retired);
        }
    }

    let _active_semantic_function_guard =
        user_functions::push_active_semantic_function(function_id.0)?;
    let mut bytecode = func.execution_bytecode(function_registry);
    let mut initially_unassigned_slots = func.initially_unassigned_slots.clone();
    for slot in &func.capture_slots {
        initially_unassigned_slots.remove(slot);
    }
    for binding in &func.inherited_storage_slots {
        initially_unassigned_slots.remove(&binding.slot);
    }
    if let Some(slot) = constructor_instance_slot {
        initially_unassigned_slots.remove(&slot);
    }
    for slot in &func.input_slots {
        if !missing_input_slots.contains(slot) {
            initially_unassigned_slots.remove(slot);
        }
    }
    if let Some(slot) = func.varargin_slot {
        initially_unassigned_slots.remove(&slot);
    }
    if let Some(slot) = func.varargout_slot {
        initially_unassigned_slots.remove(&slot);
    }
    if let Some(slot) = func.implicit_nargin_slot {
        initially_unassigned_slots.remove(&slot);
    }
    if let Some(slot) = func.implicit_nargout_slot {
        initially_unassigned_slots.remove(&slot);
    }
    bytecode.initially_unassigned_slots = initially_unassigned_slots;
    let (lexical_environment, _activation_roots) = if func.creates_lexical_environment {
        let (environment, roots) =
            runmat_runtime::call::lexical::allocate_environment(lexical_environment)?;
        (Some(environment), roots)
    } else {
        (lexical_environment, Vec::new())
    };
    for binding in &func.lexical_binding_slots {
        if (binding.kind == crate::bytecode::program::LexicalBindingKind::Transit
            && func.source_role != runmat_types::FunctionSourceRole::Anonymous)
            || (binding.kind == crate::bytecode::program::LexicalBindingKind::Inherited
                && func.source_role == runmat_types::FunctionSourceRole::Anonymous)
        {
            return Err(mex(
                "LexicalEnvironment",
                "lexical binding kind does not match callable role",
            ));
        }
        let Some(environment) = lexical_environment.as_ref() else {
            return Err(mex(
                "LexicalEnvironment",
                "shared lexical binding has no admitted activation",
            ));
        };
        match binding.kind {
            crate::bytecode::program::LexicalBindingKind::Owned => {
                if !runmat_runtime::call::lexical::reserve_owned_binding_with_context(
                    environment,
                    binding.binding,
                    &runtime,
                )? {
                    return Err(mex(
                        "LexicalEnvironment",
                        "owned shared binding was reserved twice",
                    ));
                }
                if !bytecode.initially_unassigned_slots.contains(&binding.slot) {
                    let value = vars
                        .get(binding.slot)
                        .map(|value| value.clone())
                        .ok_or_else(|| {
                            mex(
                                "LexicalEnvironment",
                                "shared binding slot is outside its frame",
                            )
                        })?;
                    runmat_runtime::call::lexical::write_owned_binding(
                        environment,
                        binding.binding,
                        value,
                    )?;
                }
            }
            crate::bytecode::program::LexicalBindingKind::Inherited => {
                if vars.get(binding.slot).is_none() {
                    return Err(mex(
                        "LexicalEnvironment",
                        "inherited binding slot is outside its frame",
                    ));
                }
                if let Some(value) =
                    runmat_runtime::call::lexical::read_binding(environment, binding.binding)?
                {
                    let retired = vars.replace_owned(binding.slot, value).ok_or_else(|| {
                        mex(
                            "LexicalEnvironment",
                            "inherited binding slot is outside its frame",
                        )
                    })?;
                    drop(retired);
                    bytecode.initially_unassigned_slots.remove(&binding.slot);
                } else {
                    bytecode.initially_unassigned_slots.insert(binding.slot);
                }
            }
            crate::bytecode::program::LexicalBindingKind::Transit => {
                if vars.get(binding.slot).is_none() {
                    return Err(mex(
                        "LexicalEnvironment",
                        "transit binding slot is outside its frame",
                    ));
                }
            }
        }
    }
    let result_vars = {
        #[cfg(target_arch = "wasm32")]
        let future = interpret_function_with_counts_in_context_with_driver(
            FunctionExecutionInvocation {
                bytecode: Cow::Borrowed(&bytecode),
                name: FunctionExecutionName::ActiveFunction,
                arity: RuntimeInvocationArity::function(prepared.nargin, requested_outputs),
                missing_input_slots,
                runtime: runtime.clone(),
                driver: Some(driver.clone()),
                lexical_environment: lexical_environment.clone(),
                lexical_binding_slots: func.lexical_binding_slots.clone(),
                inherited_storage_slots: func.inherited_storage_slots.clone(),
            },
            vars,
        );
        #[cfg(target_arch = "wasm32")]
        {
            future.await?
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            // A semantic call recursively drives another async interpreter.
            // Pure RunMat call chains use a large segmented stack so ordinary
            // nesting does not consume a small executor or test-thread stack.
            // Functions that may reach an in-process foreign runtime stay on
            // the native OS stack, which those runtimes may register or inspect.
            const SEMANTIC_CALL_STACK_BYTES: usize = 16 * 1024 * 1024;
            let execution_stack = function_registry.execution_stack(function_id);
            // Move the already-built per-call bytecode into the queued owner;
            // do not clone the program, registry, or instruction stream merely
            // to cross the process-driver boundary.
            let invocation: FunctionExecutionInvocation<'static> = FunctionExecutionInvocation {
                bytecode: Cow::Owned(bytecode),
                name: FunctionExecutionName::ActiveFunction,
                arity: RuntimeInvocationArity::function(prepared.nargin, requested_outputs),
                missing_input_slots,
                runtime: runtime.clone(),
                driver: Some(driver.clone()),
                lexical_environment: lexical_environment.clone(),
                lexical_binding_slots: func.lexical_binding_slots.clone(),
                inherited_storage_slots: func.inherited_storage_slots.clone(),
            };
            if execution_stack == runmat_types::ExecutionStackRequirement::Process {
                // Process-stack callees are siblings in the invocation-owned
                // driver. Polling the owned interpreter future directly here
                // would retain this interpreter's large poll frame while
                // entering the next one. The queue already admits and boxes
                // the exact concrete future, so it is also the canonical
                // stack-segmentation boundary for process-stack calls.
                let queue_runtime = runtime.clone();
                let queued_scope = queue_runtime.prepare_poll_scope()?;
                driver
                    .enqueue_owned_catch_scoped_in_context(
                        &queue_runtime,
                        queued_scope,
                        move || {
                            interpret_function_with_counts_in_context_with_driver(invocation, vars)
                        },
                    )?
                    .await?
            } else {
                let execution_runtime = runtime.clone();
                // Construct the large interpreter future on the temporary stack,
                // then return only its pinned heap owner to the caller. Constructing
                // it before boxing would still consume the caller's native stack.
                // No interpreter polling or foreign execution occurs here.
                let mut future = stacker::grow(SEMANTIC_CALL_STACK_BYTES, || {
                    runmat_runtime::execution::admit_native_hostcall_with_contextual_error(
                        &execution_runtime,
                        move || {
                            interpret_function_with_counts_in_context_with_driver(invocation, vars)
                        },
                    )
                })?;
                futures::future::poll_fn(move |context| {
                    stacker::grow(SEMANTIC_CALL_STACK_BYTES, || {
                        let _stack = execution_runtime.enter_execution_stack(
                            runmat_runtime::context::RuntimeExecutionStack::Segmented,
                        );
                        future.as_mut().poll(context)
                    })
                })
                .await?
            }
        }
    };
    let result_vars = result_vars.values();
    let fixed_outputs = func
        .output_slots
        .iter()
        .map(|slot| result_vars.get(*slot).cloned().unwrap_or(Value::Num(0.0)))
        .collect::<Vec<_>>();
    let varargout = func
        .varargout_slot
        .and_then(|slot| result_vars.get(slot).cloned());
    let output_values = collect_function_outputs(
        &func.display_name,
        &fixed_outputs,
        varargout.as_ref(),
        requested_outputs,
    )?;
    let updated_captures = func
        .capture_slots
        .iter()
        .map(|slot| result_vars.get(*slot).cloned().unwrap_or(Value::Num(0.0)))
        .collect::<Vec<_>>();
    let outputs = output_sequence(output_values, requested_outputs)?;
    let outputs = runmat_runtime::execution::RootedValueSequence::new(outputs)
        .map_err(|error| error.into_runtime_error())?;
    let updated_captures = runmat_runtime::execution::RootedValues::new(updated_captures)
        .map_err(|error| error.into_runtime_error())?;
    Ok((outputs, updated_captures))
}

async fn await_prior_semantic_invoker(
    driver: Option<ProcessDriverHandle>,
    runtime: runmat_runtime::context::RuntimeContext,
    future: user_functions::UserFunctionFuture,
) -> Result<runmat_runtime::execution::RootedValueSequence, RuntimeError> {
    let Some(driver) = driver else {
        return future.await;
    };
    if runtime
        .service_ports()
        .local_tasks()
        .and_then(|tasks| tasks.canonical_admission())
        .is_some()
    {
        // A prior invoker can start a second VM owner. Its selected runtime
        // and rooted inputs are already owned, so poll it as a foreground
        // sibling rather than under this VM driver's still-live poll stack.
        return runmat_runtime::context::run_selected_callback_continuation(&runtime, future).await;
    }
    // The prior invoker may itself be another VM owner. Select its call and
    // source projections before yielding this frame to the queue.
    let queued_scope = runtime.prepare_poll_scope()?;
    let (outputs, _) = driver
        .enqueue_scoped_in_context(&runtime, queued_scope, move || async move {
            let outputs = future.await?;
            let captures = runmat_runtime::execution::RootedValues::new(Vec::new())
                .map_err(|error| error.into_runtime_error())?;
            Ok((outputs, captures))
        })?
        .await?;
    Ok(outputs)
}

/// Select the previous bound invoker under the same continuation that polls
/// it. Its factory performs synchronous selected-source admission, so calling
/// the factory in the suspended interpreter frame defeats sibling placement.
/// Root the copied lexical activation before enqueueing either continuation.
async fn await_prior_bound_invoker(
    driver: Option<ProcessDriverHandle>,
    runtime: runmat_runtime::context::RuntimeContext,
    environment_roots: Option<Vec<runmat_gc::ExplicitRoot>>,
    factory: impl FnOnce() -> Result<user_functions::BoundFunctionFuture, RuntimeError> + 'static,
) -> Result<runmat_runtime::execution::RootedValueSequence, RuntimeError> {
    let Some(driver) = driver else {
        let _environment_roots = environment_roots;
        return factory()?.await;
    };
    if runtime
        .service_ports()
        .local_tasks()
        .and_then(|tasks| tasks.canonical_admission())
        .is_some()
    {
        return runmat_runtime::context::run_selected_callback_continuation(&runtime, async move {
            let _environment_roots = environment_roots;
            factory()?.await
        })
        .await;
    }
    let queued_scope = runtime.prepare_poll_scope()?;
    let (outputs, _) = driver
        .enqueue_scoped_in_context(&runtime, queued_scope, move || async move {
            let _environment_roots = environment_roots;
            let outputs = factory()?.await?;
            let captures = runmat_runtime::execution::RootedValues::new(Vec::new())
                .map_err(|error| error.into_runtime_error())?;
            Ok((outputs, captures))
        })?
        .await?;
    Ok(outputs)
}

fn output_sequence(
    output_values: Vec<Value>,
    requested_outputs: usize,
) -> Result<runmat_value::ValueSequence, RuntimeError> {
    match requested_outputs {
        0 => Ok(runmat_value::ValueSequence::empty()),
        1 => runmat_value::ValueSequence::single(
            output_values.into_iter().next().unwrap_or(Value::Num(0.0)),
        )
        .map_err(runmat_runtime::sequence::sequence_error_to_runtime),
        _ => runmat_value::ValueSequence::comma_separated(
            output_values.into_iter().take(requested_outputs).collect(),
        )
        .map_err(runmat_runtime::sequence::sequence_error_to_runtime),
    }
}

#[derive(Debug)]
pub struct WorkspaceInterpreterOutcome {
    pub outcome: VmResult<InterpreterOutcome>,
    pub workspace: Option<crate::runtime::workspace::WorkspaceExecutionCompletion>,
}

/// Execute one owned VM frame under explicit invocation/workspace authority.
///
/// The input owner is transferred, never synchronized into caller scratch.
/// Ready hands the exact completion to the caller; abandonment closes and
/// retires that frame. An embedding projection's terminal publication remains
/// its host's obligation (WorkspaceProjectionPublication), not register replay.
pub async fn interpret_workspace(
    bytecode: &Bytecode,
    initial_vars: InterpreterVars,
    current_function_name: Option<&str>,
    runtime: runmat_runtime::context::RuntimeContext,
    admission: Option<crate::runtime::workspace::PendingWorkspaceAdmission>,
    dynamic_admission: Option<DynamicEvalAdmissionPort>,
) -> WorkspaceInterpreterOutcome {
    let arity = runtime
        .invocation_arity()
        .unwrap_or(RuntimeInvocationArity::Script);
    let runtime = runtime.with_invocation_arity(arity);
    let (runtime, workspace_activation) =
        crate::runtime::workspace_service::VmWorkspaceService::install(runtime, admission);
    let frame = workspace_activation.frame();
    let runtime = with_default_acceleration_service(runtime);
    let runtime = runtime_with_bytecode_debug_frame(
        runtime,
        bytecode,
        current_function_name.unwrap_or("<main>"),
    );
    let execution = workspace_activation.scope(runmat_runtime::data::with_tx_registry_scope(
        scope_vm_pc(interpret_workspace_inner(
            bytecode,
            initial_vars,
            current_function_name,
            runtime.clone(),
            dynamic_admission,
        )),
    ));
    runtime
        .scope(workspace_execution::PendingWorkspaceExecution::new(
            execution, frame,
        ))
        .await
}

pub async fn interpret_resume_in_context(
    bytecode: &Bytecode,
    resume: InterpreterResumeState,
    current_function_name: Option<&str>,
    runtime: runmat_runtime::context::RuntimeContext,
) -> VmResult<InterpreterOutcome> {
    interpret_resume_with_environment_in_context(
        bytecode,
        resume,
        current_function_name,
        runtime,
        None,
        Vec::new(),
        Vec::new(),
    )
    .await
}

pub async fn interpret_resume_with_environment_in_context(
    bytecode: &Bytecode,
    resume: InterpreterResumeState,
    current_function_name: Option<&str>,
    runtime: runmat_runtime::context::RuntimeContext,
    lexical_environment: Option<runmat_value::LexicalEnvironment>,
    lexical_binding_slots: Vec<crate::bytecode::program::LexicalBindingSlot>,
    inherited_storage_slots: Vec<crate::bytecode::program::InheritedStorageSlot>,
) -> VmResult<InterpreterOutcome> {
    interpret_resume_with_dynamic_admission_in_context(
        bytecode,
        resume,
        current_function_name,
        runtime,
        DynamicResumeSelection {
            lexical_environment,
            lexical_binding_slots,
            inherited_storage_slots,
            dynamic_admission: None,
        },
    )
    .await
}

pub struct DynamicResumeSelection {
    pub lexical_environment: Option<runmat_value::LexicalEnvironment>,
    pub lexical_binding_slots: Vec<crate::bytecode::program::LexicalBindingSlot>,
    pub inherited_storage_slots: Vec<crate::bytecode::program::InheritedStorageSlot>,
    pub dynamic_admission: Option<DynamicEvalAdmissionPort>,
}

pub async fn interpret_resume_with_dynamic_admission_in_context(
    bytecode: &Bytecode,
    resume: InterpreterResumeState,
    current_function_name: Option<&str>,
    runtime: runmat_runtime::context::RuntimeContext,
    selected: DynamicResumeSelection,
) -> VmResult<InterpreterOutcome> {
    interpret_resume_with_owner_in_context(
        bytecode,
        None,
        resume,
        current_function_name,
        runtime,
        selected,
    )
    .await
}

pub(crate) async fn interpret_resume_shared_in_context(
    owner: crate::interpreter::bytecode_owner::BytecodeLease,
    resume: InterpreterResumeState,
    current_function_name: Option<&str>,
    runtime: runmat_runtime::context::RuntimeContext,
) -> VmResult<InterpreterOutcome> {
    interpret_resume_with_owner_in_context(
        &owner,
        Some(owner.clone()),
        resume,
        current_function_name,
        runtime,
        DynamicResumeSelection {
            lexical_environment: None,
            lexical_binding_slots: Vec::new(),
            inherited_storage_slots: Vec::new(),
            dynamic_admission: None,
        },
    )
    .await
}

async fn interpret_resume_with_owner_in_context(
    bytecode: &Bytecode,
    owner: Option<crate::interpreter::bytecode_owner::BytecodeLease>,
    resume: InterpreterResumeState,
    current_function_name: Option<&str>,
    runtime: runmat_runtime::context::RuntimeContext,
    selected: DynamicResumeSelection,
) -> VmResult<InterpreterOutcome> {
    let _environment_roots = selected
        .lexical_environment
        .as_ref()
        .map(runmat_runtime::call::lexical::root_environment)
        .transpose()?;
    // Resume authority is checked before installing workspace services or
    // reconstructing any local. A foreign continuation is not a clear.
    let selected_authority = runtime.session_variable_authority();
    let selected_function = current_function_name.unwrap_or("<main>");
    for (slot, binding) in &resume.persistent_aliases {
        let owner = selected
            .inherited_storage_slots
            .iter()
            .find(|entry| {
                entry.slot == *slot
                    && entry.kind == crate::bytecode::program::InheritedStorageKind::Persistent
                    && binding.matches_inherited_owner(&entry.owner_function, &entry.name)
            })
            .map_or(selected_function, |entry| entry.owner_function.as_str());
        binding.validate_selected(&selected_authority, owner)?;
    }
    let runtime = runtime.with_invocation_arity(RuntimeInvocationArity::function(
        resume.supplied_inputs,
        resume.requested_outputs,
    ));
    let (runtime, workspace_activation) =
        crate::runtime::workspace_service::VmWorkspaceService::install(runtime, None);
    let runtime = with_default_acceleration_service(runtime);
    let runtime = runtime_with_bytecode_debug_frame(
        runtime,
        bytecode,
        current_function_name.unwrap_or("<main>"),
    );
    workspace_activation
        .scope(
            runtime.scope(runmat_runtime::data::with_tx_registry_scope(scope_vm_pc(
                interpret_resume_inner(
                    bytecode,
                    owner,
                    resume,
                    current_function_name,
                    runtime.clone(),
                    selected,
                ),
            ))),
        )
        .await
}

fn with_default_acceleration_service(
    runtime: runmat_runtime::context::RuntimeContext,
) -> runmat_runtime::context::RuntimeContext {
    // A nested interpreter owns a distinct instruction stream and cannot
    // confirm its caller's installed region, even if it inherits other ports.
    let runtime = runtime.without_vm_fusion_host_route();
    #[cfg(feature = "native-accel")]
    let runtime = runmat_accelerate::native_auto::install_preparation_service(runtime);
    runtime
}

#[path = "runner/lexical_storage.rs"]
mod lexical_storage;
#[path = "runner/native_metadata.rs"]
mod native_metadata;
use native_metadata::register_class_metadata;
pub use native_metadata::{
    prepare_admitted_execution_metadata, prepare_admitted_execution_metadata_batch,
    prepare_admitted_execution_metadata_for_class_batch, prepare_native_execution_metadata,
    unresolved_admitted_class_parents, unresolved_admitted_class_parents_for, SelectedClassProduct,
    SelectedDeclarationProduct,
};

async fn interpret_resume_inner(
    bytecode: &Bytecode,
    owner: Option<crate::interpreter::bytecode_owner::BytecodeLease>,
    resume: InterpreterResumeState,
    current_function_name: Option<&str>,
    runtime: runmat_runtime::context::RuntimeContext,
    selected: DynamicResumeSelection,
) -> VmResult<InterpreterOutcome> {
    let DynamicResumeSelection {
        lexical_environment,
        lexical_binding_slots,
        dynamic_admission,
        ..
    } = selected;
    if resume.pc > bytecode.instructions.len() {
        return Err(mex(
            "NativeResumePoint",
            "interpreter resume PC is outside the immutable bytecode product",
        ));
    }
    if resume
        .completion_boundary
        .is_some_and(|boundary| boundary.pc() > bytecode.instructions.len())
    {
        return Err(mex(
            "NativeResumeBoundary",
            "interpreter completion boundary is outside the immutable bytecode product",
        ));
    }
    if resume.vars.len() != bytecode.var_count {
        return Err(mex(
            "NativeResumeFrame",
            "materialized native locals do not match the VM frame layout",
        ));
    }
    let vars = resume.vars;
    let assigned_slots = resume.assigned_slots;
    let invocation_arity =
        RuntimeInvocationArity::function(resume.supplied_inputs, resume.requested_outputs);
    // The count describes the dense frame extent; the exact sparse assignment
    // set stays in the resumed frame instead of modifying the compiler product.
    let initial_assigned_var_count = vars.len();
    let mut state = match owner {
        Some(owner) => InterpreterState::from_shared_vars_in_context(
            owner,
            vars,
            initial_assigned_var_count,
            current_function_name,
            invocation_arity,
            runtime,
        )?,
        None => InterpreterState::from_vars_in_context(
            bytecode.clone(),
            vars,
            initial_assigned_var_count,
            current_function_name,
            invocation_arity,
            runtime,
        )?,
    };
    state.context.lexical_environment = lexical_environment;
    state.context.dynamic_eval_admission = dynamic_admission;
    state.context.lexical_binding_slots = lexical_binding_slots;
    state.pc = resume.pc;
    state.completion_boundary = resume.completion_boundary;
    state.completion_assigned_slots = Some(assigned_slots);
    state.missing_input_slots = resume.missing_input_slots;
    state.global_aliases = runmat_runtime::workspace::bindings::FrameGlobalBindings::from_snapshot(
        resume.global_aliases,
    );
    state.persistent_aliases = resume.persistent_aliases;
    register_class_metadata(&bytecode.instructions[..resume.pc])?;
    for instruction in &bytecode.instructions[..resume.pc] {
        if let Instr::RegisterImport { path, wildcard } = instruction {
            state.imports.push((path.clone(), *wildcard));
        }
    }
    debug!(
        "interpreter resumed from native execution at pc={} side_effect_epoch={}",
        state.pc, resume.side_effect_epoch
    );
    let current_name = current_function_name.unwrap_or("<main>");
    let source_id = bytecode
        .source_id
        .and_then(|source| source.resolve_in(&state.context.runtime));
    let (source_owner, diagnostic_source) =
        selected_frame_source_basis(bytecode, &state.context.runtime)
            .map_or((None, None), |(owner, diagnostic)| {
                (Some(owner), Some(diagnostic))
            });
    // Resume already owns its InterpreterVars generation. Unlike the ordinary
    // Vec-input ABI, it has no caller-owned Vec to synchronize on completion.
    let result = run_interpreter_owned(Box::new(state)).await;
    match result {
        Ok(outcome) => Ok(outcome),
        Err(err) => {
            let err = attach_span_from_pc(bytecode, err);
            Err(attach_call_frames(
                current_name,
                source_id,
                source_owner,
                diagnostic_source,
                err,
            ))
        }
    }
}

async fn interpret_workspace_inner(
    bytecode: &Bytecode,
    initial_vars: InterpreterVars,
    current_function_name: Option<&str>,
    runtime: runmat_runtime::context::RuntimeContext,
    dynamic_admission: Option<DynamicEvalAdmissionPort>,
) -> VmResult<InterpreterOutcome> {
    bytecode
        .validate_selected_source_owners(&runtime)
        .map_err(RuntimeError::new)?;
    crate::bytecode::validate_sequence_register_flow(&bytecode.instructions)
        .map_err(RuntimeError::new)?;
    crate::bytecode::validate_stack_geometry(&bytecode.instructions, bytecode.stack_geometry)
        .map_err(RuntimeError::new)?;
    let source_id = bytecode
        .source_id
        .and_then(|source| source.resolve_in(&runtime));
    let (source_owner, diagnostic_source) = selected_frame_source_basis(bytecode, &runtime)
        .map_or((None, None), |(owner, diagnostic)| {
            (Some(owner), Some(diagnostic))
        });
    let invocation_arity = runtime
        .invocation_arity()
        .unwrap_or(RuntimeInvocationArity::Script);
    let initial_assigned_var_count = initial_vars.len();
    let borrowed_inputs =
        crate::runtime::gc::ManagedRoots::from_traced(&initial_vars).map_err(RuntimeError::new)?;
    let initial_vars = initial_vars.prepare_extent(&runtime, bytecode.var_count)?;
    let mut state = Box::new(InterpreterState::from_vars_in_context(
        bytecode.clone(),
        initial_vars,
        initial_assigned_var_count,
        current_function_name,
        invocation_arity,
        runtime,
    )?);
    state.context.dynamic_eval_admission = dynamic_admission;
    let result = Box::pin(run_interpreter_owned(state)).await;
    drop(borrowed_inputs);
    match result {
        Ok(outcome) => Ok(outcome),
        Err(err) => {
            let err = attach_span_from_pc(bytecode, err);
            let current_name = current_function_name.unwrap_or("<main>");
            Err(attach_call_frames(
                current_name,
                source_id,
                source_owner,
                diagnostic_source,
                err,
            ))
        }
    }
}

/// Attach an error-frame token only from the selected executable's issued
/// source table. A function body uses its exact retained definition owner;
/// scripts and native resume without an active function use the explicitly
/// bound invocation table. Display names and numeric SourceIds never select
/// an owner.
fn selected_frame_source_basis(
    bytecode: &Bytecode,
    runtime: &runmat_runtime::context::RuntimeContext,
) -> Option<(
    runmat_runtime::CallFrameSourceOwner,
    runmat_runtime::CallFrameDiagnosticSource,
)> {
    let source = bytecode.source_id?;
    let issued = source.runtime()?;
    issued.resolve_in(runtime)?;
    if let Some(function) = bytecode.active_function {
        let body = bytecode.function_registry.get(function)?;
        if body.source_id != Some(source) {
            return None;
        }
        let owner = body.selected_source_owner.as_ref()?;
        let ordinal = owner.source_ordinal_for(issued)?;
        let token = owner.call_frame_source_owner_for(ordinal)?;
        let info = owner.issued_source_info_for(ordinal)?;
        Some((
            token,
            runmat_runtime::CallFrameDiagnosticSource::new(
                std::sync::Arc::clone(&info.name),
                std::sync::Arc::clone(&info.text),
            ),
        ))
    } else {
        let ordinal = runtime.portable_ordinal_for_source(issued)?;
        let token = runtime.call_frame_source_owner_for_program_source(ordinal)?;
        let source = issued.resolve_in(runtime)?;
        let info = runmat_runtime::source_context::source_info_in_context(runtime, source)?;
        Some((
            token,
            runmat_runtime::CallFrameDiagnosticSource::new(info.name, info.text),
        ))
    }
}

fn bytecode_frame_span(bytecode: &Bytecode) -> Option<(usize, usize)> {
    bytecode
        .instr_spans
        .first()
        .map(|span| (span.start, span.end))
}

fn runtime_with_bytecode_debug_frame(
    runtime: runmat_runtime::context::RuntimeContext,
    bytecode: &Bytecode,
    name: &str,
) -> runmat_runtime::context::RuntimeContext {
    let source_id = bytecode
        .source_id
        .and_then(|source| source.resolve_in(&runtime));
    runtime.with_debug_frame(runmat_runtime::debug_context::DebugFrame {
        function: name.to_string(),
        source_id,
        span: bytecode_frame_span(bytecode),
    })
}

async fn run_interpreter_owned(mut state: Box<InterpreterState>) -> VmResult<InterpreterOutcome> {
    let runtime = state.context.runtime.clone();
    let owner = if state.context.process_driver.is_none() {
        let owner = ProcessContinuationDriver::new_with_dynamic_admission_in_context(
            &runtime,
            state.context.dynamic_eval_admission.clone(),
        )?;
        state.context.process_driver = Some(owner.handle());
        Some(owner)
    } else {
        None
    };
    state
        .bytecode
        .validate_catch_scope_metadata()
        .map_err(RuntimeError::new)?;
    state
        .bytecode
        .validate_selected_source_owners(&state.context.runtime)
        .map_err(RuntimeError::new)?;
    let catch_runtime = state.context.runtime.clone();
    let inherited_catch = catch_runtime.active_catch_exception_shared();
    let run_span = info_span!("interpreter.run", function = state.bytecode.display_name());
    #[cfg(feature = "native-accel")]
    let fusion_plan = state.fusion_plan.clone();
    let make_interpreter = move || {
        let interpreter = catch_runtime.scope_active_catch_shared(
            inherited_catch,
            run_interpreter_inner(*state).instrument(run_span),
        );
        #[cfg(feature = "native-accel")]
        let interpreter = runmat_accelerate::scope_fusion_plan(fusion_plan, interpreter);
        interpreter
    };
    if let Some(owner) = owner {
        owner.drive_in_context(&runtime, make_interpreter)?.await
    } else {
        make_interpreter().await
    }
}

async fn run_interpreter_inner(state: InterpreterState) -> VmResult<InterpreterOutcome> {
    ensure_wasm_builtins_registered();
    ensure_workspace_resolver_registered();
    let InterpreterState {
        mut stack,
        mut vars,
        mut pc,
        completion_boundary,
        mut completion_assigned_slots,
        mut context,
        mut try_stack,
        mut last_exception,
        mut imports,
        global_aliases,
        mut persistent_aliases,
        mut missing_input_slots,
        initial_assigned_var_count,
        #[cfg(feature = "native-accel")]
        fusion_plan,
        #[cfg(feature = "native-accel")]
        fusion_accel_graph,
        #[cfg(feature = "native-accel")]
        fusion_installation,
        #[cfg(all(
            test,
            not(target_arch = "wasm32"),
            feature = "native-accel",
            feature = "wgpu"
        ))]
        mut fusion_publication_fault,
        bytecode,
    } = state;
    context.bytecode_owner = bytecode.shared_lease();
    let current_function_name = bytecode.display_name();
    let _source_context_guard = runmat_runtime::source_context::replace_current_source_id(
        bytecode
            .source_id
            .and_then(|source| source.resolve_in(&context.runtime)),
    );
    let invocation_arity = context
        .runtime
        .invocation_arity()
        .expect("interpreter state has selected script or function arity");
    let function_registry = std::rc::Rc::new(bytecode.function_registry());
    let catch_runtime = context.runtime.clone();
    let mut catch_exceptions: HashMap<
        runmat_mir::CatchScopeId,
        std::rc::Rc<runmat_value::MException>,
    > = HashMap::new();
    let catch_entries: HashMap<usize, runmat_mir::CatchScopeId> = bytecode
        .instructions
        .iter()
        .filter_map(|instruction| match instruction {
            Instr::EnterTry {
                scope, catch_pc, ..
            } => Some((*catch_pc, runmat_mir::CatchScopeId(*scope))),
            _ => None,
        })
        .collect();
    let nested_runtime = context.runtime.clone();
    let nested_driver = context.process_driver.clone();
    let previous_semantic_invoker = user_functions::current_semantic_function_invoker();
    // Numeric/name calls select the currently published registry. An old
    // bound body's static edges are routed through the distinct typed bound
    // invoker below; an overlay must never shadow a dynamic name lookup.
    let registry_for_function_invoker = std::rc::Rc::new(function_registry.current_view());
    let _semantic_function_guard = user_functions::install_local_semantic_function_invoker(
        std::rc::Rc::new(
            move |selected,
                  function: usize,
                  args: runmat_runtime::call::RootedFunctionArguments,
                  requested_outputs: usize| {
                let lexical_environment = args.callable().and_then(|callable| {
                    let Value::Closure(closure) = callable.value() else {
                        return None;
                    };
                    closure.lexical_environment.clone()
                });
                let previous_invoker = previous_semantic_invoker.clone();
                let function_registry = std::rc::Rc::clone(&registry_for_function_invoker);
                let same_owner = selected.as_ref().is_none_or(|context| {
                    context.local_identity() == nested_runtime.local_identity()
                });
                let runtime = if same_owner {
                    selected.map_or_else(
                        || nested_runtime.clone(),
                        |context| {
                            nested_runtime
                                .clone()
                                .try_with_inherited_callback_invocation_authority_from(&context)
                                .expect(
                                    "callback Runtime owner checked before authority projection",
                                )
                        },
                    )
                } else {
                    nested_runtime.clone()
                };
                let driver = nested_driver.clone();
                let admission_runtime = runtime.clone();
                if same_owner
                    && function_registry
                        .requires_selected_constructor_admission(runmat_hir::FunctionId(function))
                {
                    return user_functions::admit_user_function_with(
                        Some(&admission_runtime),
                        move || async move {
                            let binding = function_registry
                                .exact_bound_binding(runmat_hir::FunctionId(function))
                                .ok_or_else(|| mex(
                                    "ConstructorOwnerInvalid",
                                    "selected semantic constructor lacks its exact definition lease",
                                ))?;
                            user_functions::try_call_rooted_bound_function_with_environment_in_context(
                                &runtime, binding, args, requested_outputs, lexical_environment,
                            ).await.ok_or_else(|| mex(
                                "UnavailableBoundFunction",
                                "selected semantic constructor has no bound executor",
                            ))?
                        },
                    );
                }
                user_functions::admit_user_function_with(
                    Some(&admission_runtime),
                    move || async move {
                        if !same_owner {
                            return Err(runmat_runtime::build_runtime_error(
                                "semantic callback selected another VM runtime owner",
                            )
                            .with_identifier("RunMat:VmCallbackOwner")
                            .build());
                        }
                        let local_function = function_registry
                            .get(runmat_hir::FunctionId(function))
                            .is_some();
                        if !local_function {
                            if let Some(invoker) = previous_invoker {
                                let future = invoker(
                                    Some(runtime.clone()),
                                    function,
                                    args,
                                    requested_outputs,
                                )?;
                                return await_prior_semantic_invoker(driver, runtime, future).await;
                            }
                        }
                        invoke_semantic_function_with_rooted_captures(
                            function,
                            args,
                            requested_outputs,
                            function_registry.as_ref().clone(),
                            runtime,
                            driver,
                            lexical_environment,
                        )
                        .await
                        .map(|(value, _)| value)
                    },
                )
            },
        ),
    )?;
    let previous_bound_invoker = user_functions::current_bound_function_invoker();
    let registry_for_bound_invoker = std::rc::Rc::clone(&function_registry);
    let bound_driver = context.process_driver.clone();
    let _bound_function_guard =
        user_functions::install_local_bound_function_invoker(std::rc::Rc::new(
            move |runtime, binding, args, requested_outputs, lexical_environment| {
                let previous = previous_bound_invoker.clone();
                let registry = std::rc::Rc::clone(&registry_for_bound_invoker);
                let driver = bound_driver.clone();
                runmat_runtime::execution::admit_executable_sequence_with(
                    &runtime.clone(),
                    move || {
                        bound_invocation::prepare(
                            bound_invocation::Invocation {
                                runtime,
                                binding,
                                args,
                                requested_outputs,
                                driver,
                                lexical_environment,
                            },
                            registry,
                            previous,
                        )
                    },
                )
            },
        ))?;
    let previous_semantic_resolver = user_functions::current_semantic_function_resolver();
    let previous_semantic_candidates = user_functions::current_semantic_name_candidates();
    let registry_for_function_resolver = function_registry.current_view();
    let published_name_resolution = registry_for_function_resolver.private_name_resolution_view();
    let selected_private_name_resolution = function_registry.private_name_resolution_view();
    let registry_for_scoped_names = std::rc::Rc::clone(&function_registry);
    let _scoped_name_target_guard = user_functions::install_scoped_name_target_resolver(Some(
        std::rc::Rc::new(move |name: &str| {
            user_functions::current_active_semantic_function().and_then(|caller| {
                registry_for_scoped_names
                    .scoped_private_name_target_for_caller(runmat_hir::FunctionId(caller), name)
            })
        }),
    ))?;
    let current_name_facts = Arc::new(
        registry_for_function_resolver.semantic_name_candidate_facts_in(
            &context.runtime,
            registry_for_function_resolver.functions.keys().copied(),
        ),
    );
    let retained_name_facts = Arc::new(
        function_registry.semantic_name_candidate_facts_in(
            &context.runtime,
            user_functions::current_active_semantic_function()
                .and_then(|caller| {
                    function_registry.definition_dependency_owners(runmat_hir::FunctionId(caller))
                })
                .unwrap_or_default()
                .into_iter()
                .map(|(id, _)| id),
        ),
    );
    let semantic_candidates: Arc<user_functions::SemanticNameCandidates> =
        Arc::new(move |name: &str| {
            let mut candidates = Vec::new();
            if let Some(active_function) = user_functions::current_active_semantic_function() {
                if let Some(scoped_function) = selected_private_name_resolution
                    .resolve_private_name_for_caller(runmat_hir::FunctionId(active_function), name)
                {
                    if let Some(candidate) = retained_name_facts
                        .get(&scoped_function)
                        .or_else(|| current_name_facts.get(&scoped_function))
                    {
                        candidates.push(candidate.clone());
                    }
                }
            }
            if let Some(function) = published_name_resolution.resolve_name(name) {
                if let Some(candidate) = current_name_facts.get(&function) {
                    if !candidates
                        .iter()
                        .any(|existing| existing.same_definition(candidate))
                    {
                        candidates.push(candidate.clone());
                    }
                }
            }
            if let Some(previous) = &previous_semantic_candidates {
                for candidate in previous(name) {
                    if !candidates
                        .iter()
                        .any(|existing| existing.same_definition(&candidate))
                    {
                        candidates.push(candidate);
                    }
                }
            }
            if let Some(previous) = previous_semantic_resolver
                .as_ref()
                .and_then(|resolver| resolver(name))
            {
                if !candidates
                    .iter()
                    .any(|candidate| candidate.function == previous)
                {
                    candidates.push(user_functions::SemanticNameCandidate {
                        function: previous,
                        source: None,
                        definition_owner: None,
                    });
                }
            }
            candidates
        });
    let resolver_candidates = Arc::clone(&semantic_candidates);
    let _semantic_resolver_guard =
        user_functions::install_semantic_function_resolver(Some(Arc::new(move |name: &str| {
            resolver_candidates(name)
                .into_iter()
                .next()
                .map(|candidate| candidate.function)
        })))?;
    let _semantic_candidates_guard =
        user_functions::install_semantic_name_candidates(Some(semantic_candidates))?;
    let execution_sources = function_registry.execution_source_resolver_in(&context.runtime);
    let _execution_source_guard = user_functions::install_execution_source_resolver(Some(
        std::rc::Rc::clone(&execution_sources),
    ))?;
    let mut source_function_catalog = function_registry
        .functions
        .values()
        .filter_map(|function| {
            function
                .source_id
                .and_then(|source| source.resolve_in(&context.runtime))
                .map(
                    |source_id| runmat_runtime::user_functions::SourceFunctionInfo {
                        source_id,
                        name: function.display_name.clone(),
                        function: function.function.0,
                        role: function.source_role,
                        binding_origin: function.binding_origin,
                        source_file: function.source_file.clone(),
                        definition_owners: None,
                    },
                )
        })
        .collect::<Vec<_>>();
    source_function_catalog.sort_by_key(|info| info.function);
    let _source_function_catalog_guard =
        user_functions::install_source_function_catalog(Some(Arc::new(source_function_catalog)))?;
    super_constructor::bind_selected(
        &mut context,
        &bytecode,
        &function_registry,
        &execution_sources,
    )?;
    let initial_assignment = match completion_assigned_slots.as_ref() {
        Some(assigned) => interp_engine::InitialFrameAssignment::Resumed { assigned },
        None => interp_engine::InitialFrameAssignment::Compiled {
            count: initial_assigned_var_count,
            unassigned: &bytecode.initially_unassigned_slots,
        },
    };
    let _workspace_guard = interp_engine::prepare_workspace_guard(
        &bytecode.var_names,
        &mut vars,
        initial_assignment,
        global_aliases.clone(),
    )?;
    let inherited_storage = std::mem::take(&mut context.inherited_storage_slots);
    let mut _storage_roots = Vec::new();
    if !inherited_storage.is_empty() {
        _storage_roots.push(
            runmat_runtime::execution::RootedValueLease::for_traced(&vars)
                .map_err(|error| error.into_runtime_error())?,
        );
    }
    for binding in inherited_storage {
        if binding.slot >= vars.len() {
            return Err(mex(
                "LexicalStorage",
                "inherited storage slot is outside its frame",
            ));
        }
        match binding.kind {
            crate::bytecode::program::InheritedStorageKind::Global => {
                crate::runtime::globals::declare_global_named(
                    std::slice::from_ref(&binding.slot),
                    std::slice::from_ref(&binding.name),
                    &mut vars,
                    &global_aliases,
                )?;
                crate::runtime::workspace::reconcile_workspace_vars(&mut vars);
            }
            crate::bytecode::program::InheritedStorageKind::Persistent => {
                let prepared = crate::runtime::workspace::prepare_workspace_persistent_slot(
                    context.runtime.local_identity(),
                    binding.slot,
                    &context.runtime,
                )?;
                if let Some(prepared) = &prepared {
                    prepared.checkpoint_before_publication()?;
                }
                crate::interpreter::persistent::declare(
                    &mut persistent_aliases,
                    &mut vars,
                    &context.runtime.session_variable_authority(),
                    &binding.owner_function,
                    binding.slot,
                    crate::interpreter::persistent::PersistentAlias::InheritedNamed(binding.name),
                )?;
                if let Some(prepared) = prepared {
                    let value = vars
                        .get(binding.slot)
                        .expect("persistent binding slot lies within frame extent")
                        .clone();
                    prepared.commit_owned_value(value);
                }
            }
        }
        _storage_roots.push(
            runmat_runtime::execution::RootedValueLease::for_traced(&vars)
                .map_err(|error| error.into_runtime_error())?,
        );
    }
    crate::runtime::workspace::attach_lexical_workspace_bindings(
        context.lexical_environment.as_ref(),
        &context.lexical_binding_slots,
    )
    .map_err(|error| mex("LexicalEnvironment", &error))?;
    let thread_roots: Vec<Value> = runtime_globals::collect_thread_roots();
    let mut sequence_register = interp_dispatch::SequenceState::new_in_context(&context.runtime)?;
    // All owner objects stay at these frame addresses through the inner
    // instruction future. Register them together before admitting mutations;
    // the residency guard drops first, then these registrations, then owners.
    let _gc_context = interp_engine::create_gc_context(
        &stack,
        &vars,
        &context.locals,
        &sequence_register,
        thread_roots,
    )?;
    // GC registrations retain managed frame objects during suspension. GPU
    // values own their allocation leases directly, including returned aliases.
    let debug_stack = interp_engine::debug_stack_enabled();
    let mut interpreter_timing = InterpreterTiming::new();
    #[cfg(feature = "native-accel")]
    let mut fusion_observations = super::fusion_observation::FusionRegionObservations::default();
    let cancellation = context.runtime.cancellation();
    // One wakeable scope owns all instruction execution, including provider
    // fusion and idioms that execute outside ordinary bytecode dispatch. Keep
    // roots and frame state outside it until suspended work has been dropped.
    #[cfg(feature = "native-accel")]
    let image_installation = bytecode.prepare_vm_image_plan_installation(&context.runtime);
    let instruction_loop = instruction_loop::boxed(instruction_loop::InstructionLoopContext {
        bytecode: &bytecode,
        pc: &mut pc,
        stack: &mut stack,
        vars: &mut vars,
        completion_boundary,
        completion_assigned_slots: &mut completion_assigned_slots,
        context: &mut context,
        try_stack: &mut try_stack,
        last_exception: &mut last_exception,
        imports: &mut imports,
        global_aliases: &global_aliases,
        persistent_aliases: &mut persistent_aliases,
        missing_input_slots: &mut missing_input_slots,
        current_function_name,
        invocation_arity,
        function_registry: &function_registry,
        catch_runtime: &catch_runtime,
        catch_exceptions: &mut catch_exceptions,
        catch_entries: &catch_entries,
        sequence_register: &mut sequence_register,
        debug_stack,
        interpreter_timing: &mut interpreter_timing,
        #[cfg(feature = "native-accel")]
        fusion_accel_graph: &fusion_accel_graph,
        #[cfg(feature = "native-accel")]
        fusion_plan: &fusion_plan,
        #[cfg(feature = "native-accel")]
        fusion_installation: &fusion_installation,
        #[cfg(feature = "native-accel")]
        image_installation: &image_installation,
        #[cfg(feature = "native-accel")]
        fusion_observations: &mut fusion_observations,
        #[cfg(all(
            test,
            not(target_arch = "wasm32"),
            feature = "native-accel",
            feature = "wgpu"
        ))]
        fusion_publication_fault: &mut fusion_publication_fault,
    })?;
    let execution_result =
        super::dispatch_cancellation::until_cancelled(&cancellation, instruction_loop).await;
    // The suspended operation is gone before cleanup touches its frame.
    // Cancellation bypasses language catch handlers and never completes an
    // unfinished placement observation successfully.
    let Some(execution_result) = execution_result else {
        #[cfg(feature = "native-accel")]
        fusion_observations.abort();
        return Err(interp_engine::cancellation_error());
    };
    execution_result?;
    // A completion boundary or return can stop inside a region; never infer
    // success from the function's eventual return value.
    #[cfg(feature = "native-accel")]
    fusion_observations.abort();
    interpreter_timing.flush_host_span("loop_complete", None);
    crate::interpreter::persistent::finish_success(
        &persistent_aliases,
        &mut vars,
        &context.runtime.session_variable_authority(),
        current_function_name,
    )?;
    crate::runtime::workspace::reconcile_workspace_vars(&mut vars);
    let ans_slot = crate::runtime::workspace::current_workspace_assigned_slot("ans");
    let mut assigned_slots = completion_assigned_slots
        .unwrap_or_else(crate::runtime::workspace::current_workspace_assigned_slots);
    if let Some(slot) = ans_slot {
        assigned_slots.insert(slot);
    }
    Ok(InterpreterOutcome::Completed(
        runmat_runtime::execution::RootedResult::new(
            crate::interpreter::state::InterpreterCompletion {
                ans_slot,
                assigned_slots,
                values: vars,
            },
        )
        .map_err(|error| error.into_runtime_error())?,
    ))
}

pub async fn interpret(
    bytecode: &Bytecode,
) -> Result<runmat_runtime::execution::RootedValues, RuntimeError> {
    // Frame preparation reserves compiled slots. This entry supplies no
    // existing values: padding here would mark every binding as assigned.
    let runtime = active_or_standalone_runtime_context();
    match interpret_workspace(
        bytecode,
        InterpreterVars::from_owned(Vec::new()),
        Some("<main>"),
        runtime,
        None,
        None,
    )
    .await
    .outcome
    {
        Ok(InterpreterOutcome::Completed(completion)) => completion.with_owned(|completion| {
            let values = completion.values.into_owned().map_err(|_| {
                mex(
                    "InterpreterCompletion",
                    "ordinary script completion retained workspace-backed slots",
                )
            })?;
            runmat_runtime::execution::RootedValues::new(values)
                .map_err(|error| error.into_runtime_error())
        }),
        Err(e) => Err(e),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ParforTaskMode {
    Sequential,
    IndependentIterations,
}

pub(crate) struct ParforTaskExecution<'a> {
    pub bytecode: &'a Bytecode,
    pub bytecode_owner: Option<crate::interpreter::bytecode_owner::BytecodeLease>,
    pub region: &'a crate::BytecodeParforRegion,
    pub inputs: runmat_value::OutputListStorage,
    pub iterations: runmat_value::OutputListStorage,
    pub randomness: runmat_execution::ParallelRandomnessContext,
    pub mode: ParforTaskMode,
    pub current_function_name: &'a str,
    pub runtime: runmat_runtime::context::RuntimeContext,
    pub input_roots: Option<runmat_runtime::execution::RootedValueLease>,
}

pub(crate) async fn interpret_parfor_task_in_context(
    mut execution: ParforTaskExecution<'_>,
) -> Result<runmat_runtime::execution::RootedValues, RuntimeError> {
    // Bind the lease before splitting its Value owners into locals, a frame,
    // or an iterator. On every validation error those later owners retire
    // before this registration; an unpolled queued task retains the struct's
    // last-field lease independently.
    let _input_roots = match execution.input_roots.take() {
        Some(roots) => roots,
        None => runmat_runtime::execution::RootedValueLease::for_value_slices_in_context(
            &[execution.inputs.as_slice(), execution.iterations.as_slice()],
            &execution.runtime,
        )?,
    };
    let ParforTaskExecution {
        bytecode,
        bytecode_owner,
        region,
        inputs,
        iterations,
        input_roots: _,
        randomness,
        mode,
        current_function_name,
        runtime,
    } = execution;
    let inputs = inputs.try_into_owned().map_err(|_| {
        mex(
            "ParallelTaskInput",
            "parallel inputs require their original unique collection owner",
        )
    })?;
    let iterations = iterations.try_into_owned().map_err(|_| {
        mex(
            "ParallelTaskInput",
            "parallel iterations require their original unique collection owner",
        )
    })?;
    randomness
        .validate_for_iterations(iterations.len())
        .map_err(|error| mex("ParallelRandomness", &error.to_string()))?;
    let deterministic = randomness.is_deterministic();
    let runtime = if deterministic {
        runtime.with_parallel_random_scope()?
    } else {
        runtime
    };
    let input_variables = region.input_variables().collect::<Vec<_>>();
    if inputs.len() != input_variables.len() {
        return Err(mex(
            "ParallelTaskInput",
            "parallel task inputs do not match the compiler-owned region layout",
        ));
    }
    let mut frame = InterpreterVars::from_owned(vec![Value::Num(0.0); bytecode.var_count]);
    let mut frame_assigned = HashSet::new();
    for (variable, value) in input_variables.into_iter().zip(inputs) {
        if frame.get(variable.slot).is_none() {
            return Err(mex(
                "ParallelFrame",
                "parallel task input is outside its compiled VM frame",
            ));
        };
        let _retired = frame.replace_owned(variable.slot, value);
        frame_assigned.insert(variable.slot);
    }
    if region.loop_slot >= frame.len() {
        return Err(mex(
            "ParallelFrame",
            "parallel loop value is outside its compiled VM frame",
        ));
    }
    let reduction_inputs = region
        .variables
        .iter()
        .filter(|variable| {
            matches!(
                variable.contract.role,
                runmat_types::ParallelVariableRole::Reduction { .. }
            )
        })
        .map(|variable| {
            frame
                .get(variable.slot)
                .filter(|_| frame_assigned.contains(&variable.slot))
                .map(|value| (*value).clone())
                .map(|value| (variable.slot, value))
                .ok_or_else(|| {
                    mex(
                        "ParallelTaskInput",
                        "parallel reduction is missing its typed identity value",
                    )
                })
        })
        .collect::<Result<HashMap<_, _>, _>>()?;
    let mut reduction_outputs = reduction_inputs
        .keys()
        .copied()
        .map(|slot| {
            (
                slot,
                runmat_runtime::execution::RootedValuesBuilder::default(),
            )
        })
        .collect::<HashMap<_, _>>();
    let mut frame_roots = runmat_runtime::execution::RootedValueLease::for_traced(&frame)
        .map_err(|error| error.into_runtime_error())?;

    for (iteration_index, iteration) in iterations.into_iter().enumerate() {
        for variable in &region.variables {
            if matches!(
                variable.contract.role,
                runmat_types::ParallelVariableRole::Temporary
                    | runmat_types::ParallelVariableRole::Private
            ) {
                if frame.get(variable.slot).is_none() {
                    return Err(mex(
                        "ParallelFrame",
                        "parallel private value is outside its compiled VM frame",
                    ));
                };
                let _retired = frame.replace_owned(variable.slot, Value::Num(0.0));
                frame_assigned.remove(&variable.slot);
            }
        }
        if mode == ParforTaskMode::IndependentIterations {
            for (slot, identity) in &reduction_inputs {
                let _retired = frame.replace_owned(*slot, identity.clone());
                frame_assigned.insert(*slot);
            }
        }
        let _retired = frame.replace_owned(region.loop_slot, iteration);
        frame_assigned.insert(region.loop_slot);
        let _random_stream = if let Some(stream) = randomness.deterministic_stream(iteration_index)
        {
            Some(
                runmat_runtime::builtins::common::random::enter_parallel_random_stream_async(
                    &runtime, stream,
                )
                .await?,
            )
        } else {
            None
        };
        let resume = InterpreterResumeState {
            pc: region.body.pc,
            completion_boundary: Some(
                crate::interpreter::state::InterpreterCompletionBoundary::before(region.header.pc),
            ),
            vars: frame,
            assigned_slots: frame_assigned,
            supplied_inputs: 0,
            requested_outputs: 0,
            missing_input_slots: HashSet::new(),
            global_aliases: HashMap::new(),
            persistent_aliases: HashMap::new(),
            side_effect_epoch: 0,
        };
        let resumed = match &bytecode_owner {
            Some(owner) => {
                interpret_resume_shared_in_context(
                    owner.clone(),
                    resume,
                    Some(current_function_name),
                    runtime.clone(),
                )
                .await?
            }
            None => {
                interpret_resume_in_context(
                    bytecode,
                    resume,
                    Some(current_function_name),
                    runtime.clone(),
                )
                .await?
            }
        };
        let (next_frame, next_assigned) = match resumed {
            InterpreterOutcome::Completed(completion) => completion.with_owned(|completion| {
                let next_frame = completion.values;
                let next_assigned = completion.assigned_slots;
                let next_roots =
                    runmat_runtime::execution::RootedValueLease::for_traced(&next_frame)
                        .map_err(|error| error.into_runtime_error())?;
                frame_roots = next_roots;
                Ok::<_, RuntimeError>((next_frame, next_assigned))
            })?,
        };
        frame = next_frame;
        frame_assigned = next_assigned;
        if mode == ParforTaskMode::IndependentIterations {
            for (slot, values) in &mut reduction_outputs {
                let value = frame
                    .get(*slot)
                    .filter(|_| frame_assigned.contains(slot))
                    .map(|value| (*value).clone())
                    .ok_or_else(|| {
                        mex(
                            "ParallelTaskOutput",
                            "parallel task did not assign a declared reduction output",
                        )
                    })?;
                values
                    .push(value)
                    .map_err(|error| error.into_runtime_error())?;
            }
        }
    }

    let mut outputs = runmat_runtime::execution::RootedValuesBuilder::default();
    for variable in region.output_variables() {
        if mode == ParforTaskMode::IndependentIterations
            && matches!(
                variable.contract.role,
                runmat_types::ParallelVariableRole::Reduction { .. }
            )
        {
            let values = reduction_outputs.remove(&variable.slot).ok_or_else(|| {
                mex(
                    "ParallelTaskOutput",
                    "parallel task lost a declared reduction output",
                )
            })?;
            values
                .finish()
                .map_err(|error| error.into_runtime_error())?
                .with_unbounded_values(|values| {
                    let length = values.len();
                    let value = runmat_value::CellArray::new(values, 1, length)
                        .map(Value::Cell)
                        .map_err(|error| mex("ParallelTaskOutput", &error))?;
                    outputs
                        .push(value)
                        .map_err(|error| error.into_runtime_error())
                })??;
            continue;
        }
        let value = frame
            .get(variable.slot)
            .filter(|_| frame_assigned.contains(&variable.slot))
            .map(|value| (*value).clone())
            .ok_or_else(|| {
                mex(
                    "ParallelTaskOutput",
                    "parallel task did not assign a declared output",
                )
            })?;
        outputs
            .push(value)
            .map_err(|error| error.into_runtime_error())?;
    }
    let outputs = outputs
        .finish()
        .map_err(|error| error.into_runtime_error())?;
    drop(frame_roots);
    Ok(outputs)
}

pub async fn interpret_function(
    bytecode: &Bytecode,
    vars: Vec<Value>,
) -> Result<runmat_runtime::execution::RootedValues, RuntimeError> {
    interpret_function_with_counts(bytecode, vars, "<anonymous>", 0, 0, HashSet::new()).await
}

pub async fn interpret_function_with_counts(
    bytecode: &Bytecode,
    vars: Vec<Value>,
    name: &str,
    out_count: usize,
    in_count: usize,
    missing_input_slots: HashSet<usize>,
) -> Result<runmat_runtime::execution::RootedValues, RuntimeError> {
    let runtime = active_or_standalone_runtime_context();
    interpret_function_with_counts_in_context(
        bytecode,
        vars,
        name,
        out_count,
        in_count,
        missing_input_slots,
        runtime,
    )
    .await
}

async fn interpret_function_with_counts_in_context(
    bytecode: &Bytecode,
    vars: Vec<Value>,
    name: &str,
    out_count: usize,
    in_count: usize,
    missing_input_slots: HashSet<usize>,
    runtime: runmat_runtime::context::RuntimeContext,
) -> Result<runmat_runtime::execution::RootedValues, RuntimeError> {
    interpret_function_with_counts_in_context_with_driver(
        FunctionExecutionInvocation {
            bytecode: Cow::Borrowed(bytecode),
            name: FunctionExecutionName::Borrowed(name),
            arity: RuntimeInvocationArity::function(in_count, out_count),
            missing_input_slots,
            runtime,
            driver: None,
            lexical_environment: None,
            lexical_binding_slots: Vec::new(),
            inherited_storage_slots: Vec::new(),
        },
        InterpreterVars::from_owned(vars),
    )
    .await
}

/// Exact selected function frame, including its invocation-local arity and
/// continuation owner. The session runtime service remains shared.
struct FunctionExecutionInvocation<'a> {
    bytecode: Cow<'a, Bytecode>,
    name: FunctionExecutionName<'a>,
    arity: RuntimeInvocationArity,
    missing_input_slots: HashSet<usize>,
    runtime: runmat_runtime::context::RuntimeContext,
    driver: Option<ProcessDriverHandle>,
    lexical_environment: Option<runmat_value::LexicalEnvironment>,
    lexical_binding_slots: Vec<crate::bytecode::program::LexicalBindingSlot>,
    inherited_storage_slots: Vec<crate::bytecode::program::InheritedStorageSlot>,
}

#[derive(Clone, Copy)]
enum FunctionExecutionName<'a> {
    Borrowed(&'a str),
    ActiveFunction,
}

fn function_execution_name<'a>(
    bytecode: &'a Bytecode,
    name: FunctionExecutionName<'a>,
) -> Result<&'a str, RuntimeError> {
    match name {
        FunctionExecutionName::Borrowed(name) => Ok(name),
        FunctionExecutionName::ActiveFunction => bytecode
            .active_function
            .and_then(|function| bytecode.function_registry.get(function))
            .map(|function| function.display_name.as_str())
            .ok_or_else(|| {
                mex(
                    "SemanticFunctionIdentity",
                    "owned function frame has no active registry definition",
                )
            }),
    }
}

async fn interpret_function_with_counts_in_context_with_driver(
    mut invocation: FunctionExecutionInvocation<'_>,
    vars: InterpreterVars,
) -> Result<runmat_runtime::execution::RootedValues, RuntimeError> {
    let runtime = invocation.runtime.with_invocation_arity(invocation.arity);
    let (runtime, workspace_activation) =
        crate::runtime::workspace_service::VmWorkspaceService::install(runtime, None);
    let frame = workspace_activation.frame();
    let runtime = with_default_acceleration_service(runtime);
    let name = function_execution_name(invocation.bytecode.as_ref(), invocation.name)?;
    let runtime = runtime_with_bytecode_debug_frame(runtime, invocation.bytecode.as_ref(), name);
    invocation.runtime = runtime.clone();
    let result = workspace_activation
        .scope(runtime.scope(scope_vm_pc(
            interpret_function_with_counts_in_context_inner(invocation, vars),
        )))
        .await;
    let _ = crate::runtime::workspace::take_workspace_completion(frame);
    result
}

async fn interpret_function_with_counts_in_context_inner(
    invocation: FunctionExecutionInvocation<'_>,
    vars: InterpreterVars,
) -> Result<runmat_runtime::execution::RootedValues, RuntimeError> {
    let FunctionExecutionInvocation {
        bytecode,
        name,
        arity: invocation_arity,
        missing_input_slots,
        runtime,
        driver,
        lexical_environment,
        lexical_binding_slots,
        inherited_storage_slots,
    } = invocation;
    let bytecode = bytecode.as_ref();
    let name = function_execution_name(bytecode, name)?;
    let source_id = bytecode
        .source_id
        .and_then(|source| source.resolve_in(&runtime));
    let (source_owner, diagnostic_source) = selected_frame_source_basis(bytecode, &runtime)
        .map_or((None, None), |(owner, diagnostic)| {
            (Some(owner), Some(diagnostic))
        });
    let mut state = InterpreterState::from_vars_in_context(
        bytecode.clone(),
        vars,
        bytecode.var_count,
        Some(name),
        invocation_arity,
        runtime,
    )?;
    state.context.process_driver = driver;
    state.context.dynamic_eval_admission = state
        .context
        .process_driver
        .as_ref()
        .and_then(ProcessDriverHandle::dynamic_admission);
    state.context.lexical_environment = lexical_environment;
    state.context.lexical_binding_slots = lexical_binding_slots;
    state.context.inherited_storage_slots = inherited_storage_slots;
    state.missing_input_slots = missing_input_slots;
    let res = Box::pin(run_interpreter_owned(Box::new(state))).await;
    let res = match res {
        Ok(InterpreterOutcome::Completed(completion)) => completion.with_owned(|completion| {
            let values = completion.values.into_owned().map_err(|_| {
                mex(
                    "InterpreterCompletion",
                    "ordinary function completion retained workspace-backed slots",
                )
            })?;
            runmat_runtime::execution::RootedValues::new(values)
                .map_err(|error| error.into_runtime_error())
        }),
        Err(error) => {
            let error = attach_span_from_pc(bytecode, error);
            Err(attach_call_frames(
                name,
                source_id,
                source_owner,
                diagnostic_source,
                error,
            ))
        }
    }?;
    Ok(res)
}

#[cfg(test)]
#[path = "runner/native_metadata_tests.rs"]
mod native_metadata_tests;

#[cfg(test)]
#[path = "runner/process_dispatch_tests.rs"]
mod process_dispatch_tests;

#[cfg(test)]
#[path = "runner/frame_source_tests.rs"]
mod frame_source_tests;

#[cfg(test)]
#[path = "runner/persistent_resume_tests.rs"]
mod persistent_resume_tests;

#[cfg(test)]
#[path = "runner/private_name_execution_tests.rs"]
mod private_name_execution_tests;

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "runner/parfor_ownership_tests.rs"]
mod parfor_ownership_tests;

#[cfg(test)]
mod tests {
    use super::{
        interpret_function_with_counts_in_context, interpret_resume_in_context, output_sequence,
    };
    #[cfg(feature = "native-accel")]
    use super::{interpret_workspace, run_interpreter_inner, run_interpreter_owned};
    use crate::bytecode::program::Bytecode;
    use crate::bytecode::Instr;
    use crate::interpreter::api::{InterpreterResumeState, InterpreterState};
    use crate::interpreter::vars::InterpreterVars;
    use futures::executor::block_on;
    use futures::task::noop_waker;
    use runmat_runtime::builtins::common::validation::{
        value_is_empty, value_is_greater_than, value_is_greater_than_or_equal, value_is_integer,
        value_is_less_than, value_is_less_than_or_equal, value_is_negative, value_is_nonnegative,
        value_is_nonpositive, value_is_nonzero, value_is_numeric_or_logical, value_is_positive,
        value_is_real, value_is_scalar_or_empty, value_is_text,
    };
    use runmat_runtime::context::RuntimeInvocationArity;
    #[cfg(feature = "native-accel")]
    use runmat_value::HandleRef;
    use runmat_value::{CellArray, Tensor, Value};
    use std::collections::HashSet;
    use std::future::Future;
    use std::task::{Context, Poll};
    #[cfg(feature = "native-accel")]
    use {
        once_cell::sync::Lazy,
        runmat_accelerate::simple_provider::InProcessProvider,
        runmat_accelerate_api::{AccelProvider, HostTensorView, ThreadProviderGuard},
    };

    #[cfg(feature = "native-accel")]
    static TEST_PROVIDER: Lazy<InProcessProvider> = Lazy::new(InProcessProvider::new);

    #[cfg(feature = "native-accel")]
    fn upload_provider_handle(
        data: Vec<f64>,
        shape: Vec<usize>,
    ) -> runmat_accelerate_api::GpuTensorHandle {
        runmat_accelerate::ensure_residency_hooks();
        TEST_PROVIDER
            .upload(&HostTensorView {
                data: &data,
                shape: &shape,
            })
            .expect("upload should succeed")
    }

    #[test]
    fn output_sequence_zero_requested_is_empty() {
        let sequence = output_sequence(vec![Value::Num(1.0)], 0).unwrap();
        assert!(sequence.is_empty());
    }

    #[test]
    fn output_sequence_multi_requested_remains_transient() {
        let sequence = output_sequence(vec![Value::Num(1.0), Value::Num(2.0)], 2).unwrap();
        assert_eq!(sequence.as_slice(), &[Value::Num(1.0), Value::Num(2.0)]);
    }

    #[test]
    fn interpreter_validation_remains_lazy_until_the_first_poll() {
        let runtime = runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        let bytecode = Bytecode::with_instructions(
            vec![
                Instr::EnterTry {
                    scope: 7,
                    catch_pc: 1,
                    catch_var: None,
                },
                Instr::Return,
            ],
            0,
        );
        let expected_validation_error = runmat_runtime::RuntimeError::new(
            bytecode
                .validate_catch_scope_metadata()
                .expect_err("fixture has invalid catch metadata"),
        );
        let state = Box::new(InterpreterState::from_owned_in_context(
            bytecode,
            Vec::new(),
            None,
            RuntimeInvocationArity::Script,
            runtime.clone(),
        ));
        drop(run_interpreter_owned(state));
        assert!(
            crate::take_updated_workspace_state().is_none(),
            "unpolled execution has no workspace effects"
        );

        // Ownership is transferred, never synchronized into a borrowed Vec.
        // Abandoning an unpolled execution retires that actual input lease.
        let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .expect("physical input lease");
        let state = Box::new(InterpreterState::from_owned_in_context(
            Bytecode::empty(),
            vec![Value::GpuTensor(allocation.handle)],
            None,
            RuntimeInvocationArity::Script,
            runtime.clone(),
        ));
        let unpolled = run_interpreter_owned(state);
        assert_eq!(allocation.probe.retirements(), 0);
        drop(unpolled);
        assert_eq!(allocation.probe.retirements(), 1);

        let state = Box::new(InterpreterState::from_owned_in_context(
            Bytecode::with_instructions(
                vec![
                    Instr::EnterTry {
                        scope: 7,
                        catch_pc: 1,
                        catch_var: None,
                    },
                    Instr::Return,
                ],
                0,
            ),
            Vec::new(),
            None,
            RuntimeInvocationArity::Script,
            runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
                runmat_runtime::execution::RuntimeExecutionService::new(),
            )),
        ));
        let mut future = std::pin::pin!(run_interpreter_owned(state));
        let waker = noop_waker();
        let mut context = Context::from_waker(&waker);
        let Poll::Ready(Err(error)) = Future::poll(future.as_mut(), &mut context) else {
            panic!("invalid catch metadata must reject on first poll");
        };
        assert_eq!(error.message(), expected_validation_error.message());
    }

    #[test]
    fn nested_function_interpreter_reserves_and_activates_its_exact_workspace_frame() {
        let runtime = runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        let (runtime, activation) =
            crate::runtime::workspace_service::VmWorkspaceService::install(runtime, None);
        let outer_frame = activation.frame();
        let mut child = Bytecode::with_instructions(
            vec![Instr::LoadConst(2.0), Instr::StoreVar(0), Instr::Return],
            1,
        );
        child.var_names.insert(0, "local".to_string());

        block_on(activation.scope(runtime.scope(async {
            let mut outer = InterpreterVars::from_owned(vec![Value::Num(1.0)]);
            let _outer_guard = crate::runtime::workspace::set_transient_workspace_state(
                std::collections::HashMap::from([("outer".to_string(), 0)]),
                std::collections::HashSet::from(["outer".to_string()]),
                &mut outer,
                runmat_runtime::workspace::bindings::FrameGlobalBindings::default(),
            );
            let result = interpret_function_with_counts_in_context(
                &child,
                vec![Value::Num(0.0)],
                "nested_workspace_test",
                0,
                0,
                std::collections::HashSet::new(),
                runtime.clone(),
            )
            .await
            .expect("nested interpreter should own an exact child workspace frame");
            assert_eq!(result[0], Value::Num(2.0));
            assert_eq!(
                crate::runtime::workspace::workspace_lookup("outer"),
                Some(Value::Num(1.0))
            );
            assert_eq!(crate::runtime::workspace::workspace_lookup("local"), None);
        })));
        let _ = crate::runtime::workspace::take_workspace_completion(outer_frame);
    }

    #[test]
    fn zero_variable_nested_function_still_owns_an_exact_transient_workspace_frame() {
        let base_owner = runmat_runtime::workspace::BaseWorkspaceOwner::new();
        let runtime = runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ))
        .with_service_ports(
            runmat_runtime::context::RuntimeServicePorts::default().with_base_workspace(
                std::rc::Rc::new(runmat_runtime::workspace::BaseWorkspaceService::new(
                    base_owner.clone(),
                )),
            ),
        );
        let (runtime, base_activation) =
            crate::runtime::workspace_service::VmWorkspaceService::install(runtime, None);
        let base_frame = base_activation.frame();
        let child_instructions = vec![
            Instr::LoadString("child_local = 13;".to_string()),
            Instr::CallBuiltinMulti("eval".to_string(), 1, 0),
            Instr::LoadString("caller".to_string()),
            Instr::LoadString("caller_mark".to_string()),
            Instr::LoadConst(17.0),
            Instr::CallBuiltinMulti("assignin".to_string(), 3, 0),
            Instr::LoadString("base".to_string()),
            Instr::LoadString("base_mark".to_string()),
            Instr::LoadConst(19.0),
            Instr::CallBuiltinMulti("assignin".to_string(), 3, 0),
            Instr::Return,
        ];

        block_on(base_activation.scope(runtime.scope(async {
            let mut base = InterpreterVars::from_owned(vec![Value::Num(5.0)]);
            let _base_guard = crate::runtime::workspace::set_transient_workspace_state(
                std::collections::HashMap::from([("base_value".to_string(), 0)]),
                std::collections::HashSet::from(["base_value".to_string()]),
                &mut base,
                runmat_runtime::workspace::bindings::FrameGlobalBindings::default(),
            );

            let (outer_runtime, outer_activation) =
                crate::runtime::workspace_service::VmWorkspaceService::install(
                    runtime.clone(),
                    None,
                );
            let outer_frame = outer_activation.frame();
            outer_activation
                .scope(outer_runtime.scope(async {
                    let mut outer = InterpreterVars::from_owned(vec![Value::Num(7.0)]);
                    let _outer_guard = crate::runtime::workspace::set_transient_workspace_state(
                        std::collections::HashMap::from([("outer_value".to_string(), 0)]),
                        std::collections::HashSet::from(["outer_value".to_string()]),
                        &mut outer,
                        runmat_runtime::workspace::bindings::FrameGlobalBindings::default(),
                    );
                    for (phase, end) in [("current", 2), ("caller", 6), ("base", 11)] {
                        let mut instructions = child_instructions[..end].to_vec();
                        if !matches!(instructions.last(), Some(Instr::Return)) {
                            instructions.push(Instr::Return);
                        }
                        let child = Bytecode::with_instructions(instructions, 0);
                        let result = interpret_function_with_counts_in_context(
                            &child,
                            Vec::new(),
                            "zero_variable_workspace_test",
                            0,
                            0,
                            std::collections::HashSet::new(),
                            outer_runtime.clone(),
                        )
                        .await
                        .expect(
                            "zero-variable nested interpreter should own an exact child frame",
                        );
                        assert_eq!(
                            result,
                            vec![Value::Num(13.0)],
                            "dynamic {phase} prefix should retain only the Current assignment in the child frame"
                        );
                    }
                    assert_eq!(
                        crate::runtime::workspace::workspace_lookup("child_local"),
                        None
                    );
                    assert_eq!(
                        crate::runtime::workspace::workspace_lookup("caller_mark"),
                        Some(Value::Num(17.0))
                    );
                    assert_eq!(
                        base_owner.snapshot().unwrap().get("base_mark"),
                        Some(&Value::Num(19.0)),
                        "base assignment must use the explicit base authority"
                    );
                }))
                .await;
            let _ = crate::runtime::workspace::take_workspace_completion(outer_frame);

            assert_eq!(
                crate::runtime::workspace::workspace_lookup("caller_mark"),
                None
            );
            assert_eq!(
                base_owner.snapshot().unwrap().get("base_mark"),
                Some(&Value::Num(19.0))
            );
            assert_eq!(
                crate::runtime::workspace::workspace_lookup("base_value"),
                Some(Value::Num(5.0))
            );
        })));
        let _ = crate::runtime::workspace::take_workspace_completion(base_frame);
    }

    #[test]
    fn failing_zero_variable_nested_function_retires_child_and_restores_parent() {
        let runtime = runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        let (runtime, activation) =
            crate::runtime::workspace_service::VmWorkspaceService::install(runtime, None);
        let outer_frame = activation.frame();
        let child = Bytecode::with_instructions(
            vec![
                Instr::LoadString("child_ephemeral = 23;".to_string()),
                Instr::CallBuiltinMulti("eval".to_string(), 1, 0),
                Instr::LoadString("caller".to_string()),
                Instr::LoadString("before_error".to_string()),
                Instr::LoadConst(29.0),
                Instr::CallBuiltinMulti("assignin".to_string(), 3, 0),
                Instr::LoadString("nested failure".to_string()),
                Instr::CallBuiltinMulti("error".to_string(), 1, 0),
                Instr::Return,
            ],
            0,
        );

        block_on(activation.scope(runtime.scope(async {
            let mut outer = InterpreterVars::from_owned(vec![Value::Num(31.0)]);
            let _outer_guard = crate::runtime::workspace::set_transient_workspace_state(
                std::collections::HashMap::from([("outer_value".to_string(), 0)]),
                std::collections::HashSet::from(["outer_value".to_string()]),
                &mut outer,
                runmat_runtime::workspace::bindings::FrameGlobalBindings::default(),
            );
            let error = interpret_function_with_counts_in_context(
                &child,
                Vec::new(),
                "failing_zero_variable_workspace_test",
                0,
                0,
                std::collections::HashSet::new(),
                runtime.clone(),
            )
            .await
            .expect_err("nested error should leave through the exact child frame");
            assert!(error.message().contains("nested failure"));
            assert_eq!(
                crate::runtime::workspace::workspace_lookup("child_ephemeral"),
                None
            );
            assert_eq!(
                crate::runtime::workspace::workspace_lookup("before_error"),
                Some(Value::Num(29.0))
            );
            crate::runtime::workspace::workspace_assign("after_error", Value::Num(37.0))
                .expect("restored parent frame should remain writable");
            assert_eq!(
                crate::runtime::workspace::workspace_lookup("after_error"),
                Some(Value::Num(37.0))
            );
        })));
        let _ = crate::runtime::workspace::take_workspace_completion(outer_frame);
    }

    #[test]
    fn semantic_function_error_keeps_its_selected_source_frame() {
        let runtime = runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        let source = runmat_runtime::source_context::lease_source_context_in_context(
            &runtime,
            "semantic_source.m",
            Some("/workspace/private/semantic_source.m".into()),
            "error('semantic source failure');",
        )
        .expect("semantic source lease");
        let mut function = Bytecode::with_instructions(
            vec![
                Instr::LoadString("semantic source failure".to_string()),
                Instr::CallBuiltinMulti("error".to_string(), 1, 0),
                Instr::Return,
            ],
            0,
        );
        function.source_id = Some(crate::BytecodeSourceRef::Runtime(source.source_ref()));

        let error = block_on(interpret_function_with_counts_in_context(
            &function,
            Vec::new(),
            "semantic_source",
            0,
            0,
            std::collections::HashSet::new(),
            runtime,
        ))
        .expect_err("semantic function should return its language error");

        assert_eq!(error.context.call_frames.len(), 1);
        assert_eq!(error.context.call_frames[0].function, "semantic_source");
        assert_eq!(
            error.context.call_frames[0].source_id,
            Some(source.source_id().0)
        );
    }

    #[test]
    fn interpreter_resume_starts_at_exact_pc_without_replaying_prefix() {
        let bytecode = Bytecode::with_instructions(
            vec![
                Instr::LoadConst(99.0),
                Instr::StoreVar(0),
                Instr::LoadConst(2.0),
                Instr::StoreVar(1),
                Instr::Return,
            ],
            3,
        );
        let resume = InterpreterResumeState {
            pc: 2,
            completion_boundary: None,
            vars: InterpreterVars::from_owned(vec![
                Value::Num(41.0),
                Value::Num(0.0),
                Value::Num(0.0),
            ]),
            assigned_slots: HashSet::from([0]),
            supplied_inputs: 1,
            requested_outputs: 1,
            missing_input_slots: Default::default(),
            global_aliases: Default::default(),
            persistent_aliases: Default::default(),
            side_effect_epoch: 1,
        };
        let runtime = runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        let parent = runtime.clone();
        let super::InterpreterOutcome::Completed(completion) = block_on(
            interpret_resume_in_context(&bytecode, resume, Some("resume_test"), runtime),
        )
        .unwrap();
        assert_eq!(*completion.values.get(0).unwrap(), Value::Num(41.0));
        assert_eq!(*completion.values.get(1).unwrap(), Value::Num(2.0));
        assert_eq!(*completion.values.get(2).unwrap(), Value::Num(0.0));
        assert_eq!(
            completion.assigned_slots,
            std::collections::HashSet::from([0, 1])
        );
        assert_eq!(parent.invocation_arity(), None);
    }

    #[test]
    fn resumed_legacy_global_slot_keeps_its_typed_binding_and_store() {
        let mut bytecode = Bytecode::with_instructions(
            vec![
                Instr::DeclareGlobal(vec![0]),
                Instr::LoadVar(0),
                Instr::StoreVar(1),
                Instr::LoadConst(29.0),
                Instr::StoreVar(0),
                Instr::LoadVar(0),
                Instr::StoreVar(2),
                Instr::Return,
            ],
            3,
        );
        bytecode.var_names.insert(0, "var_0".to_owned());
        let runtime = runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        block_on(runtime.scope(async {
            runmat_runtime::workspace::session::store_global_slot(0, Value::Num(17.0))
                .expect("seed global slot");
        }));
        let resume = InterpreterResumeState {
            pc: 1,
            completion_boundary: None,
            vars: InterpreterVars::from_owned(vec![
                Value::Num(3.0),
                Value::Num(0.0),
                Value::Num(0.0),
            ]),
            assigned_slots: HashSet::from([0]),
            supplied_inputs: 0,
            requested_outputs: 3,
            missing_input_slots: Default::default(),
            global_aliases: std::collections::HashMap::from([(
                0,
                runmat_runtime::workspace::bindings::FrameGlobalBinding::LegacySlot(0),
            )]),
            persistent_aliases: Default::default(),
            side_effect_epoch: 0,
        };
        let super::InterpreterOutcome::Completed(completion) =
            block_on(interpret_resume_in_context(
                &bytecode,
                resume,
                Some("legacy_resume_test"),
                runtime.clone(),
            ))
            .unwrap();
        assert_eq!(*completion.values.get(1).unwrap(), Value::Num(17.0));
        assert_eq!(*completion.values.get(2).unwrap(), Value::Num(29.0));
        block_on(runtime.scope(async {
            assert_eq!(
                runmat_runtime::workspace::session::global_slot_value(0),
                Some(Value::Num(29.0))
            );
        }));
    }

    #[test]
    fn interpreter_resume_stops_before_the_declared_completion_boundary() {
        let bytecode = Bytecode::with_instructions(
            vec![
                Instr::LoadConst(7.0),
                Instr::StoreVar(0),
                Instr::LoadConst(99.0),
                Instr::StoreVar(1),
                Instr::Return,
            ],
            2,
        );
        let resume = InterpreterResumeState {
            pc: 0,
            completion_boundary: Some(
                crate::interpreter::state::InterpreterCompletionBoundary::before(2),
            ),
            vars: InterpreterVars::from_owned(vec![Value::Num(0.0), Value::Num(0.0)]),
            assigned_slots: HashSet::new(),
            supplied_inputs: 0,
            requested_outputs: 0,
            missing_input_slots: Default::default(),
            global_aliases: Default::default(),
            persistent_aliases: Default::default(),
            side_effect_epoch: 0,
        };
        let runtime = runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        let parent = runtime.clone();
        let super::InterpreterOutcome::Completed(completion) = block_on(
            interpret_resume_in_context(&bytecode, resume, Some("bounded_resume_test"), runtime),
        )
        .unwrap();
        assert_eq!(
            completion.values.clone_values_for_test(),
            vec![Value::Num(7.0), Value::Num(0.0)]
        );
        assert_eq!(
            completion.assigned_slots,
            std::collections::HashSet::from([0])
        );
        assert_eq!(parent.invocation_arity(), None);
    }

    #[test]
    fn numeric_or_logical_validator_accepts_expected_domains() {
        assert!(value_is_numeric_or_logical(&Value::Num(1.0)));
        assert!(value_is_numeric_or_logical(&Value::Bool(true)));
        assert!(value_is_numeric_or_logical(&Value::Complex(1.0, 2.0)));
        let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("tensor");
        assert!(value_is_numeric_or_logical(&Value::Tensor(tensor)));
        assert!(!value_is_numeric_or_logical(&Value::String("x".into())));
        assert!(!value_is_numeric_or_logical(&Value::CharArray(
            runmat_value::CharArray::from_unicode_scalars("x".chars().collect(), 1, 1)
                .expect("char")
        )));
    }

    #[test]
    fn text_validator_accepts_string_char_vector_and_cellstr() {
        assert!(value_is_text(&Value::String("x".into())));
        assert!(value_is_text(&Value::CharArray(
            runmat_value::CharArray::from_unicode_scalars("abc".chars().collect(), 1, 3)
                .expect("char")
        )));
        assert!(value_is_text(&Value::Cell(
            CellArray::new(
                vec![
                    Value::CharArray(
                        runmat_value::CharArray::from_unicode_scalars("a".chars().collect(), 1, 1)
                            .expect("char"),
                    ),
                    Value::String("b".into()),
                ],
                1,
                2,
            )
            .expect("cell"),
        )));
        assert!(!value_is_text(&Value::Num(1.0)));
    }

    #[test]
    fn nonempty_validator_rejects_empty_arrays_and_cells() {
        let empty_num = Tensor::new(Vec::new(), vec![0, 0]).expect("empty tensor");
        assert!(value_is_empty(&Value::Tensor(empty_num)));
        let empty_char =
            runmat_value::CharArray::from_code_units(Vec::new(), 1, 0).expect("empty char array");
        assert!(value_is_empty(&Value::CharArray(empty_char)));
        let empty_cell = CellArray::new(Vec::new(), 0, 0).expect("empty cell");
        assert!(value_is_empty(&Value::Cell(empty_cell)));
        assert!(!value_is_empty(&Value::String("".into())));
        assert!(!value_is_empty(&Value::Num(1.0)));
    }

    #[test]
    fn scalar_or_empty_validator_accepts_scalar_or_empty_shapes() {
        assert!(value_is_scalar_or_empty(&Value::Num(1.0)));
        assert!(value_is_scalar_or_empty(&Value::Bool(true)));
        let empty_num = Tensor::new(Vec::new(), vec![0, 0]).expect("empty tensor");
        assert!(value_is_scalar_or_empty(&Value::Tensor(empty_num)));
        let matrix = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("matrix");
        assert!(!value_is_scalar_or_empty(&Value::Tensor(matrix)));
    }

    #[test]
    fn real_validator_rejects_imaginary_values() {
        assert!(value_is_real(&Value::Num(1.0)));
        assert!(value_is_real(&Value::Complex(1.0, 0.0)));
        assert!(!value_is_real(&Value::Complex(1.0, 2.0)));
        let complex_real =
            runmat_value::ComplexTensor::new(vec![(1.0, 0.0)], vec![1, 1]).expect("complex tensor");
        let complex_imag =
            runmat_value::ComplexTensor::new(vec![(1.0, 2.0)], vec![1, 1]).expect("complex tensor");
        assert!(value_is_real(&Value::ComplexTensor(complex_real)));
        assert!(!value_is_real(&Value::ComplexTensor(complex_imag)));
    }

    #[test]
    fn integer_validator_accepts_integer_valued_numeric_and_logical_inputs() {
        assert!(value_is_integer(&Value::Int(runmat_value::IntValue::I64(
            3
        ))));
        assert!(value_is_integer(&Value::Num(3.0)));
        assert!(!value_is_integer(&Value::Num(3.5)));
        let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("tensor");
        assert!(value_is_integer(&Value::Tensor(tensor)));
        let non_integer = Tensor::new(vec![1.0, 2.5], vec![1, 2]).expect("tensor");
        assert!(!value_is_integer(&Value::Tensor(non_integer)));
        assert!(value_is_integer(&Value::Bool(true)));
        assert!(value_is_integer(&Value::LogicalArray(
            runmat_value::LogicalArray::new(vec![0, 1], vec![1, 2]).expect("logical array")
        )));
    }

    #[test]
    fn positive_validator_rejects_zero_and_negative_values() {
        assert!(value_is_positive(&Value::Num(1.0)));
        assert!(!value_is_positive(&Value::Num(0.0)));
        assert!(!value_is_positive(&Value::Num(-1.0)));
        assert!(value_is_positive(&Value::Int(runmat_value::IntValue::I64(
            2
        ))));
        assert!(!value_is_positive(&Value::Int(
            runmat_value::IntValue::I64(0)
        )));
        let positive = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("tensor");
        assert!(value_is_positive(&Value::Tensor(positive)));
        let mixed = Tensor::new(vec![1.0, 0.0], vec![1, 2]).expect("tensor");
        assert!(!value_is_positive(&Value::Tensor(mixed)));
    }

    #[test]
    fn negative_validator_rejects_zero_and_positive_values() {
        assert!(value_is_negative(&Value::Num(-1.0)));
        assert!(!value_is_negative(&Value::Num(0.0)));
        assert!(!value_is_negative(&Value::Num(1.0)));
        assert!(value_is_negative(&Value::Int(runmat_value::IntValue::I64(
            -2
        ))));
        let ok = Tensor::new(vec![-1.0, -2.0], vec![1, 2]).expect("tensor");
        assert!(value_is_negative(&Value::Tensor(ok)));
        let bad = Tensor::new(vec![-1.0, 0.0], vec![1, 2]).expect("tensor");
        assert!(!value_is_negative(&Value::Tensor(bad)));
    }

    #[test]
    fn nonnegative_validator_accepts_zero_and_positive_values() {
        assert!(value_is_nonnegative(&Value::Num(0.0)));
        assert!(value_is_nonnegative(&Value::Num(2.0)));
        assert!(!value_is_nonnegative(&Value::Num(-1.0)));
        assert!(value_is_nonnegative(&Value::Int(
            runmat_value::IntValue::I64(0)
        )));
        let ok = Tensor::new(vec![0.0, 1.0], vec![1, 2]).expect("tensor");
        assert!(value_is_nonnegative(&Value::Tensor(ok)));
        let bad = Tensor::new(vec![0.0, -1.0], vec![1, 2]).expect("tensor");
        assert!(!value_is_nonnegative(&Value::Tensor(bad)));
    }

    #[test]
    fn nonzero_validator_rejects_zero_values() {
        assert!(value_is_nonzero(&Value::Num(1.0)));
        assert!(!value_is_nonzero(&Value::Num(0.0)));
        assert!(value_is_nonzero(&Value::Int(runmat_value::IntValue::I64(
            2
        ))));
        assert!(!value_is_nonzero(&Value::Int(runmat_value::IntValue::I64(
            0
        ))));
        assert!(value_is_nonzero(&Value::Complex(0.0, 1.0)));
        assert!(!value_is_nonzero(&Value::Complex(0.0, 0.0)));
        let ok = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("tensor");
        assert!(value_is_nonzero(&Value::Tensor(ok)));
        let bad = Tensor::new(vec![1.0, 0.0], vec![1, 2]).expect("tensor");
        assert!(!value_is_nonzero(&Value::Tensor(bad)));
    }

    #[test]
    fn nonpositive_validator_accepts_zero_and_negative_values() {
        assert!(value_is_nonpositive(&Value::Num(0.0)));
        assert!(value_is_nonpositive(&Value::Num(-2.0)));
        assert!(!value_is_nonpositive(&Value::Num(1.0)));
        assert!(value_is_nonpositive(&Value::Int(
            runmat_value::IntValue::I64(0)
        )));
        let ok = Tensor::new(vec![0.0, -1.0], vec![1, 2]).expect("tensor");
        assert!(value_is_nonpositive(&Value::Tensor(ok)));
        let bad = Tensor::new(vec![0.0, 1.0], vec![1, 2]).expect("tensor");
        assert!(!value_is_nonpositive(&Value::Tensor(bad)));
    }

    #[test]
    fn greater_than_or_equal_validator_uses_numeric_threshold() {
        assert!(value_is_greater_than_or_equal(&Value::Num(2.0), 0.0));
        assert!(value_is_greater_than_or_equal(&Value::Num(0.0), 0.0));
        assert!(!value_is_greater_than_or_equal(&Value::Num(-1.0), 0.0));
    }

    #[test]
    fn less_than_or_equal_validator_uses_numeric_threshold() {
        assert!(value_is_less_than_or_equal(&Value::Num(-1.0), 0.0));
        assert!(value_is_less_than_or_equal(&Value::Num(0.0), 0.0));
        assert!(!value_is_less_than_or_equal(&Value::Num(1.0), 0.0));
    }

    #[test]
    fn greater_than_and_less_than_validators_use_numeric_threshold() {
        assert!(value_is_greater_than(&Value::Num(2.0), 1.0));
        assert!(!value_is_greater_than(&Value::Num(1.0), 1.0));
        assert!(value_is_less_than(&Value::Num(-2.0), -1.0));
        assert!(!value_is_less_than(&Value::Num(-1.0), -1.0));
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn cancellation_preserves_externally_owned_gpu_values() {
        runmat_gc::gc_test_context(|| {
            use runmat_accelerate::fusion_residency;
            let handle = upload_provider_handle(vec![1.], vec![1, 1]);
            let weak = handle.downgrade();
            fusion_residency::mark(&handle);
            assert!(fusion_residency::is_resident(&handle));

            let vars = vec![Value::GpuTensor(handle.clone())];
            let bytecode = Bytecode::with_instructions(vec![Instr::Return], vars.len());
            let cancelled = runmat_execution::cancellation::CancellationSource::new();
            cancelled.cancel(runmat_execution::CancellationReason::User);
            let _interrupt_guard =
                runmat_runtime::interrupt::replace_interrupt(Some(cancelled.token()));

            let err = block_on(interpret_workspace(
                &bytecode,
                crate::interpreter::vars::InterpreterVars::from_owned(vars),
                Some("<main>"),
                runmat_runtime::context::RuntimeContext::with_cancellation(
                    std::rc::Rc::new(runmat_runtime::execution::RuntimeExecutionService::new()),
                    cancelled.token(),
                ),
                None,
                None,
            ))
            .outcome
            .expect_err("cancelled execution should return error");
            assert_eq!(err.identifier(), Some("RunMat:Cancelled"));
            assert!(
                fusion_residency::is_resident(&handle),
                "cancellation must not invalidate a retained external view"
            );
            assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
            drop(handle);
            assert!(weak.acquire().is_err());
        });
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn completion_clears_stack_only_gpu_residency() {
        runmat_gc::gc_test_context(|| {
            use runmat_accelerate::fusion_residency;
            let handle = upload_provider_handle(vec![2.], vec![1, 1]);
            let weak = handle.downgrade();
            fusion_residency::mark(&handle);
            assert!(fusion_residency::is_resident(&handle));

            let bytecode = Bytecode::with_instructions(Vec::new(), 1);
            let seed_vars = vec![Value::Num(0.0)];
            let mut state = InterpreterState::from_owned_in_context(
                bytecode,
                seed_vars,
                Some("<main>"),
                RuntimeInvocationArity::Script,
                runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
                    runmat_runtime::execution::RuntimeExecutionService::new(),
                )),
            );
            state
                .stack
                .try_push(Value::GpuTensor(handle.clone()))
                .expect("test stack capacity");
            state.vars = vec![Value::Num(0.0)].into();
            drop(handle);

            let outcome =
                block_on(run_interpreter_inner(state)).expect("interpreter should complete");
            assert!(matches!(
                outcome,
                crate::interpreter::api::InterpreterOutcome::Completed(_)
            ));
            assert!(
                weak.acquire().is_err(),
                "completion retires the final stack-only owner"
            );
        });
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn pop_releases_stack_only_provider_handle() {
        runmat_gc::gc_test_context(|| {
            use runmat_accelerate::fusion_residency;

            let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
            let handle = upload_provider_handle(vec![9.0], vec![1]);
            assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
            fusion_residency::mark(&handle);

            let bytecode = Bytecode::with_instructions(vec![Instr::Pop, Instr::Return], 1);
            let seed_vars = vec![Value::Num(0.0)];
            let mut state = InterpreterState::from_owned_in_context(
                bytecode,
                seed_vars,
                Some("<main>"),
                RuntimeInvocationArity::Script,
                runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
                    runmat_runtime::execution::RuntimeExecutionService::new(),
                )),
            );
            state
                .stack
                .try_push(Value::GpuTensor(handle.clone()))
                .expect("test stack capacity");
            state.vars = vec![Value::Num(0.0)].into();

            let weak = handle.downgrade();
            drop(handle);
            let _result_vars =
                block_on(run_interpreter_inner(state)).expect("interpreter should complete");
            assert!(
                weak.acquire().is_err(),
                "the final VM-owned reference must retire"
            );
        });
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn pop_preserves_provider_handle_when_still_live_in_vars() {
        runmat_gc::gc_test_context(|| {
            use runmat_accelerate::fusion_residency;

            let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
            let handle = upload_provider_handle(vec![11.0], vec![1]);
            assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
            fusion_residency::mark(&handle);

            let bytecode = Bytecode::with_instructions(vec![Instr::Pop, Instr::Return], 1);
            let seed_vars = vec![Value::GpuTensor(handle.clone())];
            let mut state = InterpreterState::from_owned_in_context(
                bytecode,
                seed_vars,
                Some("<main>"),
                RuntimeInvocationArity::Script,
                runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
                    runmat_runtime::execution::RuntimeExecutionService::new(),
                )),
            );
            state
                .stack
                .try_push(Value::GpuTensor(handle.clone()))
                .expect("test stack capacity");
            state.vars = vec![Value::GpuTensor(handle.clone())].into();

            let _result_vars =
                block_on(run_interpreter_inner(state)).expect("interpreter should complete");
            assert!(
                fusion_residency::is_resident(&handle),
                "pop should preserve residency for handles still referenced by vars"
            );
            assert!(
                block_on(TEST_PROVIDER.download(&handle)).is_ok(),
                "pop should not release provider storage for handles still referenced by vars"
            );
            let weak = handle.downgrade();
            drop(_result_vars);
            drop(handle);
            runmat_gc::gc_collect_major().unwrap();
            assert!(weak.acquire().is_err(), "all actual owners were released");
        });
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn exit_scope_releases_local_only_provider_handle() {
        runmat_gc::gc_test_context(|| {
            use runmat_accelerate::fusion_residency;

            let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
            let handle = upload_provider_handle(vec![15.0], vec![1]);
            assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
            fusion_residency::mark(&handle);

            let bytecode = Bytecode::with_instructions(vec![Instr::ExitScope(1), Instr::Return], 1);
            let seed_vars = vec![Value::Num(0.0)];
            let mut state = InterpreterState::from_owned_in_context(
                bytecode,
                seed_vars,
                Some("<main>"),
                RuntimeInvocationArity::Script,
                runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
                    runmat_runtime::execution::RuntimeExecutionService::new(),
                )),
            );
            state.context.locals.push(Value::GpuTensor(handle.clone()));
            state.vars = vec![Value::Num(0.0)].into();

            let weak = handle.downgrade();
            drop(handle);
            let _result_vars =
                block_on(run_interpreter_inner(state)).expect("exit scope should complete");
            assert!(
                weak.acquire().is_err(),
                "the final VM-owned reference must retire"
            );
        });
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn exit_scope_preserves_provider_handle_when_still_live_in_vars() {
        runmat_gc::gc_test_context(|| {
            use runmat_accelerate::fusion_residency;

            let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
            let handle = upload_provider_handle(vec![17.0], vec![1]);
            assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
            fusion_residency::mark(&handle);

            let bytecode = Bytecode::with_instructions(vec![Instr::ExitScope(1), Instr::Return], 1);
            let seed_vars = vec![Value::GpuTensor(handle.clone())];
            let mut state = InterpreterState::from_owned_in_context(
                bytecode,
                seed_vars,
                Some("<main>"),
                RuntimeInvocationArity::Script,
                runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
                    runmat_runtime::execution::RuntimeExecutionService::new(),
                )),
            );
            state.context.locals.push(Value::GpuTensor(handle.clone()));
            state.vars = vec![Value::GpuTensor(handle.clone())].into();

            let _result_vars =
                block_on(run_interpreter_inner(state)).expect("exit scope should complete");
            assert!(
                fusion_residency::is_resident(&handle),
                "exit scope should preserve residency for handles still referenced by vars"
            );
            assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_ok(),
            "exit scope should not release provider storage for handles still referenced by vars"
        );
            let weak = handle.downgrade();
            drop(_result_vars);
            drop(handle);
            runmat_gc::gc_collect_major().unwrap();
            assert!(weak.acquire().is_err(), "all actual owners were released");
        });
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn exit_scope_releases_nested_handle_object_local_provider_handle() {
        runmat_gc::gc_test_context(|| {
            use runmat_accelerate::fusion_residency;

            let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
            let handle = upload_provider_handle(vec![18.0], vec![1]);
            assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
            fusion_residency::mark(&handle);

            let bytecode = Bytecode::with_instructions(vec![Instr::ExitScope(1), Instr::Return], 1);
            let seed_vars = vec![Value::Num(0.0)];
            let mut state = InterpreterState::from_owned_in_context(
                bytecode,
                seed_vars,
                Some("<main>"),
                RuntimeInvocationArity::Script,
                runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
                    runmat_runtime::execution::RuntimeExecutionService::new(),
                )),
            );
            let mut payload = runmat_value::UnboundedStructValueBuilder::with_capacity(1);
            payload.insert("nested", Value::GpuTensor(handle.clone()));
            let target = runmat_gc::gc_allocate(Value::Struct(payload.finish()))
                .expect("gc allocate payload");
            state.context.locals.push(Value::HandleObject(HandleRef {
                class_name: "Payload".into(),
                target,
                valid: true,
            }));
            state.vars = vec![Value::Num(0.0)].into();

            let weak = handle.downgrade();
            drop(handle);
            let _result_vars = block_on(run_interpreter_inner(state))
                .expect("exit scope should complete for nested handle-object local");
            runmat_gc::gc_collect_major().unwrap();
            assert!(
                weak.acquire().is_err(),
                "the final VM-owned reference must retire"
            );
        });
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn exit_scope_preserves_nested_handle_object_provider_handle_when_still_live_in_vars() {
        runmat_gc::gc_test_context(|| {
            use runmat_accelerate::fusion_residency;

            let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
            let handle = upload_provider_handle(vec![20.0], vec![1]);
            assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
            fusion_residency::mark(&handle);

            let bytecode = Bytecode::with_instructions(vec![Instr::ExitScope(1), Instr::Return], 1);
            let seed_vars = vec![Value::Num(0.0)];
            let mut state = InterpreterState::from_owned_in_context(
                bytecode,
                seed_vars,
                Some("<main>"),
                RuntimeInvocationArity::Script,
                runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
                    runmat_runtime::execution::RuntimeExecutionService::new(),
                )),
            );
            let mut payload = runmat_value::UnboundedStructValueBuilder::with_capacity(1);
            payload.insert("nested", Value::GpuTensor(handle.clone()));
            let target = runmat_gc::gc_allocate(Value::Struct(payload.finish()))
                .expect("gc allocate payload");
            let local_value = Value::HandleObject(HandleRef {
                class_name: "Payload".into(),
                target,
                valid: true,
            });
            state.context.locals.push(local_value.clone());
            state.vars = vec![local_value].into();

            let _result_vars = block_on(run_interpreter_inner(state))
                .expect("exit scope should complete for aliased nested handle-object local");
            assert!(
            fusion_residency::is_resident(&handle),
            "exit scope should preserve residency for nested handle-object handles still referenced by vars"
        );
            assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_ok(),
            "exit scope should not release provider storage for nested handle-object handles still referenced by vars"
        );
            let weak = handle.downgrade();
            drop(_result_vars);
            drop(handle);
            runmat_gc::gc_collect_major().unwrap();
            assert!(weak.acquire().is_err(), "all actual owners were released");
        });
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn store_var_overwrite_preserves_provider_handle_when_shared_in_other_var() {
        runmat_gc::gc_test_context(|| {
            use runmat_accelerate::fusion_residency;

            let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
            let handle = upload_provider_handle(vec![19.0], vec![1]);
            assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
            fusion_residency::mark(&handle);

            let bytecode = Bytecode::with_instructions(vec![Instr::StoreVar(0), Instr::Return], 2);
            let seed_vars = vec![
                Value::GpuTensor(handle.clone()),
                Value::GpuTensor(handle.clone()),
            ];
            let mut state = InterpreterState::from_owned_in_context(
                bytecode,
                seed_vars,
                Some("<main>"),
                RuntimeInvocationArity::Script,
                runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
                    runmat_runtime::execution::RuntimeExecutionService::new(),
                )),
            );
            state
                .stack
                .try_push(Value::Num(0.0))
                .expect("test stack capacity");
            state.vars = vec![
                Value::GpuTensor(handle.clone()),
                Value::GpuTensor(handle.clone()),
            ]
            .into();

            let _result_vars =
                block_on(run_interpreter_inner(state)).expect("store var should complete");
            assert!(
            fusion_residency::is_resident(&handle),
            "store var overwrite should preserve residency for handles still live in other vars"
        );
            assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_ok(),
            "store var overwrite should not release provider storage for handles still live in other vars"
        );
            let weak = handle.downgrade();
            drop(_result_vars);
            drop(handle);
            runmat_gc::gc_collect_major().unwrap();
            assert!(weak.acquire().is_err(), "all actual owners were released");
        });
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn store_var_overwrite_preserves_provider_handle_when_shared_in_local() {
        runmat_gc::gc_test_context(|| {
            use runmat_accelerate::fusion_residency;

            let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
            let handle = upload_provider_handle(vec![20.0], vec![1]);
            assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
            fusion_residency::mark(&handle);

            let bytecode = Bytecode::with_instructions(vec![Instr::StoreVar(0), Instr::Return], 1);
            let seed_vars = vec![Value::GpuTensor(handle.clone())];
            let mut state = InterpreterState::from_owned_in_context(
                bytecode,
                seed_vars,
                Some("<main>"),
                RuntimeInvocationArity::Script,
                runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
                    runmat_runtime::execution::RuntimeExecutionService::new(),
                )),
            );
            state
                .stack
                .try_push(Value::Num(0.0))
                .expect("test stack capacity");
            state.vars = vec![Value::GpuTensor(handle.clone())].into();
            state.context.locals.push(Value::GpuTensor(handle.clone()));

            let _result_vars = block_on(run_interpreter_inner(state))
                .expect("store var should complete when alias lives in locals");
            assert!(
                fusion_residency::is_resident(&handle),
                "store var overwrite should preserve residency for handles still live in locals"
            );
            assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_ok(),
            "store var overwrite should not release provider storage for handles still live in locals"
        );
            let weak = handle.downgrade();
            drop(_result_vars);
            drop(handle);
            runmat_gc::gc_collect_major().unwrap();
            assert!(weak.acquire().is_err(), "all actual owners were released");
        });
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn store_var_overwrite_releases_nested_handle_object_provider_handle_when_unaliased() {
        runmat_gc::gc_test_context(|| {
            use runmat_accelerate::fusion_residency;

            let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
            let handle = upload_provider_handle(vec![22.0], vec![1]);
            assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
            fusion_residency::mark(&handle);

            let bytecode = Bytecode::with_instructions(vec![Instr::StoreVar(0), Instr::Return], 1);
            let seed_vars = vec![Value::Num(0.0)];
            let mut state = InterpreterState::from_owned_in_context(
                bytecode,
                seed_vars,
                Some("<main>"),
                RuntimeInvocationArity::Script,
                runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
                    runmat_runtime::execution::RuntimeExecutionService::new(),
                )),
            );
            let mut payload = runmat_value::UnboundedStructValueBuilder::with_capacity(1);
            payload.insert("nested", Value::GpuTensor(handle.clone()));
            let target = runmat_gc::gc_allocate(Value::Struct(payload.finish()))
                .expect("gc allocate payload");
            state.vars = vec![Value::HandleObject(HandleRef {
                class_name: "Payload".into(),
                target,
                valid: true,
            })]
            .into();
            state
                .stack
                .try_push(Value::Num(0.0))
                .expect("test stack capacity");

            let weak = handle.downgrade();
            drop(handle);
            let _result_vars = block_on(run_interpreter_inner(state))
                .expect("store var overwrite should complete for nested handle-object value");
            runmat_gc::gc_collect_major().unwrap();
            assert!(
                weak.acquire().is_err(),
                "the final VM-owned reference must retire"
            );
        });
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn store_var_overwrite_preserves_nested_handle_object_provider_handle_when_shared_in_other_var()
    {
        runmat_gc::gc_test_context(|| {
            use runmat_accelerate::fusion_residency;

            let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
            let handle = upload_provider_handle(vec![24.0], vec![1]);
            assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
            fusion_residency::mark(&handle);

            let bytecode = Bytecode::with_instructions(vec![Instr::StoreVar(0), Instr::Return], 2);
            let seed_vars = vec![Value::Num(0.0), Value::Num(0.0)];
            let mut state = InterpreterState::from_owned_in_context(
                bytecode,
                seed_vars,
                Some("<main>"),
                RuntimeInvocationArity::Script,
                runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
                    runmat_runtime::execution::RuntimeExecutionService::new(),
                )),
            );
            let mut payload = runmat_value::UnboundedStructValueBuilder::with_capacity(1);
            payload.insert("nested", Value::GpuTensor(handle.clone()));
            let target = runmat_gc::gc_allocate(Value::Struct(payload.finish()))
                .expect("gc allocate payload");
            let nested = Value::HandleObject(HandleRef {
                class_name: "Payload".into(),
                target,
                valid: true,
            });
            state.vars = vec![nested.clone(), nested.clone()].into();
            state
                .stack
                .try_push(Value::Num(0.0))
                .expect("test stack capacity");

            let _result_vars = block_on(run_interpreter_inner(state)).expect(
                "store var overwrite should complete for aliased nested handle-object values",
            );
            assert!(
            fusion_residency::is_resident(&handle),
            "store var overwrite should preserve residency for nested handle-object handles still live in other vars"
        );
            assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_ok(),
            "store var overwrite should not release provider storage for nested handle-object handles still live in other vars"
        );
            let weak = handle.downgrade();
            drop(_result_vars);
            drop(handle);
            runmat_gc::gc_collect_major().unwrap();
            assert!(weak.acquire().is_err(), "all actual owners were released");
        });
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn store_var_overwrite_preserves_nested_handle_object_provider_handle_when_shared_in_local() {
        runmat_gc::gc_test_context(|| {
            use runmat_accelerate::fusion_residency;

            let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
            let handle = upload_provider_handle(vec![27.0], vec![1]);
            assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
            fusion_residency::mark(&handle);

            let bytecode = Bytecode::with_instructions(vec![Instr::StoreVar(0), Instr::Return], 1);
            let seed_vars = vec![Value::Num(0.0)];
            let mut state = InterpreterState::from_owned_in_context(
                bytecode,
                seed_vars,
                Some("<main>"),
                RuntimeInvocationArity::Script,
                runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
                    runmat_runtime::execution::RuntimeExecutionService::new(),
                )),
            );
            let mut payload = runmat_value::UnboundedStructValueBuilder::with_capacity(1);
            payload.insert("nested", Value::GpuTensor(handle.clone()));
            let target = runmat_gc::gc_allocate(Value::Struct(payload.finish()))
                .expect("gc allocate payload");
            let nested = Value::HandleObject(HandleRef {
                class_name: "Payload".into(),
                target,
                valid: true,
            });
            state.vars = vec![nested.clone()].into();
            state
                .stack
                .try_push(Value::Num(0.0))
                .expect("test stack capacity");
            state.context.locals.push(nested);

            let _result_vars = block_on(run_interpreter_inner(state))
                .expect("store var overwrite should complete when alias lives in locals");
            assert!(
            fusion_residency::is_resident(&handle),
            "store var overwrite should preserve residency for nested handle-object handles still live in locals"
        );
            assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_ok(),
            "store var overwrite should not release provider storage for nested handle-object handles still live in locals"
        );
            let weak = handle.downgrade();
            drop(_result_vars);
            drop(handle);
            runmat_gc::gc_collect_major().unwrap();
            assert!(weak.acquire().is_err(), "all actual owners were released");
        });
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn store_local_overwrite_preserves_provider_handle_when_shared_in_var() {
        runmat_gc::gc_test_context(|| {
            use runmat_accelerate::fusion_residency;

            let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
            let handle = upload_provider_handle(vec![23.0], vec![1]);
            assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
            fusion_residency::mark(&handle);

            let bytecode =
                Bytecode::with_instructions(vec![Instr::StoreLocal(0), Instr::Return], 1);
            let seed_vars = vec![Value::GpuTensor(handle.clone())];
            let mut state = InterpreterState::from_owned_in_context(
                bytecode,
                seed_vars,
                Some("<main>"),
                RuntimeInvocationArity::Script,
                runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
                    runmat_runtime::execution::RuntimeExecutionService::new(),
                )),
            );
            state
                .stack
                .try_push(Value::Num(0.0))
                .expect("test stack capacity");
            state.vars = vec![Value::GpuTensor(handle.clone())].into();
            state
                .context
                .call_stack
                .push(crate::bytecode::program::CallFrame {
                    function_name: "<local>".into(),
                    return_address: 0,
                    locals_start: 0,
                    locals_count: 1,
                    expected_outputs: 0,
                });
            state.context.locals.push(Value::GpuTensor(handle.clone()));

            let _result_vars =
                block_on(run_interpreter_inner(state)).expect("store local should complete");
            assert!(
                fusion_residency::is_resident(&handle),
                "store local overwrite should preserve residency for handles still live in vars"
            );
            assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_ok(),
            "store local overwrite should not release provider storage for handles still live in vars"
        );
            let weak = handle.downgrade();
            drop(_result_vars);
            drop(handle);
            runmat_gc::gc_collect_major().unwrap();
            assert!(weak.acquire().is_err(), "all actual owners were released");
        });
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn store_local_overwrite_preserves_provider_handle_when_shared_in_other_local() {
        runmat_gc::gc_test_context(|| {
            use runmat_accelerate::fusion_residency;

            let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
            let handle = upload_provider_handle(vec![24.0], vec![1]);
            assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
            fusion_residency::mark(&handle);

            let bytecode =
                Bytecode::with_instructions(vec![Instr::StoreLocal(0), Instr::Return], 1);
            let seed_vars = vec![Value::Num(0.0)];
            let mut state = InterpreterState::from_owned_in_context(
                bytecode,
                seed_vars,
                Some("<main>"),
                RuntimeInvocationArity::Script,
                runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
                    runmat_runtime::execution::RuntimeExecutionService::new(),
                )),
            );
            state
                .stack
                .try_push(Value::Num(0.0))
                .expect("test stack capacity");
            state.vars = vec![Value::Num(0.0)].into();
            state
                .context
                .call_stack
                .push(crate::bytecode::program::CallFrame {
                    function_name: "<local>".into(),
                    return_address: 0,
                    locals_start: 0,
                    locals_count: 2,
                    expected_outputs: 0,
                });
            state.context.locals.push(Value::GpuTensor(handle.clone()));
            state.context.locals.push(Value::GpuTensor(handle.clone()));

            let _result_vars = block_on(run_interpreter_inner(state))
                .expect("store local should complete when alias lives in other local");
            assert!(
            fusion_residency::is_resident(&handle),
            "store local overwrite should preserve residency for handles still live in other locals"
        );
            assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_ok(),
            "store local overwrite should not release provider storage for handles still live in other locals"
        );
            let weak = handle.downgrade();
            drop(_result_vars);
            drop(handle);
            runmat_gc::gc_collect_major().unwrap();
            assert!(weak.acquire().is_err(), "all actual owners were released");
        });
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn store_local_overwrite_releases_provider_handle_when_unaliased() {
        runmat_gc::gc_test_context(|| {
            use runmat_accelerate::fusion_residency;

            let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
            let handle = upload_provider_handle(vec![25.0], vec![1]);
            assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
            fusion_residency::mark(&handle);

            let bytecode =
                Bytecode::with_instructions(vec![Instr::StoreLocal(0), Instr::Return], 1);
            let seed_vars = vec![Value::Num(0.0)];
            let mut state = InterpreterState::from_owned_in_context(
                bytecode,
                seed_vars,
                Some("<main>"),
                RuntimeInvocationArity::Script,
                runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
                    runmat_runtime::execution::RuntimeExecutionService::new(),
                )),
            );
            state
                .stack
                .try_push(Value::Num(0.0))
                .expect("test stack capacity");
            state.vars = vec![Value::Num(0.0)].into();
            state
                .context
                .call_stack
                .push(crate::bytecode::program::CallFrame {
                    function_name: "<local>".into(),
                    return_address: 0,
                    locals_start: 0,
                    locals_count: 1,
                    expected_outputs: 0,
                });
            state.context.locals.push(Value::GpuTensor(handle.clone()));

            let weak = handle.downgrade();
            drop(handle);
            let _result_vars = block_on(run_interpreter_inner(state))
                .expect("store local overwrite should complete");
            assert!(
                weak.acquire().is_err(),
                "the final VM-owned reference must retire"
            );
        });
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn store_local_overwrite_releases_nested_handle_object_provider_handle_when_unaliased() {
        runmat_gc::gc_test_context(|| {
            use runmat_accelerate::fusion_residency;

            let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
            let handle = upload_provider_handle(vec![26.0], vec![1]);
            assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
            fusion_residency::mark(&handle);

            let bytecode =
                Bytecode::with_instructions(vec![Instr::StoreLocal(0), Instr::Return], 1);
            let seed_vars = vec![Value::Num(0.0)];
            let mut state = InterpreterState::from_owned_in_context(
                bytecode,
                seed_vars,
                Some("<main>"),
                RuntimeInvocationArity::Script,
                runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
                    runmat_runtime::execution::RuntimeExecutionService::new(),
                )),
            );
            let mut payload = runmat_value::UnboundedStructValueBuilder::with_capacity(1);
            payload.insert("nested", Value::GpuTensor(handle.clone()));
            let target = runmat_gc::gc_allocate(Value::Struct(payload.finish()))
                .expect("gc allocate payload");
            state
                .stack
                .try_push(Value::Num(0.0))
                .expect("test stack capacity");
            state.vars = vec![Value::Num(0.0)].into();
            state
                .context
                .call_stack
                .push(crate::bytecode::program::CallFrame {
                    function_name: "<local>".into(),
                    return_address: 0,
                    locals_start: 0,
                    locals_count: 1,
                    expected_outputs: 0,
                });
            state.context.locals.push(Value::HandleObject(HandleRef {
                class_name: "Payload".into(),
                target,
                valid: true,
            }));

            let weak = handle.downgrade();
            drop(handle);
            let _result_vars = block_on(run_interpreter_inner(state))
                .expect("store local overwrite should complete for nested handle-object value");
            runmat_gc::gc_collect_major().unwrap();
            assert!(
                weak.acquire().is_err(),
                "the final VM-owned reference must retire"
            );
        });
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn store_local_overwrite_preserves_nested_handle_object_provider_handle_when_shared_in_var() {
        runmat_gc::gc_test_context(|| {
            use runmat_accelerate::fusion_residency;

            let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
            let handle = upload_provider_handle(vec![28.0], vec![1]);
            assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
            fusion_residency::mark(&handle);

            let bytecode =
                Bytecode::with_instructions(vec![Instr::StoreLocal(0), Instr::Return], 1);
            let seed_vars = vec![Value::Num(0.0)];
            let mut state = InterpreterState::from_owned_in_context(
                bytecode,
                seed_vars,
                Some("<main>"),
                RuntimeInvocationArity::Script,
                runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
                    runmat_runtime::execution::RuntimeExecutionService::new(),
                )),
            );
            let mut payload = runmat_value::UnboundedStructValueBuilder::with_capacity(1);
            payload.insert("nested", Value::GpuTensor(handle.clone()));
            let target = runmat_gc::gc_allocate(Value::Struct(payload.finish()))
                .expect("gc allocate payload");
            let local_value = Value::HandleObject(HandleRef {
                class_name: "Payload".into(),
                target,
                valid: true,
            });
            state
                .stack
                .try_push(Value::Num(0.0))
                .expect("test stack capacity");
            state.vars = vec![local_value.clone()].into();
            state
                .context
                .call_stack
                .push(crate::bytecode::program::CallFrame {
                    function_name: "<local>".into(),
                    return_address: 0,
                    locals_start: 0,
                    locals_count: 1,
                    expected_outputs: 0,
                });
            state.context.locals.push(local_value);

            let _result_vars = block_on(run_interpreter_inner(state)).expect(
                "store local overwrite should complete for aliased nested handle-object value",
            );
            assert!(
            fusion_residency::is_resident(&handle),
            "store local overwrite should preserve residency for nested handle-object handles still live in vars"
        );
            assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_ok(),
            "store local overwrite should not release provider storage for nested handle-object handles still live in vars"
        );
            let weak = handle.downgrade();
            drop(_result_vars);
            drop(handle);
            runmat_gc::gc_collect_major().unwrap();
            assert!(weak.acquire().is_err(), "all actual owners were released");
        });
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn store_local_overwrite_preserves_nested_handle_object_provider_alias() {
        runmat_gc::gc_test_context(|| {
            use runmat_accelerate::fusion_residency;

            let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
            let handle = upload_provider_handle(vec![30.0], vec![1]);
            assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
            fusion_residency::mark(&handle);

            let bytecode =
                Bytecode::with_instructions(vec![Instr::StoreLocal(0), Instr::Return], 1);
            let seed_vars = vec![Value::Num(0.0)];
            let mut state = InterpreterState::from_owned_in_context(
                bytecode,
                seed_vars,
                Some("<main>"),
                RuntimeInvocationArity::Script,
                runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
                    runmat_runtime::execution::RuntimeExecutionService::new(),
                )),
            );
            let mut payload = runmat_value::UnboundedStructValueBuilder::with_capacity(1);
            payload.insert("nested", Value::GpuTensor(handle.clone()));
            let target = runmat_gc::gc_allocate(Value::Struct(payload.finish()))
                .expect("gc allocate payload");
            let nested = Value::HandleObject(HandleRef {
                class_name: "Payload".into(),
                target,
                valid: true,
            });
            state
                .stack
                .try_push(Value::Num(0.0))
                .expect("test stack capacity");
            state.vars = vec![Value::Num(0.0)].into();
            state
                .context
                .call_stack
                .push(crate::bytecode::program::CallFrame {
                    function_name: "<local>".into(),
                    return_address: 0,
                    locals_start: 0,
                    locals_count: 2,
                    expected_outputs: 0,
                });
            state.context.locals.push(nested.clone());
            state.context.locals.push(nested);

            let _result_vars = block_on(run_interpreter_inner(state))
                .expect("store local overwrite should complete when alias lives in other local");
            assert!(
            fusion_residency::is_resident(&handle),
            "store local overwrite should preserve residency for nested handle-object handles still live in other locals"
        );
            assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_ok(),
            "store local overwrite should not release provider storage for nested handle-object handles still live in other locals"
        );
            let weak = handle.downgrade();
            drop(_result_vars);
            drop(handle);
            runmat_gc::gc_collect_major().unwrap();
            assert!(weak.acquire().is_err(), "all actual owners were released");
        });
    }
}
