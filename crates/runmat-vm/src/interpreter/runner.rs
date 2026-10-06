use crate::accel::fusion as accel_fusion;
use crate::bytecode::{Bytecode, FunctionRegistry, Instr};
use crate::interpreter::api::{InterpreterOutcome, InterpreterResumeState, InterpreterState};
use crate::interpreter::dispatch::{self as interp_dispatch, DispatchDecision};
use crate::interpreter::engine as interp_engine;
use crate::interpreter::errors::{attach_span_from_pc, mex, set_vm_pc};
use crate::interpreter::timing::InterpreterTiming;
use crate::runtime::call_stack::attach_call_frames;
use crate::runtime::globals as runtime_globals;
use crate::runtime::workspace::{
    refresh_workspace_state, workspace_assign, workspace_clear, workspace_lookup, workspace_remove,
    workspace_snapshot,
};
use runmat_runtime::call::function_abi::{
    collect_function_outputs, prepare_function_inputs_async, FunctionInputSpec,
};
use runmat_runtime::{
    user_functions,
    workspace::{self as runtime_workspace, WorkspaceResolver},
    RuntimeError,
};
use runmat_value::{CellArray, Value};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::Once;
use tracing::{debug, info_span};

#[cfg(not(target_arch = "wasm32"))]
use std::future::Future;

#[cfg(feature = "native-accel")]
use runmat_accelerate::placement::{
    begin_trace, fusion_operation_token, PlacementEventKind, PlacementVariant,
};
#[cfg(feature = "native-accel")]
use runmat_accelerate::{
    activate_fusion_plan, active_group_plan_clone, deactivate_fusion_plan, set_current_pc,
};
#[cfg(feature = "native-accel")]
use runmat_time::{duration_ns_saturating, Instant};

#[cfg(feature = "native-accel")]
struct FusionPlanGuard;

#[cfg(feature = "native-accel")]
impl Drop for FusionPlanGuard {
    fn drop(&mut self) {
        deactivate_fusion_plan();
    }
}

type VmResult<T> = Result<T, RuntimeError>;
runmat_thread_local::runmat_thread_local! {
    static CALL_COUNTS: RefCell<Vec<(usize, usize)>> = const { RefCell::new(Vec::new()) };
}

fn sync_initial_vars(initial: &mut Vec<Value>, vars: &[Value]) {
    initial.clear();
    initial.extend_from_slice(vars);
}

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
) -> Result<runmat_value::ValueSequence, RuntimeError> {
    let runtime = active_or_standalone_runtime_context();
    let (value, _) = invoke_semantic_function_with_captures(
        function,
        args,
        requested_outputs,
        function_registry,
        runtime,
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
) -> Result<runmat_value::ValueSequence, RuntimeError> {
    let (value, _) = invoke_semantic_function_with_captures(
        function,
        args,
        requested_outputs,
        function_registry,
        runtime,
    )
    .await?;
    Ok(value)
}

pub(crate) async fn invoke_semantic_function_value_with_capture_updates(
    function: usize,
    args: &[Value],
    requested_outputs: usize,
    function_registry: &FunctionRegistry,
    runtime: runmat_runtime::context::RuntimeContext,
) -> Result<(runmat_value::ValueSequence, Vec<Value>), RuntimeError> {
    invoke_semantic_function_with_captures(
        function,
        args,
        requested_outputs,
        function_registry,
        runtime,
    )
    .await
}

async fn invoke_semantic_function_with_captures(
    function: usize,
    args: &[Value],
    requested_outputs: usize,
    function_registry: &FunctionRegistry,
    runtime: runmat_runtime::context::RuntimeContext,
) -> Result<(runmat_value::ValueSequence, Vec<Value>), RuntimeError> {
    let runtime = with_default_acceleration_service(runtime);
    runtime
        .scope(invoke_semantic_function_with_captures_inner(
            function,
            args,
            requested_outputs,
            function_registry,
            runtime.clone(),
        ))
        .await
}

async fn invoke_semantic_function_with_captures_inner(
    function: usize,
    args: &[Value],
    requested_outputs: usize,
    function_registry: &FunctionRegistry,
    runtime: runmat_runtime::context::RuntimeContext,
) -> Result<(runmat_value::ValueSequence, Vec<Value>), RuntimeError> {
    let _arguments = crate::call::root_values(args)?;
    let function_id = runmat_hir::FunctionId(function);
    let func = function_registry.get(function_id).ok_or_else(|| {
        let message = format!("Undefined semantic function: {function}");
        mex("UndefinedSemanticFunction", &message)
    })?;
    if args.len() < func.capture_slots.len() {
        let message = format!(
            "semantic function {} received too few arguments",
            func.display_name
        );
        return Err(mex("SemanticFunctionArity", &message));
    }
    let runtime_args = &args[func.capture_slots.len()..];
    let mut vars = vec![Value::Num(0.0); func.var_count];
    for (slot, value) in func.capture_slots.iter().zip(args.iter()) {
        if *slot < vars.len() {
            vars[*slot] = value.clone();
        }
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
    let prepared = prepare_function_inputs_async(
        &func.display_name,
        runtime_args,
        func.input_slots.len(),
        func.varargin_slot.is_some(),
        &input_specs,
    )
    .await?;
    let mut missing_input_slots = HashSet::new();
    for (slot, value) in func.input_slots.iter().zip(&prepared.fixed) {
        if let Some(value) = value {
            if *slot < vars.len() {
                vars[*slot] = value.clone();
            }
        } else {
            missing_input_slots.insert(*slot);
        }
    }
    if let (Some(slot), Some(cell)) = (func.varargin_slot, prepared.varargin) {
        if slot < vars.len() {
            vars[slot] = Value::Cell(cell);
        }
    }
    if let Some(slot) = func.varargout_slot {
        if slot < vars.len() {
            let cell = CellArray::new(Vec::new(), 1, 0)
                .map_err(|err| mex("VarargoutPack", &format!("varargout: {err}")))?;
            vars[slot] = Value::Cell(cell);
        }
    }
    if let Some(slot) = func.implicit_nargin_slot {
        if slot < vars.len() {
            vars[slot] = Value::Num(prepared.nargin as f64);
        }
    }
    if let Some(slot) = func.implicit_nargout_slot {
        if slot < vars.len() {
            vars[slot] = Value::Num(requested_outputs as f64);
        }
    }

    let _active_semantic_function_guard =
        user_functions::push_active_semantic_function(function_id.0);
    let mut bytecode = func.execution_bytecode(function_registry);
    let mut initially_unassigned_slots = func.initially_unassigned_slots.clone();
    for slot in &func.capture_slots {
        initially_unassigned_slots.remove(slot);
    }
    for (slot, value) in func.input_slots.iter().zip(&prepared.fixed) {
        if value.is_some() {
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
    let result_vars = {
        #[cfg(not(target_arch = "wasm32"))]
        let execution_runtime = runtime.clone();
        let future = interpret_function_with_counts_in_context(
            &bytecode,
            vars,
            &func.display_name,
            requested_outputs,
            prepared.nargin,
            missing_input_slots,
            runtime.clone(),
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
            if execution_stack == runmat_types::ExecutionStackRequirement::Process {
                // The async interpreter state is substantially larger than a
                // call frame. Heap-allocate it so nested dispatch does not
                // consume the native stack needed by the foreign runtime.
                Box::pin(future).await?
            } else {
                let mut future = Box::pin(future);
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
    let projected = super::CompletedFrameProjection::new(result_vars)?;
    let result_vars = projected.values();
    let fixed_outputs = func
        .output_slots
        .iter()
        .map(|slot| result_vars.get(*slot).cloned().unwrap_or(Value::Num(0.0)))
        .collect::<Vec<_>>();
    let varargout = func.varargout_slot.and_then(|slot| result_vars.get(slot));
    let output_values = collect_function_outputs(
        &func.display_name,
        &fixed_outputs,
        varargout,
        requested_outputs,
    )?;
    let updated_captures = func
        .capture_slots
        .iter()
        .map(|slot| result_vars.get(*slot).cloned().unwrap_or(Value::Num(0.0)))
        .collect::<Vec<_>>();
    let outputs = output_sequence(output_values, requested_outputs)?;
    drop(projected);
    Ok((outputs, updated_captures))
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

pub async fn interpret_with_vars(
    bytecode: &Bytecode,
    initial_vars: &mut Vec<Value>,
    current_function_name: Option<&str>,
) -> VmResult<InterpreterOutcome> {
    let runtime = active_or_standalone_runtime_context();
    interpret_with_vars_in_context(bytecode, initial_vars, current_function_name, runtime).await
}

pub async fn interpret_with_vars_in_context(
    bytecode: &Bytecode,
    initial_vars: &mut Vec<Value>,
    current_function_name: Option<&str>,
    runtime: runmat_runtime::context::RuntimeContext,
) -> VmResult<InterpreterOutcome> {
    interpret_with_workspace_in_context(
        bytecode,
        initial_vars,
        current_function_name,
        runtime,
        None,
    )
    .await
    .outcome
}

pub struct WorkspaceInterpreterOutcome {
    pub outcome: VmResult<InterpreterOutcome>,
    pub workspace: Option<crate::runtime::workspace::WorkspaceExecutionCompletion>,
}

pub async fn interpret_with_workspace_in_context(
    bytecode: &Bytecode,
    initial_vars: &mut Vec<Value>,
    current_function_name: Option<&str>,
    runtime: runmat_runtime::context::RuntimeContext,
    admission: Option<crate::runtime::workspace::PendingWorkspaceAdmission>,
) -> WorkspaceInterpreterOutcome {
    let (runtime, workspace_activation) =
        crate::runtime::workspace_service::VmWorkspaceService::install(runtime, admission);
    let frame = workspace_activation.frame();
    let runtime = with_default_acceleration_service(runtime);
    let outcome = workspace_activation
        .scope(runtime.scope(runmat_runtime::data::with_tx_registry_scope(
            interpret_with_vars_inner(
                bytecode,
                initial_vars,
                current_function_name,
                runtime.clone(),
            ),
        )))
        .await;
    let workspace = crate::runtime::workspace::take_workspace_completion(frame);
    WorkspaceInterpreterOutcome { outcome, workspace }
}

pub async fn interpret_resume_in_context(
    bytecode: &Bytecode,
    resume: InterpreterResumeState,
    current_function_name: Option<&str>,
    runtime: runmat_runtime::context::RuntimeContext,
) -> VmResult<InterpreterOutcome> {
    let (runtime, workspace_activation) =
        crate::runtime::workspace_service::VmWorkspaceService::install(runtime, None);
    let runtime = with_default_acceleration_service(runtime);
    workspace_activation
        .scope(runtime.scope(runmat_runtime::data::with_tx_registry_scope(
            interpret_resume_inner(bytecode, resume, current_function_name, runtime.clone()),
        )))
        .await
}

fn with_default_acceleration_service(
    runtime: runmat_runtime::context::RuntimeContext,
) -> runmat_runtime::context::RuntimeContext {
    #[cfg(feature = "native-accel")]
    let runtime = runmat_accelerate::native_auto::install_preparation_service(runtime);
    runtime
}

#[path = "runner/native_metadata.rs"]
mod native_metadata;
pub use native_metadata::prepare_native_execution_metadata;
use native_metadata::register_class_metadata;

async fn interpret_resume_inner(
    bytecode: &Bytecode,
    resume: InterpreterResumeState,
    current_function_name: Option<&str>,
    runtime: runmat_runtime::context::RuntimeContext,
) -> VmResult<InterpreterOutcome> {
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
    let mut resumed_bytecode = bytecode.clone();
    resumed_bytecode.initially_unassigned_slots.clear();
    let mut vars = Vec::with_capacity(resume.vars.len());
    let mut assigned_slots = HashSet::new();
    for (slot, value) in resume.vars.into_iter().enumerate() {
        match value {
            Some(value) => {
                assigned_slots.insert(slot);
                vars.push(value);
            }
            None => {
                vars.push(Value::Num(0.0));
                resumed_bytecode.initially_unassigned_slots.insert(slot);
            }
        }
    }
    let previous_call_counts = CALL_COUNTS.with(|counts| counts.borrow().clone());
    let mut call_counts = previous_call_counts.clone();
    call_counts.push((resume.supplied_inputs, resume.requested_outputs));
    let mut state = InterpreterState::new_in_context(
        resumed_bytecode,
        &mut vars,
        current_function_name,
        call_counts,
        runtime,
    );
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
    let _debug_frame_guard = runmat_runtime::debug_context::push_frame(
        current_name,
        source_id,
        bytecode_frame_span(bytecode),
    );
    let mut output_vars = vars;
    let result = run_interpreter(Box::new(state), &mut output_vars).await;
    CALL_COUNTS.with(|counts| *counts.borrow_mut() = previous_call_counts);
    match result {
        Ok(outcome) => Ok(outcome),
        Err(err) => {
            let err = attach_span_from_pc(bytecode, err);
            Err(attach_call_frames(current_name, source_id, err))
        }
    }
}

async fn interpret_with_vars_inner(
    bytecode: &Bytecode,
    initial_vars: &mut Vec<Value>,
    current_function_name: Option<&str>,
    runtime: runmat_runtime::context::RuntimeContext,
) -> VmResult<InterpreterOutcome> {
    bytecode
        .validate_selected_source_owners(&runtime)
        .map_err(RuntimeError::new)?;
    crate::bytecode::validate_sequence_register_flow(&bytecode.instructions)
        .map_err(RuntimeError::new)?;
    let source_id = bytecode
        .source_id
        .and_then(|source| source.resolve_in(&runtime));
    let _debug_frame_guard = runmat_runtime::debug_context::push_frame(
        current_function_name.unwrap_or("<main>"),
        source_id,
        bytecode_frame_span(bytecode),
    );
    let call_counts = CALL_COUNTS.with(|cc| cc.borrow().clone());
    let _borrowed_inputs =
        crate::runtime::gc::ManagedRoots::new(initial_vars).map_err(RuntimeError::new)?;
    let state = Box::new(InterpreterState::new_in_context(
        bytecode.clone(),
        initial_vars,
        current_function_name,
        call_counts,
        runtime,
    ));
    match Box::pin(run_interpreter(state, initial_vars)).await {
        Ok(outcome) => Ok(outcome),
        Err(err) => {
            let err = attach_span_from_pc(bytecode, err);
            let current_name = current_function_name.unwrap_or("<main>");
            Err(attach_call_frames(current_name, source_id, err))
        }
    }
}

fn bytecode_frame_span(bytecode: &Bytecode) -> Option<(usize, usize)> {
    bytecode
        .instr_spans
        .first()
        .map(|span| (span.start, span.end))
}

async fn run_interpreter(
    state: Box<InterpreterState>,
    initial_vars: &mut Vec<Value>,
) -> VmResult<InterpreterOutcome> {
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
    let state = *state;
    catch_runtime
        .scope_active_catch_shared(inherited_catch, run_interpreter_inner(state, initial_vars))
        .await
}

async fn run_interpreter_inner(
    state: InterpreterState,
    initial_vars: &mut Vec<Value>,
) -> VmResult<InterpreterOutcome> {
    let run_span = info_span!(
        "interpreter.run",
        function = state.current_function_name.as_str()
    );
    let _run_guard = run_span.enter();
    ensure_wasm_builtins_registered();
    ensure_workspace_resolver_registered();
    #[cfg(feature = "native-accel")]
    activate_fusion_plan(state.fusion_plan.clone());
    #[cfg(feature = "native-accel")]
    let _fusion_guard = FusionPlanGuard;
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
        current_function_name,
        call_counts,
        initial_assigned_var_count,
        #[cfg(feature = "native-accel")]
            fusion_plan: _,
        #[cfg(feature = "native-accel")]
        fusion_accel_graph,
        bytecode,
    } = state;
    let _source_context_guard = runmat_runtime::source_context::replace_current_source_id(
        bytecode
            .source_id
            .and_then(|source| source.resolve_in(&context.runtime)),
    );
    let _arity_call_counts_guard =
        runmat_runtime::builtins::introspection::arity_check::replace_call_counts(
            call_counts.clone(),
        );
    let function_registry = Arc::new(bytecode.function_registry());
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
    let previous_semantic_invoker = user_functions::current_semantic_function_invoker();
    // Numeric/name calls select the currently published registry. An old
    // bound body's static edges are routed through the distinct typed bound
    // invoker below; an overlay must never shadow a dynamic name lookup.
    let registry_for_function_invoker = Arc::new(function_registry.current_view());
    let _semantic_function_guard =
        user_functions::install_local_semantic_function_invoker(std::rc::Rc::new(
            move |function: usize, args: &[Value], requested_outputs: usize| {
                let args = args.to_vec();
                let previous_invoker = previous_semantic_invoker.clone();
                let function_registry = Arc::clone(&registry_for_function_invoker);
                let runtime = nested_runtime.clone();
                Box::pin(async move {
                    let local_function = function_registry
                        .get(runmat_hir::FunctionId(function))
                        .is_some();
                    if !local_function {
                        if let Some(invoker) = previous_invoker {
                            return invoker(function, &args, requested_outputs).await;
                        }
                    }
                    invoke_semantic_function_value_with_capture_updates(
                        function,
                        &args,
                        requested_outputs,
                        &function_registry,
                        runtime,
                    )
                    .await
                    .map(|(value, _)| value)
                })
            },
        ));
    let previous_bound_invoker = user_functions::current_bound_function_invoker();
    let registry_for_bound_invoker = Arc::clone(&function_registry);
    let bound_runtime = context.runtime.clone();
    let _bound_function_guard = user_functions::install_local_bound_function_invoker(
        std::rc::Rc::new(move |binding, args, requested_outputs| {
            let args = args.to_vec();
            let previous = previous_bound_invoker.clone();
            let registry = Arc::clone(&registry_for_bound_invoker);
            let runtime = bound_runtime.clone();
            Box::pin(async move {
                let function = runmat_hir::FunctionId(binding.function);
                if let runmat_value::BoundFunctionAuthority::Portable { origin } =
                    &binding.authority
                {
                    if registry
                        .get_current(function)
                        .is_some_and(|body| body.binding_origin == Some(*origin))
                    {
                        return invoke_semantic_function_value_with_capture_updates(
                            binding.function,
                            &args,
                            requested_outputs,
                            &registry,
                            runtime,
                        )
                        .await
                        .map(|(value, _)| value);
                    }
                }
                let view = registry.owned_view(function, &binding.authority);
                if let Some(view) = view {
                    return invoke_semantic_function_value_with_capture_updates(
                        binding.function,
                        &args,
                        requested_outputs,
                        &view,
                        runtime,
                    )
                    .await
                    .map(|(value, _)| value);
                }
                if let Some(previous) = previous {
                    return previous(binding, &args, requested_outputs).await;
                }
                Err(mex(
                    "UnavailableBoundFunction",
                    "bound function definition is unavailable in this execution owner",
                ))
            })
        }),
    );
    let previous_semantic_resolver = user_functions::current_semantic_function_resolver();
    let registry_for_function_resolver = Arc::new(function_registry.current_view());
    let active_definition_registry = Arc::clone(&function_registry);
    let _semantic_resolver_guard =
        user_functions::install_semantic_function_resolver(Some(Arc::new(move |name: &str| {
            if let Some(active_function) = user_functions::current_active_semantic_function() {
                if let Some(function) =
                    active_definition_registry.get(runmat_hir::FunctionId(active_function))
                {
                    if let Some(scoped_function) = registry_for_function_resolver
                        .resolve_name_in_private_scope(&function.private_owner_scope, name)
                    {
                        return Some(scoped_function.0);
                    }
                }
            }
            if let Some(function) = registry_for_function_resolver.resolve_name(name) {
                return Some(function.0);
            }
            previous_semantic_resolver
                .as_ref()
                .and_then(|resolver| resolver(name))
        })));
    let registry_for_execution_source = Arc::clone(&function_registry);
    let registry_for_execution_group = Arc::clone(&function_registry);
    let registry_for_execution_owners = Arc::clone(&function_registry);
    let source_runtime = context.runtime.clone();
    let group_runtime = context.runtime.clone();
    let _execution_source_guard = user_functions::install_execution_source_resolver(Some(
        std::rc::Rc::new(user_functions::ExecutionSourceFacts::new(
            std::rc::Rc::new(move |function| {
                registry_for_execution_source
                    .execution_source_fact_in(&source_runtime, runmat_hir::FunctionId(function))
            }),
            std::rc::Rc::new(move |source| {
                registry_for_execution_group.execution_source_group_facts_in(&group_runtime, source)
            }),
            std::rc::Rc::new(move |function| {
                registry_for_execution_owners
                    .execution_definition_owners(runmat_hir::FunctionId(function))
            }),
        )),
    ));
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
        user_functions::install_source_function_catalog(Some(Arc::new(source_function_catalog)));
    CALL_COUNTS.with(|cc| {
        *cc.borrow_mut() = call_counts.clone();
    });
    let _workspace_guard = interp_engine::prepare_workspace_guard(
        &bytecode.var_names,
        &mut vars,
        initial_assigned_var_count,
        &bytecode.initially_unassigned_slots,
        global_aliases.clone(),
    );
    let thread_roots: Vec<Value> = runtime_globals::collect_thread_roots();
    let mut sequence_register = interp_dispatch::SequenceState::default();
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
    let execution_result = super::dispatch_cancellation::until_cancelled(&cancellation, async {
        while pc < bytecode.instructions.len() {
            crate::runtime::workspace::reconcile_workspace_vars(&mut vars);
            if let (Some(scope), Some(exception)) =
                (catch_entries.get(&pc), last_exception.as_ref())
            {
                catch_exceptions.insert(*scope, std::rc::Rc::new(exception.clone()));
            }
            if completion_boundary.is_some_and(|boundary| boundary.pc() == pc) {
                break;
            }
            if let Some(sites) = bytecode.coverage_sites.get(pc) {
                runmat_runtime::coverage::hit_sites(sites);
            }
            set_vm_pc(pc);
            #[cfg(feature = "native-accel")]
            set_current_pc(pc);
            interp_engine::check_cancelled()?;
            #[cfg(feature = "native-accel")]
            if let (Some(plan), Some(graph)) =
                (active_group_plan_clone(), fusion_accel_graph.as_ref())
            {
                if plan.group.span.start == pc && !fusion_observations.contains_start(pc) {
                    #[cfg(feature = "native-accel")]
                    {
                        interp_engine::note_fusion_gate(
                            &mut interpreter_timing,
                            &plan,
                            &bytecode,
                            pc,
                            accel_fusion::fusion_span_has_vm_barrier(
                                &bytecode.instructions,
                                &plan.group.span,
                            ),
                            accel_fusion::fusion_span_live_result_count(
                                &bytecode.instructions,
                                &plan.group.span,
                            ),
                        );
                    }
                    let span = plan.group.span.clone();
                    let has_barrier =
                        accel_fusion::fusion_span_has_vm_barrier(&bytecode.instructions, &span);
                    let _fusion_span = info_span!(
                        "fusion.execute",
                        span_start = plan.group.span.start,
                        span_end = plan.group.span.end,
                        kind = ?plan.group.kind
                    )
                    .entered();
                    let fusion_started = Instant::now();
                    let trace = begin_trace(fusion_operation_token(&plan.group.kind), None);
                    trace.event(
                        PlacementEventKind::Candidate,
                        Some(PlacementVariant::ProviderFusion),
                        Some("legacy_fusion_candidate"),
                        None,
                        None,
                        &[],
                    );
                    trace.event(
                        PlacementEventKind::Candidate,
                        Some(PlacementVariant::SharedRuntime),
                        Some("bytecode_fallback_candidate"),
                        None,
                        None,
                        &[],
                    );
                    if !has_barrier {
                        let _consumed_fusion = crate::runtime::gc::ManagedRoots::new(&stack)
                            .map_err(RuntimeError::new)?;
                        match accel_fusion::try_execute_fusion_group(
                        &plan,
                        graph,
                        &mut stack,
                        &mut vars,
                        &mut context,
                        Some(trace.correlation()),
                    )
                    .await
                    {
                        Ok(runmat_accelerate::fusion_exec::FusionExecutionOutcome::Executed(
                            result,
                        )) => {
                            trace.submitted(
                                PlacementVariant::ProviderFusion,
                                Some(duration_ns_saturating(fusion_started.elapsed())),
                            );
                            stack.push(result);
                            let executed_pc = pc;
                            pc = plan.group.span.end + 1;
                            fusion_observations.instruction_exit(executed_pc, pc);
                            continue;
                        }
                        Ok(
                            runmat_accelerate::fusion_exec::FusionExecutionOutcome::SharedRuntime(
                                ticket,
                            ),
                        ) => {
                            fusion_observations.begin(span.clone(), Some(ticket), trace);
                            // Planning may yield or request cancellation. Install
                            // the observation first so cancellation retires the
                            // selected ticket as failure, even for a one-instruction
                            // CPU region that could otherwise complete this turn.
                            interp_engine::check_cancelled()?;
                        }
                        Err(err) => {
                            trace.event(
                                PlacementEventKind::Fallback,
                                Some(PlacementVariant::SharedRuntime),
                                Some("fusion_execution_failed"),
                                Some(duration_ns_saturating(fusion_started.elapsed())),
                                None,
                                &[],
                            );
                            fusion_observations.begin(span.clone(), None, trace);
                            log::debug!("fusion fallback at pc {}: {}", pc, err);
                        }
                    }
                    } else {
                        trace.event(
                            PlacementEventKind::Selected,
                            Some(PlacementVariant::SharedRuntime),
                            Some("fusion_vm_barrier"),
                            None,
                            None,
                            &[],
                        );
                        fusion_observations.begin(span.clone(), None, trace);
                        interp_engine::note_fusion_skip(pc, &span);
                    }
                }
            }
            interp_engine::note_pre_dispatch(
                &mut interpreter_timing,
                debug_stack,
                pc,
                &bytecode.instructions[pc],
                stack.len(),
            );
            let call_counts_snapshot = CALL_COUNTS.with(|cc| cc.borrow().clone());
            let store_var_global_aliases = match &bytecode.instructions[pc] {
                Instr::StoreVar(_) => Some(global_aliases.clone()),
                _ => None,
            };
            let store_local_global_aliases = match &bytecode.instructions[pc] {
                Instr::StoreLocal(_) => Some(global_aliases.clone()),
                _ => None,
            };
            let mut store_var_before_overwrite = |_current: &Value, _incoming: &Value| {};
            let mut store_var_after_store = |stored_index: usize, stored_value: &Value| {
                if let Some(assigned_slots) = completion_assigned_slots.as_mut() {
                    assigned_slots.insert(stored_index);
                }
                if let Some(ref aliases) = store_var_global_aliases {
                    runtime_globals::update_global_store(stored_index, stored_value, aliases);
                }
            };
            let mut store_local_before_local_overwrite = |_current: &Value, _incoming: &Value| {};
            let mut store_local_before_var_overwrite = |_current: &Value, _incoming: &Value| {};
            let mut store_local_after_store = |stored_offset: usize, stored_value: &Value| {
                if let Some(ref aliases) = store_local_global_aliases {
                    runtime_globals::update_global_store(stored_offset, stored_value, aliases);
                }
            };
            let mut store_local_after_fallback_store =
                |func_name: &str, stored_offset: usize, stored_value: &Value| {
                    if let Some(ref aliases) = store_local_global_aliases {
                        runtime_globals::update_global_store(stored_offset, stored_value, aliases);
                    }
                    runtime_globals::update_persistent_local_store(
                        func_name,
                        stored_offset,
                        stored_value,
                    );
                };
            #[cfg(feature = "native-accel")]
            let executed_pc = pc;
            let active_catch = bytecode
                .catch_scopes
                .get(pc)
                .copied()
                .flatten()
                .and_then(|scope| catch_exceptions.get(&scope).cloned());
            let dispatch_future = interp_dispatch::dispatch_instruction(
                interp_dispatch::DispatchMeta {
                    instr: &bytecode.instructions[pc],
                    bytecode: &bytecode,
                    var_names: &bytecode.var_names,
                    function_registry: &function_registry,
                    source_id: bytecode
                        .source_id
                        .and_then(|source| source.resolve_in(&context.runtime)),
                    call_arg_spans: bytecode.call_arg_spans.get(pc).cloned().flatten(),
                    call_counts: &call_counts_snapshot,
                    current_function_name: &current_function_name,
                },
                interp_dispatch::DispatchState {
                    stack: &mut stack,
                    vars: &mut vars,
                    sequence_register: &mut sequence_register,
                    context: &mut context,
                    try_stack: &mut try_stack,
                    last_exception: &mut last_exception,
                    imports: &mut imports,
                    global_aliases: &global_aliases,
                    persistent_aliases: &mut persistent_aliases,
                    missing_input_slots: &mut missing_input_slots,
                    pc: &mut pc,
                },
                interp_dispatch::DispatchHooks {
                    store_var_before_overwrite: &mut store_var_before_overwrite,
                    store_var_after_store: &mut store_var_after_store,
                    store_local_before_local_overwrite: &mut store_local_before_local_overwrite,
                    store_local_before_var_overwrite: &mut store_local_before_var_overwrite,
                    store_local_after_store: &mut store_local_after_store,
                    store_local_after_fallback_store: &mut store_local_after_fallback_store,
                },
            );
            let dispatch_result = if let Some(exception) = active_catch {
                catch_runtime
                    .scope_active_catch_unboxed_shared(Some(exception), dispatch_future)
                    .await
            } else {
                dispatch_future.await
            };
            let dispatch_result = match dispatch_result {
                Ok(result) => result,
                Err(err) => {
                    #[cfg(feature = "native-accel")]
                    fusion_observations.abort();
                    sequence_register.clear(&mut stack);
                    match interp_dispatch::redirect_exception_to_catch(
                        err,
                        &mut try_stack,
                        &mut vars,
                        &mut last_exception,
                        &mut pc,
                        refresh_workspace_state,
                    ) {
                        interp_dispatch::ExceptionHandling::Caught => {
                            continue;
                        }
                        interp_dispatch::ExceptionHandling::Uncaught(err) => return Err(*err),
                    }
                }
            };
            if let Some(decision) = dispatch_result {
                match decision {
                    interp_dispatch::DispatchHandled::Generic(DispatchDecision::ContinueLoop) => {
                        #[cfg(feature = "native-accel")]
                        fusion_observations.instruction_exit(executed_pc, pc);
                        continue;
                    }
                    interp_dispatch::DispatchHandled::Generic(DispatchDecision::FallThrough) => {
                        pc += 1;
                        #[cfg(feature = "native-accel")]
                        fusion_observations.instruction_exit(executed_pc, pc);
                        continue;
                    }
                    interp_dispatch::DispatchHandled::Generic(DispatchDecision::Return) => {
                        interpreter_timing.flush_host_span("return", None);
                        break;
                    }
                    interp_dispatch::DispatchHandled::ReturnValue(
                        DispatchDecision::ContinueLoop,
                    )
                    | interp_dispatch::DispatchHandled::Return(DispatchDecision::ContinueLoop) => {
                        #[cfg(feature = "native-accel")]
                        fusion_observations.instruction_exit(executed_pc, pc);
                        continue;
                    }
                    interp_dispatch::DispatchHandled::ReturnValue(DispatchDecision::Return) => {
                        interpreter_timing.flush_host_span("return_value", None);
                        break;
                    }
                    interp_dispatch::DispatchHandled::Return(DispatchDecision::Return) => {
                        interpreter_timing.flush_host_span("return", None);
                        break;
                    }
                    interp_dispatch::DispatchHandled::ReturnValue(
                        DispatchDecision::FallThrough,
                    )
                    | interp_dispatch::DispatchHandled::Return(DispatchDecision::FallThrough) => {
                        pc += 1;
                        #[cfg(feature = "native-accel")]
                        fusion_observations.instruction_exit(executed_pc, pc);
                        continue;
                    }
                }
            }
            match bytecode.instructions[pc].clone() {
                Instr::EmitStackTop { .. }
                | Instr::EmitVar { .. }
                | Instr::AndAnd(_)
                | Instr::OrOr(_)
                | Instr::JumpIfFalse(_)
                | Instr::Jump(_)
                | Instr::LoadConst(_)
                | Instr::LoadInt(_)
                | Instr::LoadComplex(_, _)
                | Instr::LoadBool(_)
                | Instr::LoadString(_)
                | Instr::LoadCharRow(_)
                | Instr::LoadLocal(_)
                | Instr::LoadVar(_)
                | Instr::LoadVarForIndexAssignment(_)
                | Instr::LoadVarForStructAssignment(_)
                | Instr::StoreVar(_)
                | Instr::StoreLocal(_)
                | Instr::Swap
                | Instr::Pop
                | Instr::EnterTry { .. }
                | Instr::LeaveTry(_)
                | Instr::ReturnValue
                | Instr::Return
                | Instr::EnterScope(_)
                | Instr::LoadMember(_)
                | Instr::LoadMemberOrInit(_)
                | Instr::LoadMemberDynamic
                | Instr::LoadMemberDynamicOrInit
                | Instr::LoadMemberSequence { .. }
                | Instr::LoadMemberDynamicSequence { .. }
                | Instr::MemberSequenceCardinality
                | Instr::LoadMemberSequenceUsingOutputSlot { .. }
                | Instr::LoadMemberDynamicSequenceUsingOutputSlot { .. }
                | Instr::CaptureCallOutputSequence
                | Instr::CaptureScalarSequence
                | Instr::CaptureMemberSequence { .. }
                | Instr::CaptureMemberDynamicSequence { .. }
                | Instr::CaptureCellContentsSequence { .. }
                | Instr::CaptureReturnedOutputsSequence { .. }
                | Instr::ReadSubscriptPath { .. }
                | Instr::CaptureSubscriptPath { .. }
                | Instr::BeginSubscriptEndReceiver { .. }
                | Instr::LoadSubscriptEnd { .. }
                | Instr::FinishSubscriptEndReceiver { .. }
                | Instr::FinishDottedInvokeReceiver { .. }
                | Instr::BeginOutputAssignment { .. }
                | Instr::PrepareFixedOutputTarget
                | Instr::PrepareDiscardOutputTarget
                | Instr::PrepareMemberSequenceOutputTarget { .. }
                | Instr::PrepareMemberDynamicSequenceOutputTarget { .. }
                | Instr::PrepareIndexedMemberSequenceOutputTarget { .. }
                | Instr::PrepareIndexedMemberDynamicSequenceOutputTarget { .. }
                | Instr::PrepareCellContentsSequenceOutputTarget { .. }
                | Instr::BeginSequenceOutputTarget { .. }
                | Instr::PrepareMemberPathStep(_)
                | Instr::PrepareDynamicMemberPathStep
                | Instr::BeginPreparedIndexSelectors { .. }
                | Instr::LoadPreparedIndexEnd { .. }
                | Instr::BeginContextualIndexSelectors { .. }
                | Instr::LoadContextualIndexEnd { .. }
                | Instr::FinishContextualIndexSelectors { .. }
                | Instr::PrepareParenthesesPathStep { .. }
                | Instr::PrepareBracesPathStep { .. }
                | Instr::FinishMemberSequenceOutputTarget(_)
                | Instr::FinishDynamicMemberSequenceOutputTarget
                | Instr::FinishCellContentsSequenceOutputTarget { .. }
                | Instr::LoadPreparedOutputCardinality
                | Instr::CommitPreparedOutputTargets { .. }
                | Instr::StoreMember(_)
                | Instr::StoreMemberOrInit(_)
                | Instr::StoreMemberDynamic
                | Instr::StoreMemberDynamicOrInit
                | Instr::StoreMemberSequence(_)
                | Instr::StoreMemberDynamicSequence
                | Instr::Index(_)
                | Instr::IndexSlice(_, _, _, _)
                | Instr::IndexCell { .. }
                | Instr::IndexCellExpand { .. }
                | Instr::IndexCellList { .. }
                | Instr::StoreIndex(_)
                | Instr::StoreIndexCell { .. }
                | Instr::StoreIndexDelete(_)
                | Instr::StoreIndexCellDelete { .. }
                | Instr::StoreSlice(_, _, _, _)
                | Instr::StoreSliceDelete(_, _, _, _)
                | Instr::CallMethodOrMemberIndexMulti { .. }
                | Instr::CallMethodOrMemberIndexExpandMultiOutput { .. }
                | Instr::LoadMethod(_)
                | Instr::CreateFunctionHandle(_)
                | Instr::CreateExternalFunctionHandle(_)
                | Instr::CreateMethodFunctionHandle(_)
                | Instr::CreateBoundFunctionHandle(_, _, _)
                | Instr::CreateExternalBoundFunctionHandle(_, _, _)
                | Instr::CreateClosure(_, _)
                | Instr::CreateSemanticClosure(_, _, _, _)
                | Instr::LoadStaticProperty(_, _)
                | Instr::LoadWorkspaceFirstStaticProperty { .. }
                | Instr::RegisterClass { .. }
                | Instr::CallFevalMulti(_, _)
                | Instr::CallFevalMultiUsingOutputSlot(_, _)
                | Instr::CallFevalExpandMultiOutput(_, _)
                | Instr::CallFevalExpandMultiOutputUsingOutputSlot(_, _)
                | Instr::CreateSemanticFuture(_, _, _)
                | Instr::CreateSemanticFutureExpandMultiOutput(_, _, _)
                | Instr::ScheduleFeval { .. }
                | Instr::Spawn
                | Instr::SpawnOn
                | Instr::Await
                | Instr::FetchOutputs { .. }
                | Instr::FetchNext { .. }
                | Instr::EnsurePool(_)
                | Instr::CurrentPool(_)
                | Instr::CallBuiltinMulti(_, _, _)
                | Instr::CallBuiltinMultiUsingOutputSlot(_, _, _)
                | Instr::CallSuperConstructorMulti { .. }
                | Instr::CallSuperMethodMulti { .. }
                | Instr::CallSemanticFunctionMulti(_, _, _)
                | Instr::CallSemanticFunctionMultiUsingOutputSlot(_, _, _)
                | Instr::CallSemanticNestedFunctionMulti { .. }
                | Instr::CallSemanticNestedFunctionMultiUsingOutputSlot { .. }
                | Instr::CallFunctionMulti { .. }
                | Instr::CallFunctionMultiUsingOutputSlot { .. }
                | Instr::CallFunctionExpandMultiOutput { .. }
                | Instr::CallWorkspaceFirstMulti { .. }
                | Instr::CallWorkspaceFirstMultiUsingOutputSlot { .. }
                | Instr::CallWorkspaceFirstExpandMultiOutput { .. }
                | Instr::CallWorkspaceFirstExpandMultiOutputUsingOutputSlot { .. }
                | Instr::CallSemanticFunctionExpandMultiOutput(_, _, _)
                | Instr::CallSemanticNestedFunctionExpandMultiOutput { .. }
                | Instr::CallBuiltinExpandMultiOutput(_, _, _)
                | Instr::CallSuperConstructorExpandMultiOutput { .. }
                | Instr::CallSuperMethodExpandMultiOutput { .. }
                | Instr::ExitScope(_)
                | Instr::RegisterImport { .. }
                | Instr::DeclareGlobal(_)
                | Instr::DeclareGlobalNamed(_, _)
                | Instr::DeclarePersistent(_)
                | Instr::DeclarePersistentNamed(_, _)
                | Instr::CreateCell2D(_, _)
                | Instr::CreateCellFromSequences { .. }
                | Instr::CreateStructLiteral(_)
                | Instr::CreateObjectLiteral { .. }
                | Instr::Add
                | Instr::Sub
                | Instr::Mul
                | Instr::ElemMul
                | Instr::ElemDiv
                | Instr::ElemPow
                | Instr::ElemLeftDiv
                | Instr::Neg
                | Instr::UPlus
                | Instr::Transpose
                | Instr::ConjugateTranspose
                | Instr::Pow
                | Instr::RightDiv
                | Instr::LeftDiv
                | Instr::LessEqual
                | Instr::Less
                | Instr::Greater
                | Instr::GreaterEqual
                | Instr::Equal
                | Instr::NotEqual
                | Instr::LogicalNot
                | Instr::LogicalAnd
                | Instr::LogicalOr
                | Instr::ExecuteParfor { .. }
                | Instr::ExecuteSpmd { .. }
                | Instr::Distributed(_)
                | Instr::Collective { .. }
                | Instr::Unpack(_)
                | Instr::CreateMatrix(_, _)
                | Instr::CreateMatrixFromSequences { .. }
                | Instr::CreateMatrixDynamic(_)
                | Instr::CreateRange(_)
                | Instr::PackToRow(_)
                | Instr::PackToCol(_) => unreachable!("handled by dispatch_instruction"),
                Instr::StochasticEvolution => {
                    let steps_value = stack
                        .pop()
                        .ok_or(mex("StackUnderflow", "stack underflow"))?;
                    let scale_value = stack
                        .pop()
                        .ok_or(mex("StackUnderflow", "stack underflow"))?;
                    let drift_value = stack
                        .pop()
                        .ok_or(mex("StackUnderflow", "stack underflow"))?;
                    let state_value = stack
                        .pop()
                        .ok_or(mex("StackUnderflow", "stack underflow"))?;
                    let evolved =
                        crate::accel::idioms::stochastic_evolution::execute_stochastic_evolution(
                            state_value,
                            drift_value,
                            scale_value,
                            steps_value,
                        )
                        .await?;
                    stack.push(evolved);
                }
            }
            if debug_stack {
                debug!(pc, stack_len = stack.len(), "[vm] after exec");
            }
            pc += 1;
            #[cfg(feature = "native-accel")]
            fusion_observations.instruction_exit(executed_pc, pc);
        }
        Ok::<(), RuntimeError>(())
    })
    .await;
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
    crate::runtime::workspace::reconcile_workspace_vars(&mut vars);
    sync_initial_vars(initial_vars, &vars);
    Ok(InterpreterOutcome::Completed(
        crate::interpreter::state::InterpreterCompletion {
            assigned_slots: completion_assigned_slots
                .unwrap_or_else(crate::runtime::workspace::current_workspace_assigned_slots),
            values: vars,
        },
    ))
}

pub async fn interpret(bytecode: &Bytecode) -> Result<Vec<Value>, RuntimeError> {
    let mut vars = vec![Value::Num(0.0); bytecode.var_count];
    match interpret_with_vars(bytecode, &mut vars, Some("<main>")).await {
        Ok(InterpreterOutcome::Completed(completion)) => Ok(completion.values),
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
    pub region: &'a crate::BytecodeParforRegion,
    pub inputs: Vec<Value>,
    pub iterations: Vec<Value>,
    pub randomness: runmat_execution::ParallelRandomnessContext,
    pub mode: ParforTaskMode,
    pub current_function_name: &'a str,
    pub runtime: runmat_runtime::context::RuntimeContext,
}

pub(crate) async fn interpret_parfor_task_in_context(
    execution: ParforTaskExecution<'_>,
) -> Result<Vec<Value>, RuntimeError> {
    let ParforTaskExecution {
        bytecode,
        region,
        inputs,
        iterations,
        randomness,
        mode,
        current_function_name,
        runtime,
    } = execution;
    randomness
        .validate_for_iterations(iterations.len())
        .map_err(|error| mex("ParallelRandomness", &error.to_string()))?;
    let deterministic_streams = match randomness {
        runmat_execution::ParallelRandomnessContext::Deterministic { streams } => Some(streams),
        runmat_execution::ParallelRandomnessContext::Inherit
        | runmat_execution::ParallelRandomnessContext::Nondeterministic => None,
    };
    let input_variables = region.input_variables().collect::<Vec<_>>();
    if inputs.len() != input_variables.len() {
        return Err(mex(
            "ParallelTaskInput",
            "parallel task inputs do not match the compiler-owned region layout",
        ));
    }
    let mut frame = vec![None; bytecode.var_count];
    for (variable, value) in input_variables.into_iter().zip(inputs) {
        let Some(slot) = frame.get_mut(variable.slot) else {
            return Err(mex(
                "ParallelFrame",
                "parallel task input is outside its compiled VM frame",
            ));
        };
        *slot = Some(value);
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
                .and_then(Clone::clone)
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
        .map(|slot| (slot, Vec::new()))
        .collect::<HashMap<_, _>>();

    for (iteration_index, iteration) in iterations.into_iter().enumerate() {
        for variable in &region.variables {
            if matches!(
                variable.contract.role,
                runmat_types::ParallelVariableRole::Temporary
                    | runmat_types::ParallelVariableRole::Private
            ) {
                let Some(slot) = frame.get_mut(variable.slot) else {
                    return Err(mex(
                        "ParallelFrame",
                        "parallel private value is outside its compiled VM frame",
                    ));
                };
                *slot = None;
            }
        }
        if mode == ParforTaskMode::IndependentIterations {
            for (slot, identity) in &reduction_inputs {
                frame[*slot] = Some(identity.clone());
            }
        }
        frame[region.loop_slot] = Some(iteration);
        let _random_stream = deterministic_streams
            .as_ref()
            .and_then(|streams| streams.get(iteration_index).copied())
            .map(|stream| {
                runmat_runtime::builtins::common::random::enter_parallel_random_stream(
                    &runtime, stream,
                )
            })
            .transpose()?;
        let resume = InterpreterResumeState {
            pc: region.body.pc,
            completion_boundary: Some(
                crate::interpreter::state::InterpreterCompletionBoundary::before(region.header.pc),
            ),
            vars: frame,
            supplied_inputs: 0,
            requested_outputs: 0,
            missing_input_slots: HashSet::new(),
            global_aliases: HashMap::new(),
            persistent_aliases: HashMap::new(),
            side_effect_epoch: 0,
        };
        frame = match Box::pin(interpret_resume_in_context(
            bytecode,
            resume,
            Some(current_function_name),
            runtime.clone(),
        ))
        .await?
        {
            InterpreterOutcome::Completed(completion) => {
                completion.values.into_iter().map(Some).collect()
            }
        };
        if mode == ParforTaskMode::IndependentIterations {
            for (slot, values) in &mut reduction_outputs {
                let value = frame.get(*slot).and_then(Clone::clone).ok_or_else(|| {
                    mex(
                        "ParallelTaskOutput",
                        "parallel task did not assign a declared reduction output",
                    )
                })?;
                values.push(value);
            }
        }
    }

    region
        .output_variables()
        .map(|variable| {
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
                let length = values.len();
                return runmat_value::CellArray::new(values, 1, length)
                    .map(Value::Cell)
                    .map_err(|error| mex("ParallelTaskOutput", &error));
            }
            frame
                .get(variable.slot)
                .and_then(Clone::clone)
                .ok_or_else(|| {
                    mex(
                        "ParallelTaskOutput",
                        "parallel task did not assign a declared output",
                    )
                })
        })
        .collect()
}

pub async fn interpret_function(
    bytecode: &Bytecode,
    vars: Vec<Value>,
) -> Result<Vec<Value>, RuntimeError> {
    interpret_function_with_counts(bytecode, vars, "<anonymous>", 0, 0, HashSet::new()).await
}

pub async fn interpret_function_with_counts(
    bytecode: &Bytecode,
    vars: Vec<Value>,
    name: &str,
    out_count: usize,
    in_count: usize,
    missing_input_slots: HashSet<usize>,
) -> Result<Vec<Value>, RuntimeError> {
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
) -> Result<Vec<Value>, RuntimeError> {
    let (runtime, workspace_activation) =
        crate::runtime::workspace_service::VmWorkspaceService::install(runtime, None);
    let frame = workspace_activation.frame();
    let runtime = with_default_acceleration_service(runtime);
    let result = workspace_activation
        .scope(
            runtime.scope(interpret_function_with_counts_in_context_inner(
                bytecode,
                vars,
                name,
                out_count,
                in_count,
                missing_input_slots,
                runtime.clone(),
            )),
        )
        .await;
    let _ = crate::runtime::workspace::take_workspace_completion(frame);
    result
}

async fn interpret_function_with_counts_in_context_inner(
    bytecode: &Bytecode,
    vars: Vec<Value>,
    name: &str,
    out_count: usize,
    in_count: usize,
    missing_input_slots: HashSet<usize>,
    runtime: runmat_runtime::context::RuntimeContext,
) -> Result<Vec<Value>, RuntimeError> {
    let mut vars = vars;
    CALL_COUNTS.with(|cc| {
        cc.borrow_mut().push((in_count, out_count));
    });
    let call_counts = CALL_COUNTS.with(|cc| cc.borrow().clone());
    let mut state = InterpreterState::new_in_context(
        bytecode.clone(),
        &mut vars,
        Some(name),
        call_counts,
        runtime,
    );
    state.missing_input_slots = missing_input_slots;
    let _debug_frame_guard = runmat_runtime::debug_context::push_frame(
        name,
        bytecode
            .source_id
            .and_then(|source| source.resolve_in(&state.context.runtime)),
        bytecode_frame_span(bytecode),
    );
    let res = Box::pin(run_interpreter(Box::new(state), &mut vars)).await;
    CALL_COUNTS.with(|cc| {
        cc.borrow_mut().pop();
    });
    let res = match res {
        Ok(InterpreterOutcome::Completed(completion)) => Ok(completion.values),
        Err(e) => Err(e),
    }?;
    runtime_globals::persist_declared_for_bytecode(bytecode, name, &vars);
    Ok(res)
}

#[cfg(test)]
#[path = "runner/native_metadata_tests.rs"]
mod native_metadata_tests;

#[cfg(test)]
mod tests {
    use super::{
        interpret_function_with_counts_in_context, interpret_resume_in_context,
        interpret_with_vars, output_sequence, run_interpreter_inner,
    };
    use crate::bytecode::program::Bytecode;
    use crate::bytecode::Instr;
    use crate::interpreter::api::{InterpreterResumeState, InterpreterState};
    use futures::executor::block_on;
    use runmat_runtime::builtins::common::validation::{
        value_is_empty, value_is_greater_than, value_is_greater_than_or_equal, value_is_integer,
        value_is_less_than, value_is_less_than_or_equal, value_is_negative, value_is_nonnegative,
        value_is_nonpositive, value_is_nonzero, value_is_numeric_or_logical, value_is_positive,
        value_is_real, value_is_scalar_or_empty, value_is_text,
    };
    use runmat_value::{CellArray, HandleRef, StructValue, Tensor, Value};
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
        assert_eq!(
            sequence.into_values(),
            vec![Value::Num(1.0), Value::Num(2.0)]
        );
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
            let mut outer = vec![Value::Num(1.0)];
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
        let runtime = runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        let (runtime, base_activation) =
            crate::runtime::workspace_service::VmWorkspaceService::install(runtime, None);
        let base_frame = base_activation.frame();
        let child = Bytecode::with_instructions(
            vec![
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
            ],
            0,
        );

        block_on(base_activation.scope(runtime.scope(async {
            let mut base = vec![Value::Num(5.0)];
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
                    let mut outer = vec![Value::Num(7.0)];
                    let _outer_guard = crate::runtime::workspace::set_transient_workspace_state(
                        std::collections::HashMap::from([("outer_value".to_string(), 0)]),
                        std::collections::HashSet::from(["outer_value".to_string()]),
                        &mut outer,
                        runmat_runtime::workspace::bindings::FrameGlobalBindings::default(),
                    );
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
                    .expect("zero-variable nested interpreter should own an exact child frame");
                    assert!(result.is_empty());
                    assert_eq!(
                        crate::runtime::workspace::workspace_lookup("child_local"),
                        None
                    );
                    assert_eq!(
                        crate::runtime::workspace::workspace_lookup("caller_mark"),
                        Some(Value::Num(17.0))
                    );
                    let base = crate::runtime::workspace::workspace_target_snapshot(
                        crate::runtime::workspace::WorkspaceTarget::Base,
                    )
                    .expect("nested child should retain its inherited base lineage");
                    assert_eq!(
                        base.names
                            .get("base_mark")
                            .and_then(|slot| base.vars.get(*slot)),
                        Some(&Value::Num(19.0))
                    );
                }))
                .await;
            let _ = crate::runtime::workspace::take_workspace_completion(outer_frame);

            assert_eq!(
                crate::runtime::workspace::workspace_lookup("caller_mark"),
                None
            );
            assert_eq!(
                crate::runtime::workspace::workspace_lookup("base_mark"),
                Some(Value::Num(19.0))
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
            let mut outer = vec![Value::Num(31.0)];
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
    fn interpreter_resume_starts_at_exact_pc_without_replaying_prefix() {
        super::CALL_COUNTS.with(|counts| assert!(counts.borrow().is_empty()));
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
            vars: vec![Some(Value::Num(41.0)), None, None],
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
        let super::InterpreterOutcome::Completed(completion) = block_on(
            interpret_resume_in_context(&bytecode, resume, Some("resume_test"), runtime),
        )
        .unwrap();
        assert_eq!(completion.values[0], Value::Num(41.0));
        assert_eq!(completion.values[1], Value::Num(2.0));
        assert_eq!(completion.values[2], Value::Num(0.0));
        assert_eq!(
            completion.assigned_slots,
            std::collections::HashSet::from([0, 1])
        );
        super::CALL_COUNTS.with(|counts| assert!(counts.borrow().is_empty()));
    }

    #[test]
    fn resumed_legacy_global_slot_keeps_its_typed_binding_and_store() {
        let bytecode = Bytecode::with_instructions(
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
        let runtime = runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        block_on(runtime.scope(async {
            runmat_runtime::workspace::session::store_global_slot(0, Value::Num(17.0));
        }));
        let resume = InterpreterResumeState {
            pc: 1,
            completion_boundary: None,
            vars: vec![Some(Value::Num(3.0)), None, None],
            supplied_inputs: 0,
            requested_outputs: 3,
            missing_input_slots: Default::default(),
            global_aliases: std::collections::HashMap::from([(
                0,
                runmat_runtime::workspace::bindings::FrameGlobalBinding::LegacySlot,
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
        assert_eq!(completion.values[1], Value::Num(17.0));
        assert_eq!(completion.values[2], Value::Num(29.0));
        block_on(runtime.scope(async {
            assert_eq!(
                runmat_runtime::workspace::session::global_slot_value(0),
                Some(Value::Num(29.0))
            );
        }));
    }

    #[test]
    fn interpreter_resume_stops_before_the_declared_completion_boundary() {
        super::CALL_COUNTS.with(|counts| assert!(counts.borrow().is_empty()));
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
            vars: vec![None, None],
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
        let super::InterpreterOutcome::Completed(completion) = block_on(
            interpret_resume_in_context(&bytecode, resume, Some("bounded_resume_test"), runtime),
        )
        .unwrap();
        assert_eq!(completion.values, vec![Value::Num(7.0), Value::Num(0.0)]);
        assert_eq!(
            completion.assigned_slots,
            std::collections::HashSet::from([0])
        );
        super::CALL_COUNTS.with(|counts| assert!(counts.borrow().is_empty()));
    }

    #[test]
    fn numeric_or_logical_validator_accepts_expected_domains() {
        assert!(value_is_numeric_or_logical(&Value::Num(1.0)));
        assert!(value_is_numeric_or_logical(&Value::Bool(true)));
        assert!(value_is_numeric_or_logical(&Value::Complex(1.0, 2.0)));
        let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("tensor");
        assert!(value_is_numeric_or_logical(&Value::Tensor(tensor)));
        assert!(!value_is_numeric_or_logical(&Value::String(
            "x".to_string()
        )));
        assert!(!value_is_numeric_or_logical(&Value::CharArray(
            runmat_value::CharArray::new("x".chars().collect(), 1, 1).expect("char")
        )));
    }

    #[test]
    fn text_validator_accepts_string_char_vector_and_cellstr() {
        assert!(value_is_text(&Value::String("x".to_string())));
        assert!(value_is_text(&Value::CharArray(
            runmat_value::CharArray::new("abc".chars().collect(), 1, 3).expect("char")
        )));
        assert!(value_is_text(&Value::Cell(
            CellArray::new(
                vec![
                    Value::CharArray(
                        runmat_value::CharArray::new("a".chars().collect(), 1, 1).expect("char"),
                    ),
                    Value::String("b".to_string()),
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
        let empty_char = runmat_value::CharArray::new(Vec::new(), 1, 0).expect("empty char array");
        assert!(value_is_empty(&Value::CharArray(empty_char)));
        let empty_cell = CellArray::new(Vec::new(), 0, 0).expect("empty cell");
        assert!(value_is_empty(&Value::Cell(empty_cell)));
        assert!(!value_is_empty(&Value::String("".to_string())));
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

            let mut vars = vec![Value::GpuTensor(handle.clone())];
            let bytecode = Bytecode::with_instructions(vec![Instr::Return], vars.len());
            let cancelled = runmat_execution::cancellation::CancellationSource::new();
            cancelled.cancel(runmat_execution::CancellationReason::User);
            let _interrupt_guard =
                runmat_runtime::interrupt::replace_interrupt(Some(cancelled.token()));

            let err = block_on(interpret_with_vars(&bytecode, &mut vars, Some("<main>")))
                .expect_err("cancelled execution should return error");
            assert_eq!(err.identifier(), Some("RunMat:ExecutionCancelled"));
            assert!(
                fusion_residency::is_resident(&handle),
                "cancellation must not invalidate a retained external view"
            );
            assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
            drop(vars);
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
            let mut seed_vars = vec![Value::Num(0.0)];
            let mut state =
                InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
            state.stack.push(Value::GpuTensor(handle.clone()));
            state.vars = vec![Value::Num(0.0)];
            drop(handle);

            let mut result_vars = vec![Value::Num(0.0)];
            let outcome = block_on(run_interpreter_inner(state, &mut result_vars))
                .expect("interpreter should complete");
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
            let mut seed_vars = vec![Value::Num(0.0)];
            let mut state =
                InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
            state.stack.push(Value::GpuTensor(handle.clone()));
            state.vars = vec![Value::Num(0.0)];

            let mut result_vars = vec![Value::Num(0.0)];
            let weak = handle.downgrade();
            drop(handle);
            let _ = block_on(run_interpreter_inner(state, &mut result_vars))
                .expect("interpreter should complete");
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
            let mut seed_vars = vec![Value::GpuTensor(handle.clone())];
            let mut state =
                InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
            state.stack.push(Value::GpuTensor(handle.clone()));
            state.vars = vec![Value::GpuTensor(handle.clone())];

            let mut result_vars = vec![Value::GpuTensor(handle.clone())];
            let _ = block_on(run_interpreter_inner(state, &mut result_vars))
                .expect("interpreter should complete");
            assert!(
                fusion_residency::is_resident(&handle),
                "pop should preserve residency for handles still referenced by vars"
            );
            assert!(
                block_on(TEST_PROVIDER.download(&handle)).is_ok(),
                "pop should not release provider storage for handles still referenced by vars"
            );
            let weak = handle.downgrade();
            drop(seed_vars);
            drop(result_vars);
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
            let mut seed_vars = vec![Value::Num(0.0)];
            let mut state =
                InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
            state.context.locals.push(Value::GpuTensor(handle.clone()));
            state.vars = vec![Value::Num(0.0)];

            let mut result_vars = vec![Value::Num(0.0)];
            let weak = handle.downgrade();
            drop(handle);
            let _ = block_on(run_interpreter_inner(state, &mut result_vars))
                .expect("exit scope should complete");
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
            let mut seed_vars = vec![Value::GpuTensor(handle.clone())];
            let mut state =
                InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
            state.context.locals.push(Value::GpuTensor(handle.clone()));
            state.vars = vec![Value::GpuTensor(handle.clone())];

            let mut result_vars = vec![Value::GpuTensor(handle.clone())];
            let _ = block_on(run_interpreter_inner(state, &mut result_vars))
                .expect("exit scope should complete");
            assert!(
                fusion_residency::is_resident(&handle),
                "exit scope should preserve residency for handles still referenced by vars"
            );
            assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_ok(),
            "exit scope should not release provider storage for handles still referenced by vars"
        );
            let weak = handle.downgrade();
            drop(seed_vars);
            drop(result_vars);
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
            let mut seed_vars = vec![Value::Num(0.0)];
            let mut state =
                InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
            let mut payload = StructValue::new();
            payload
                .fields
                .insert("nested".to_string(), Value::GpuTensor(handle.clone()));
            let target =
                runmat_gc::gc_allocate(Value::Struct(payload)).expect("gc allocate payload");
            state.context.locals.push(Value::HandleObject(HandleRef {
                class_name: "Payload".into(),
                target,
                valid: true,
            }));
            state.vars = vec![Value::Num(0.0)];

            let mut result_vars = vec![Value::Num(0.0)];
            let weak = handle.downgrade();
            drop(handle);
            let _ = block_on(run_interpreter_inner(state, &mut result_vars))
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
            let mut seed_vars = vec![Value::Num(0.0)];
            let mut state =
                InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
            let mut payload = StructValue::new();
            payload
                .fields
                .insert("nested".to_string(), Value::GpuTensor(handle.clone()));
            let target =
                runmat_gc::gc_allocate(Value::Struct(payload)).expect("gc allocate payload");
            let local_value = Value::HandleObject(HandleRef {
                class_name: "Payload".into(),
                target,
                valid: true,
            });
            state.context.locals.push(local_value.clone());
            state.vars = vec![local_value.clone()];

            let mut result_vars = vec![local_value];
            let _ = block_on(run_interpreter_inner(state, &mut result_vars))
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
            drop(seed_vars);
            drop(result_vars);
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
            let mut seed_vars = vec![
                Value::GpuTensor(handle.clone()),
                Value::GpuTensor(handle.clone()),
            ];
            let mut state =
                InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
            state.stack.push(Value::Num(0.0));
            state.vars = vec![
                Value::GpuTensor(handle.clone()),
                Value::GpuTensor(handle.clone()),
            ];

            let mut result_vars = state.vars.clone();
            let _ = block_on(run_interpreter_inner(state, &mut result_vars))
                .expect("store var should complete");
            assert!(
            fusion_residency::is_resident(&handle),
            "store var overwrite should preserve residency for handles still live in other vars"
        );
            assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_ok(),
            "store var overwrite should not release provider storage for handles still live in other vars"
        );
            let weak = handle.downgrade();
            drop(seed_vars);
            drop(result_vars);
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
            let mut seed_vars = vec![Value::GpuTensor(handle.clone())];
            let mut state =
                InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
            state.stack.push(Value::Num(0.0));
            state.vars = vec![Value::GpuTensor(handle.clone())];
            state.context.locals.push(Value::GpuTensor(handle.clone()));

            let mut result_vars = state.vars.clone();
            let _ = block_on(run_interpreter_inner(state, &mut result_vars))
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
            drop(seed_vars);
            drop(result_vars);
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
            let mut seed_vars = vec![Value::Num(0.0)];
            let mut state =
                InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
            let mut payload = StructValue::new();
            payload
                .fields
                .insert("nested".to_string(), Value::GpuTensor(handle.clone()));
            let target =
                runmat_gc::gc_allocate(Value::Struct(payload)).expect("gc allocate payload");
            state.vars = vec![Value::HandleObject(HandleRef {
                class_name: "Payload".into(),
                target,
                valid: true,
            })];
            state.stack.push(Value::Num(0.0));

            let mut result_vars = state.vars.clone();
            let weak = handle.downgrade();
            drop(handle);
            let _ = block_on(run_interpreter_inner(state, &mut result_vars))
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
            let mut seed_vars = vec![Value::Num(0.0), Value::Num(0.0)];
            let mut state =
                InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
            let mut payload = StructValue::new();
            payload
                .fields
                .insert("nested".to_string(), Value::GpuTensor(handle.clone()));
            let target =
                runmat_gc::gc_allocate(Value::Struct(payload)).expect("gc allocate payload");
            let nested = Value::HandleObject(HandleRef {
                class_name: "Payload".into(),
                target,
                valid: true,
            });
            state.vars = vec![nested.clone(), nested.clone()];
            state.stack.push(Value::Num(0.0));

            let mut result_vars = state.vars.clone();
            let _ = block_on(run_interpreter_inner(state, &mut result_vars)).expect(
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
            drop(seed_vars);
            drop(result_vars);
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
            let mut seed_vars = vec![Value::Num(0.0)];
            let mut state =
                InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
            let mut payload = StructValue::new();
            payload
                .fields
                .insert("nested".to_string(), Value::GpuTensor(handle.clone()));
            let target =
                runmat_gc::gc_allocate(Value::Struct(payload)).expect("gc allocate payload");
            let nested = Value::HandleObject(HandleRef {
                class_name: "Payload".into(),
                target,
                valid: true,
            });
            state.vars = vec![nested.clone()];
            state.stack.push(Value::Num(0.0));
            state.context.locals.push(nested);

            let mut result_vars = state.vars.clone();
            let _ = block_on(run_interpreter_inner(state, &mut result_vars))
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
            drop(seed_vars);
            drop(result_vars);
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
            let mut seed_vars = vec![Value::GpuTensor(handle.clone())];
            let mut state =
                InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
            state.stack.push(Value::Num(0.0));
            state.vars = vec![Value::GpuTensor(handle.clone())];
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

            let mut result_vars = state.vars.clone();
            let _ = block_on(run_interpreter_inner(state, &mut result_vars))
                .expect("store local should complete");
            assert!(
                fusion_residency::is_resident(&handle),
                "store local overwrite should preserve residency for handles still live in vars"
            );
            assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_ok(),
            "store local overwrite should not release provider storage for handles still live in vars"
        );
            let weak = handle.downgrade();
            drop(seed_vars);
            drop(result_vars);
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
            let mut seed_vars = vec![Value::Num(0.0)];
            let mut state =
                InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
            state.stack.push(Value::Num(0.0));
            state.vars = vec![Value::Num(0.0)];
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

            let mut result_vars = state.vars.clone();
            let _ = block_on(run_interpreter_inner(state, &mut result_vars))
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
            drop(seed_vars);
            drop(result_vars);
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
            let mut seed_vars = vec![Value::Num(0.0)];
            let mut state =
                InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
            state.stack.push(Value::Num(0.0));
            state.vars = vec![Value::Num(0.0)];
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

            let mut result_vars = state.vars.clone();
            let weak = handle.downgrade();
            drop(handle);
            let _ = block_on(run_interpreter_inner(state, &mut result_vars))
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
            let mut seed_vars = vec![Value::Num(0.0)];
            let mut state =
                InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
            let mut payload = StructValue::new();
            payload
                .fields
                .insert("nested".to_string(), Value::GpuTensor(handle.clone()));
            let target =
                runmat_gc::gc_allocate(Value::Struct(payload)).expect("gc allocate payload");
            state.stack.push(Value::Num(0.0));
            state.vars = vec![Value::Num(0.0)];
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

            let mut result_vars = state.vars.clone();
            let weak = handle.downgrade();
            drop(handle);
            let _ = block_on(run_interpreter_inner(state, &mut result_vars))
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
            let mut seed_vars = vec![Value::Num(0.0)];
            let mut state =
                InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
            let mut payload = StructValue::new();
            payload
                .fields
                .insert("nested".to_string(), Value::GpuTensor(handle.clone()));
            let target =
                runmat_gc::gc_allocate(Value::Struct(payload)).expect("gc allocate payload");
            let local_value = Value::HandleObject(HandleRef {
                class_name: "Payload".into(),
                target,
                valid: true,
            });
            state.stack.push(Value::Num(0.0));
            state.vars = vec![local_value.clone()];
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

            let mut result_vars = state.vars.clone();
            let _ = block_on(run_interpreter_inner(state, &mut result_vars)).expect(
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
            drop(seed_vars);
            drop(result_vars);
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
            let mut seed_vars = vec![Value::Num(0.0)];
            let mut state =
                InterpreterState::new(bytecode, &mut seed_vars, Some("<main>"), Vec::new());
            let mut payload = StructValue::new();
            payload
                .fields
                .insert("nested".to_string(), Value::GpuTensor(handle.clone()));
            let target =
                runmat_gc::gc_allocate(Value::Struct(payload)).expect("gc allocate payload");
            let nested = Value::HandleObject(HandleRef {
                class_name: "Payload".into(),
                target,
                valid: true,
            });
            state.stack.push(Value::Num(0.0));
            state.vars = vec![Value::Num(0.0)];
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

            let mut result_vars = state.vars.clone();
            let _ = block_on(run_interpreter_inner(state, &mut result_vars))
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
            drop(seed_vars);
            drop(result_vars);
            drop(handle);
            runmat_gc::gc_collect_major().unwrap();
            assert!(weak.acquire().is_err(), "all actual owners were released");
        });
    }
}
