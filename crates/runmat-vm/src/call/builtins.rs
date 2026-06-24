use crate::bytecode::{Bytecode, FunctionRegistry, Instr};
use crate::interpreter::api::InterpreterOutcome;
use crate::interpreter::errors::mex;
use crate::interpreter::runner::interpret_with_vars;
use crate::interpreter::stack::pop_args;
use crate::runtime::workspace::{
    replace_workspace_target_vars_and_state, workspace_assign_target, workspace_target_snapshot,
    WorkspaceTarget, WorkspaceValueSnapshot,
};
use runmat_builtins::Value;
use runmat_hir::{CallableIdentity, FunctionId, QualifiedName, SymbolName};
use runmat_parser::{parse_with_options, CompatMode, ParserOptions};
use runmat_runtime::{build_runtime_error, RuntimeError};
use runmat_thread_local::runmat_thread_local;
use std::collections::{HashMap, HashSet};

#[cfg(feature = "native-accel")]
fn map_prepare_builtin_args_error(err: impl std::fmt::Display) -> RuntimeError {
    mex(
        "AccelerationOperationFailed",
        &format!("prepare builtin args: {err}"),
    )
}

#[derive(Clone, Copy)]
enum VmIntrinsicBuiltin {
    Nargin,
    Nargout,
    Narginchk,
    Nargoutchk,
}

#[derive(Clone, Copy)]
enum VmDynamicWorkspaceBuiltin {
    Eval,
    Evalin,
    Assignin,
    Run,
}

impl VmDynamicWorkspaceBuiltin {
    fn classify(name: &str) -> Option<Self> {
        match name {
            runmat_hir::EVAL_BUILTIN_NAME => Some(Self::Eval),
            runmat_hir::EVALIN_BUILTIN_NAME => Some(Self::Evalin),
            runmat_hir::ASSIGNIN_BUILTIN_NAME => Some(Self::Assignin),
            runmat_hir::RUN_BUILTIN_NAME => Some(Self::Run),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Eval => runmat_hir::EVAL_BUILTIN_NAME,
            Self::Evalin => runmat_hir::EVALIN_BUILTIN_NAME,
            Self::Assignin => runmat_hir::ASSIGNIN_BUILTIN_NAME,
            Self::Run => runmat_hir::RUN_BUILTIN_NAME,
        }
    }
}

#[derive(Clone, Copy)]
struct DynamicEvalOptions {
    compat_mode: CompatMode,
    runmat_extensions_enabled: bool,
    top_level_await_enabled: bool,
    dynamic_eval_enabled: bool,
}

impl Default for DynamicEvalOptions {
    fn default() -> Self {
        Self {
            compat_mode: CompatMode::Matlab,
            runmat_extensions_enabled: true,
            top_level_await_enabled: true,
            dynamic_eval_enabled: true,
        }
    }
}

runmat_thread_local! {
    static DYNAMIC_EVAL_OPTIONS: std::cell::RefCell<DynamicEvalOptions> =
        std::cell::RefCell::new(DynamicEvalOptions::default());
}

pub fn set_dynamic_eval_options(
    compat_mode: CompatMode,
    runmat_extensions_enabled: bool,
    top_level_await_enabled: bool,
    dynamic_eval_enabled: bool,
) {
    DYNAMIC_EVAL_OPTIONS.with(|slot| {
        *slot.borrow_mut() = DynamicEvalOptions {
            compat_mode,
            runmat_extensions_enabled,
            top_level_await_enabled,
            dynamic_eval_enabled,
        };
    });
}

pub struct DynamicEvalOptionsGuard {
    previous: DynamicEvalOptions,
}

impl Drop for DynamicEvalOptionsGuard {
    fn drop(&mut self) {
        let previous = self.previous;
        DYNAMIC_EVAL_OPTIONS.with(|slot| {
            *slot.borrow_mut() = previous;
        });
    }
}

pub fn push_dynamic_eval_options(
    compat_mode: CompatMode,
    runmat_extensions_enabled: bool,
    top_level_await_enabled: bool,
    dynamic_eval_enabled: bool,
) -> DynamicEvalOptionsGuard {
    let previous = current_dynamic_eval_options();
    set_dynamic_eval_options(
        compat_mode,
        runmat_extensions_enabled,
        top_level_await_enabled,
        dynamic_eval_enabled,
    );
    DynamicEvalOptionsGuard { previous }
}

fn current_dynamic_eval_options() -> DynamicEvalOptions {
    DYNAMIC_EVAL_OPTIONS.with(|slot| *slot.borrow())
}

impl VmIntrinsicBuiltin {
    fn classify(name: &str) -> Option<Self> {
        match name {
            runmat_hir::NARGIN_BUILTIN_NAME => Some(Self::Nargin),
            runmat_hir::NARGOUT_BUILTIN_NAME => Some(Self::Nargout),
            runmat_hir::NARGINCHK_BUILTIN_NAME => Some(Self::Narginchk),
            runmat_hir::NARGOUTCHK_BUILTIN_NAME => Some(Self::Nargoutchk),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Nargin => runmat_hir::NARGIN_BUILTIN_NAME,
            Self::Nargout => runmat_hir::NARGOUT_BUILTIN_NAME,
            Self::Narginchk => runmat_hir::NARGINCHK_BUILTIN_NAME,
            Self::Nargoutchk => runmat_hir::NARGOUTCHK_BUILTIN_NAME,
        }
    }
}

pub fn is_vm_intrinsic_builtin(name: &str) -> bool {
    VmIntrinsicBuiltin::classify(name).is_some()
}

pub fn is_vm_dynamic_workspace_builtin(name: &str) -> bool {
    VmDynamicWorkspaceBuiltin::classify(name).is_some()
}

#[derive(Clone, Copy)]
enum VmIntrinsicExceptionBuiltin {
    Rethrow,
}

impl VmIntrinsicExceptionBuiltin {
    fn classify(name: &str) -> Option<Self> {
        match name {
            "rethrow" => Some(Self::Rethrow),
            _ => None,
        }
    }
}

#[cfg(feature = "native-accel")]
pub async fn prepare_builtin_args(name: &str, args: &[Value]) -> Result<Vec<Value>, RuntimeError> {
    runmat_accelerate::prepare_builtin_args(name, args)
        .await
        .map_err(map_prepare_builtin_args_error)
}

#[cfg(not(feature = "native-accel"))]
pub async fn prepare_builtin_args(_name: &str, args: &[Value]) -> Result<Vec<Value>, RuntimeError> {
    Ok(args.to_vec())
}

pub fn collect_call_args(
    stack: &mut Vec<Value>,
    arg_count: usize,
) -> Result<Vec<Value>, RuntimeError> {
    pop_args(stack, arg_count)
}

#[derive(Clone, Copy)]
enum ArityBound {
    Finite(usize),
    Unbounded,
}

impl ArityBound {
    fn permits(self, actual: usize) -> bool {
        match self {
            Self::Finite(max) => actual <= max,
            Self::Unbounded => true,
        }
    }
}

fn parse_finite_arity_bound(
    value: &Value,
    builtin: &str,
    name: &str,
) -> Result<usize, RuntimeError> {
    let number = match value {
        Value::Num(value) => *value,
        Value::Int(value) => value.to_f64(),
        Value::Tensor(tensor) if tensor.data.len() == 1 => tensor.data[0],
        other => {
            return Err(mex(
                &format!("{builtin}ArgumentInvalid"),
                &format!("{builtin}: {name} must be a nonnegative integer scalar, got {other:?}"),
            ));
        }
    };

    if !number.is_finite() || number < 0.0 || number.fract() != 0.0 {
        return Err(mex(
            &format!("{builtin}ArgumentInvalid"),
            &format!("{builtin}: {name} must be a nonnegative integer scalar"),
        ));
    }
    if number > usize::MAX as f64 {
        return Err(mex(
            &format!("{builtin}ArgumentInvalid"),
            &format!("{builtin}: {name} exceeds the platform argument-count range"),
        ));
    }
    Ok(number as usize)
}

fn parse_max_arity_bound(value: &Value, builtin: &str) -> Result<ArityBound, RuntimeError> {
    match value {
        Value::Num(value) if value.is_infinite() && value.is_sign_positive() => {
            Ok(ArityBound::Unbounded)
        }
        Value::Tensor(tensor)
            if tensor.data.len() == 1
                && tensor.data[0].is_infinite()
                && tensor.data[0].is_sign_positive() =>
        {
            Ok(ArityBound::Unbounded)
        }
        _ => parse_finite_arity_bound(value, builtin, "maxArgs").map(ArityBound::Finite),
    }
}

fn validate_arity_bounds(
    builtin: &str,
    min: &Value,
    max: &Value,
) -> Result<(usize, ArityBound), RuntimeError> {
    let min = parse_finite_arity_bound(min, builtin, "minArgs")?;
    let max = parse_max_arity_bound(max, builtin)?;
    if let ArityBound::Finite(max_value) = max {
        if min > max_value {
            return Err(mex(
                &format!("{builtin}BoundsInvalid"),
                &format!("{builtin}: minArgs must be less than or equal to maxArgs"),
            ));
        }
    }
    Ok((min, max))
}

fn validate_narginchk(args: &[Value], actual: usize) -> Result<(), RuntimeError> {
    let builtin = runmat_hir::NARGINCHK_BUILTIN_NAME;
    let (min, max) = validate_arity_bounds("Narginchk", &args[0], &args[1])?;
    if actual < min {
        return Err(mex(
            "NotEnoughInputs",
            &format!("{builtin}: expected at least {min} input arguments, got {actual}"),
        ));
    }
    if !max.permits(actual) {
        return Err(mex(
            "TooManyInputs",
            &format!("{builtin}: input argument count {actual} exceeds maxArgs"),
        ));
    }
    Ok(())
}

fn validate_nargoutchk(args: &[Value], actual: usize) -> Result<(), RuntimeError> {
    let builtin = runmat_hir::NARGOUTCHK_BUILTIN_NAME;
    let (min, max) = validate_arity_bounds("Nargoutchk", &args[0], &args[1])?;
    if actual < min {
        return Err(mex(
            "NotEnoughOutputs",
            &format!("{builtin}: expected at least {min} output arguments, got {actual}"),
        ));
    }
    if !max.permits(actual) {
        return Err(mex(
            "TooManyOutputs",
            &format!("{builtin}: output argument count {actual} exceeds maxArgs"),
        ));
    }
    Ok(())
}

fn validate_intrinsic_arg_count(
    builtin: &str,
    actual: usize,
    expected: usize,
) -> Result<(), RuntimeError> {
    if actual < expected {
        return Err(mex(
            "NotEnoughInputs",
            &format!("{builtin} takes {expected} arguments"),
        ));
    }
    if actual > expected {
        return Err(mex(
            "TooManyInputs",
            &format!("{builtin} takes {expected} arguments"),
        ));
    }
    Ok(())
}

pub fn vm_intrinsic_builtin(
    stack: &mut Vec<Value>,
    name: &str,
    arg_count: usize,
    call_counts: &[(usize, usize)],
) -> Result<Value, RuntimeError> {
    let Some(intrinsic) = VmIntrinsicBuiltin::classify(name) else {
        return Err(mex(
            "UndefinedFunction",
            &format!("unknown VM intrinsic builtin '{name}'"),
        ));
    };
    let args = collect_call_args(stack, arg_count)?;
    match intrinsic {
        VmIntrinsicBuiltin::Nargin => {
            validate_intrinsic_arg_count(intrinsic.name(), args.len(), 0)?;
            let (nin, _) = call_counts.last().cloned().unwrap_or((0, 0));
            Ok(Value::Num(nin as f64))
        }
        VmIntrinsicBuiltin::Nargout => {
            validate_intrinsic_arg_count(intrinsic.name(), args.len(), 0)?;
            let (_, nout) = call_counts.last().cloned().unwrap_or((0, 0));
            Ok(Value::Num(nout as f64))
        }
        VmIntrinsicBuiltin::Narginchk => {
            validate_intrinsic_arg_count(intrinsic.name(), args.len(), 2)?;
            let (nin, _) = call_counts.last().cloned().unwrap_or((0, 0));
            validate_narginchk(&args, nin)?;
            Ok(Value::Num(0.0))
        }
        VmIntrinsicBuiltin::Nargoutchk => {
            validate_intrinsic_arg_count(intrinsic.name(), args.len(), 2)?;
            let (_, nout) = call_counts.last().cloned().unwrap_or((0, 0));
            validate_nargoutchk(&args, nout)?;
            Ok(Value::Num(0.0))
        }
    }
}

pub async fn vm_dynamic_workspace_builtin(
    stack: &mut Vec<Value>,
    name: &str,
    arg_count: usize,
    requested_outputs: usize,
    function_registry: &FunctionRegistry,
    source_id: Option<runmat_hir::SourceId>,
) -> Result<Value, RuntimeError> {
    let Some(builtin) = VmDynamicWorkspaceBuiltin::classify(name) else {
        return Err(mex(
            "UndefinedFunction",
            &format!("unknown VM dynamic workspace builtin '{name}'"),
        ));
    };
    let args = collect_call_args(stack, arg_count)?;
    match builtin {
        VmDynamicWorkspaceBuiltin::Eval => {
            validate_intrinsic_arg_count(builtin.name(), args.len(), 1)?;
            let source = workspace_text_arg(builtin.name(), &args[0])?;
            eval_workspace_source(
                WorkspaceEvalRequest {
                    builtin: builtin.name(),
                    target: WorkspaceTarget::Current,
                    source,
                    source_label: "<eval>".to_string(),
                    requested_outputs,
                    source_id,
                    source_context: None,
                    commit_workspace_on_error: false,
                },
                function_registry,
            )
            .await
        }
        VmDynamicWorkspaceBuiltin::Evalin => {
            validate_intrinsic_arg_count(builtin.name(), args.len(), 2)?;
            let target = workspace_target_arg(builtin.name(), &args[0])?;
            let source = workspace_text_arg(builtin.name(), &args[1])?;
            eval_workspace_source(
                WorkspaceEvalRequest {
                    builtin: builtin.name(),
                    target,
                    source,
                    source_label: "<eval>".to_string(),
                    requested_outputs,
                    source_id,
                    source_context: None,
                    commit_workspace_on_error: false,
                },
                function_registry,
            )
            .await
        }
        VmDynamicWorkspaceBuiltin::Assignin => {
            validate_intrinsic_arg_count(builtin.name(), args.len(), 3)?;
            let target = workspace_target_arg(builtin.name(), &args[0])?;
            let name = workspace_text_arg(builtin.name(), &args[1])?;
            if !is_valid_workspace_identifier(&name) {
                return Err(mex(
                    "DynamicWorkspaceName",
                    &format!(
                        "{}: invalid workspace variable name '{name}'",
                        builtin.name()
                    ),
                ));
            }
            workspace_assign_target(target, &name, args[2].clone()).map_err(|err| {
                mex(
                    "DynamicWorkspaceUnavailable",
                    &format!("{}: {err}", builtin.name()),
                )
            })?;
            Ok(Value::Num(0.0))
        }
        VmDynamicWorkspaceBuiltin::Run => {
            validate_intrinsic_arg_count(builtin.name(), args.len(), 1)?;
            if requested_outputs > 0 {
                return Err(runmat_runtime::builtins::io::repl_fs::run::too_many_outputs_error());
            }
            let source =
                runmat_runtime::builtins::io::repl_fs::run::resolve_run_source(&args[0]).await?;
            let dynamic_source_id = next_dynamic_source_id();
            let source_context = DynamicSourceContext {
                source_id: dynamic_source_id,
                name: source.display_name.clone(),
                text: source.text.clone(),
            };
            eval_workspace_source(
                WorkspaceEvalRequest {
                    builtin: builtin.name(),
                    target: WorkspaceTarget::Current,
                    source: source.text,
                    source_label: source.display_name,
                    requested_outputs: 0,
                    source_id: Some(source_context.source_id),
                    source_context: Some(source_context),
                    commit_workspace_on_error: true,
                },
                function_registry,
            )
            .await
        }
    }
}

fn workspace_text_arg(builtin: &str, value: &Value) -> Result<String, RuntimeError> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::CharArray(chars) if chars.rows == 1 => Ok(chars.data.iter().collect()),
        Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
        other => Err(mex(
            "DynamicWorkspaceArgumentType",
            &format!("{builtin}: expected a string scalar or character row, got {other:?}"),
        )),
    }
}

fn workspace_target_arg(builtin: &str, value: &Value) -> Result<WorkspaceTarget, RuntimeError> {
    let selector = workspace_text_arg(builtin, value)?.to_ascii_lowercase();
    match selector.as_str() {
        "base" => Ok(WorkspaceTarget::Base),
        "caller" => Ok(WorkspaceTarget::Caller),
        _ => Err(mex(
            "DynamicWorkspaceSelector",
            &format!("{builtin}: workspace selector must be 'base' or 'caller'"),
        )),
    }
}

fn is_valid_workspace_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    (first == '_' || first.is_ascii_alphabetic())
        && chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
}

struct WorkspaceEvalRequest {
    builtin: &'static str,
    target: WorkspaceTarget,
    source: String,
    source_label: String,
    requested_outputs: usize,
    source_id: Option<runmat_hir::SourceId>,
    source_context: Option<DynamicSourceContext>,
    commit_workspace_on_error: bool,
}

async fn eval_workspace_source(
    request: WorkspaceEvalRequest,
    function_registry: &FunctionRegistry,
) -> Result<Value, RuntimeError> {
    let WorkspaceEvalRequest {
        builtin,
        target,
        source,
        source_label,
        requested_outputs,
        source_id,
        source_context,
        commit_workspace_on_error,
    } = request;
    if !current_dynamic_eval_options().dynamic_eval_enabled {
        return Err(mex(
            "DynamicEvalDisabled",
            &format!("{builtin}: dynamic eval is disabled by host execution policy"),
        ));
    }
    let frame = workspace_target_snapshot(target)
        .map_err(|err| mex("DynamicWorkspaceUnavailable", &format!("{builtin}: {err}")))?;
    let _source_catalog_guard = source_context.as_ref().map(install_dynamic_source_context);
    let Some((bytecode, result_slot)) = compile_dynamic_workspace_eval(
        builtin,
        &source,
        &frame.names,
        function_registry,
        source_id,
        requested_outputs,
    )?
    else {
        return Ok(Value::OutputList(Vec::new()));
    };
    let target_vars = unsafe { (*frame.vars_ptr).clone() };
    let (outcome, updated_workspace, effects) = interpret_dynamic_workspace_eval(
        bytecode,
        target_vars,
        frame.names.clone(),
        frame.assigned,
        builtin,
        source_label,
    )
    .await?;
    let vars = match outcome {
        Ok(crate::interpreter::api::InterpreterOutcome::Completed(vars)) => {
            if let Some(snapshot) = updated_workspace {
                let result_vars = snapshot.vars.clone();
                replace_workspace_target_vars_and_state(
                    target,
                    snapshot.vars,
                    snapshot.names,
                    snapshot.assigned,
                )
                .map_err(|err| mex("DynamicWorkspaceUnavailable", &format!("{builtin}: {err}")))?;
                result_vars
            } else {
                vars
            }
        }
        Err(err) => {
            if commit_workspace_on_error {
                if let Some(snapshot) = updated_workspace {
                    replace_workspace_target_vars_and_state(
                        target,
                        snapshot.vars,
                        snapshot.names,
                        snapshot.assigned,
                    )
                    .map_err(|replace_err| {
                        mex(
                            "DynamicWorkspaceUnavailable",
                            &format!("{builtin}: {replace_err}"),
                        )
                    })?;
                }
            }
            effects.replay();
            return Err(err);
        }
    };
    effects.replay();
    let result = if requested_outputs == 0 {
        Value::OutputList(Vec::new())
    } else if let Some(slot) = result_slot {
        vars.get(slot).cloned().unwrap_or(Value::Num(0.0))
    } else {
        Value::Num(0.0)
    };
    Ok(result)
}

#[derive(Clone)]
struct DynamicSourceContext {
    source_id: runmat_hir::SourceId,
    name: String,
    text: String,
}

fn install_dynamic_source_context(
    context: &DynamicSourceContext,
) -> runmat_runtime::source_context::SourceCatalogGuard {
    let mut entries = runmat_runtime::source_context::source_catalog_entries_with_fullpaths();
    entries.retain(|(source_id, _, _, _)| *source_id != context.source_id);
    entries.push((
        context.source_id,
        context.name.clone(),
        None,
        context.text.clone(),
    ));
    runmat_runtime::source_context::replace_source_catalog_with_fullpaths(entries)
}

fn next_dynamic_source_id() -> runmat_hir::SourceId {
    runmat_runtime::source_context::source_catalog_entries_with_fullpaths()
        .into_iter()
        .map(|(source_id, _, _, _)| source_id.0)
        .max()
        .map(|id| runmat_hir::SourceId(id + 1))
        .unwrap_or(runmat_hir::SourceId(0))
}

#[derive(Default)]
struct DynamicThreadEffects {
    console: Vec<runmat_runtime::console::ConsoleEntry>,
    warnings: Vec<runmat_runtime::warning_store::RuntimeWarning>,
    recent_figures: Vec<u32>,
}

impl DynamicThreadEffects {
    fn replay(self) {
        runmat_runtime::console::append_thread_buffer(self.console);
        runmat_runtime::warning_store::extend(self.warnings);
        runmat_runtime::plotting_hooks::record_recent_figures(self.recent_figures);
    }
}

async fn interpret_dynamic_workspace_eval(
    bytecode: Bytecode,
    mut target_vars: Vec<Value>,
    names: HashMap<String, usize>,
    assigned: HashSet<String>,
    _builtin: &'static str,
    source_label: String,
) -> Result<
    (
        Result<InterpreterOutcome, RuntimeError>,
        Option<WorkspaceValueSnapshot>,
        DynamicThreadEffects,
    ),
    RuntimeError,
> {
    let _pending_workspace = crate::runtime::workspace::push_pending_workspace(names, assigned);
    let outcome = interpret_with_vars(&bytecode, &mut target_vars, Some(&source_label)).await;
    let updated_workspace = crate::runtime::workspace::take_updated_workspace_state();
    let _ = crate::runtime::workspace::take_updated_workspace_assigned_report();
    Ok((outcome, updated_workspace, DynamicThreadEffects::default()))
}

fn compile_dynamic_workspace_eval(
    builtin: &str,
    source: &str,
    workspace_bindings: &HashMap<String, usize>,
    function_registry: &FunctionRegistry,
    source_id: Option<runmat_hir::SourceId>,
    requested_outputs: usize,
) -> Result<Option<(Bytecode, Option<usize>)>, RuntimeError> {
    let options = current_dynamic_eval_options();
    let ast =
        parse_with_options(source, ParserOptions::new(options.compat_mode)).map_err(|err| {
            mex(
                "DynamicWorkspaceParse",
                &format!("{builtin}: parse error: {err}"),
            )
        })?;
    let function_output_arities = function_registry
        .functions
        .iter()
        .map(|(id, function)| {
            (
                *id,
                runmat_hir::FunctionOutputArity::new(
                    function.output_slots.len(),
                    function.varargout_slot.is_some(),
                ),
            )
        })
        .collect::<HashMap<_, _>>();
    let lowering = runmat_hir::lower(
        &ast,
        &runmat_hir::LoweringContext::new(workspace_bindings)
            .with_bound_functions(&function_registry.names)
            .with_function_output_arities(&function_output_arities)
            .with_runmat_extensions_enabled(options.runmat_extensions_enabled)
            .with_top_level_await_enabled(options.top_level_await_enabled),
    )
    .map_err(|err| {
        mex(
            "DynamicWorkspaceLower",
            &format!("{builtin}: lowering error: {err}"),
        )
    })?;
    let Some(entrypoint) = lowering.assembly.entrypoints.first() else {
        return Ok(None);
    };
    let mir = runmat_mir::lowering::lower_assembly(&lowering.assembly).map_err(|err| {
        mex(
            "DynamicWorkspaceLower",
            &format!("{builtin}: MIR lowering error: {err}"),
        )
    })?;
    let mut bytecode =
        crate::bytecode::compile(&lowering.assembly, &mir, entrypoint.id).map_err(|err| {
            mex(
                "DynamicWorkspaceCompile",
                &format!("{builtin}: compile error: {err}"),
            )
        })?;
    if source_id.is_some() {
        bytecode.source_id = source_id;
    }
    merge_eval_function_registry(&mut bytecode, function_registry);
    remap_dynamic_workspace_slots(&mut bytecode, workspace_bindings);
    let result_slot =
        if requested_outputs > 0 && matches!(bytecode.instructions.last(), Some(Instr::Pop)) {
            bytecode.instructions.pop();
            let next_workspace_slot = workspace_bindings
                .values()
                .copied()
                .max()
                .map(|slot| slot + 1)
                .unwrap_or(0);
            let temp_slot = bytecode.var_count.max(next_workspace_slot);
            bytecode.instructions.push(Instr::StoreVar(temp_slot));
            bytecode.var_count = temp_slot + 1;
            Some(temp_slot)
        } else {
            None
        };
    Ok(Some((bytecode, result_slot)))
}

fn remap_dynamic_workspace_slots(
    bytecode: &mut Bytecode,
    workspace_bindings: &HashMap<String, usize>,
) {
    if bytecode.var_count == 0 {
        return;
    }

    let original_var_count = bytecode.var_count;
    let original_var_names = bytecode.var_names.clone();
    let original_initially_unassigned_slots = bytecode.initially_unassigned_slots.clone();
    let mut used_slots = workspace_bindings.values().copied().collect::<HashSet<_>>();
    let mut next_slot = used_slots
        .iter()
        .copied()
        .max()
        .map(|slot| slot + 1)
        .unwrap_or(0);
    let mut slot_remap = HashMap::new();

    for old_slot in 0..original_var_count {
        let new_slot = original_var_names
            .get(&old_slot)
            .and_then(|name| workspace_bindings.get(name).copied())
            .unwrap_or_else(|| allocate_dynamic_workspace_slot(&mut used_slots, &mut next_slot));
        used_slots.insert(new_slot);
        slot_remap.insert(old_slot, new_slot);
    }

    for instr in &mut bytecode.instructions {
        remap_instr_slots(instr, &slot_remap);
    }

    let mut remapped_var_names = HashMap::new();
    for (old_slot, name) in original_var_names {
        if let Some(new_slot) = slot_remap.get(&old_slot).copied() {
            remapped_var_names.insert(new_slot, name);
        }
    }
    bytecode.var_names = remapped_var_names;
    bytecode.initially_unassigned_slots = original_initially_unassigned_slots
        .into_iter()
        .filter_map(|old_slot| slot_remap.get(&old_slot).copied())
        .collect();

    let new_var_count = slot_remap
        .values()
        .copied()
        .max()
        .map(|slot| slot + 1)
        .unwrap_or(original_var_count);
    let old_var_types = std::mem::take(&mut bytecode.var_types);
    let mut new_var_types = vec![runmat_builtins::Type::Unknown; new_var_count];
    for (old_slot, new_slot) in slot_remap {
        if let (Some(old_ty), Some(new_ty)) =
            (old_var_types.get(old_slot), new_var_types.get_mut(new_slot))
        {
            *new_ty = old_ty.clone();
        }
    }
    bytecode.var_types = new_var_types;
    bytecode.var_count = new_var_count;
}

fn allocate_dynamic_workspace_slot(
    used_slots: &mut HashSet<usize>,
    next_slot: &mut usize,
) -> usize {
    while used_slots.contains(next_slot) {
        *next_slot += 1;
    }
    let slot = *next_slot;
    used_slots.insert(slot);
    *next_slot += 1;
    slot
}

fn remap_slot(slot: &mut usize, slot_remap: &HashMap<usize, usize>) {
    if let Some(new_slot) = slot_remap.get(slot).copied() {
        *slot = new_slot;
    }
}

fn remap_emit_label(label: &mut crate::bytecode::EmitLabel, slot_remap: &HashMap<usize, usize>) {
    if let crate::bytecode::EmitLabel::Var(slot) = label {
        remap_slot(slot, slot_remap);
    }
}

fn remap_end_expr_slots(expr: &mut crate::bytecode::EndExpr, slot_remap: &HashMap<usize, usize>) {
    match expr {
        crate::bytecode::EndExpr::Var(slot) => remap_slot(slot, slot_remap),
        crate::bytecode::EndExpr::ResolvedCall { args, .. } => {
            for arg in args {
                remap_end_expr_slots(arg, slot_remap);
            }
        }
        crate::bytecode::EndExpr::Add(left, right)
        | crate::bytecode::EndExpr::Sub(left, right)
        | crate::bytecode::EndExpr::Mul(left, right)
        | crate::bytecode::EndExpr::Div(left, right)
        | crate::bytecode::EndExpr::LeftDiv(left, right)
        | crate::bytecode::EndExpr::Pow(left, right) => {
            remap_end_expr_slots(left, slot_remap);
            remap_end_expr_slots(right, slot_remap);
        }
        crate::bytecode::EndExpr::Neg(inner)
        | crate::bytecode::EndExpr::Pos(inner)
        | crate::bytecode::EndExpr::Floor(inner)
        | crate::bytecode::EndExpr::Ceil(inner)
        | crate::bytecode::EndExpr::Round(inner)
        | crate::bytecode::EndExpr::Fix(inner) => remap_end_expr_slots(inner, slot_remap),
        crate::bytecode::EndExpr::End | crate::bytecode::EndExpr::Const(_) => {}
    }
}

fn remap_optional_end_expr_slots(
    exprs: &mut [Option<crate::bytecode::EndExpr>],
    slot_remap: &HashMap<usize, usize>,
) {
    for expr in exprs.iter_mut().flatten() {
        remap_end_expr_slots(expr, slot_remap);
    }
}

fn remap_indexed_end_expr_slots(
    exprs: &mut [(usize, crate::bytecode::EndExpr)],
    slot_remap: &HashMap<usize, usize>,
) {
    for (_, expr) in exprs {
        remap_end_expr_slots(expr, slot_remap);
    }
}

fn remap_end_expr_vec_slots(
    exprs: &mut [crate::bytecode::EndExpr],
    slot_remap: &HashMap<usize, usize>,
) {
    for expr in exprs {
        remap_end_expr_slots(expr, slot_remap);
    }
}

fn remap_instr_slots(instr: &mut Instr, slot_remap: &HashMap<usize, usize>) {
    match instr {
        Instr::LoadVar(slot) | Instr::LoadVarForIndexAssignment(slot) | Instr::StoreVar(slot) => {
            remap_slot(slot, slot_remap);
        }
        Instr::IndexSliceExpr {
            range_start_exprs,
            range_step_exprs,
            range_end_exprs,
            end_numeric_exprs,
            ..
        }
        | Instr::StoreSliceExpr {
            range_start_exprs,
            range_step_exprs,
            range_end_exprs,
            end_numeric_exprs,
            ..
        }
        | Instr::StoreSliceExprDelete {
            range_start_exprs,
            range_step_exprs,
            range_end_exprs,
            end_numeric_exprs,
            ..
        } => {
            remap_optional_end_expr_slots(range_start_exprs, slot_remap);
            remap_optional_end_expr_slots(range_step_exprs, slot_remap);
            remap_end_expr_vec_slots(range_end_exprs, slot_remap);
            remap_indexed_end_expr_slots(end_numeric_exprs, slot_remap);
        }
        Instr::IndexCell { end_exprs, .. }
        | Instr::IndexCellExpand { end_exprs, .. }
        | Instr::IndexCellList { end_exprs, .. }
        | Instr::StoreIndexCell { end_exprs, .. }
        | Instr::StoreIndexCellDelete { end_exprs, .. } => {
            remap_indexed_end_expr_slots(end_exprs, slot_remap);
        }
        Instr::CallFevalMultiUsingOutputSlot(_, slot)
        | Instr::CallFevalExpandMultiOutputUsingOutputSlot(_, slot)
        | Instr::CallBuiltinMultiUsingOutputSlot(_, _, slot)
        | Instr::CallSemanticFunctionMultiUsingOutputSlot(_, _, slot) => {
            remap_slot(slot, slot_remap);
        }
        Instr::CallFunctionMultiUsingOutputSlot { out_count_slot, .. } => {
            remap_slot(out_count_slot, slot_remap);
        }
        Instr::CallWorkspaceFirstMultiUsingOutputSlot { out_count_slot, .. }
        | Instr::CallWorkspaceFirstExpandMultiOutputUsingOutputSlot { out_count_slot, .. } => {
            remap_slot(out_count_slot, slot_remap);
        }
        Instr::CallSemanticNestedFunctionMultiUsingOutputSlot {
            capture_slots,
            out_count_slot,
            ..
        } => {
            for slot in capture_slots {
                remap_slot(slot, slot_remap);
            }
            remap_slot(out_count_slot, slot_remap);
        }
        Instr::CallSemanticNestedFunctionMulti { capture_slots, .. }
        | Instr::CallSemanticNestedFunctionExpandMultiOutput { capture_slots, .. }
        | Instr::DeclareGlobal(capture_slots)
        | Instr::DeclarePersistent(capture_slots)
        | Instr::DeclareGlobalNamed(capture_slots, _)
        | Instr::DeclarePersistentNamed(capture_slots, _) => {
            for slot in capture_slots {
                remap_slot(slot, slot_remap);
            }
        }
        Instr::EmitStackTop { label } => remap_emit_label(label, slot_remap),
        Instr::EmitVar { var_index, label } => {
            remap_slot(var_index, slot_remap);
            remap_emit_label(label, slot_remap);
        }
        _ => {}
    }
}

fn merge_eval_function_registry(bytecode: &mut Bytecode, parent_registry: &FunctionRegistry) {
    let eval_registry = bytecode.function_registry.clone();
    let mut merged = parent_registry.clone();
    let mut ids = eval_registry.functions.keys().copied().collect::<Vec<_>>();
    ids.sort_by_key(|id| id.0);
    let remap = eval_function_id_remap(&ids, parent_registry);
    for instr in &mut bytecode.instructions {
        remap_eval_local_function_instr(instr, &remap);
    }
    for id in ids {
        if let Some(function) = eval_registry.functions.get(&id) {
            let mut function = function.clone();
            if let Some(new_id) = remap.get(&id).copied() {
                function.function = new_id;
                for instr in &mut function.instructions {
                    remap_eval_local_function_instr(instr, &remap);
                }
            }
            merged.insert_replacing_name(function);
        }
    }
    bytecode.bound_functions = merged.functions.clone();
    bytecode.function_registry = merged;
}

fn eval_function_id_remap(
    eval_ids: &[FunctionId],
    parent_registry: &FunctionRegistry,
) -> HashMap<FunctionId, FunctionId> {
    let mut used_ids = parent_registry
        .functions
        .keys()
        .map(|id| id.0)
        .collect::<HashSet<_>>();
    let mut next_id = used_ids.iter().copied().max().map(|id| id + 1).unwrap_or(0);
    let mut remap = HashMap::new();
    for old_id in eval_ids {
        while used_ids.contains(&next_id) {
            next_id += 1;
        }
        let new_id = FunctionId(next_id);
        used_ids.insert(next_id);
        next_id += 1;
        remap.insert(*old_id, new_id);
    }
    remap
}

fn remap_eval_local_function_instr(instr: &mut Instr, remap: &HashMap<FunctionId, FunctionId>) {
    match instr {
        Instr::CreateSemanticClosure(function, _, _)
        | Instr::CreateBoundFunctionHandle(function, _)
        | Instr::CreateSemanticFuture(function, _, _)
        | Instr::CreateSemanticFutureExpandMultiOutput(function, _, _)
        | Instr::CallSemanticFunctionMulti(function, _, _)
        | Instr::CallSemanticFunctionMultiUsingOutputSlot(function, _, _)
        | Instr::CallSemanticFunctionExpandMultiOutput(function, _, _) => {
            remap_function_id(function, remap);
        }
        Instr::CallSemanticNestedFunctionMulti { function, .. }
        | Instr::CallSemanticNestedFunctionMultiUsingOutputSlot { function, .. }
        | Instr::CallSemanticNestedFunctionExpandMultiOutput { function, .. } => {
            remap_function_id(function, remap);
        }
        Instr::CallFunctionMulti { identity, .. }
        | Instr::CallFunctionMultiUsingOutputSlot { identity, .. }
        | Instr::CallFunctionExpandMultiOutput { identity, .. } => {
            remap_eval_local_callable_identity(identity, remap);
        }
        Instr::IndexSliceExpr {
            range_start_exprs,
            range_step_exprs,
            range_end_exprs,
            end_numeric_exprs,
            ..
        }
        | Instr::StoreSliceExpr {
            range_start_exprs,
            range_step_exprs,
            range_end_exprs,
            end_numeric_exprs,
            ..
        }
        | Instr::StoreSliceExprDelete {
            range_start_exprs,
            range_step_exprs,
            range_end_exprs,
            end_numeric_exprs,
            ..
        } => {
            for expr in range_start_exprs.iter_mut().flatten() {
                remap_eval_local_end_expr(expr, remap);
            }
            for expr in range_step_exprs.iter_mut().flatten() {
                remap_eval_local_end_expr(expr, remap);
            }
            for expr in range_end_exprs {
                remap_eval_local_end_expr(expr, remap);
            }
            for (_, expr) in end_numeric_exprs {
                remap_eval_local_end_expr(expr, remap);
            }
        }
        Instr::IndexCell { end_exprs, .. }
        | Instr::IndexCellExpand { end_exprs, .. }
        | Instr::IndexCellList { end_exprs, .. }
        | Instr::StoreIndexCell { end_exprs, .. }
        | Instr::StoreIndexCellDelete { end_exprs, .. } => {
            for (_, expr) in end_exprs {
                remap_eval_local_end_expr(expr, remap);
            }
        }
        _ => {}
    }
}

fn remap_function_id(function: &mut FunctionId, remap: &HashMap<FunctionId, FunctionId>) {
    if let Some(new_id) = remap.get(function).copied() {
        *function = new_id;
    }
}

fn remap_eval_local_callable_identity(
    identity: &mut CallableIdentity,
    remap: &HashMap<FunctionId, FunctionId>,
) {
    match identity {
        CallableIdentity::BoundFunction(function)
        | CallableIdentity::AnonymousFunction(function) => remap_function_id(function, remap),
        CallableIdentity::ExternalFunction { .. }
        | CallableIdentity::Builtin(_)
        | CallableIdentity::Imported(_)
        | CallableIdentity::Method(_)
        | CallableIdentity::DynamicName(_)
        | CallableIdentity::ExternalName(_) => {}
    }
}

fn remap_eval_local_end_expr(
    expr: &mut crate::bytecode::EndExpr,
    remap: &HashMap<FunctionId, FunctionId>,
) {
    match expr {
        crate::bytecode::EndExpr::ResolvedCall { identity, args, .. } => {
            remap_eval_local_callable_identity(identity, remap);
            for arg in args {
                remap_eval_local_end_expr(arg, remap);
            }
        }
        crate::bytecode::EndExpr::Add(lhs, rhs)
        | crate::bytecode::EndExpr::Sub(lhs, rhs)
        | crate::bytecode::EndExpr::Mul(lhs, rhs)
        | crate::bytecode::EndExpr::Div(lhs, rhs)
        | crate::bytecode::EndExpr::LeftDiv(lhs, rhs)
        | crate::bytecode::EndExpr::Pow(lhs, rhs) => {
            remap_eval_local_end_expr(lhs, remap);
            remap_eval_local_end_expr(rhs, remap);
        }
        crate::bytecode::EndExpr::Neg(inner)
        | crate::bytecode::EndExpr::Pos(inner)
        | crate::bytecode::EndExpr::Floor(inner)
        | crate::bytecode::EndExpr::Ceil(inner)
        | crate::bytecode::EndExpr::Round(inner)
        | crate::bytecode::EndExpr::Fix(inner) => remap_eval_local_end_expr(inner, remap),
        crate::bytecode::EndExpr::End
        | crate::bytecode::EndExpr::Const(_)
        | crate::bytecode::EndExpr::Var(_) => {}
    }
}

pub enum ImportedBuiltinResolution {
    Resolved(Value),
    Ambiguous(RuntimeError),
    NotFound,
}

fn imported_builtin_qualified_name(path: &[String], leaf: Option<&str>) -> Option<String> {
    if path.iter().any(|segment| segment.is_empty()) {
        return None;
    }
    let mut segments: Vec<SymbolName> = path
        .iter()
        .map(|segment| SymbolName(segment.clone()))
        .collect();
    if let Some(leaf) = leaf {
        segments.push(SymbolName(leaf.to_string()));
    }
    QualifiedName(segments).display_name()
}

pub async fn resolve_imported_builtin(
    name: &str,
    imports: &[(Vec<String>, bool)],
    prepared_primary: &[Value],
    requested_outputs: usize,
) -> Result<ImportedBuiltinResolution, RuntimeError> {
    let mut specific_matches: Vec<(String, Value)> = Vec::new();
    for (path, wildcard) in imports {
        if *wildcard {
            continue;
        }
        if path.last().map(|s| s.as_str()) == Some(name) {
            let Some(qual) = imported_builtin_qualified_name(path, None) else {
                continue;
            };
            let qual_args = prepare_builtin_args(&qual, prepared_primary).await?;
            let result = runmat_runtime::call_builtin_async_with_outputs(
                &qual,
                &qual_args,
                requested_outputs,
            )
            .await;
            if let Ok(value) = result {
                specific_matches.push((qual, value));
            }
        }
    }
    if specific_matches.len() > 1 {
        let msg = specific_matches
            .iter()
            .map(|(q, _)| q.clone())
            .collect::<Vec<_>>()
            .join(", ");
        return Ok(ImportedBuiltinResolution::Ambiguous(mex(
            "AmbiguousBuiltinImport",
            &format!("ambiguous builtin '{}' via imports: {}", name, msg),
        )));
    }
    if let Some((_, value)) = specific_matches.pop() {
        return Ok(ImportedBuiltinResolution::Resolved(value));
    }

    let mut wildcard_matches: Vec<(String, Value)> = Vec::new();
    for (path, wildcard) in imports {
        if !*wildcard || path.is_empty() {
            continue;
        }
        let Some(qual) = imported_builtin_qualified_name(path, Some(name)) else {
            continue;
        };
        let qual_args = prepare_builtin_args(&qual, prepared_primary).await?;
        let result =
            runmat_runtime::call_builtin_async_with_outputs(&qual, &qual_args, requested_outputs)
                .await;
        if let Ok(value) = result {
            wildcard_matches.push((qual, value));
        }
    }
    if wildcard_matches.len() > 1 {
        let msg = wildcard_matches
            .iter()
            .map(|(q, _)| q.clone())
            .collect::<Vec<_>>()
            .join(", ");
        return Ok(ImportedBuiltinResolution::Ambiguous(mex(
            "AmbiguousBuiltinImport",
            &format!("ambiguous builtin '{}' via wildcard imports: {}", name, msg),
        )));
    }
    if let Some((_, value)) = wildcard_matches.pop() {
        return Ok(ImportedBuiltinResolution::Resolved(value));
    }

    Ok(ImportedBuiltinResolution::NotFound)
}

pub fn rethrow_without_explicit_exception(
    name: &str,
    args: &[Value],
    last_identifier: Option<&str>,
    last_message: Option<&str>,
) -> Option<RuntimeError> {
    match VmIntrinsicExceptionBuiltin::classify(name) {
        Some(VmIntrinsicExceptionBuiltin::Rethrow) if args.is_empty() => {
            if let (Some(identifier), Some(message)) = (last_identifier, last_message) {
                return Some(
                    build_runtime_error(message.to_string())
                        .with_identifier(identifier.to_string())
                        .build(),
                );
            }
            None
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{imported_builtin_qualified_name, rethrow_without_explicit_exception};

    #[test]
    fn imported_builtin_qualified_name_uses_typed_segments() {
        let specific = imported_builtin_qualified_name(&["PkgF".into(), "foo".into()], None);
        assert_eq!(specific.as_deref(), Some("PkgF.foo"));

        let wildcard = imported_builtin_qualified_name(&["PkgF".into()], Some("foo"));
        assert_eq!(wildcard.as_deref(), Some("PkgF.foo"));
    }

    #[test]
    fn imported_builtin_qualified_name_ignores_empty_segments_without_fallback() {
        let empty = imported_builtin_qualified_name(&[], None);
        assert_eq!(empty, None);

        let only_empty = imported_builtin_qualified_name(&["".into()], None);
        assert_eq!(only_empty, None);

        let mixed_empty = imported_builtin_qualified_name(&["PkgF".into(), "".into()], Some("foo"));
        assert_eq!(mixed_empty, None);
    }

    #[test]
    fn rethrow_preserves_last_exception_identifier() {
        let err = rethrow_without_explicit_exception(
            "rethrow",
            &[],
            Some("RunMat:Original"),
            Some("boom"),
        )
        .expect("rethrow should preserve prior exception");
        assert_eq!(err.identifier(), Some("RunMat:Original"));
        assert_eq!(err.message(), "boom");
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn prepare_builtin_args_error_maps_to_accel_identifier() {
        let err = super::map_prepare_builtin_args_error("boom");
        assert_eq!(err.identifier(), Some("RunMat:AccelerationOperationFailed"));
        assert!(err.message().contains("prepare builtin args"));
        assert!(err.message().contains("boom"));
    }
}
