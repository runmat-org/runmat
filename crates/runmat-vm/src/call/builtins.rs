use crate::bytecode::{Bytecode, FunctionRegistry, Instr};
use crate::interpreter::api::InterpreterOutcome;
use crate::interpreter::errors::mex;
use crate::interpreter::stack::pop_args;
use crate::runtime::workspace::{
    base_target_authority, capture_workspace_target_in_context, has_workspace_target_in_context,
    AdmittedDynamicInput, BaseTargetAuthority, WorkspaceAssignedReport, WorkspaceFrameFuture,
    WorkspaceTarget, WorkspaceTargetIdentity, WorkspaceTargetSnapshot, WorkspaceValueSnapshot,
};
use runmat_builtins::DynamicWorkspaceBuiltin;
use runmat_hir::{FunctionId, QualifiedName, SymbolName};
use runmat_parser::{parse_with_options, CompatMode, ParserOptions};
use runmat_runtime::RuntimeError;
use runmat_thread_local::runmat_thread_local;
use runmat_value::Value;
use std::collections::{HashMap, HashSet};

#[path = "builtins/dynamic_base_target.rs"]
mod dynamic_base_target;
#[path = "builtins/eval_hir_sites.rs"]
mod eval_hir_sites;
#[path = "builtins/eval_static_selection.rs"]
mod eval_static_selection;
#[path = "builtins/native_target.rs"]
mod native_target;
#[path = "builtins/result_capture.rs"]
mod result_capture;
pub(crate) use eval_hir_sites::PendingClosurePlan;
pub(crate) use eval_hir_sites::SelectedClosureCapture;
pub use result_capture::DynamicEvalResult;
use result_capture::{
    prepare_dynamic_result_capture, retain_dynamic_result_slot, DynamicEvalResultCapture,
    DynamicTerminalOutput, PreparedDynamicEvalResultCapture,
};
mod source_evaluation;
pub use eval_static_selection::EvalSelectedSource;
pub use source_evaluation::{
    evaluate_numeric_text_source_in_context, evaluate_session_source_in_context,
    evaluate_session_source_with_dynamic_admission_in_context, DynamicSessionEvalAdmission,
};

#[derive(Clone, Copy)]
enum VmIntrinsicBuiltin {
    Nargin,
    Nargout,
}

#[derive(Clone, Copy, PartialEq, Eq)]
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
            runmat_extensions_enabled: false,
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
    if let Some(runtime) = runmat_runtime::context::legacy::active() {
        runtime.set_language_mode(match compat_mode {
            CompatMode::Matlab => runmat_runtime::context::RuntimeLanguageMode::Matlab,
            CompatMode::RunMat => runmat_runtime::context::RuntimeLanguageMode::RunMat,
            CompatMode::Strict => runmat_runtime::context::RuntimeLanguageMode::Strict,
        });
        runtime.set_runmat_extensions_enabled(runmat_extensions_enabled);
        runtime.set_top_level_await_enabled(top_level_await_enabled);
        runtime.set_dynamic_eval_enabled(dynamic_eval_enabled);
    }
    runmat_runtime::compatibility::set_runmat_extensions_enabled(runmat_extensions_enabled);
    DYNAMIC_EVAL_OPTIONS.with(|slot| {
        *slot.borrow_mut() = DynamicEvalOptions {
            compat_mode,
            runmat_extensions_enabled,
            top_level_await_enabled,
            dynamic_eval_enabled,
        };
    });
}

#[must_use]
pub struct DynamicEvalOptionsGuard {
    previous: DynamicEvalOptions,
}

impl Drop for DynamicEvalOptionsGuard {
    fn drop(&mut self) {
        let previous = self.previous;
        set_dynamic_eval_options(
            previous.compat_mode,
            previous.runmat_extensions_enabled,
            previous.top_level_await_enabled,
            previous.dynamic_eval_enabled,
        );
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
    if let Some(runtime) = runmat_runtime::context::legacy::active() {
        let compat_mode = match runtime.language_mode() {
            runmat_runtime::context::RuntimeLanguageMode::Matlab => CompatMode::Matlab,
            runmat_runtime::context::RuntimeLanguageMode::RunMat => CompatMode::RunMat,
            runmat_runtime::context::RuntimeLanguageMode::Strict => CompatMode::Strict,
        };
        return DynamicEvalOptions {
            compat_mode,
            runmat_extensions_enabled: runtime.runmat_extensions_enabled(),
            top_level_await_enabled: runtime.top_level_await_enabled(),
            dynamic_eval_enabled: runtime.dynamic_eval_enabled(),
        };
    }
    DYNAMIC_EVAL_OPTIONS.with(|slot| *slot.borrow())
}

impl VmIntrinsicBuiltin {
    fn classify(name: &str) -> Option<Self> {
        match name {
            runmat_hir::NARGIN_BUILTIN_NAME => Some(Self::Nargin),
            runmat_hir::NARGOUT_BUILTIN_NAME => Some(Self::Nargout),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Nargin => runmat_hir::NARGIN_BUILTIN_NAME,
            Self::Nargout => runmat_hir::NARGOUT_BUILTIN_NAME,
        }
    }
}

pub fn is_vm_intrinsic_builtin(name: &str) -> bool {
    VmIntrinsicBuiltin::classify(name).is_some()
}

pub fn is_vm_dynamic_workspace_builtin(name: &str) -> bool {
    DynamicWorkspaceBuiltin::for_canonical_name(name).is_some()
}

#[cfg(test)]
mod dispatch_tests;

pub fn collect_call_args(
    stack: &mut crate::interpreter::stack::InterpreterStack,
    arg_count: usize,
) -> Result<Vec<Value>, RuntimeError> {
    pop_args(stack, arg_count)
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
    stack: &mut crate::interpreter::stack::InterpreterStack,
    name: &str,
    arg_count: usize,
    invocation_arity: runmat_runtime::context::RuntimeInvocationArity,
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
            let (nin, _) = invocation_arity.intrinsic_counts();
            Ok(Value::Num(nin as f64))
        }
        VmIntrinsicBuiltin::Nargout => {
            validate_intrinsic_arg_count(intrinsic.name(), args.len(), 0)?;
            let (_, nout) = invocation_arity.intrinsic_counts();
            Ok(Value::Num(nout as f64))
        }
    }
}

#[cfg(test)]
pub async fn vm_dynamic_workspace_builtin(
    runtime: &runmat_runtime::context::RuntimeContext,
    stack: &mut crate::interpreter::stack::InterpreterStack,
    name: &str,
    arg_count: usize,
    requested_outputs: usize,
    function_registry: &FunctionRegistry,
    source_id: Option<runmat_hir::SourceId>,
) -> Result<runmat_runtime::execution::RootedValue, RuntimeError> {
    vm_dynamic_workspace_builtin_with_driver(
        runtime,
        stack,
        name,
        arg_count,
        requested_outputs,
        function_registry,
        DynamicBuiltinSource {
            source_id,
            driver: None,
            dynamic_admission: None,
        },
    )
    .await
}

pub(crate) struct DynamicBuiltinSource {
    pub source_id: Option<runmat_hir::SourceId>,
    pub driver: Option<crate::interpreter::process_driver::ProcessDriverHandle>,
    pub dynamic_admission: Option<crate::interpreter::dynamic_admission::DynamicEvalAdmissionPort>,
}

pub(crate) async fn vm_dynamic_workspace_builtin_with_driver(
    runtime: &runmat_runtime::context::RuntimeContext,
    stack: &mut crate::interpreter::stack::InterpreterStack,
    name: &str,
    arg_count: usize,
    requested_outputs: usize,
    function_registry: &FunctionRegistry,
    selected: DynamicBuiltinSource,
) -> Result<runmat_runtime::execution::RootedValue, RuntimeError> {
    let Some(builtin) = DynamicWorkspaceBuiltin::for_canonical_name(name) else {
        return Err(mex(
            "UndefinedFunction",
            &format!("unknown VM dynamic workspace builtin '{name}'"),
        ));
    };
    let args = collect_call_args(stack, arg_count)?;
    let _roots = crate::call::root_values(&args)?;
    execute_dynamic_workspace_builtin(
        runtime,
        runmat_runtime::call::FunctionArgumentView::from_slice(&args),
        builtin,
        requested_outputs,
        function_registry,
        selected,
    )
    .await
}

async fn execute_dynamic_workspace_builtin(
    runtime: &runmat_runtime::context::RuntimeContext,
    args: runmat_runtime::call::FunctionArgumentView<'_>,
    builtin: DynamicWorkspaceBuiltin,
    requested_outputs: usize,
    function_registry: &FunctionRegistry,
    selected: DynamicBuiltinSource,
) -> Result<runmat_runtime::execution::RootedValue, RuntimeError> {
    let DynamicBuiltinSource {
        source_id,
        driver,
        dynamic_admission,
    } = selected;
    let dynamic_admission = dynamic_admission.or_else(|| {
        driver
            .as_ref()
            .and_then(|driver| driver.dynamic_admission())
    });
    match builtin {
        DynamicWorkspaceBuiltin::Eval => {
            validate_intrinsic_arg_count(builtin.name(), args.len(), 1)?;
            let source = workspace_text_arg(builtin.name(), &args[0])?;
            let source_lease = dynamic_source_context(runtime, "<eval>", &source, source_id)?;
            eval_workspace_source(
                WorkspaceEvalRequest {
                    builtin: builtin.name(),
                    target: WorkspaceTarget::Current,
                    source,
                    source_label: "<eval>".to_string(),
                    requested_outputs,
                    source_ref: Some(source_lease.source_ref()),
                    source_lease: Some(source_lease),
                    caller_source_id: source_id,
                    dynamic_admission: dynamic_admission.clone(),
                    driver: driver.clone(),
                    runtime: runtime.clone(),
                    commit_workspace_on_error: false,
                    terminal_output: DynamicTerminalOutput::Preserve,
                    empty_value_when_no_result: false,
                },
                function_registry,
            )
            .await
        }
        DynamicWorkspaceBuiltin::Evalc => {
            validate_intrinsic_arg_count(builtin.name(), args.len(), 1)?;
            let policy = runmat_runtime::context::host_allocation::HostAllocationReceipt::policy(
                Some(runtime),
            )
            .map_err(|source| {
                runmat_runtime::context::host_allocation::host_allocation_runtime_error(
                    Some(runtime),
                    "RunMat:evalc:HostAllocation",
                    "evalc capture admission failed",
                    source,
                )
            })?;
            if policy == runmat_runtime::context::host_allocation::HostAllocationPolicy::Managed {
                return Err(runmat_runtime::build_runtime_error(
                    "evalc cannot yet publish captured text under finite managed admission",
                )
                .with_identifier("RunMat:evalc:ManagedTextUnavailable")
                .build());
            }
            let source = workspace_text_arg(builtin.name(), &args[0])?;
            let source_lease = dynamic_source_context(runtime, "<evalc>", &source, source_id)?;
            let eval_requested_outputs = if requested_outputs == 1 {
                1
            } else {
                requested_outputs.saturating_sub(1)
            };
            let capture = runmat_runtime::console::begin_capture()?;
            let eval_result = eval_workspace_source(
                WorkspaceEvalRequest {
                    builtin: builtin.name(),
                    target: WorkspaceTarget::Current,
                    source,
                    source_label: "<evalc>".to_string(),
                    requested_outputs: eval_requested_outputs,
                    source_ref: Some(source_lease.source_ref()),
                    source_lease: Some(source_lease),
                    caller_source_id: source_id,
                    dynamic_admission: dynamic_admission.clone(),
                    driver: driver.clone(),
                    runtime: runtime.clone(),
                    commit_workspace_on_error: false,
                    terminal_output: if requested_outputs == 1 {
                        DynamicTerminalOutput::Capture
                    } else {
                        DynamicTerminalOutput::Suppress
                    },
                    empty_value_when_no_result: requested_outputs > 1,
                },
                function_registry,
            )
            .await;
            let captured = capture.finish()?;
            eval_result?.try_map_owned(
                |result| evalc_output_value(captured, result, requested_outputs),
                |error| error.into_runtime_error(),
            )
        }
        DynamicWorkspaceBuiltin::Evalin => {
            validate_intrinsic_arg_count(builtin.name(), args.len(), 2)?;
            let target = workspace_target_arg(builtin.name(), &args[0])?;
            let source = workspace_text_arg(builtin.name(), &args[1])?;
            let source_lease = dynamic_source_context(runtime, "<eval>", &source, source_id)?;
            eval_workspace_source(
                WorkspaceEvalRequest {
                    builtin: builtin.name(),
                    target,
                    source,
                    source_label: "<eval>".to_string(),
                    requested_outputs,
                    source_ref: Some(source_lease.source_ref()),
                    source_lease: Some(source_lease),
                    caller_source_id: source_id,
                    dynamic_admission: dynamic_admission.clone(),
                    driver: driver.clone(),
                    runtime: runtime.clone(),
                    commit_workspace_on_error: false,
                    terminal_output: DynamicTerminalOutput::Preserve,
                    empty_value_when_no_result: false,
                },
                function_registry,
            )
            .await
        }
        DynamicWorkspaceBuiltin::Assignin => {
            runmat_runtime::context::execute_assignin_effect_view(
                runtime,
                args,
                requested_outputs,
            )?;
            runmat_runtime::execution::RootedValue::new(Value::Num(0.0))
                .map_err(|error| error.into_runtime_error())
        }
        DynamicWorkspaceBuiltin::Run => {
            validate_intrinsic_arg_count(builtin.name(), args.len(), 1)?;
            if requested_outputs > 0 {
                return Err(runmat_runtime::builtins::io::repl_fs::run::too_many_outputs_error());
            }
            let source =
                runmat_runtime::builtins::io::repl_fs::run::resolve_run_source(&args[0]).await?;
            let source_lease = runmat_runtime::source_context::lease_source_context_in_context(
                runtime,
                source.display_name.as_str(),
                Some(std::sync::Arc::from(source.path.to_string_lossy().as_ref())),
                source.text.as_str(),
            )
            .map_err(runmat_runtime::source_context::NestedSourceIdentityError::Exhausted)
            .map_err(dynamic_source_identity_error)?;
            eval_workspace_source(
                WorkspaceEvalRequest {
                    builtin: builtin.name(),
                    target: WorkspaceTarget::Current,
                    source: source.text,
                    source_label: source.display_name,
                    requested_outputs: 0,
                    source_ref: Some(source_lease.source_ref()),
                    source_lease: Some(source_lease),
                    caller_source_id: source_id,
                    dynamic_admission,
                    driver,
                    runtime: runtime.clone(),
                    commit_workspace_on_error: true,
                    terminal_output: DynamicTerminalOutput::Preserve,
                    empty_value_when_no_result: false,
                },
                function_registry,
            )
            .await
        }
    }
}

/// Shared executor engine entered by the catalog-selected Runtime builtin
/// body. Core supplies the immutable executable generation and its selected
/// dynamic-source admission; the engine never rediscovers a callable name.
pub async fn evaluate_dynamic_workspace_in_context(
    runtime: &runmat_runtime::context::RuntimeContext,
    invocation: runmat_runtime::context::DynamicWorkspaceInvocation<'_>,
    function_registry: &FunctionRegistry,
    dynamic_admission: Option<crate::interpreter::dynamic_admission::DynamicEvalAdmissionPort>,
) -> Result<runmat_runtime::execution::RootedValueSequence, RuntimeError> {
    let source_id = invocation
        .callsite
        .as_ref()
        .and_then(|callsite| callsite.source_id);
    let value = execute_dynamic_workspace_builtin(
        runtime,
        invocation.arguments,
        invocation.builtin,
        invocation.requested_outputs,
        function_registry,
        DynamicBuiltinSource {
            source_id,
            driver: None,
            dynamic_admission,
        },
    )
    .await?;
    value.try_map_owned(
        runmat_runtime::call::arguments::adapt_legacy_builtin_result,
        |error| error.into_runtime_error(),
    )
}

fn evalc_output_value(
    captured: String,
    result: Value,
    requested_outputs: usize,
) -> Result<Value, RuntimeError> {
    if requested_outputs == 0 {
        return Ok(Value::output_list(Vec::new()));
    }

    let mut outputs = Vec::with_capacity(requested_outputs.max(1));
    outputs.push(Value::String(captured.into()));
    if requested_outputs > 1 {
        match result {
            Value::OutputList(values) => {
                let values = values.try_into_owned().map_err(|_| {
                    mex(
                        "SharedOutputStorage",
                        "evalc result storage is shared and cannot be materialized implicitly",
                    )
                })?;
                outputs.extend(values.into_iter().take(requested_outputs - 1))
            }
            value => outputs.push(value),
        }
        if outputs.len() < requested_outputs {
            return Err(mex(
                "EvalcTooManyOutputs",
                &format!(
                    "evalc: requested {} output arguments, but evaluated source produced {}",
                    requested_outputs,
                    outputs.len()
                ),
            ));
        }
    }

    if requested_outputs == 1 {
        Ok(outputs.remove(0))
    } else {
        Ok(Value::output_list(outputs))
    }
}

fn workspace_text_arg(builtin: &str, value: &Value) -> Result<String, RuntimeError> {
    match value {
        Value::String(text) => text.as_str().map(str::to_owned).ok_or_else(|| {
            mex(
                "DynamicWorkspaceArgumentType",
                &format!("{builtin}: expected nonmissing text"),
            )
        }),
        Value::CharArray(chars) if chars.rows == 1 => chars
            .try_row_string()
            .map_err(|source| {
                runmat_runtime::runtime_error::semantic_error_with_source(
                    "DynamicWorkspaceArgumentType",
                    format!("{builtin}: character row is not valid UTF-16 text"),
                    source,
                )
            })?
            .ok_or_else(|| {
                mex(
                    "DynamicWorkspaceArgumentType",
                    &format!("{builtin}: expected a character row"),
                )
            }),
        Value::StringArray(array) if array.data().len() == 1 => {
            array.data()[0].as_str().map(str::to_owned).ok_or_else(|| {
                mex(
                    "DynamicWorkspaceArgumentType",
                    &format!("{builtin}: expected nonmissing text"),
                )
            })
        }
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

struct WorkspaceEvalRequest {
    builtin: &'static str,
    target: WorkspaceTarget,
    source: String,
    source_label: String,
    requested_outputs: usize,
    source_ref: Option<runmat_runtime::source_context::RuntimeSourceRef>,
    source_lease: Option<runmat_runtime::source_context::RuntimeSourceLease>,
    caller_source_id: Option<runmat_types::SourceId>,
    dynamic_admission: Option<crate::interpreter::dynamic_admission::DynamicEvalAdmissionPort>,
    driver: Option<crate::interpreter::process_driver::ProcessDriverHandle>,
    runtime: runmat_runtime::context::RuntimeContext,
    commit_workspace_on_error: bool,
    terminal_output: DynamicTerminalOutput,
    empty_value_when_no_result: bool,
}

/// One compiler invocation, before and after eval's caller-workspace binding.
/// The canonical local product is retained separately from the interpreted
/// projection so a selected-product owner can verify each typed remap without
/// reparsing source or treating merged parent definitions as eval-local code.
pub struct DynamicEvalCompileProduct {
    pub compat_mode: CompatMode,
    pub hir: runmat_hir::HirAssembly,
    pub mir: runmat_mir::MirAssembly,
    pub canonical_bytecode: Bytecode,
    pub execution_bytecode: Bytecode,
    pub function_remap: HashMap<FunctionId, FunctionId>,
    pub workspace_slot_remap: HashMap<usize, usize>,
    pub result: Option<DynamicEvalResult>,
    /// Compiler-only aliases paired with exact no-invoke source selections.
    /// Core consumes these at sealed-site admission; they never enter names.
    pub selected_sources: Vec<EvalSelectedSource>,
    pub selected_external_sources: Vec<eval_static_selection::EvalSelectedExternalSource>,
    /// Installed native selections observed at this closure's unresolved HIR
    /// sites. Core must reprove them against the exact prepared project scope.
    pub(crate) selected_natives: Vec<eval_static_selection::EvalSelectedNative>,
    workspace_extent: usize,
    requested_outputs: usize,
    result_capture: PreparedDynamicEvalResultCapture,
    source_ref: Option<runmat_runtime::source_context::RuntimeSourceRef>,
}

impl DynamicEvalCompileProduct {
    /// The observed installed-native fact is useful only if the recompiled
    /// closure still has that same typed unresolved call at its proven
    /// first-effect frontier. Names are not used to discover authority here.
    pub fn selected_native_interop_for(
        &self,
        function: FunctionId,
    ) -> Result<runmat_types::InteropManifest, &'static str> {
        let selected = self
            .hir
            .functions
            .iter()
            .find(|candidate| candidate.id == function)
            .ok_or("selected native closure has no recompiled HIR body")?;
        let sites = eval_hir_sites::pre_effect_sites(
            selected,
            eval_hir_sites::all_unresolved_sites(selected),
        );
        let mut manifests = Vec::new();
        for native in &self.selected_natives {
            if native.site.caller != function || !sites.contains(&native.site) {
                return Err("selected native site changed during closure recompilation");
            }
            manifests.push(native.artifact.interop.clone());
        }
        runmat_types::InteropManifest::merge(manifests)
            .map_err(|_| "selected native interop facts conflict")
    }
    /// Apply the already recorded eval projection to the decorated, source-bound
    /// canonical unit. The returned body is the admitted body that will run;
    /// an independently remapped pre-decoration clone never becomes authority.
    pub fn project_admitted_bytecode(
        &self,
        mut admitted: Bytecode,
        inherited: &FunctionRegistry,
    ) -> Result<Bytecode, RuntimeError> {
        if let Some(source_ref) = self.source_ref {
            if admitted.source_id != Some(crate::BytecodeSourceRef::Runtime(source_ref)) {
                return Err(mex(
                    "DynamicWorkspaceCompile",
                    "admitted dynamic source differs from the compiled source lease",
                ));
            }
        }
        // Plan against the compiler-local layout before decorating it. The
        // decorated layout already contains published IDs, which would be
        // mistaken for additional reserved local IDs by a second plan.
        let (local_ids, remap) = eval_function_merge_plan(&admitted, inherited);
        if remap != self.function_remap {
            return Err(mex(
                "DynamicWorkspaceCompile",
                "admitted eval function identities differ from the compiler projection",
            ));
        }
        crate::bytecode::program::function_remap::remap_decorated_eval_metadata(
            &mut admitted,
            &remap,
        )
        .map_err(|error| {
            mex(
                "DynamicWorkspaceCompile",
                &format!("admitted eval metadata remap: {error}"),
            )
        })?;
        merge_eval_function_registry_with_plan(&mut admitted, inherited, local_ids, &remap)
            .map_err(|error| {
                mex(
                    "DynamicWorkspaceCompile",
                    &format!("admitted eval function remap: {error}"),
                )
            })?;
        remap_dynamic_workspace_slots_with_plan(
            &mut admitted,
            &self.workspace_slot_remap,
            self.workspace_extent,
        )?;
        let result = retain_dynamic_result_slot(
            &mut admitted,
            self.requested_outputs,
            self.workspace_extent,
            self.result_capture,
        )?;
        if result != self.result {
            return Err(mex(
                "DynamicWorkspaceCompile",
                "admitted eval output slot differs from the compiler projection",
            ));
        }
        let same_instructions = |left: &[Instr], right: &[Instr]| -> Result<bool, RuntimeError> {
            let encode = |instructions: &[Instr]| {
                serde_json::to_vec(instructions).map_err(|error| {
                    mex(
                        "DynamicWorkspaceCompile",
                        &format!("cannot verify admitted eval instructions: {error}"),
                    )
                })
            };
            Ok(encode(left)? == encode(right)?)
        };
        if !same_instructions(
            &admitted.instructions,
            &self.execution_bytecode.instructions,
        )? {
            return Err(mex(
                "DynamicWorkspaceCompile",
                "admitted eval instructions differ from the compiler projection",
            ));
        }
        for published in self.function_remap.values() {
            let selected = admitted.function_registry.get_current(*published);
            let compiled = self
                .execution_bytecode
                .function_registry
                .get_current(*published);
            let same_body = match (selected, compiled) {
                (Some(left), Some(right)) => {
                    if self.source_ref.is_some_and(|source_ref| {
                        left.source_id != Some(crate::BytecodeSourceRef::Runtime(source_ref))
                    }) {
                        return Err(mex(
                            "DynamicWorkspaceCompile",
                            "admitted eval function source differs from its issued lease",
                        ));
                    }
                    // Core may add only the source and analyzed execution
                    // decoration listed here. Every remaining semantic body
                    // field (including lexical/capture/output/class facts)
                    // must still match the compiler's one-pass projection.
                    let mut normalized = left.clone();
                    normalized.source_id = right.source_id;
                    normalized.private_source_owner = right.private_source_owner.clone();
                    normalized.coverage_sites = right.coverage_sites.clone();
                    normalized.regions = right.regions.clone();
                    normalized.parfor_regions = right.parfor_regions.clone();
                    normalized.spmd_regions = right.spmd_regions.clone();
                    normalized.distributed_values = right.distributed_values.clone();
                    normalized.collective_contracts = right.collective_contracts.clone();
                    #[cfg(feature = "native-accel")]
                    {
                        normalized.fusion_stage_projection = right.fusion_stage_projection.clone();
                    }
                    let fingerprint = |body: &crate::bytecode::FunctionBytecode| {
                        crate::bytecode::program::semantic_definition_fingerprint(body).map_err(
                            |error| {
                                mex(
                                    "DynamicWorkspaceCompile",
                                    &format!("cannot verify admitted eval function: {error}"),
                                )
                            },
                        )
                    };
                    left.lexical_parent == right.lexical_parent
                        && fingerprint(&normalized)? == fingerprint(right)?
                }
                _ => false,
            };
            if !same_body {
                return Err(mex(
                    "DynamicWorkspaceCompile",
                    "admitted eval function body differs from the compiler projection",
                ));
            }
        }
        // Function fingerprints above prove every selected local body, but
        // they do not cover the containing script frame. Normalize only
        // Core's admitted execution decorations, then compare the complete
        // serialized top-level product. In particular, workspace names,
        // types, unassigned slots, catch scopes, and async facts may not be
        // silently substituted while leaving the instructions unchanged.
        let mut normalized = admitted.clone();
        normalized.coverage_sites = self.execution_bytecode.coverage_sites.clone();
        normalized.regions = self.execution_bytecode.regions.clone();
        normalized.parfor_regions = self.execution_bytecode.parfor_regions.clone();
        normalized.spmd_regions = self.execution_bytecode.spmd_regions.clone();
        normalized.distributed_values = self.execution_bytecode.distributed_values.clone();
        normalized.collective_contracts = self.execution_bytecode.collective_contracts.clone();
        #[cfg(feature = "native-accel")]
        {
            normalized.accel_graph = self.execution_bytecode.accel_graph.clone();
            normalized.fusion_groups = self.execution_bytecode.fusion_groups.clone();
            normalized.fusion_metadata = self.execution_bytecode.fusion_metadata.clone();
        }
        for published in self.function_remap.values() {
            let expected = self
                .execution_bytecode
                .function_registry
                .get_current(*published)
                .ok_or_else(|| {
                    mex(
                        "DynamicWorkspaceCompile",
                        "compiler projection lacks its selected function",
                    )
                })?;
            let bound = normalized.bound_functions.get(published).ok_or_else(|| {
                mex(
                    "DynamicWorkspaceCompile",
                    "admitted eval lacks its bound function mirror",
                )
            })?;
            let selected = normalized
                .function_registry
                .get_current(*published)
                .ok_or_else(|| {
                    mex(
                        "DynamicWorkspaceCompile",
                        "admitted eval lacks its selected function",
                    )
                })?;
            if !std::sync::Arc::ptr_eq(&bound.definition_owner, &selected.definition_owner)
                || bound.binding_origin != selected.binding_origin
                || bound.source_id != selected.source_id
                || !match (
                    &bound.selected_source_owner,
                    &selected.selected_source_owner,
                ) {
                    (None, None) => true,
                    (Some(left), Some(right)) => left.same_issuance(right),
                    _ => false,
                }
                || !match (
                    &bound.selected_class_product,
                    &selected.selected_class_product,
                ) {
                    (None, None) => true,
                    (Some(left), Some(right)) => std::rc::Rc::ptr_eq(left, right),
                    _ => false,
                }
            {
                return Err(mex(
                    "DynamicWorkspaceCompile",
                    "admitted bound function differs from the selected body",
                ));
            }
            normalized
                .function_registry
                .functions
                .insert(*published, expected.clone());
            normalized
                .bound_functions
                .insert(*published, expected.clone());
        }
        // Compilation finalizes resume points in the derived layout. Compare
        // the admitted unit with that finalized one-pass compiler product,
        // not a fresh pre-compilation derivation that lacks those points.
        let mut expected_layout = self.canonical_bytecode.layout.clone().ok_or_else(|| {
            mex(
                "DynamicWorkspaceCompile",
                "compiler projection lacks its canonical eval layout",
            )
        })?;
        crate::layout::remap_layout_function_ids(&mut expected_layout, &self.function_remap)
            .map_err(|error| {
                mex(
                    "DynamicWorkspaceCompile",
                    &format!("cannot remap admitted eval layout: {error:?}"),
                )
            })?;
        if admitted.layout.as_ref() != Some(&expected_layout) {
            return Err(mex(
                "DynamicWorkspaceCompile",
                "admitted eval layout differs from its compiler derivation",
            ));
        }
        // The runtime source reference is an issuer identity, not portable
        // JSON. Prove it before normalizing the admitted source decoration.
        if self.source_ref.is_none() && admitted.source_id != self.execution_bytecode.source_id {
            return Err(mex(
                "DynamicWorkspaceCompile",
                "admitted eval source differs from the compiler projection",
            ));
        }
        normalized.source_id = self.execution_bytecode.source_id;
        normalized.layout = self.execution_bytecode.layout.clone();
        let same_owner = |left: &crate::bytecode::FunctionBytecode,
                          right: &crate::bytecode::FunctionBytecode| {
            left.source_id == right.source_id
                && std::sync::Arc::ptr_eq(&left.definition_owner, &right.definition_owner)
                && left.private_source_owner == right.private_source_owner
                && match (&left.selected_source_owner, &right.selected_source_owner) {
                    (None, None) => true,
                    (Some(left), Some(right)) => left.same_issuance(right),
                    _ => false,
                }
                && match (&left.selected_class_product, &right.selected_class_product) {
                    (None, None) => true,
                    (Some(left), Some(right)) => std::rc::Rc::ptr_eq(left, right),
                    _ => false,
                }
        };
        let same_body_owners =
            |left: &HashMap<FunctionId, crate::bytecode::FunctionBytecode>,
             right: &HashMap<FunctionId, crate::bytecode::FunctionBytecode>| {
                left.len() == right.len()
                    && left.iter().all(|(id, body)| {
                        right
                            .get(id)
                            .is_some_and(|expected| same_owner(body, expected))
                    })
            };
        let semantic_fingerprint = |bytecode: &Bytecode| {
            crate::bytecode::program::local_bytecode_semantic_fingerprint(bytecode).map_err(
                |error| {
                    mex(
                        "DynamicWorkspaceCompile",
                        &format!("cannot verify admitted eval semantics: {error}"),
                    )
                },
            )
        };
        if normalized.active_function != self.execution_bytecode.active_function
            || normalized.active_class_method_owner
                != self.execution_bytecode.active_class_method_owner
            || {
                #[cfg(feature = "native-accel")]
                {
                    normalized.function_private_layout.is_some()
                        || self.execution_bytecode.function_private_layout.is_some()
                }
                #[cfg(not(feature = "native-accel"))]
                {
                    false
                }
            }
            || !same_body_owners(
                &normalized.bound_functions,
                &self.execution_bytecode.bound_functions,
            )
            || !same_body_owners(
                &normalized.function_registry.functions,
                &self.execution_bytecode.function_registry.functions,
            )
            || semantic_fingerprint(&normalized)? != semantic_fingerprint(&self.execution_bytecode)?
        {
            return Err(mex(
                "DynamicWorkspaceCompile",
                "admitted eval top-level metadata differs from the compiler projection",
            ));
        }
        Ok(admitted)
    }
}

enum DynamicEvalCompiled {
    Ordinary(Box<Bytecode>, Option<DynamicEvalResult>),
    Selected(Box<DynamicEvalCompileProduct>),
}

/// Keep the source-lowering cause until a consumer chooses its conversion
/// policy. Existing eval callers project the identical RuntimeError; numeric
/// text may recognize only explicitly typed invalid-source causes.
struct DynamicEvalCompileFailure {
    error: RuntimeError,
    source_kind: Option<runmat_hir::HirErrorKind>,
}

impl From<RuntimeError> for DynamicEvalCompileFailure {
    fn from(error: RuntimeError) -> Self {
        Self {
            error,
            source_kind: None,
        }
    }
}

impl From<DynamicEvalCompileFailure> for RuntimeError {
    fn from(failure: DynamicEvalCompileFailure) -> Self {
        failure.error
    }
}

type DynamicProductSelector<'a> =
    &'a dyn Fn(&Bytecode, &FunctionRegistry) -> Result<bool, RuntimeError>;

struct DynamicEvalCompileRequest<'a> {
    builtin: &'a str,
    ast: &'a runmat_parser::Program,
    workspace_bindings: &'a dyn runmat_hir::WorkspaceBindingView,
    workspace_value_count: usize,
    lowering_bindings: &'a dyn runmat_hir::WorkspaceBindingView,
    function_registry: &'a FunctionRegistry,
    source_ref: Option<runmat_runtime::source_context::RuntimeSourceRef>,
    requested_outputs: usize,
    result_capture: DynamicEvalResultCapture,
    select_product: Option<DynamicProductSelector<'a>>,
    selected_sources: eval_static_selection::EvalSelectedSources,
}

fn parse_dynamic_eval(builtin: &str, source: &str) -> Result<runmat_parser::Program, RuntimeError> {
    let options = current_dynamic_eval_options();
    parse_with_options(source, ParserOptions::new(options.compat_mode)).map_err(|err| {
        mex(
            "DynamicWorkspaceParse",
            &format!("{builtin}: parse error: {err}"),
        )
    })
}

#[derive(Clone)]
enum DynamicEvalTarget {
    Frame {
        identity: WorkspaceTargetIdentity,
        explicit_base: Option<std::rc::Rc<dynamic_base_target::ExplicitBaseTarget>>,
    },
    Service(runmat_runtime::workspace::DynamicWorkspaceTargetCapture),
    Base(std::rc::Rc<dynamic_base_target::ExplicitBaseTarget>),
}

impl DynamicEvalTarget {
    fn scope<F: std::future::Future>(&self, future: F) -> WorkspaceFrameFuture<F> {
        match self {
            Self::Frame { identity, .. } => identity.scope(future),
            Self::Service(_) | Self::Base(_) => WorkspaceFrameFuture::without_frame(future),
        }
    }

    fn ensure_effect_authority(
        &self,
        runtime: &runmat_runtime::context::RuntimeContext,
    ) -> Result<(), RuntimeError> {
        match self {
            Self::Frame {
                explicit_base: Some(base),
                ..
            }
            | Self::Base(base) => base.ensure_effect_authority(),
            Self::Service(captured) => {
                let live = captured
                    .issuer()
                    .capture_workspace_completion_identity_with_context(runtime)?;
                if live != Some(captured.identity()) {
                    return Err(mex(
                        "DynamicWorkspaceUnavailable",
                        "captured workspace target changed before execution",
                    ));
                }
                Ok(())
            }
            Self::Frame {
                explicit_base: None,
                ..
            } => Ok(()),
        }
    }

    fn commit(
        &self,
        runtime: &runmat_runtime::context::RuntimeContext,
        snapshot: WorkspaceValueSnapshot,
        report: WorkspaceAssignedReport,
    ) -> Result<(), RuntimeError> {
        match self {
            Self::Frame {
                identity,
                explicit_base,
            } => commit_dynamic_workspace_target(
                runtime,
                *identity,
                explicit_base.as_deref(),
                snapshot,
                report,
            ),
            Self::Base(base) => base.commit(runtime, snapshot, report),
            Self::Service(captured) => {
                let source = native_target::NativeCompletionSource::prepare(
                    runtime, &snapshot, &report, captured,
                )?;
                captured.issuer().apply_workspace_completion_with_context(
                    runtime,
                    captured.identity(),
                    &source,
                )
            }
        }
    }
}

enum DynamicEvalInput {
    Frame(RootedDynamicEvalInput),
    Admitted(AdmittedDynamicInput),
}

async fn eval_workspace_source(
    request: WorkspaceEvalRequest,
    function_registry: &FunctionRegistry,
) -> Result<runmat_runtime::execution::RootedValue, RuntimeError> {
    let WorkspaceEvalRequest {
        builtin,
        target,
        source,
        source_label,
        requested_outputs,
        source_ref,
        source_lease,
        caller_source_id,
        dynamic_admission,
        driver,
        runtime,
        commit_workspace_on_error,
        terminal_output,
        empty_value_when_no_result,
    } = request;
    let options = current_dynamic_eval_options();
    if !options.dynamic_eval_enabled {
        return Err(mex(
            "DynamicEvalDisabled",
            &format!("{builtin}: dynamic eval is disabled by host execution policy"),
        ));
    }
    // An explicit base selector in a detached source-evaluation frame must
    // not alias that frame. Its typed admission selects the session owner;
    // ordinary VM frames retain their installed base-frame authority.
    let (target, frame) = if has_workspace_target_in_context(&runtime, target) {
        if target == WorkspaceTarget::Base
            && matches!(
                base_target_authority(),
                Some(BaseTargetAuthority::CanonicalService)
            )
        {
            let (identity, _) =
                capture_workspace_target_in_context(&runtime, WorkspaceTarget::Current).map_err(
                    |err| mex("DynamicWorkspaceUnavailable", &format!("{builtin}: {err}")),
                )?;
            let (base, frame) = dynamic_base_target::ExplicitBaseTarget::capture(&runtime)?;
            (
                DynamicEvalTarget::Frame {
                    identity,
                    explicit_base: Some(std::rc::Rc::new(base)),
                },
                DynamicEvalInput::Frame(RootedDynamicEvalInput::new(frame)?),
            )
        } else {
            let (identity, frame) = capture_workspace_target_in_context(&runtime, target)
                .map_err(|err| mex("DynamicWorkspaceUnavailable", &format!("{builtin}: {err}")))?;
            (
                DynamicEvalTarget::Frame {
                    identity,
                    explicit_base: None,
                },
                DynamicEvalInput::Frame(RootedDynamicEvalInput::new(frame)?),
            )
        }
    } else if target == WorkspaceTarget::Base {
        let (base, frame) = dynamic_base_target::ExplicitBaseTarget::capture(&runtime)?;
        (
            DynamicEvalTarget::Base(std::rc::Rc::new(base)),
            DynamicEvalInput::Frame(RootedDynamicEvalInput::new(frame)?),
        )
    } else {
        let service = runtime
            .service_ports()
            .require_workspace("capture dynamic workspace target")
            .map_err(runmat_runtime::context::RuntimeCapabilityError::into_runtime_error)?
            .clone();
        let service = if target == WorkspaceTarget::Caller {
            service
                .caller_workspace_with_context(&runtime)?
                .ok_or_else(|| {
                    mex(
                        "DynamicWorkspaceUnavailable",
                        "caller workspace is unavailable",
                    )
                })?
        } else {
            service
        };
        let captured = service
            .capture_dynamic_workspace_target_with_context(&runtime)?
            .ok_or_else(|| {
                mex(
                    "DynamicWorkspaceUnavailable",
                    "exact workspace target is unavailable",
                )
            })?;
        let admitted = AdmittedDynamicInput::capture(&runtime, &captured)?;
        (
            DynamicEvalTarget::Service(captured),
            DynamicEvalInput::Admitted(admitted),
        )
    };
    // FunctionRegistry is an Rc-backed immutable execution snapshot. This
    // clone carries the exact pre-eval code owner into the queued request; it
    // does not duplicate bodies or resolve a later published name.
    let function_registry = function_registry.clone();
    let continuation_runtime = runtime.clone();
    let work_runtime = runtime.clone();
    let execution_target = target.clone();
    let work = async move {
        execution_target.ensure_effect_authority(&runtime)?;
        if current_dynamic_eval_options() != options {
            return Err(mex(
                "DynamicWorkspaceCompile",
                "eval language options changed before its owned request began",
            ));
        }
        // The frame reserves slots for identifiers that have never been assigned.
        // Such a slot must not make a function call into a captured workspace
        // variable. Keep all slots for the later remap, but expose only assigned
        // bindings to HIR name classification.
        let legacy_lowering = match &frame {
            DynamicEvalInput::Frame(frame) => Some(
                frame
                    .frame
                    .names
                    .iter()
                    .filter(|(name, _)| frame.frame.assigned.contains(*name))
                    .map(|(name, slot)| (name.clone(), *slot))
                    .collect::<HashMap<_, _>>(),
            ),
            DynamicEvalInput::Admitted(_) => None,
        };
        let admitted_bindings = match &frame {
            DynamicEvalInput::Admitted(input) => Some(input.bindings()),
            DynamicEvalInput::Frame(_) => None,
        };
        let admitted_lowering = match &frame {
            DynamicEvalInput::Admitted(input) => Some(input.assigned_bindings()),
            DynamicEvalInput::Frame(_) => None,
        };
        let workspace_bindings: &dyn runmat_hir::WorkspaceBindingView = match &frame {
            DynamicEvalInput::Frame(frame) => &frame.frame.names,
            DynamicEvalInput::Admitted(_) => admitted_bindings
                .as_ref()
                .expect("admitted binding view belongs to admitted input"),
        };
        let lowering_bindings: &dyn runmat_hir::WorkspaceBindingView = match &frame {
            DynamicEvalInput::Frame(_) => legacy_lowering
                .as_ref()
                .expect("legacy lowering map belongs to legacy input"),
            DynamicEvalInput::Admitted(_) => admitted_lowering
                .as_ref()
                .expect("admitted lowering view belongs to admitted input"),
        };
        let workspace_value_count = match &frame {
            DynamicEvalInput::Frame(frame) => frame.frame.vars.len(),
            DynamicEvalInput::Admitted(input) => input.slot_count(),
        };
        let requires_product = |canonical: &Bytecode, inherited: &FunctionRegistry| {
            dynamic_admission.as_ref().map_or(Ok(false), |port| {
                port.host().requires_selected_product(canonical, inherited)
            })
        };
        let ast = std::rc::Rc::new(parse_dynamic_eval(builtin, &source)?);
        // A closure inside this source may be created only after earlier script
        // effects mutate path or workspace state. Source selection therefore
        // belongs to CreateSemanticClosure, never this eval-entry boundary.
        let selected_sources = eval_static_selection::EvalSelectedSources::default();
        let Some(compiled) = compile_dynamic_workspace_eval_inner(DynamicEvalCompileRequest {
            builtin,
            ast: ast.as_ref(),
            workspace_bindings,
            workspace_value_count,
            lowering_bindings,
            function_registry: &function_registry,
            source_ref,
            requested_outputs,
            result_capture: DynamicEvalResultCapture::TerminalExpression(terminal_output),
            select_product: dynamic_admission
                .as_ref()
                .map(|_| &requires_product as DynamicProductSelector<'_>),
            selected_sources,
        })?
        else {
            return Ok(None);
        };
        let mut dynamic_admission = dynamic_admission;
        let (bytecode, result, runtime, publication) = match compiled {
            DynamicEvalCompiled::Ordinary(bytecode, result) => (
                *bytecode,
                result,
                runtime.without_selected_graph_program(),
                None,
            ),
            DynamicEvalCompiled::Selected(product) => {
                let lease = source_lease.as_ref().ok_or_else(|| {
                    mex(
                        "DynamicWorkspaceSource",
                        "selected source has no issued lease",
                    )
                })?;
                let port = dynamic_admission
                    .as_ref()
                    .cloned()
                    .expect("selected product requires the explicit admission port");
                let pending = PendingClosurePlan::from_compiled(
                    std::rc::Rc::clone(&ast),
                    lease.clone(),
                    caller_source_id,
                    &product,
                    &function_registry,
                )
                .map_err(|error| mex("DynamicWorkspaceCompile", error))?;
                let admitted = port.host().admit(
                    &runtime,
                    lease,
                    &source,
                    caller_source_id,
                    *product,
                    &function_registry,
                )?;
                if !pending.is_empty() {
                    dynamic_admission = Some(port.clone().with_pending(std::rc::Rc::new(pending)));
                }
                (
                    admitted.execution_bytecode,
                    admitted.result,
                    admitted.runtime,
                    Some(admitted.publication),
                )
            }
        };
        let execution = DynamicEvalRuntime {
            runtime,
            admission: dynamic_admission,
        };
        let (outcome, updated_workspace, effects) = match frame {
            DynamicEvalInput::Frame(frame) => {
                let frame =
                    frame.prepare_compiled_extent(&execution.runtime, bytecode.var_count)?;
                let (_frame_roots, frame) = frame.into_parts();
                interpret_dynamic_workspace_eval(
                    bytecode,
                    frame.vars,
                    frame.names,
                    frame.assigned,
                    frame.global_aliases,
                    source_label,
                    execution,
                )
                .await?
            }
            DynamicEvalInput::Admitted(frame) => {
                interpret_admitted_dynamic_workspace_eval(bytecode, frame, source_label, execution)
                    .await?
            }
        };
        Ok(Some(DynamicEvalWorkResult {
            result,
            outcome,
            updated_workspace,
            effects,
            publication,
            _source_lease: source_lease,
        }))
    };
    // A VM instruction already owns a portable ProcessContinuationDriver on
    // native and WASM. Submit the whole parse/compile/admit/interpret request
    // to that driver, not merely its final interpreter phase. Bare callers
    // retain the existing canonical local-host/direct placement.
    let completed = if let Some(driver) = driver {
        let queued_scope = work_runtime.prepare_poll_scope()?;
        let queued_target = target.clone();
        driver
            .enqueue_owned_scoped_in_context(&work_runtime, queued_scope, move || {
                queued_target.scope(work)
            })?
            .await?
    } else {
        let work = target.scope(work);
        runmat_runtime::context::run_owned_request_continuation(&continuation_runtime, work).await?
    };
    let Some(DynamicEvalWorkResult {
        result: captured_result,
        outcome,
        updated_workspace,
        effects,
        publication,
        _source_lease,
    }) = completed
    else {
        return runmat_runtime::execution::RootedValue::new(Value::output_list(Vec::new()))
            .map_err(|error| error.into_runtime_error());
    };
    let result = match outcome {
        Ok(crate::interpreter::api::InterpreterOutcome::Completed(completion)) => completion
            .with_owned(|completion| {
                let vars = completion.values;
                // Target completion deliberately trims the child's private
                // compiler slots. The result slot is one of those slots, and
                // `vars` shares its backing with the workspace snapshot.
                // Root the selected value before committing that trim, while
                // retaining the existing single public-result Value clone.
                let value = if requested_outputs == 0 {
                    Ok(Value::output_list(Vec::new()))
                } else if let Some(result) = captured_result {
                    vars.get(result.slot)
                        .map(|value| value.clone())
                        .ok_or_else(|| {
                            mex(
                                "DynamicWorkspaceResult",
                                "selected dynamic workspace result slot is missing",
                            )
                        })
                } else if empty_value_when_no_result {
                    Ok(Value::output_list(Vec::new()))
                } else {
                    Ok(Value::Num(0.0))
                };
                let rooted_result = value.and_then(|value| {
                    runmat_runtime::execution::RootedValue::new(value)
                        .map_err(|error| error.into_runtime_error())
                });
                if let Some(snapshot) = updated_workspace {
                    let (_workspace_roots, snapshot, report) = snapshot.into_parts();
                    target
                        .commit(&continuation_runtime, snapshot, report)
                        .map_err(|err| dynamic_workspace_commit_error(builtin, err))?;
                }
                let result = rooted_result?;
                Ok::<_, RuntimeError>(result)
            })?,
        Err(err) => {
            if commit_workspace_on_error {
                if let Some(snapshot) = updated_workspace {
                    let (_workspace_roots, snapshot, report) = snapshot.into_parts();
                    target
                        .commit(&continuation_runtime, snapshot, report)
                        .map_err(|err| dynamic_workspace_commit_error(builtin, err))?;
                }
            }
            let replay = effects.replay();
            drop(publication);
            return runmat_runtime::runtime_error::complete_with_cleanup(Err(err), replay);
        }
    };
    effects.replay()?;
    // The source's earlier effects precede its captured final expression.
    // Use the same provider-aware emitter as ordinary statement output while
    // the rooted result and workspace publication remain live.
    let emission = if requested_outputs != 0
        && captured_result.is_some_and(|result| result.source_display)
        && terminal_output == DynamicTerminalOutput::Capture
    {
        continuation_runtime
            .scope(runmat_runtime::console::emit_statement_value(
                &continuation_runtime,
                None,
                result.value(),
            ))
            .await
    } else {
        Ok(())
    };
    drop(publication);
    emission?;
    Ok(result)
}

fn dynamic_source_context(
    runtime: &runmat_runtime::context::RuntimeContext,
    name: impl Into<String>,
    text: &str,
    lexical_source_id: Option<runmat_hir::SourceId>,
) -> Result<runmat_runtime::source_context::RuntimeSourceLease, RuntimeError> {
    let name = std::sync::Arc::<str>::from(name.into());
    runmat_runtime::source_context::lease_nested_source_context_in_context(
        runtime,
        name,
        text,
        lexical_source_id,
    )
    .map_err(dynamic_source_identity_error)
}

fn dynamic_source_identity_error(
    error: runmat_runtime::source_context::NestedSourceIdentityError,
) -> RuntimeError {
    runmat_runtime::build_runtime_error(error.to_string())
        .with_identifier("RunMat:DynamicSourceIdentity")
        .with_source(error)
        .build()
}

#[derive(Default)]
struct DynamicThreadEffects {
    console: Vec<runmat_runtime::console::ConsoleEntry>,
    warnings: Vec<runmat_runtime::warning_store::RuntimeWarning>,
    recent_figures: Vec<u32>,
}

struct DynamicEvalWorkResult {
    result: Option<DynamicEvalResult>,
    outcome: Result<InterpreterOutcome, RuntimeError>,
    updated_workspace: Option<RootedDynamicWorkspaceSnapshot>,
    effects: DynamicThreadEffects,
    publication: Option<Box<dyn crate::interpreter::dynamic_admission::DynamicEvalPublication>>,
    _source_lease: Option<runmat_runtime::source_context::RuntimeSourceLease>,
}

// The interpreter detaches this snapshot from its frame. Keep its existing
// Value storage rooted across the owned-request response and the parent-side
// commit, including the error-commit route, without cloning the workspace.
struct RootedDynamicWorkspaceSnapshot {
    state: WorkspaceValueSnapshot,
    report: WorkspaceAssignedReport,
    roots: runmat_runtime::execution::RootedValueLease,
}

// The caller's workspace frame can be cleared while a submitted sibling is
// queued. Root the one existing input copy before submission, including the
// never-polled cancellation path; there is no second workspace copy.
struct RootedDynamicEvalInput {
    frame: WorkspaceTargetSnapshot,
    roots: runmat_runtime::execution::RootedValueLease,
}

impl RootedDynamicEvalInput {
    fn new(frame: WorkspaceTargetSnapshot) -> Result<Self, RuntimeError> {
        let roots = runmat_runtime::execution::RootedValueLease::for_traced(&frame.vars)
            .map_err(|error| error.into_runtime_error())?;
        Ok(Self { frame, roots })
    }

    fn prepare_compiled_extent(
        mut self,
        context: &runmat_runtime::context::RuntimeContext,
        extent: usize,
    ) -> Result<Self, RuntimeError> {
        let vars = std::mem::replace(
            &mut self.frame.vars,
            crate::interpreter::vars::InterpreterVars::from_owned(Vec::new()),
        );
        self.frame.vars = vars.prepare_extent(context, extent)?;
        Ok(self)
    }

    fn into_parts(
        self,
    ) -> (
        runmat_runtime::execution::RootedValueLease,
        WorkspaceTargetSnapshot,
    ) {
        (self.roots, self.frame)
    }
}

impl RootedDynamicWorkspaceSnapshot {
    fn new(
        state: WorkspaceValueSnapshot,
        report: WorkspaceAssignedReport,
    ) -> Result<Self, RuntimeError> {
        let roots = runmat_runtime::execution::RootedValueLease::for_traced(&state.vars)
            .map_err(|error| error.into_runtime_error())?;
        Ok(Self {
            state,
            report,
            roots,
        })
    }

    fn into_parts(
        self,
    ) -> (
        runmat_runtime::execution::RootedValueLease,
        WorkspaceValueSnapshot,
        WorkspaceAssignedReport,
    ) {
        (self.roots, self.state, self.report)
    }
}

impl DynamicThreadEffects {
    fn replay(self) -> Result<(), RuntimeError> {
        runmat_runtime::console::append_thread_buffer(self.console)?;
        runmat_runtime::warning_store::extend(self.warnings);
        runmat_runtime::plotting_hooks::record_recent_figures(self.recent_figures);
        Ok(())
    }
}

struct DynamicEvalRuntime {
    runtime: runmat_runtime::context::RuntimeContext,
    admission: Option<crate::interpreter::dynamic_admission::DynamicEvalAdmissionPort>,
}

async fn interpret_dynamic_workspace_eval(
    bytecode: Bytecode,
    target_vars: crate::interpreter::vars::InterpreterVars,
    names: HashMap<String, usize>,
    assigned: HashSet<String>,
    global_aliases: HashMap<usize, runmat_runtime::workspace::bindings::FrameGlobalBinding>,
    source_label: String,
    selected: DynamicEvalRuntime,
) -> Result<
    (
        Result<InterpreterOutcome, RuntimeError>,
        Option<RootedDynamicWorkspaceSnapshot>,
        DynamicThreadEffects,
    ),
    RuntimeError,
> {
    let admission =
        crate::runtime::workspace::transactional_workspace_admission_with_global_bindings(
            names,
            assigned,
            runmat_runtime::workspace::bindings::FrameGlobalBindings::from_snapshot(global_aliases),
        );
    interpret_dynamic_workspace_eval_with_admission(
        bytecode,
        target_vars,
        source_label,
        selected,
        admission,
    )
    .await
}

async fn interpret_admitted_dynamic_workspace_eval(
    bytecode: Bytecode,
    input: AdmittedDynamicInput,
    source_label: String,
    selected: DynamicEvalRuntime,
) -> Result<
    (
        Result<InterpreterOutcome, RuntimeError>,
        Option<RootedDynamicWorkspaceSnapshot>,
        DynamicThreadEffects,
    ),
    RuntimeError,
> {
    let input = input.prepare_compiled_extent(&selected.runtime, bytecode.var_count)?;
    let (vars, metadata) = input.into_execution_parts();
    interpret_dynamic_workspace_eval_with_admission(
        bytecode,
        vars,
        source_label,
        selected,
        crate::runtime::workspace::PendingWorkspaceAdmission::admitted(metadata),
    )
    .await
}

async fn interpret_dynamic_workspace_eval_with_admission(
    bytecode: Bytecode,
    target_vars: crate::interpreter::vars::InterpreterVars,
    source_label: String,
    selected: DynamicEvalRuntime,
    admission: crate::runtime::workspace::PendingWorkspaceAdmission,
) -> Result<
    (
        Result<InterpreterOutcome, RuntimeError>,
        Option<RootedDynamicWorkspaceSnapshot>,
        DynamicThreadEffects,
    ),
    RuntimeError,
> {
    let execution = crate::interpreter::runner::interpret_workspace(
        &bytecode,
        target_vars,
        Some(&source_label),
        selected.runtime,
        Some(admission),
        selected.admission,
    )
    .await;
    let mut outcome = execution.outcome;
    let updated_workspace = match execution.workspace {
        Some(completion) => {
            match RootedDynamicWorkspaceSnapshot::new(completion.state, completion.assigned) {
                Ok(snapshot) => Some(snapshot),
                Err(error) => {
                    outcome =
                        runmat_runtime::runtime_error::complete_with_cleanup(outcome, Err(error));
                    None
                }
            }
        }
        None => None,
    };
    Ok((outcome, updated_workspace, DynamicThreadEffects::default()))
}

fn dynamic_workspace_commit_error(builtin: &str, source: RuntimeError) -> RuntimeError {
    runmat_runtime::build_runtime_error(format!("{builtin}: {source}"))
        .with_identifier("RunMat:DynamicWorkspaceUnavailable")
        .with_source(source)
        .build()
}

fn commit_dynamic_workspace_target(
    runtime: &runmat_runtime::context::RuntimeContext,
    identity: WorkspaceTargetIdentity,
    explicit_base: Option<&dynamic_base_target::ExplicitBaseTarget>,
    snapshot: WorkspaceValueSnapshot,
    report: WorkspaceAssignedReport,
) -> Result<(), RuntimeError> {
    if let Some(base) = explicit_base {
        identity.ensure_current(runtime)?;
        base.commit(runtime, snapshot, report)
    } else {
        identity.commit(runtime, snapshot, report)
    }
}

#[cfg(test)]
fn compile_dynamic_workspace_eval(
    builtin: &str,
    source: &str,
    workspace_bindings: &HashMap<String, usize>,
    function_registry: &FunctionRegistry,
    source_ref: Option<runmat_runtime::source_context::RuntimeSourceRef>,
    requested_outputs: usize,
) -> Result<Option<(Bytecode, Option<DynamicEvalResult>)>, RuntimeError> {
    let ast = parse_dynamic_eval(builtin, source)?;
    compile_dynamic_workspace_eval_inner(DynamicEvalCompileRequest {
        builtin,
        ast: &ast,
        workspace_bindings,
        workspace_value_count: workspace_binding_extent(workspace_bindings)?,
        lowering_bindings: workspace_bindings,
        function_registry,
        source_ref,
        requested_outputs,
        result_capture: DynamicEvalResultCapture::TerminalExpression(
            DynamicTerminalOutput::Preserve,
        ),
        select_product: None,
        selected_sources: eval_static_selection::EvalSelectedSources::default(),
    })
    .map(|compiled| {
        compiled.map(|compiled| match compiled {
            DynamicEvalCompiled::Ordinary(bytecode, result) => (*bytecode, result),
            DynamicEvalCompiled::Selected(_) => unreachable!("ordinary eval requested no product"),
        })
    })
}

/// The selected-product route retains both canonical and workspace-execution
/// views from one compilation. Ordinary eval uses the move-only route above.
pub fn compile_dynamic_workspace_eval_product(
    builtin: &str,
    source: &str,
    workspace_bindings: &HashMap<String, usize>,
    function_registry: &FunctionRegistry,
    source_ref: Option<runmat_runtime::source_context::RuntimeSourceRef>,
    requested_outputs: usize,
) -> Result<Option<DynamicEvalCompileProduct>, RuntimeError> {
    let ast = parse_dynamic_eval(builtin, source)?;
    compile_dynamic_workspace_eval_inner(DynamicEvalCompileRequest {
        builtin,
        ast: &ast,
        workspace_bindings,
        workspace_value_count: workspace_binding_extent(workspace_bindings)?,
        lowering_bindings: workspace_bindings,
        function_registry,
        source_ref,
        requested_outputs,
        result_capture: DynamicEvalResultCapture::TerminalExpression(
            DynamicTerminalOutput::Preserve,
        ),
        select_product: Some(&|_, _| Ok(true)),
        selected_sources: eval_static_selection::EvalSelectedSources::default(),
    })
    .map(|compiled| {
        compiled.map(|compiled| match compiled {
            DynamicEvalCompiled::Selected(product) => *product,
            DynamicEvalCompiled::Ordinary(_, _) => unreachable!("selected eval requested product"),
        })
    })
}

fn compile_dynamic_workspace_eval_inner(
    request: DynamicEvalCompileRequest<'_>,
) -> Result<Option<DynamicEvalCompiled>, RuntimeError> {
    compile_dynamic_workspace_eval_inner_typed(request).map_err(Into::into)
}

fn compile_dynamic_workspace_eval_inner_typed(
    request: DynamicEvalCompileRequest<'_>,
) -> Result<Option<DynamicEvalCompiled>, DynamicEvalCompileFailure> {
    let DynamicEvalCompileRequest {
        builtin,
        ast,
        workspace_bindings,
        workspace_value_count,
        lowering_bindings,
        function_registry,
        source_ref,
        requested_outputs,
        result_capture,
        select_product,
        selected_sources,
    } = request;
    let options = current_dynamic_eval_options();
    let mut function_output_arities = function_registry
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
    for source in &selected_sources.sources {
        if function_output_arities
            .insert(
                source.alias,
                runmat_hir::FunctionOutputArity::new(source.fixed_outputs, source.has_varargout),
            )
            .is_some()
        {
            return Err(mex(
                "DynamicWorkspaceCompile",
                "selected source alias collides with an existing function signature",
            )
            .into());
        }
    }
    for source in &selected_sources.external_sources {
        let (fixed_outputs, has_varargout) = source.target.output_arity();
        if function_output_arities
            .insert(
                source.alias,
                runmat_hir::FunctionOutputArity::new(fixed_outputs, has_varargout),
            )
            .is_some()
        {
            return Err(mex(
                "DynamicWorkspaceCompile",
                "selected external source alias collides with a function signature",
            )
            .into());
        }
    }
    let mut bound_names = function_registry.names.clone();
    bound_names.extend(
        selected_sources
            .names
            .iter()
            .map(|(name, id)| (name.clone(), *id)),
    );
    let lowering = runmat_hir::lower(
        ast,
        &runmat_hir::LoweringContext::new(lowering_bindings)
            .with_bound_functions(&bound_names)
            .with_function_output_arities(&function_output_arities)
            .with_runmat_extensions_enabled(options.runmat_extensions_enabled)
            .with_top_level_await_enabled(options.top_level_await_enabled),
    )
    .map_err(|err| DynamicEvalCompileFailure {
        source_kind: Some(err.kind),
        error: mex(
            "DynamicWorkspaceLower",
            &format!("{builtin}: lowering error: {err}"),
        ),
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
    let result_capture =
        prepare_dynamic_result_capture(result_capture, &lowering.assembly, entrypoint.target)?;
    let mut bytecode =
        crate::bytecode::compile(&lowering.assembly, &mir, entrypoint.id).map_err(|err| {
            mex(
                "DynamicWorkspaceCompile",
                &format!("{builtin}: compile error: {err}"),
            )
        })?;
    // Ask only after canonical compilation, before workspace/source rewriting.
    // Numeric ordinary eval keeps the move-only path and never clones a full
    // instruction/function registry merely because a host port is installed.
    let retain_product = select_product
        .map(|select| select(&bytecode, function_registry))
        .transpose()?
        .unwrap_or(false);
    if !retain_product
        && (!selected_sources.sources.is_empty()
            || !selected_sources.external_sources.is_empty()
            || !selected_sources.natives.is_empty())
    {
        return Err(mex(
            "DynamicWorkspaceCompile",
            "selected eval source or native artifact has no product admission",
        )
        .into());
    }
    let canonical_bytecode = retain_product.then(|| bytecode.clone());
    if source_ref.is_some() {
        bytecode.source_id = source_ref.map(crate::BytecodeSourceRef::Runtime);
    }
    let function_remap =
        merge_eval_function_registry(&mut bytecode, function_registry).map_err(|error| {
            mex(
                "DynamicWorkspaceCompile",
                &format!("{builtin}: eval function remap: {error}"),
            )
        })?;
    // The ordinary move-only execution product needs the same function-layout
    // rebinding as the selected decorated product. Otherwise its retained
    // layout still names compiler-local bodies after the registry is merged.
    if let Some(layout) = &mut bytecode.layout {
        crate::layout::remap_layout_function_ids(layout, &function_remap).map_err(|error| {
            mex(
                "DynamicWorkspaceCompile",
                &format!("{builtin}: eval layout remap: {error:?}"),
            )
        })?;
    }
    let workspace_extent = workspace_value_count.max(workspace_binding_extent(workspace_bindings)?);
    let workspace_slot_remap =
        remap_dynamic_workspace_slots(&mut bytecode, workspace_bindings, workspace_extent)?;
    let result = retain_dynamic_result_slot(
        &mut bytecode,
        requested_outputs,
        workspace_extent,
        result_capture,
    )?;
    Ok(Some(match canonical_bytecode {
        Some(canonical_bytecode) => {
            DynamicEvalCompiled::Selected(Box::new(DynamicEvalCompileProduct {
                compat_mode: options.compat_mode,
                hir: lowering.assembly,
                mir,
                canonical_bytecode,
                execution_bytecode: bytecode,
                function_remap,
                workspace_slot_remap,
                result,
                selected_sources: selected_sources.sources,
                selected_external_sources: selected_sources.external_sources,
                selected_natives: selected_sources.natives,
                workspace_extent,
                requested_outputs,
                result_capture,
                source_ref,
            }))
        }
        None => DynamicEvalCompiled::Ordinary(Box::new(bytecode), result),
    }))
}

/// Recompile one pending closure only after its creation opcode is reached.
/// This is a compiler product, never a second execution of the eval script.
/// Core admits only the checked function body, not the surrounding script or
/// sibling definitions. The live frame contributes name/assignment metadata
/// only; no workspace Value is copied or interpreted here.
pub(crate) struct CompiledPendingClosure {
    // `DynamicEvalCompiled::Selected` already owns this allocation. Keep it
    // boxed through the nested VM poll and Core admission instead of moving
    // two complete Bytecode views onto the interpreter thread's stack.
    pub product: Box<DynamicEvalCompileProduct>,
    pub captures: Vec<eval_hir_sites::SelectedClosureCapture>,
    pub issuing_frame: crate::runtime::workspace::WorkspaceFrameId,
}

pub(crate) async fn compile_pending_closure(
    plan: &PendingClosurePlan,
    function: &eval_hir_sites::PendingClosureFunction,
    runtime: &runmat_runtime::context::RuntimeContext,
    execution_registry: &FunctionRegistry,
    execution_id: FunctionId,
    admission: &crate::interpreter::dynamic_admission::DynamicEvalAdmissionPort,
) -> Result<CompiledPendingClosure, RuntimeError> {
    if plan
        .function(execution_id)
        .is_none_or(|selected| !std::ptr::eq(selected, function))
        || execution_registry
            .get_current(execution_id)
            .is_none_or(|body| {
                !std::sync::Arc::ptr_eq(&body.definition_owner, &function.execution_owner)
                    || body.source_id != function.execution_source
            })
    {
        return Err(mex(
            "DynamicWorkspaceCompile",
            "pending closure differs from its running definition",
        ));
    }
    if current_dynamic_eval_options() != plan.options {
        return Err(mex(
            "DynamicWorkspaceCompile",
            "eval language options changed before closure issuance",
        ));
    }
    let crate::runtime::workspace::WorkspaceBindingState {
        frame: issuing_frame,
        names: workspace_bindings,
        assigned,
        value_count: workspace_value_count,
    } = crate::runtime::workspace::workspace_binding_state().ok_or_else(|| {
        mex(
            "DynamicWorkspaceUnavailable",
            "pending eval closure has no live workspace frame",
        )
    })?;
    let lowering_bindings = workspace_bindings
        .iter()
        .filter(|(name, _)| assigned.contains(*name))
        .map(|(name, slot)| (name.clone(), *slot))
        .collect::<HashMap<_, _>>();
    let selected_sources = eval_static_selection::select_sites(
        &function.sites,
        plan.alias_floor,
        runtime,
        &plan.inherited,
        &lowering_bindings,
        admission,
    )
    .await?;
    let compiled = compile_dynamic_workspace_eval_inner(DynamicEvalCompileRequest {
        builtin: "eval",
        ast: plan.ast.as_ref(),
        workspace_bindings: &workspace_bindings,
        workspace_value_count,
        lowering_bindings: &lowering_bindings,
        function_registry: &plan.inherited,
        source_ref: plan.source_ref,
        requested_outputs: plan.requested_outputs,
        result_capture: DynamicEvalResultCapture::TerminalExpression(
            DynamicTerminalOutput::Preserve,
        ),
        select_product: Some(&|_, _| Ok(true)),
        selected_sources,
    })?
    .ok_or_else(|| {
        mex(
            "DynamicWorkspaceCompile",
            "pending eval closure has no compiler product",
        )
    })?;
    let DynamicEvalCompiled::Selected(product) = compiled else {
        return Err(mex(
            "DynamicWorkspaceCompile",
            "pending eval closure was not selected",
        ));
    };
    for sibling in &function.siblings {
        if eval_hir_sites::sibling_hir_digest(&product.hir, sibling.local_function)
            != Some(sibling.hir_digest)
        {
            return Err(mex(
                "DynamicWorkspaceCompile",
                "pending closure Local dependency changed before issuance",
            ));
        }
    }
    let body = product
        .canonical_bytecode
        .function_registry
        .get_current(function.local_id)
        .ok_or_else(|| {
            mex(
                "DynamicWorkspaceCompile",
                "pending closure body disappeared",
            )
        })?;
    let captures = function
        .abi
        .capture_plan(&product.hir, body, &lowering_bindings)
        .ok_or_else(|| {
            mex(
                "DynamicWorkspaceCompile",
                "pending closure compiler ABI changed before issuance",
            )
        })?;
    if captures.len() != body.capture_slots.len() {
        return Err(mex(
            "DynamicWorkspaceCompile",
            "pending closure compiler ABI changed before issuance",
        ));
    }
    Ok(CompiledPendingClosure {
        product,
        captures,
        issuing_frame,
    })
}

fn remap_dynamic_workspace_slots(
    bytecode: &mut Bytecode,
    workspace_bindings: &dyn runmat_hir::WorkspaceBindingView,
    workspace_extent: usize,
) -> Result<HashMap<usize, usize>, RuntimeError> {
    let mut alias_slots = HashSet::with_capacity(workspace_bindings.len());
    let mut duplicate_slot = false;
    workspace_bindings.visit(&mut |_, slot| {
        if !alias_slots.insert(slot) {
            duplicate_slot = true;
        }
    });
    if duplicate_slot {
        return Err(mex(
            "DynamicWorkspaceCompile",
            "dynamic workspace names share a target slot",
        ));
    }
    if bytecode.var_count == 0 {
        // Keep the type table sparse rather than allocating one Unknown entry
        // per inherited slot. Type consumers canonically treat a missing
        // entry as Unknown; there is no compiler-local type to remap here.
        bytecode.var_count = workspace_extent;
        return Ok(HashMap::new());
    }

    let original_var_count = bytecode.var_count;
    let original_var_names = &bytecode.var_names;
    let mut next_slot = workspace_extent;
    let mut slot_remap = HashMap::new();

    // Map language-visible names before compiler-only storage. New dynamic
    // names thereby occupy the target frame's next slots, while temporaries
    // remain a trailing execution detail that completion can retire.
    let named = (0..original_var_count).filter(|slot| original_var_names.contains_key(slot));
    let unnamed = (0..original_var_count).filter(|slot| !original_var_names.contains_key(slot));
    for old_slot in named.chain(unnamed) {
        let new_slot = original_var_names
            .get(&old_slot)
            .and_then(|name| workspace_bindings.get(name))
            .map(Ok)
            .unwrap_or_else(|| {
                let slot = next_slot;
                next_slot = next_slot
                    .checked_add(1)
                    .ok_or_else(dynamic_workspace_slot_overflow)?;
                Ok::<usize, RuntimeError>(slot)
            })?;
        slot_remap.insert(old_slot, new_slot);
    }

    remap_dynamic_workspace_slots_with_plan(bytecode, &slot_remap, workspace_extent)?;
    Ok(slot_remap)
}

/// Replay the exact compiler-derived workspace projection on a separately
/// admitted copy of the same canonical bytecode. The selected product retains
/// this numeric plan rather than cloning the source workspace name map.
fn remap_dynamic_workspace_slots_with_plan(
    bytecode: &mut Bytecode,
    slot_remap: &HashMap<usize, usize>,
    workspace_extent: usize,
) -> Result<(), RuntimeError> {
    let original_var_count = bytecode.var_count;
    if slot_remap.len() != original_var_count
        || (0..original_var_count).any(|slot| !slot_remap.contains_key(&slot))
    {
        return Err(mex(
            "DynamicWorkspaceCompile",
            "admitted eval workspace layout differs from the compiler projection",
        ));
    }
    let mut projected_slots = HashSet::with_capacity(slot_remap.len());
    if slot_remap
        .values()
        .any(|slot| !projected_slots.insert(*slot))
    {
        return Err(mex(
            "DynamicWorkspaceCompile",
            "admitted eval workspace projection aliases compiler slots",
        ));
    }

    if original_var_count == 0 {
        bytecode.var_count = workspace_extent;
        return Ok(());
    }

    let original_var_names = std::mem::take(&mut bytecode.var_names);
    let original_initially_unassigned_slots =
        std::mem::take(&mut bytecode.initially_unassigned_slots);

    for instr in &mut bytecode.instructions {
        remap_instr_slots(instr, slot_remap);
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

    let new_var_count = match slot_remap.values().copied().max() {
        Some(slot) => workspace_extent.max(
            slot.checked_add(1)
                .ok_or_else(dynamic_workspace_slot_overflow)?,
        ),
        None => workspace_extent.max(original_var_count),
    };
    let old_var_types = std::mem::take(&mut bytecode.var_types);
    let mut new_var_types = vec![runmat_builtins::Type::Unknown; new_var_count];
    for (old_slot, new_slot) in slot_remap {
        if let (Some(old_ty), Some(new_ty)) = (
            old_var_types.get(*old_slot),
            new_var_types.get_mut(*new_slot),
        ) {
            *new_ty = old_ty.clone();
        }
    }
    bytecode.var_types = new_var_types;
    bytecode.var_count = new_var_count;
    Ok(())
}

fn workspace_binding_extent(
    workspace_bindings: &dyn runmat_hir::WorkspaceBindingView,
) -> Result<usize, RuntimeError> {
    let mut maximum = None;
    workspace_bindings.visit(&mut |_, slot| {
        maximum = Some(maximum.map_or(slot, |current: usize| current.max(slot)));
    });
    match maximum {
        Some(slot) => slot
            .checked_add(1)
            .ok_or_else(dynamic_workspace_slot_overflow),
        None => Ok(0),
    }
}

fn dynamic_workspace_slot_overflow() -> RuntimeError {
    mex(
        "DynamicWorkspaceCompile",
        "dynamic workspace slot extent exceeds the supported range",
    )
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

fn remap_instr_slots(instr: &mut Instr, slot_remap: &HashMap<usize, usize>) {
    match instr {
        Instr::LoadVar(slot)
        | Instr::LoadVarForIndexAssignment(slot)
        | Instr::LoadVarForCellAssignment(slot)
        | Instr::LoadVarForStructAssignment(slot)
        | Instr::StoreVar(slot) => {
            remap_slot(slot, slot_remap);
        }
        Instr::CallFevalMultiUsingOutputSlot(_, slot)
        | Instr::CallFevalExpandMultiOutputUsingOutputSlot(_, slot)
        | Instr::CallBuiltinMultiUsingOutputSlot(_, _, slot)
        | Instr::CallBuiltinExpandMultiOutputUsingOutputSlot(_, _, slot)
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

fn merge_eval_function_registry(
    bytecode: &mut Bytecode,
    parent_registry: &FunctionRegistry,
) -> Result<HashMap<FunctionId, FunctionId>, String> {
    let (ids, remap) = eval_function_merge_plan(bytecode, parent_registry);
    merge_eval_function_registry_with_plan(bytecode, parent_registry, ids, &remap)?;
    Ok(remap)
}

fn eval_function_merge_plan(
    bytecode: &Bytecode,
    parent_registry: &FunctionRegistry,
) -> (Vec<FunctionId>, HashMap<FunctionId, FunctionId>) {
    let mut ids = bytecode
        .function_registry
        .functions
        .keys()
        .copied()
        .collect::<Vec<_>>();
    ids.sort_by_key(|id| id.0);
    let remap = eval_function_id_remap(&ids, bytecode.layout.as_ref(), parent_registry);
    (ids, remap)
}

fn merge_eval_function_registry_with_plan(
    bytecode: &mut Bytecode,
    parent_registry: &FunctionRegistry,
    ids: Vec<FunctionId>,
    remap: &HashMap<FunctionId, FunctionId>,
) -> Result<(), String> {
    let eval_registry = bytecode.function_registry.clone();
    let mut merged = parent_registry.clone();
    for instr in &mut bytecode.instructions {
        crate::bytecode::program::function_remap::remap_eval_local_instruction(instr, remap);
    }
    for id in ids {
        if let Some(function) = eval_registry.functions.get(&id) {
            let mut function = function.clone();
            crate::bytecode::program::function_remap::remap_eval_local_function(
                &mut function,
                remap,
            )?;
            merged.insert_replacing_name(function);
        }
    }
    bytecode.bound_functions = merged.functions.clone();
    bytecode.function_registry = merged;
    Ok(())
}

fn eval_function_id_remap(
    eval_ids: &[FunctionId],
    layout: Option<&crate::layout::VmAssemblyLayout>,
    parent_registry: &FunctionRegistry,
) -> HashMap<FunctionId, FunctionId> {
    let mut used_ids = parent_registry
        .functions
        .keys()
        .map(|id| id.0)
        .collect::<HashSet<_>>();
    // The retained layout also includes script/entrypoint functions that are
    // not published into the eval-local registry. A local body cannot be
    // remapped onto one of those IDs even when the parent registry is empty.
    if let Some(layout) = layout {
        let eval_ids = eval_ids.iter().copied().collect::<HashSet<_>>();
        used_ids.extend(
            layout
                .functions
                .keys()
                .filter(|id| !eval_ids.contains(*id))
                .map(|id| id.0),
        );
    }
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

pub enum ImportedBuiltinResolution {
    Resolved(runmat_runtime::execution::RootedValueSequence),
    Ambiguous(RuntimeError),
    NotFound,
}

/// A typed no-target result, not a body's error text, authorizes import
/// selection. Every selected direct or imported implementation runs once.
pub async fn call_builtin_or_imported(
    runtime: &runmat_runtime::context::RuntimeContext,
    name: &str,
    imports: &[(Vec<String>, bool)],
    args: &[Value],
    requested_outputs: usize,
) -> Result<runmat_runtime::execution::RootedValueSequence, RuntimeError> {
    if let Some(result) =
        crate::call::try_call_builtin_async_with_outputs(runtime, name, args, requested_outputs)
            .await?
    {
        return Ok(result);
    }
    match resolve_imported_builtin(runtime, name, imports, args, requested_outputs).await? {
        ImportedBuiltinResolution::Resolved(result) => Ok(result),
        ImportedBuiltinResolution::Ambiguous(error) => Err(error),
        ImportedBuiltinResolution::NotFound => {
            Err(runmat_runtime::dispatcher::undefined_builtin_error(name))
        }
    }
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
    runtime: &runmat_runtime::context::RuntimeContext,
    name: &str,
    imports: &[(Vec<String>, bool)],
    args: &[Value],
    requested_outputs: usize,
) -> Result<ImportedBuiltinResolution, RuntimeError> {
    let _runtime_scope = runtime.enter();
    for wildcard_tier in [false, true] {
        let mut seen = HashSet::new();
        let mut matches = Vec::new();
        for (path, wildcard) in imports {
            if *wildcard != wildcard_tier || (wildcard_tier && path.is_empty()) {
                continue;
            }
            if !wildcard_tier && path.last().map(String::as_str) != Some(name) {
                continue;
            }
            let Some(qualified) =
                imported_builtin_qualified_name(path, wildcard_tier.then_some(name))
            else {
                continue;
            };
            let Some(target) = runmat_runtime::dispatcher::observe_builtin_dispatch_target(
                runtime, &qualified, args,
            )?
            else {
                continue;
            };
            if seen.insert(target.identity()) {
                matches.push(qualified);
            }
        }
        if matches.len() > 1 {
            let tier = if wildcard_tier {
                "wildcard imports"
            } else {
                "imports"
            };
            return Ok(ImportedBuiltinResolution::Ambiguous(mex(
                "AmbiguousBuiltinImport",
                &format!(
                    "ambiguous builtin '{}' via {}: {}",
                    name,
                    tier,
                    matches.join(", ")
                ),
            )));
        }
        if let Some(qualified) = matches.pop() {
            return crate::call::call_builtin_async_with_outputs(
                runtime,
                &qualified,
                args,
                requested_outputs,
            )
            .await
            .map(ImportedBuiltinResolution::Resolved);
        }
    }
    Ok(ImportedBuiltinResolution::NotFound)
}

#[cfg(test)]
mod tests {
    use super::{
        compile_dynamic_workspace_eval, compile_dynamic_workspace_eval_product,
        current_dynamic_eval_options, dynamic_source_context, imported_builtin_qualified_name,
        merge_eval_function_registry, push_dynamic_eval_options, remap_dynamic_workspace_slots,
        retain_dynamic_result_slot, workspace_text_arg, DynamicEvalResult,
        PreparedDynamicEvalResultCapture,
    };
    use crate::bytecode::{FunctionBytecode, FunctionRegistry, Instr};
    use crate::{Bytecode, DynamicEvalCompileProduct};
    use runmat_hir::FunctionId;
    use runmat_parser::CompatMode;
    use runmat_types::{CallableFallbackPolicy, CallableIdentity};
    use runmat_value::{CharArray, Utf16CodeUnit, Utf16DecodeError, Value};
    use std::collections::HashMap;

    #[test]
    fn dynamic_workspace_text_decodes_full_utf16_or_preserves_invalid_cause() {
        let text = Value::CharArray(CharArray::new_row("A🙂"));
        assert_eq!(workspace_text_arg("evalin", &text).unwrap(), "A🙂");

        let malformed = Value::CharArray(
            CharArray::from_code_units(vec![Utf16CodeUnit::from_raw(0xD800)], 1, 1)
                .expect("raw UTF-16 row"),
        );
        let error = workspace_text_arg("evalin", &malformed).unwrap_err();
        assert_eq!(
            error.identifier.as_deref(),
            Some("RunMat:DynamicWorkspaceArgumentType")
        );
        assert!(std::error::Error::source(&error)
            .is_some_and(|source| source.downcast_ref::<Utf16DecodeError>().is_some()));
    }

    fn compile_test_source(source: &str) -> crate::Bytecode {
        let ast = runmat_parser::parse(source).expect("parse test source");
        let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty())
            .expect("lower test source");
        let mir = runmat_mir::lowering::lower_assembly(&hir.assembly).expect("lower test MIR");
        crate::bytecode::compile(&hir.assembly, &mir, hir.assembly.entrypoints[0].id)
            .expect("compile test source")
    }

    fn admitted_eval_candidate(product: &DynamicEvalCompileProduct) -> Bytecode {
        // Core retains the compiler-finalized layout, including resume points.
        product.canonical_bytecode.clone()
    }

    #[test]
    fn dynamic_eval_keeps_canonical_and_workspace_execution_products_separate() {
        let bindings = HashMap::from([("value".to_string(), 7)]);
        let registry = FunctionRegistry::default();
        let product = compile_dynamic_workspace_eval_product(
            "eval",
            "value = 1;",
            &bindings,
            &registry,
            None,
            0,
        )
        .expect("compile eval once")
        .expect("nonempty eval product");
        let canonical_slot = product
            .canonical_bytecode
            .var_names
            .iter()
            .find_map(|(slot, name)| (name == "value").then_some(*slot))
            .expect("canonical value binding");
        assert_eq!(product.workspace_slot_remap.get(&canonical_slot), Some(&7));
        assert_eq!(
            product.execution_bytecode.var_names.get(&7),
            Some(&"value".into())
        );
        assert_eq!(
            product.canonical_bytecode.var_names.get(&canonical_slot),
            Some(&"value".into())
        );
        assert_eq!(product.hir.entrypoints.len(), 1);
        assert_eq!(product.mir.entrypoints.len(), 1);
        assert!(product.function_remap.is_empty());
        assert!(product.result.is_none());
        let admitted = product
            .project_admitted_bytecode(admitted_eval_candidate(&product), &registry)
            .expect("exact decorated canonical body projects to eval execution");
        assert_eq!(admitted.var_names, product.execution_bytecode.var_names);
        assert_eq!(
            serde_json::to_vec(&admitted.instructions).expect("admitted instructions"),
            serde_json::to_vec(&product.execution_bytecode.instructions)
                .expect("compiled execution instructions")
        );
        let mut wrong_unit = admitted_eval_candidate(&product);
        wrong_unit.instructions.push(Instr::Pop);
        assert!(product
            .project_admitted_bytecode(wrong_unit, &registry)
            .is_err());
        let (ordinary, ordinary_result) =
            compile_dynamic_workspace_eval("eval", "value = 1;", &bindings, &registry, None, 0)
                .expect("compile ordinary eval")
                .expect("nonempty ordinary eval");
        assert_eq!(
            serde_json::to_vec(&ordinary.instructions).expect("ordinary instruction encoding"),
            serde_json::to_vec(&product.execution_bytecode.instructions)
                .expect("selected instruction encoding")
        );
        assert_eq!(ordinary.var_names, product.execution_bytecode.var_names);
        assert_eq!(ordinary_result, product.result);

        let mut literal = compile_test_source("1;");
        let empty_bindings = HashMap::new();
        remap_dynamic_workspace_slots(&mut literal, &empty_bindings, 8)
            .expect("reserve unnamed target extent");
        let literal_result = retain_dynamic_result_slot(
            &mut literal,
            1,
            8,
            PreparedDynamicEvalResultCapture::TerminalExpression {
                source_display: false,
                output: super::DynamicTerminalOutput::Preserve,
            },
        )
        .expect("capture literal result beyond target extent");
        assert_eq!(
            literal_result,
            Some(DynamicEvalResult {
                slot: 8,
                source_display: false,
            })
        );
        assert_eq!(literal.var_count, 9);

        let mut named = compile_test_source("x = 1;");
        let named_binding = HashMap::from([("x".to_string(), 0)]);
        remap_dynamic_workspace_slots(&mut named, &named_binding, 8)
            .expect("preserve target extent around an existing named slot");
        assert_eq!(named.var_names.get(&0), Some(&"x".into()));
        assert!(named.var_count >= 8);
        assert!(named.var_types.len() >= 8);
    }

    #[test]
    fn selected_eval_projection_proves_remapped_local_body_after_parent_id_collision() {
        let parent = FunctionRegistry::new(HashMap::from([(
            FunctionId(0),
            FunctionBytecode {
                function: FunctionId(0),
                display_name: "prior".into(),
                ..Default::default()
            },
        )]));
        let source = "out=helper(1); function y=helper(x); y=x+1; end";
        let product = compile_dynamic_workspace_eval_product(
            "eval",
            source,
            &HashMap::new(),
            &parent,
            None,
            1,
        )
        .expect("compile selected eval")
        .expect("selected eval body");
        assert!(!product.function_remap.is_empty());
        let local_layout = product
            .canonical_bytecode
            .layout
            .as_ref()
            .expect("canonical eval layout");
        for reserved in local_layout
            .functions
            .keys()
            .filter(|id| !product.function_remap.contains_key(id))
        {
            assert!(
                !product
                    .function_remap
                    .values()
                    .any(|target| target == reserved),
                "published local body must not shadow a retained script frame"
            );
        }
        let admitted = product
            .project_admitted_bytecode(admitted_eval_candidate(&product), &parent)
            .expect("exact local body and remap");
        assert_eq!(admitted.layout, product.execution_bytecode.layout);
        let (ordinary, _) =
            compile_dynamic_workspace_eval("eval", source, &HashMap::new(), &parent, None, 1)
                .expect("compile ordinary eval")
                .expect("nonempty ordinary eval");
        assert_eq!(ordinary.layout, product.execution_bytecode.layout);
        assert_eq!(admitted.var_count, product.execution_bytecode.var_count);
        let mut wrong = admitted_eval_candidate(&product);
        let local = *wrong
            .function_registry
            .functions
            .keys()
            .next()
            .expect("eval-local function");
        wrong
            .function_registry
            .functions
            .get_mut(&local)
            .expect("local body")
            .instructions
            .push(Instr::LoadConst(99.0));
        assert!(product.project_admitted_bytecode(wrong, &parent).is_err());
        let mut wrong_metadata = admitted_eval_candidate(&product);
        wrong_metadata
            .function_registry
            .functions
            .get_mut(&local)
            .expect("local body")
            .output_slots
            .push(99);
        assert!(product
            .project_admitted_bytecode(wrong_metadata, &parent)
            .is_err());
        let mut wrong_script_metadata = admitted_eval_candidate(&product);
        if !wrong_script_metadata.initially_unassigned_slots.insert(0) {
            wrong_script_metadata.initially_unassigned_slots.remove(&0);
        }
        assert!(product
            .project_admitted_bytecode(wrong_script_metadata, &parent)
            .is_err());
    }

    #[test]
    fn eval_merge_preserves_parent_external_identity_when_local_id_collides() {
        let overlapping = FunctionId(7);
        let parent = FunctionRegistry::new(HashMap::from([(
            overlapping,
            FunctionBytecode {
                function: overlapping,
                display_name: "parent_helper".into(),
                ..Default::default()
            },
        )]));
        let mut bytecode = compile_test_source("value = 1;");
        bytecode.instructions = vec![Instr::CallFunctionMulti {
            identity: CallableIdentity::ExternalFunction {
                function: overlapping,
                display_name: "parent_helper".into(),
            },
            fallback_policy: CallableFallbackPolicy::ExternalBoundary,
            arg_count: 0,
            out_count: 1,
        }];
        bytecode.function_registry = FunctionRegistry::new(HashMap::from([(
            overlapping,
            FunctionBytecode {
                function: overlapping,
                display_name: "eval_local".into(),
                ..Default::default()
            },
        )]));

        merge_eval_function_registry(&mut bytecode, &parent).expect("merge eval functions");

        let Instr::CallFunctionMulti { identity, .. } = &bytecode.instructions[0] else {
            panic!("expected compiler-bound parent call");
        };
        assert_eq!(
            identity,
            &CallableIdentity::ExternalFunction {
                function: overlapping,
                display_name: "parent_helper".into(),
            }
        );
        assert_eq!(
            bytecode.function_registry.resolve_name("parent_helper"),
            Some(overlapping)
        );
        let eval_local = bytecode
            .function_registry
            .resolve_name("eval_local")
            .expect("eval-local body remains published");
        assert_ne!(eval_local, overlapping);
        assert_eq!(
            bytecode.function_registry.get(eval_local).unwrap().function,
            eval_local
        );
    }

    #[test]
    fn dynamic_eval_guard_restores_runtime_extension_policy() {
        let original = current_dynamic_eval_options();
        let original_runtime = runmat_runtime::compatibility::runmat_extensions_enabled();
        {
            let _guard = push_dynamic_eval_options(CompatMode::RunMat, true, false, false);
            assert!(current_dynamic_eval_options().runmat_extensions_enabled);
            assert!(runmat_runtime::compatibility::runmat_extensions_enabled());
        }
        assert_eq!(
            current_dynamic_eval_options().runmat_extensions_enabled,
            original.runmat_extensions_enabled
        );
        assert_eq!(
            runmat_runtime::compatibility::runmat_extensions_enabled(),
            original_runtime
        );
    }

    #[test]
    fn dynamic_eval_guard_updates_and_restores_active_runtime_policy() {
        let runtime = runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        runtime.set_language_mode(runmat_runtime::context::RuntimeLanguageMode::Strict);
        runtime.set_runmat_extensions_enabled(false);
        runtime.set_top_level_await_enabled(true);
        runtime.set_dynamic_eval_enabled(true);
        futures::executor::block_on(runtime.scope(async {
            {
                let _guard = push_dynamic_eval_options(CompatMode::RunMat, true, false, false);
                let current = current_dynamic_eval_options();
                assert_eq!(current.compat_mode, CompatMode::RunMat);
                assert!(current.runmat_extensions_enabled);
                assert!(!current.top_level_await_enabled);
                assert!(!current.dynamic_eval_enabled);
            }
            let restored = current_dynamic_eval_options();
            assert_eq!(restored.compat_mode, CompatMode::Strict);
            assert!(!restored.runmat_extensions_enabled);
            assert!(restored.top_level_await_enabled);
            assert!(restored.dynamic_eval_enabled);
        }));
    }

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
    fn dynamic_eval_has_distinct_source_and_inherits_calling_file_identity() {
        let runtime = runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        let _scope = runtime.enter();
        let source_id = runmat_hir::SourceId(41);
        let _catalog =
            runmat_runtime::source_context::replace_source_catalog_with_fullpaths(vec![(
                source_id,
                "analysis.m".to_string(),
                Some("/project/analysis.m".to_string()),
                "value = eval('mfilename()');".to_string(),
            )])
            .expect("dynamic eval source catalog admission");
        let _current = runmat_runtime::source_context::replace_current_source_id(Some(source_id));
        let dynamic = dynamic_source_context(&runtime, "<eval>", "mfilename()", Some(source_id))
            .expect("session-owned source lease");
        let info = dynamic.source_info();
        assert_eq!(info.name.as_ref(), "<eval>");
        assert_eq!(info.fullpath_name.as_deref(), None);
        assert_eq!(info.text.as_ref(), "mfilename()");
        assert_eq!(info.lexical_file_name(), "analysis.m");
        assert_eq!(
            info.lexical_file_fullpath_name(),
            Some("/project/analysis.m")
        );
        assert_ne!(dynamic.source_id(), source_id);
    }
}
