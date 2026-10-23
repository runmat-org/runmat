use crate::bytecode::instr::Instr;
#[cfg(feature = "native-accel")]
use crate::bytecode::program::Bytecode;
use crate::interpreter::timing::InterpreterTiming;
use crate::interpreter::vars::InterpreterVars;
use crate::runtime::gc::InterpretContext;
use crate::runtime::workspace::{
    has_exact_pending_workspace_service_frame, take_pending_workspace_state,
    try_set_transient_workspace_state, PendingWorkspaceInput, WorkspaceStateGuard,
};
use runmat_runtime::RuntimeError;
use runmat_value::Value;
use std::collections::{HashMap, HashSet};

pub enum InitialFrameAssignment<'a> {
    Compiled {
        count: usize,
        unassigned: &'a HashSet<usize>,
    },
    Resumed {
        assigned: &'a HashSet<usize>,
    },
}

pub fn prepare_workspace_guard(
    var_names: &HashMap<usize, String>,
    vars: &mut InterpreterVars,
    initial_assignment: InitialFrameAssignment<'_>,
    global_aliases: runmat_runtime::workspace::bindings::FrameGlobalBindings,
) -> Result<Option<WorkspaceStateGuard>, RuntimeError> {
    let pending_state = take_pending_workspace_state();
    let guard = pending_state
        .map(|pending| match pending.input {
            PendingWorkspaceInput::Legacy {
                names,
                assigned,
                global_aliases: pending_aliases,
            } => {
                global_aliases.replace_with_owned(pending_aliases);
                let filtered_assigned: HashSet<String> = assigned
                    .into_iter()
                    .filter(|name| names.contains_key(name))
                    .collect();
                crate::runtime::workspace::try_set_workspace_state_with_policy(
                    names,
                    filtered_assigned,
                    vars,
                    pending.policy,
                    pending.base_target,
                    global_aliases.clone(),
                )
            }
            PendingWorkspaceInput::Admitted(metadata) => {
                global_aliases.replace_with_owned(metadata.aliases.clone());
                crate::runtime::workspace::install_admitted_workspace_state(
                    metadata,
                    vars,
                    pending.policy,
                    pending.base_target,
                )
            }
        })
        .transpose()?;
    if guard.is_some() {
        return Ok(guard);
    }
    if var_names.is_empty() {
        if !has_exact_pending_workspace_service_frame() {
            return Ok(None);
        }
        let guard = try_set_transient_workspace_state(
            HashMap::new(),
            HashSet::new(),
            vars,
            global_aliases,
        )?;
        return Ok(Some(guard));
    }
    let mut names = HashMap::new();
    let mut assigned = HashSet::new();
    for (slot, name) in var_names {
        names.insert(name.clone(), *slot);
        let is_assigned = match &initial_assignment {
            InitialFrameAssignment::Compiled { count, unassigned } => {
                *slot < *count && !unassigned.contains(slot)
            }
            InitialFrameAssignment::Resumed { assigned } => assigned.contains(slot),
        };
        if is_assigned {
            assigned.insert(name.clone());
        }
    }
    let guard = try_set_transient_workspace_state(names, assigned, vars, global_aliases)?;
    Ok(Some(guard))
}

pub fn create_gc_context<'roots>(
    stack: &'roots crate::interpreter::stack::InterpreterStack,
    vars: &'roots InterpreterVars,
    locals: &'roots Vec<Value>,
    sequence: &'roots crate::interpreter::dispatch::SequenceState,
    thread_roots: Vec<Value>,
) -> Result<InterpretContext<'roots>, String> {
    let mut gc_context =
        InterpretContext::new_frame(stack.gc_root_owner(), vars, locals, sequence)?;
    let _ = gc_context.register_global_values(thread_roots, "thread_globals_persistents");
    Ok(gc_context)
}

pub fn debug_stack_enabled() -> bool {
    std::env::var("RUNMAT_DEBUG_STACK")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

pub fn check_cancelled(
    runtime: &runmat_runtime::context::RuntimeContext,
) -> Result<(), RuntimeError> {
    if runmat_runtime::interrupt::stop_reason()
        == Some(runmat_execution::CancellationReason::Deadline)
    {
        return Err(control_cancellation_error(
            "ExecutionDeadline",
            "Execution deadline elapsed",
        ));
    }
    if runmat_runtime::interrupt::is_cancelled() || runtime.is_cancelled() {
        return Err(cancellation_error());
    }
    Ok(())
}

pub(super) fn cancellation_error() -> RuntimeError {
    control_cancellation_error("ExecutionCancelled", "Execution cancelled by user")
}

fn control_cancellation_error(identifier: &str, message: &str) -> RuntimeError {
    let mut error = runmat_runtime::runtime_error::semantic_error_with_source(
        identifier,
        message,
        runmat_runtime::execution::ExecutionServiceError::Cancelled,
    );
    // Only the actual execution-control producer establishes this origin.
    // A language failure retaining the same source must not be reclassified.
    error.failure_origin =
        runmat_runtime::runtime_error::RuntimeFailureOrigin::ExecutionControlCancellation;
    error.gpu_gather_retry = runmat_runtime::runtime_error::GpuGatherRetry::Never;
    error
}

pub fn note_pre_dispatch(
    interpreter_timing: &mut InterpreterTiming,
    debug_stack: bool,
    pc: usize,
    instr: &Instr,
    stack_len: usize,
) {
    interpreter_timing.note_host_instr(pc);
    if debug_stack {
        log::debug!(
            "[vm] instr pc={} instr={:?} stack_len={}",
            pc,
            instr,
            stack_len
        );
    }
}

#[cfg(feature = "native-accel")]
#[inline]
pub fn fusion_debug_enabled() -> bool {
    static FLAG: once_cell::sync::OnceCell<bool> = once_cell::sync::OnceCell::new();
    *FLAG.get_or_init(|| match std::env::var("RUNMAT_DEBUG_FUSION") {
        Ok(v) => v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes"),
        Err(_) => false,
    })
}

#[cfg(feature = "native-accel")]
pub fn log_fusion_span_window(
    plan: &runmat_accelerate::FusionGroupPlan,
    bytecode: &Bytecode,
    pc: usize,
) {
    if !fusion_debug_enabled() || !log::log_enabled!(log::Level::Debug) {
        return;
    }
    if bytecode.instructions.is_empty() {
        return;
    }
    let window = 3usize;
    let span = plan.group.span.clone();
    let total = bytecode.instructions.len();
    let start = span.start.saturating_sub(window);
    let mut end = span.end + window;
    if end >= total {
        end = total.saturating_sub(1);
    }
    if end < span.end {
        end = span.end;
    }
    let mut ops: Vec<String> = Vec::new();
    for idx in start..=end {
        let instr = &bytecode.instructions[idx];
        let mut tags: Vec<&'static str> = Vec::new();
        if idx == pc {
            tags.push("pc");
        }
        if idx == span.start {
            tags.push("start");
        }
        if idx == span.end {
            tags.push("end");
        }
        let tag_str = if tags.is_empty() {
            String::new()
        } else {
            format!("<{}>", tags.join(","))
        };
        ops.push(format!("{}{} {:?}", idx, tag_str, instr));
    }
    log::debug!(
        "fusion plan {} span window [{}..{}]: {}",
        plan.index,
        start,
        end,
        ops.join(" | ")
    );
}

#[cfg(feature = "native-accel")]
pub fn note_fusion_gate(
    interpreter_timing: &mut InterpreterTiming,
    plan: &runmat_accelerate::FusionGroupPlan,
    bytecode: &Bytecode,
    pc: usize,
    has_barrier: bool,
    live_result_count: Option<usize>,
) {
    let detail = format!(
        "plan={} kind={:?} span=[{}..{}]",
        plan.index, plan.group.kind, plan.group.span.start, plan.group.span.end
    );
    interpreter_timing.flush_host_span("before_fusion", Some(detail.as_str()));
    log_fusion_span_window(plan, bytecode, pc);
    if fusion_debug_enabled() {
        log::trace!(
            "fusion gate pc={} kind={:?} span={}..{} has_barrier={} live_results={:?}",
            pc,
            plan.group.kind,
            plan.group.span.start,
            plan.group.span.end,
            has_barrier,
            live_result_count
        );
    }
}

#[cfg(feature = "native-accel")]
pub fn note_fusion_skip(pc: usize, span: &runmat_accelerate::InstrSpan) {
    if fusion_debug_enabled() {
        log::debug!(
            "fusion skip at pc {}: side-effecting instrs in span {}..{}",
            pc,
            span.start,
            span.end
        );
    }
}
