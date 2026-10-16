use crate::bytecode::instr::Instr;
#[cfg(feature = "native-accel")]
use crate::bytecode::program::Bytecode;
use crate::interpreter::timing::InterpreterTiming;
use crate::runtime::gc::InterpretContext;
use crate::runtime::workspace::{
    has_exact_pending_workspace_service_frame, refresh_workspace_state,
    set_transient_workspace_state, take_pending_workspace_state, WorkspaceStateGuard,
};
use runmat_runtime::RuntimeError;
use runmat_value::Value;
use std::collections::{HashMap, HashSet};

pub fn prepare_workspace_guard(
    var_names: &HashMap<usize, String>,
    vars: &mut [Value],
    initial_assigned_var_count: usize,
    initially_unassigned_slots: &HashSet<usize>,
    global_aliases: runmat_runtime::workspace::bindings::FrameGlobalBindings,
) -> Option<WorkspaceStateGuard> {
    let pending_state = take_pending_workspace_state();
    let guard = pending_state.map(|pending| {
        let names = pending.names;
        let assigned = pending.assigned;
        global_aliases.replace_with_snapshot(pending.global_aliases.snapshot());
        let filtered_assigned: HashSet<String> = assigned
            .into_iter()
            .filter(|name| names.contains_key(name))
            .collect();
        crate::runtime::workspace::set_workspace_state_with_policy(
            names,
            filtered_assigned,
            vars,
            pending.policy,
            global_aliases.clone(),
        )
    });
    if guard.is_some() {
        refresh_workspace_state(vars);
        return guard;
    }
    if var_names.is_empty() {
        if !has_exact_pending_workspace_service_frame() {
            return None;
        }
        let guard =
            set_transient_workspace_state(HashMap::new(), HashSet::new(), vars, global_aliases);
        refresh_workspace_state(vars);
        return Some(guard);
    }
    let mut names = HashMap::new();
    let mut assigned = HashSet::new();
    for (slot, name) in var_names {
        names.insert(name.clone(), *slot);
        if *slot < initial_assigned_var_count && !initially_unassigned_slots.contains(slot) {
            assigned.insert(name.clone());
        }
    }
    let guard = set_transient_workspace_state(names, assigned, vars, global_aliases);
    refresh_workspace_state(vars);
    Some(guard)
}

pub fn create_gc_context<'roots>(
    stack: &'roots Vec<Value>,
    vars: &'roots Vec<Value>,
    locals: &'roots Vec<Value>,
    sequence: &'roots crate::interpreter::dispatch::SequenceState,
    thread_roots: Vec<Value>,
) -> Result<InterpretContext<'roots>, String> {
    let mut gc_context = InterpretContext::new_frame(stack, vars, locals, sequence)?;
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
    if runmat_runtime::interrupt::is_cancelled() || runtime.is_cancelled() {
        return Err(cancellation_error());
    }
    Ok(())
}

pub(super) fn cancellation_error() -> RuntimeError {
    crate::interpreter::errors::mex("ExecutionCancelled", "Execution cancelled by user")
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
