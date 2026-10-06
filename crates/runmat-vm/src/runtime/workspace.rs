use runmat_runtime::workspace::bindings::{FrameGlobalBinding, FrameGlobalBindings};
use runmat_thread_local::runmat_thread_local;
use runmat_value::Value;
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

mod frame;
pub(crate) use frame::{
    reconcile_workspace_frame, release_pending_workspace_service_frame,
    reserve_workspace_service_frame, workspace_frame_assign, workspace_frame_assign_from_executor,
    workspace_frame_clear, workspace_frame_lookup, workspace_frame_name_is_global,
    workspace_frame_publishes, workspace_frame_remove, workspace_frame_snapshot, WorkspaceFrameId,
    WorkspaceFrameLifecycle,
};
use frame::{take_service_frame, PendingServiceFrame};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum WorkspaceOwner {
    Runtime(runmat_runtime::context::RuntimeContextLocalId),
    Ambient,
}

fn current_owner() -> WorkspaceOwner {
    runmat_runtime::context::legacy::active()
        .map(|runtime| WorkspaceOwner::Runtime(runtime.local_identity()))
        .unwrap_or(WorkspaceOwner::Ambient)
}

#[derive(Debug, Clone)]
enum SlotLifecycle {
    Assigned(String),
    Unassigned(String),
}

impl SlotLifecycle {
    fn name(&self) -> &str {
        match self {
            SlotLifecycle::Assigned(name) | SlotLifecycle::Unassigned(name) => name,
        }
    }

    fn is_assigned(&self) -> bool {
        matches!(self, SlotLifecycle::Assigned(_))
    }
}

struct WorkspaceState {
    names: HashMap<String, usize>,
    assigned: HashSet<String>,
    assigned_names_this_execution: HashSet<String>,
    assigned_ids_this_execution: HashSet<usize>,
    removed_slots_this_execution: HashMap<usize, String>,
    slot_lifecycle: HashMap<usize, SlotLifecycle>,
    values: Vec<Value>,
    pending_var_reconciliation: HashSet<usize>,
}

struct WorkspaceFrame {
    id: WorkspaceFrameId,
    lifecycle: Rc<Cell<WorkspaceFrameLifecycle>>,
    delegate: Rc<RefCell<Option<Rc<dyn runmat_runtime::context::RuntimeWorkspaceService>>>>,
    state: WorkspaceState,
    global_aliases: FrameGlobalBindings,
    policy: WorkspaceFramePolicy,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum WorkspaceFramePolicy {
    /// A top-level execution publishes admitted mutations as they occur and
    /// retains its completed slot state for the host result adapter.
    TopLevel,
    /// Dynamic evaluation captures its completed slot state for an explicit
    /// commit by the caller, but never publishes mutations while evaluating.
    Transactional,
    /// Procedure-local frames neither publish nor become a host completion.
    Transient,
}

impl WorkspaceFramePolicy {
    fn publishes_mutations(self) -> bool {
        matches!(self, Self::TopLevel)
    }

    fn captures_completion(self) -> bool {
        matches!(self, Self::TopLevel | Self::Transactional)
    }
}

pub(crate) struct PendingWorkspace {
    pub(crate) names: HashMap<String, usize>,
    pub(crate) assigned: HashSet<String>,
    pub(crate) policy: WorkspaceFramePolicy,
}

#[derive(Debug, Clone)]
pub struct WorkspaceValueSnapshot {
    pub vars: Vec<Value>,
    pub names: HashMap<String, usize>,
    pub assigned: HashSet<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkspaceTarget {
    Current,
    Caller,
    Base,
}

#[derive(Debug, Clone)]
pub struct WorkspaceTargetSnapshot {
    pub names: HashMap<String, usize>,
    pub assigned: HashSet<String>,
    pub vars: Vec<Value>,
}

#[derive(Debug, Clone)]
pub struct WorkspaceAssignedReport {
    pub ids: HashSet<usize>,
    pub names: HashSet<String>,
    pub removed_ids: HashSet<usize>,
    pub removed_names: HashSet<String>,
}

runmat_thread_local! {
    static WORKSPACE_STACKS: RefCell<HashMap<WorkspaceOwner, Vec<WorkspaceFrame>>> = RefCell::new(HashMap::new());
    static PENDING_WORKSPACES: RefCell<HashMap<WorkspaceOwner, PendingWorkspace>> = RefCell::new(HashMap::new());
    static LAST_WORKSPACE_STATES: RefCell<HashMap<WorkspaceOwner, WorkspaceValueSnapshot>> = RefCell::new(HashMap::new());
    static LAST_WORKSPACE_ASSIGNED_REPORTS: RefCell<HashMap<WorkspaceOwner, WorkspaceAssignedReport>> = RefCell::new(HashMap::new());
    static PENDING_SERVICE_FRAMES: RefCell<HashMap<WorkspaceOwner, Vec<PendingServiceFrame>>> = RefCell::new(HashMap::new());
    static NEXT_WORKSPACE_FRAME_ID: Cell<usize> = const { Cell::new(1) };
}

fn with_stack<R>(owner: WorkspaceOwner, f: impl FnOnce(&[WorkspaceFrame]) -> R) -> R {
    WORKSPACE_STACKS.with(|stacks| {
        let stacks = stacks.borrow();
        f(stacks.get(&owner).map(Vec::as_slice).unwrap_or_default())
    })
}

fn with_stack_mut<R>(owner: WorkspaceOwner, f: impl FnOnce(&mut Vec<WorkspaceFrame>) -> R) -> R {
    WORKSPACE_STACKS.with(|stacks| {
        let mut stacks = stacks.borrow_mut();
        let result = f(stacks.entry(owner).or_default());
        if stacks.get(&owner).is_some_and(Vec::is_empty) {
            stacks.remove(&owner);
        }
        result
    })
}

fn mark_slot_unassigned(ws: &mut WorkspaceState, index: usize, name: String) {
    ws.slot_lifecycle
        .insert(index, SlotLifecycle::Unassigned(name.clone()));
    ws.removed_slots_this_execution.insert(index, name);
}

fn mark_slot_assigned(ws: &mut WorkspaceState, index: usize, name: String) {
    ws.slot_lifecycle
        .insert(index, SlotLifecycle::Assigned(name.clone()));
    ws.removed_slots_this_execution.remove(&index);
}

fn find_unassigned_slot_for_name(ws: &WorkspaceState, name: &str) -> Option<usize> {
    ws.slot_lifecycle.iter().find_map(|(idx, state)| {
        matches!(state, SlotLifecycle::Unassigned(slot_name) if slot_name == name).then_some(*idx)
    })
}

fn upsert_slot_lifecycle_name(ws: &mut WorkspaceState, index: usize, name: &str) {
    if let Some(existing_index) = ws.names.insert(name.to_string(), index) {
        if existing_index != index {
            mark_slot_unassigned(ws, existing_index, name.to_string());
            ws.assigned.remove(name);
        }
    }

    match ws.slot_lifecycle.get_mut(&index) {
        Some(state) => {
            let was_assigned = state.is_assigned();
            let old_name = state.name().to_string();
            if old_name != name {
                if ws.names.get(&old_name).copied() == Some(index) {
                    ws.names.remove(&old_name);
                }
                ws.assigned.remove(&old_name);
            }
            *state = if was_assigned {
                SlotLifecycle::Assigned(name.to_string())
            } else {
                SlotLifecycle::Unassigned(name.to_string())
            };
        }
        None => {
            ws.slot_lifecycle
                .insert(index, SlotLifecycle::Unassigned(name.to_string()));
        }
    }
}

fn target_frame_index(len: usize, target: WorkspaceTarget) -> Option<usize> {
    if len == 0 {
        return None;
    }
    match target {
        WorkspaceTarget::Current => Some(len - 1),
        WorkspaceTarget::Caller => Some(len.saturating_sub(2)),
        WorkspaceTarget::Base => Some(0),
    }
}

fn lifecycle_from_names(
    names: &HashMap<String, usize>,
    assigned: &HashSet<String>,
) -> HashMap<usize, SlotLifecycle> {
    names
        .iter()
        .map(|(name, idx)| {
            let lifecycle = if assigned.contains(name) {
                SlotLifecycle::Assigned(name.clone())
            } else {
                SlotLifecycle::Unassigned(name.clone())
            };
            (*idx, lifecycle)
        })
        .collect()
}

pub struct WorkspaceStateGuard {
    owner: WorkspaceOwner,
}

impl Drop for WorkspaceStateGuard {
    fn drop(&mut self) {
        let Some(frame) = with_stack_mut(self.owner, |stack| stack.pop()) else {
            return;
        };
        frame.lifecycle.set(WorkspaceFrameLifecycle::Closed);
        let retired_delegate = frame.delegate.borrow_mut().take();
        if frame.policy.captures_completion() {
            let ws = frame.state;
            let removed_ids = ws.removed_slots_this_execution.keys().copied().collect();
            let removed_names = ws.removed_slots_this_execution.values().cloned().collect();
            LAST_WORKSPACE_ASSIGNED_REPORTS.with(|reports| {
                reports.borrow_mut().insert(
                    self.owner,
                    WorkspaceAssignedReport {
                        ids: ws.assigned_ids_this_execution,
                        names: ws.assigned_names_this_execution,
                        removed_ids,
                        removed_names,
                    },
                );
            });
            LAST_WORKSPACE_STATES.with(|states| {
                states.borrow_mut().insert(
                    self.owner,
                    WorkspaceValueSnapshot {
                        vars: ws.values,
                        names: ws.names,
                        assigned: ws.assigned,
                    },
                );
            });
        }
        drop(retired_delegate);
    }
}

pub struct PendingWorkspaceGuard {
    owner: WorkspaceOwner,
}

impl Drop for PendingWorkspaceGuard {
    fn drop(&mut self) {
        PENDING_WORKSPACES.with(|workspaces| {
            workspaces.borrow_mut().remove(&self.owner);
        });
    }
}

pub fn push_pending_workspace(
    names: HashMap<String, usize>,
    assigned: HashSet<String>,
) -> PendingWorkspaceGuard {
    push_pending_workspace_with_policy(names, assigned, WorkspaceFramePolicy::TopLevel)
}

pub(crate) fn push_transactional_pending_workspace(
    names: HashMap<String, usize>,
    assigned: HashSet<String>,
) -> PendingWorkspaceGuard {
    push_pending_workspace_with_policy(names, assigned, WorkspaceFramePolicy::Transactional)
}

fn push_pending_workspace_with_policy(
    names: HashMap<String, usize>,
    assigned: HashSet<String>,
    policy: WorkspaceFramePolicy,
) -> PendingWorkspaceGuard {
    let owner = current_owner();
    PENDING_WORKSPACES.with(|workspaces| {
        workspaces.borrow_mut().insert(
            owner,
            PendingWorkspace {
                names,
                assigned,
                policy,
            },
        );
    });
    PendingWorkspaceGuard { owner }
}

pub(crate) fn take_pending_workspace_state() -> Option<PendingWorkspace> {
    let owner = current_owner();
    PENDING_WORKSPACES.with(|workspaces| workspaces.borrow_mut().remove(&owner))
}

pub fn workspace_global_names() -> Vec<String> {
    runmat_runtime::workspace::session::global_names()
}

pub fn take_updated_workspace_state() -> Option<WorkspaceValueSnapshot> {
    let owner = current_owner();
    LAST_WORKSPACE_STATES.with(|states| states.borrow_mut().remove(&owner))
}

pub fn take_updated_workspace_assigned_report() -> Option<WorkspaceAssignedReport> {
    let owner = current_owner();
    LAST_WORKSPACE_ASSIGNED_REPORTS.with(|reports| reports.borrow_mut().remove(&owner))
}

#[cfg(test)]
pub fn set_workspace_state(
    names: HashMap<String, usize>,
    assigned: HashSet<String>,
    vars: &mut Vec<Value>,
    global_aliases: FrameGlobalBindings,
) -> WorkspaceStateGuard {
    set_workspace_state_with_policy(
        names,
        assigned,
        vars,
        WorkspaceFramePolicy::TopLevel,
        global_aliases,
    )
}

pub fn set_transient_workspace_state(
    names: HashMap<String, usize>,
    assigned: HashSet<String>,
    vars: &mut Vec<Value>,
    global_aliases: FrameGlobalBindings,
) -> WorkspaceStateGuard {
    set_workspace_state_with_policy(
        names,
        assigned,
        vars,
        WorkspaceFramePolicy::Transient,
        global_aliases,
    )
}

fn install_workspace_state(
    names: HashMap<String, usize>,
    assigned: HashSet<String>,
    vars: &mut Vec<Value>,
    policy: WorkspaceFramePolicy,
    global_aliases: FrameGlobalBindings,
) -> WorkspaceStateGuard {
    let mut slot_lifecycle = HashMap::new();
    for (name, idx) in &names {
        let lifecycle = if assigned.contains(name) {
            SlotLifecycle::Assigned(name.clone())
        } else {
            SlotLifecycle::Unassigned(name.clone())
        };
        slot_lifecycle.insert(*idx, lifecycle);
    }
    let owner = current_owner();
    let service_frame = take_service_frame(owner);
    service_frame.lifecycle.set(WorkspaceFrameLifecycle::Active);
    with_stack_mut(owner, |stack| {
        stack.push(WorkspaceFrame {
            id: service_frame.id,
            lifecycle: service_frame.lifecycle,
            delegate: service_frame.delegate,
            global_aliases,
            state: WorkspaceState {
                names,
                assigned,
                assigned_names_this_execution: HashSet::new(),
                assigned_ids_this_execution: HashSet::new(),
                removed_slots_this_execution: HashMap::new(),
                slot_lifecycle,
                values: vars.clone(),
                pending_var_reconciliation: HashSet::new(),
            },
            policy,
        });
    });
    WorkspaceStateGuard { owner }
}

pub(crate) fn set_workspace_state_with_policy(
    names: HashMap<String, usize>,
    assigned: HashSet<String>,
    vars: &mut Vec<Value>,
    policy: WorkspaceFramePolicy,
    global_aliases: FrameGlobalBindings,
) -> WorkspaceStateGuard {
    install_workspace_state(names, assigned, vars, policy, global_aliases)
}

pub fn refresh_workspace_state(vars: &[Value]) {
    with_stack_mut(current_owner(), |stack| {
        if let Some(frame) = stack.last_mut() {
            frame.state.values.clone_from(&vars.to_vec());
        }
    });
}

/// Refresh one execution-slot cache entry without cloning the full frame.
/// This does not change assignment lifecycle; declaration/import paths use it
/// before the corresponding language binding is admitted.
pub fn refresh_workspace_slot(index: usize, value: &Value) {
    with_stack_mut(current_owner(), |stack| {
        if let Some(frame) = stack.last_mut() {
            if frame.state.values.len() <= index {
                frame.state.values.resize(index + 1, Value::Num(0.0));
            }
            frame.state.values[index] = value.clone();
        }
    });
}

pub fn reconcile_workspace_vars(vars: &mut Vec<Value>) {
    with_stack_mut(current_owner(), |stack| {
        let Some(frame) = stack.last_mut() else {
            return;
        };
        let changed = std::mem::take(&mut frame.state.pending_var_reconciliation);
        for slot in changed {
            if vars.len() <= slot {
                vars.resize(slot + 1, Value::Num(0.0));
            }
            vars[slot] = frame
                .state
                .values
                .get(slot)
                .cloned()
                .unwrap_or(Value::Num(0.0));
        }
    });
}

pub fn workspace_lookup(name: &str) -> Option<Value> {
    with_stack(current_owner(), |stack| {
        let frame = stack.last()?;
        let ws = &frame.state;
        let idx = ws.names.get(name)?;
        if !ws.assigned.contains(name) {
            return None;
        }
        ws.values.get(*idx).cloned()
    })
}

pub fn workspace_slot_assigned(index: usize) -> Option<bool> {
    with_stack(current_owner(), |stack| {
        let frame = stack.last()?;
        let ws = &frame.state;
        ws.slot_lifecycle
            .get(&index)
            .map(SlotLifecycle::is_assigned)
    })
}

pub fn workspace_slot_name(index: usize) -> Option<String> {
    with_stack(current_owner(), |stack| {
        let frame = stack.last()?;
        let ws = &frame.state;
        ws.slot_lifecycle
            .get(&index)
            .map(|state| state.name().to_string())
    })
}

pub fn workspace_assign(name: &str, value: Value) -> Result<(), String> {
    workspace_assign_target(WorkspaceTarget::Current, name, value)
}

pub fn workspace_assign_target(
    target: WorkspaceTarget,
    name: &str,
    value: Value,
) -> Result<(), String> {
    with_stack_mut(current_owner(), |stack| {
        let index = target_frame_index(stack.len(), target)
            .ok_or_else(|| "load: workspace state unavailable".to_string())?;
        let frame = stack
            .get_mut(index)
            .ok_or_else(|| "load: workspace state unavailable".to_string())?;
        set_workspace_variable_in_frame(frame, name, value)
    })
}

pub fn workspace_clear() -> Result<(), String> {
    with_stack_mut(current_owner(), |stack| {
        let frame = stack
            .last_mut()
            .ok_or_else(|| "clear: workspace state unavailable".to_string())?;
        clear_workspace_frame(frame);
        Ok(())
    })
}

fn clear_workspace_frame(frame: &mut WorkspaceFrame) {
    let ws = &mut frame.state;
    for (name, idx) in ws.names.clone() {
        if let Some(value) = ws.values.get_mut(idx) {
            *value = Value::Num(0.0);
        }
        mark_slot_unassigned(ws, idx, name);
        ws.pending_var_reconciliation.insert(idx);
    }
    ws.names.clear();
    ws.assigned.clear();
    frame.global_aliases.unbind_all();
}

pub fn workspace_remove(name: &str) -> Result<(), String> {
    with_stack_mut(current_owner(), |stack| {
        let frame = stack
            .last_mut()
            .ok_or_else(|| "clear: workspace state unavailable".to_string())?;
        remove_workspace_name(frame, name);
        Ok(())
    })
}

fn remove_workspace_name(frame: &mut WorkspaceFrame, name: &str) {
    let ws = &mut frame.state;
    if let Some(idx) = ws.names.remove(name) {
        if let Some(value) = ws.values.get_mut(idx) {
            *value = Value::Num(0.0);
        }
        ws.assigned.remove(name);
        frame.global_aliases.unbind_slot(idx);
        mark_slot_unassigned(ws, idx, name.to_string());
        ws.pending_var_reconciliation.insert(idx);
    }
}

pub fn workspace_snapshot() -> Vec<(String, Value)> {
    with_stack(current_owner(), |stack| {
        if let Some(frame) = stack.last() {
            let ws = &frame.state;
            let mut entries: Vec<(String, Value)> = ws
                .names
                .iter()
                .filter_map(|(name, idx)| {
                    if !ws.assigned.contains(name) {
                        return None;
                    }
                    ws.values
                        .get(*idx)
                        .cloned()
                        .map(|value| (name.clone(), value))
                })
                .collect();
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            entries
        } else {
            Vec::new()
        }
    })
}

pub fn workspace_target_snapshot(
    target: WorkspaceTarget,
) -> Result<WorkspaceTargetSnapshot, String> {
    with_stack(current_owner(), |stack| {
        let index = target_frame_index(stack.len(), target)
            .ok_or_else(|| "workspace state unavailable".to_string())?;
        let frame = stack
            .get(index)
            .ok_or_else(|| "workspace state unavailable".to_string())?;
        Ok(WorkspaceTargetSnapshot {
            names: frame.state.names.clone(),
            assigned: frame.state.assigned.clone(),
            vars: frame.state.values.clone(),
        })
    })
}

pub fn replace_workspace_target_vars_and_state(
    target: WorkspaceTarget,
    vars: Vec<Value>,
    names: HashMap<String, usize>,
    assigned: HashSet<String>,
) -> Result<(), String> {
    with_stack_mut(current_owner(), |stack| {
        let index = target_frame_index(stack.len(), target)
            .ok_or_else(|| "workspace state unavailable".to_string())?;
        let frame = stack
            .get_mut(index)
            .ok_or_else(|| "workspace state unavailable".to_string())?;
        let mut changed_slots = frame
            .state
            .slot_lifecycle
            .keys()
            .copied()
            .collect::<HashSet<_>>();
        changed_slots.extend(names.values().copied());
        frame.state.values = vars;
        for (slot, binding) in frame.global_aliases.snapshot() {
            let still_named = match binding {
                FrameGlobalBinding::Named(name) => names.get(&name) == Some(&slot),
                FrameGlobalBinding::LegacySlot => names.values().any(|active| *active == slot),
            };
            if !still_named {
                frame.global_aliases.unbind_slot(slot);
            }
        }
        frame.state.names = names;
        frame.state.assigned = assigned;
        frame.state.slot_lifecycle =
            lifecycle_from_names(&frame.state.names, &frame.state.assigned);
        frame.state.pending_var_reconciliation.extend(changed_slots);
        Ok(())
    })
}

#[cfg(test)]
pub fn set_workspace_variable(
    name: &str,
    value: Value,
    vars: &mut Vec<Value>,
) -> Result<(), String> {
    with_stack_mut(current_owner(), |stack| match stack.last_mut() {
        Some(frame) => set_workspace_variable_in_frame(frame, name, value),
        None => Err("load: workspace state unavailable".to_string()),
    })?;
    if let Some(value) = workspace_lookup(name) {
        let index = with_stack(current_owner(), |stack| {
            stack
                .last()
                .and_then(|frame| frame.state.names.get(name).copied())
        })
        .ok_or_else(|| "load: workspace slot unavailable".to_string())?;
        if vars.len() <= index {
            vars.resize(index + 1, Value::Num(0.0));
        }
        vars[index] = value;
    }
    Ok(())
}

fn set_workspace_variable_in_frame(
    frame: &mut WorkspaceFrame,
    name: &str,
    value: Value,
) -> Result<(), String> {
    set_workspace_variable_in_frame_with_reconciliation(frame, name, value, true)
}

fn set_workspace_variable_in_frame_with_reconciliation(
    frame: &mut WorkspaceFrame,
    name: &str,
    value: Value,
    reconcile_var: bool,
) -> Result<(), String> {
    let ws = &mut frame.state;
    let idx = if let Some(idx) = ws.names.get(name).copied() {
        idx
    } else if let Some(idx) = find_unassigned_slot_for_name(ws, name) {
        ws.names.insert(name.to_string(), idx);
        idx
    } else {
        let idx = ws.values.len();
        ws.names.insert(name.to_string(), idx);
        idx
    };
    if idx >= ws.values.len() {
        ws.values.resize(idx + 1, Value::Num(0.0));
    }
    ws.values[idx] = value;
    if reconcile_var {
        ws.pending_var_reconciliation.insert(idx);
    }
    match frame.global_aliases.binding_for_slot(idx) {
        Some(FrameGlobalBinding::Named(bound)) if bound == name => {
            runmat_runtime::workspace::session::store_global_named(name, ws.values[idx].clone());
        }
        Some(FrameGlobalBinding::LegacySlot) => {
            runmat_runtime::workspace::session::store_global_slot(idx, ws.values[idx].clone());
        }
        Some(FrameGlobalBinding::Named(_)) | None => {}
    }
    ws.assigned.insert(name.to_string());
    ws.assigned_names_this_execution.insert(name.to_string());
    ws.assigned_ids_this_execution.insert(idx);
    mark_slot_assigned(ws, idx, name.to_string());
    Ok(())
}

pub fn ensure_workspace_slot_name(index: usize, name: &str) {
    with_stack_mut(current_owner(), |stack| {
        if let Some(frame) = stack.last_mut() {
            upsert_slot_lifecycle_name(&mut frame.state, index, name);
        }
    });
}

pub fn mark_workspace_assigned(index: usize, value: &Value) {
    with_stack_mut(current_owner(), |stack| {
        if let Some(frame) = stack.last_mut() {
            let ws = &mut frame.state;
            if let Some(name) = ws
                .slot_lifecycle
                .get(&index)
                .map(|slot| slot.name().to_string())
            {
                ws.assigned.insert(name.clone());
                ws.assigned_names_this_execution.insert(name.clone());
                ws.assigned_ids_this_execution.insert(index);
                mark_slot_assigned(ws, index, name);
                if ws.values.len() <= index {
                    ws.values.resize(index + 1, Value::Num(0.0));
                }
                ws.values[index] = value.clone();
            }
        }
    });
}

/// Admit a language-visible named-slot write before changing the VM register
/// file. Installed VM workspace services update both the exact frame and, for
/// a top-level invocation, its Runtime-owned canonical projection.
pub fn admit_workspace_slot_write(
    index: usize,
    value: &Value,
) -> Result<(), runmat_runtime::RuntimeError> {
    let name = workspace_slot_name(index);
    let Some(name) = name else {
        mark_workspace_assigned(index, value);
        return Ok(());
    };
    if let Some(context) = runmat_runtime::context::legacy::active() {
        if let Some(service) = context.service_ports().workspace() {
            return service.assign_executor_slot(index, &name, value.clone());
        }
    }
    mark_workspace_assigned(index, value);
    Ok(())
}

/// Read the authoritative language-visible value for a named workspace slot.
/// Compiler temporaries and procedure locals are intentionally absent.
pub fn workspace_slot_value(index: usize) -> Option<Value> {
    with_stack(current_owner(), |stack| {
        let frame = stack.last()?;
        let lifecycle = frame.state.slot_lifecycle.get(&index)?;
        lifecycle
            .is_assigned()
            .then(|| frame.state.values.get(index).cloned())
            .flatten()
    })
}

/// Snapshot the slots currently assigned in the innermost VM workspace.
/// Internal placeholder values are deliberately excluded.
pub(crate) fn current_workspace_assigned_slots() -> HashSet<usize> {
    with_stack(current_owner(), |stack| {
        stack
            .last()
            .map(|frame| {
                frame
                    .state
                    .slot_lifecycle
                    .iter()
                    .filter_map(|(slot, lifecycle)| lifecycle.is_assigned().then_some(*slot))
                    .collect()
            })
            .unwrap_or_default()
    })
}

pub(crate) fn reset_thread_state_for_tests() {
    WORKSPACE_STACKS.with(|stacks| stacks.borrow_mut().clear());
    PENDING_WORKSPACES.with(|workspaces| workspaces.borrow_mut().clear());
    LAST_WORKSPACE_STATES.with(|states| states.borrow_mut().clear());
    LAST_WORKSPACE_ASSIGNED_REPORTS.with(|reports| reports.borrow_mut().clear());
    PENDING_SERVICE_FRAMES.with(|frames| frames.borrow_mut().clear());
    NEXT_WORKSPACE_FRAME_ID.with(|next| next.set(1));
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::future::Future;
    use std::pin::Pin;
    use std::rc::Rc;
    use std::sync::Arc;
    use std::task::{Context, Poll, Wake};

    struct NoopWake;

    impl Wake for NoopWake {
        fn wake(self: Arc<Self>) {}
    }

    fn take_report_after(f: impl FnOnce(&mut Vec<Value>)) -> WorkspaceAssignedReport {
        let _ = take_updated_workspace_assigned_report();
        let _ = take_updated_workspace_state();

        let mut vars = Vec::new();
        {
            let _guard = set_workspace_state(
                HashMap::new(),
                HashSet::new(),
                &mut vars,
                FrameGlobalBindings::default(),
            );
            f(&mut vars);
        }

        take_updated_workspace_assigned_report().expect("workspace report should be recorded")
    }

    #[test]
    fn transactional_frames_capture_completion_without_live_publication() {
        let _ = take_updated_workspace_state();
        let _ = take_updated_workspace_assigned_report();
        let mut vars = vec![Value::Num(1.0)];
        {
            let _guard = install_workspace_state(
                HashMap::from([("x".to_string(), 0)]),
                HashSet::from(["x".to_string()]),
                &mut vars,
                WorkspaceFramePolicy::Transactional,
                FrameGlobalBindings::default(),
            );
            let frame = with_stack(current_owner(), |stack| stack.last().unwrap().id);
            assert_eq!(workspace_frame_publishes(frame), Some(false));
            workspace_frame_assign_from_executor(frame, "x", Value::Num(7.0)).unwrap();
        }

        let completed = take_updated_workspace_state()
            .expect("transactional caller must receive the completed frame");
        assert_eq!(completed.vars, vec![Value::Num(7.0)]);
        assert_eq!(completed.names.get("x"), Some(&0));
        assert!(completed.assigned.contains("x"));
    }

    #[test]
    fn transient_frames_do_not_become_host_completions() {
        let _ = take_updated_workspace_state();
        let mut vars = vec![Value::Num(1.0)];
        {
            let _guard = install_workspace_state(
                HashMap::from([("x".to_string(), 0)]),
                HashSet::from(["x".to_string()]),
                &mut vars,
                WorkspaceFramePolicy::Transient,
                FrameGlobalBindings::default(),
            );
        }
        assert!(take_updated_workspace_state().is_none());
    }

    #[test]
    fn interleaved_runtime_contexts_keep_workspace_frames_isolated() {
        async fn session(label: &'static str) {
            let mut vars = vec![Value::String(label.into())];
            let names = HashMap::from([(String::from("value"), 0)]);
            let assigned = HashSet::from([String::from("value")]);
            let _guard = set_transient_workspace_state(
                names,
                assigned,
                &mut vars,
                FrameGlobalBindings::default(),
            );
            let mut yielded = false;
            std::future::poll_fn(move |_| {
                if yielded {
                    Poll::Ready(())
                } else {
                    yielded = true;
                    Poll::Pending
                }
            })
            .await;
            assert_eq!(workspace_lookup("value"), Some(Value::String(label.into())));
        }

        fn poll(future: Pin<&mut impl Future<Output = ()>>) -> Poll<()> {
            let waker = std::task::Waker::from(Arc::new(NoopWake));
            future.poll(&mut Context::from_waker(&waker))
        }

        let first = runmat_runtime::context::RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        let second = runmat_runtime::context::RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        assert_ne!(first.local_identity(), second.local_identity());
        let mut first = Box::pin(first.scope(session("first")));
        let mut second = Box::pin(second.scope(session("second")));
        assert!(poll(first.as_mut()).is_pending());
        assert!(poll(second.as_mut()).is_pending());
        assert!(poll(first.as_mut()).is_ready());
        assert!(poll(second.as_mut()).is_ready());
    }

    #[test]
    fn remove_preserves_assignment_report_and_records_removal() {
        let report = take_report_after(|vars| {
            set_workspace_variable("x", Value::Num(1.0), vars).unwrap();
            workspace_remove("x").unwrap();
        });

        assert!(report.names.contains("x"));
        assert!(report.ids.contains(&0));
        assert!(report.removed_names.contains("x"));
        assert!(report.removed_ids.contains(&0));
    }

    #[test]
    fn clear_preserves_assignment_report_and_records_removal() {
        let report = take_report_after(|vars| {
            set_workspace_variable("x", Value::Num(1.0), vars).unwrap();
            workspace_clear().unwrap();
        });

        assert!(report.names.contains("x"));
        assert!(report.ids.contains(&0));
        assert!(report.removed_names.contains("x"));
        assert!(report.removed_ids.contains(&0));
    }

    #[test]
    fn clear_preserves_active_frame_slots() {
        let _ = take_updated_workspace_state();
        let mut vars = vec![Value::Num(11.0), Value::Num(0.0), Value::Num(0.0)];
        let mut names = HashMap::new();
        names.insert("x".to_string(), 0);
        let assigned = HashSet::from(["x".to_string()]);

        {
            let _guard =
                set_workspace_state(names, assigned, &mut vars, FrameGlobalBindings::default());
            workspace_clear().unwrap();

            assert_eq!(vars.len(), 3);
            assert_eq!(vars[0], Value::Num(11.0));
            assert_eq!(vars[1], Value::Num(0.0));
            assert_eq!(vars[2], Value::Num(0.0));
            assert_eq!(workspace_lookup("x"), None);
            assert_eq!(workspace_slot_value(0), None);
            assert_eq!(workspace_slot_assigned(0), Some(false));
        }

        let snapshot = take_updated_workspace_state().expect("workspace state should be recorded");
        assert_eq!(snapshot.vars.len(), 3);
        assert!(snapshot.names.is_empty());
        assert!(snapshot.assigned.is_empty());
    }

    #[test]
    fn service_mutations_reconcile_only_changed_register_slots() {
        let runtime = runmat_runtime::context::RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        let runtime_id = runtime.local_identity();
        let _reserved = reserve_workspace_service_frame(runtime_id);
        futures::executor::block_on(runtime.scope(async {
            let mut vars = vec![Value::Num(1.0), Value::Num(2.0)];
            let guard = set_workspace_state(
                HashMap::from([("x".to_string(), 0), ("y".to_string(), 1)]),
                HashSet::from(["x".to_string(), "y".to_string()]),
                &mut vars,
                FrameGlobalBindings::default(),
            );
            let frame = with_stack(current_owner(), |stack| stack.last().unwrap().id);

            workspace_frame_assign(frame, "x", Value::Num(9.0)).unwrap();
            assert_eq!(vars, vec![Value::Num(1.0), Value::Num(2.0)]);
            reconcile_workspace_vars(&mut vars);
            assert_eq!(vars, vec![Value::Num(9.0), Value::Num(2.0)]);

            workspace_frame_clear(frame).unwrap();
            reconcile_workspace_vars(&mut vars);
            assert_eq!(vars, vec![Value::Num(0.0), Value::Num(0.0)]);
            drop(guard);
        }));
    }

    #[test]
    fn executor_admission_does_not_schedule_duplicate_register_reconciliation() {
        let runtime = runmat_runtime::context::RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        let runtime_id = runtime.local_identity();
        let _reserved = reserve_workspace_service_frame(runtime_id);
        futures::executor::block_on(runtime.scope(async {
            let mut vars = vec![Value::Num(1.0)];
            let guard = set_workspace_state(
                HashMap::from([("x".to_string(), 0)]),
                HashSet::from(["x".to_string()]),
                &mut vars,
                FrameGlobalBindings::default(),
            );
            let frame = with_stack(current_owner(), |stack| stack.last().unwrap().id);
            workspace_frame_assign_from_executor(frame, "x", Value::Num(7.0)).unwrap();
            vars[0] = Value::Num(7.0);
            reconcile_workspace_vars(&mut vars);
            assert_eq!(vars, vec![Value::Num(7.0)]);
            drop(guard);
        }));
    }

    #[test]
    fn assignment_after_clear_clears_final_removal_marker() {
        let report = take_report_after(|vars| {
            set_workspace_variable("x", Value::Num(1.0), vars).unwrap();
            workspace_clear().unwrap();
            set_workspace_variable("x", Value::Num(2.0), vars).unwrap();
        });

        assert!(report.names.contains("x"));
        assert!(report.removed_names.is_empty());
        assert!(report.removed_ids.is_empty());
    }

    #[test]
    fn assignment_after_remove_reuses_previous_slot() {
        let mut vars = Vec::new();
        let _ = take_updated_workspace_state();
        {
            let _guard = set_workspace_state(
                HashMap::new(),
                HashSet::new(),
                &mut vars,
                FrameGlobalBindings::default(),
            );
            set_workspace_variable("x", Value::Num(1.0), &mut vars).unwrap();
            set_workspace_variable("z", Value::Num(9.0), &mut vars).unwrap();
            workspace_remove("x").unwrap();
            set_workspace_variable("x", Value::Num(42.0), &mut vars).unwrap();
            assert_eq!(workspace_lookup("x"), Some(Value::Num(42.0)));
            assert_eq!(vars[0], Value::Num(42.0));
        }

        let snapshot = take_updated_workspace_state().expect("workspace state should be recorded");
        assert_eq!(snapshot.names.get("x"), Some(&0));
        assert_eq!(snapshot.names.get("z"), Some(&1));
        assert!(snapshot.assigned.contains("x"));
        assert!(snapshot.assigned.contains("z"));
        assert_eq!(snapshot.vars[0], Value::Num(42.0));
    }

    #[test]
    fn dynamic_assignment_publishes_active_global_but_clear_only_unbinds_frame() {
        let runtime = runmat_runtime::context::RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        futures::executor::block_on(runtime.scope(async {
            let aliases = FrameGlobalBindings::default();
            aliases.declare_named(0, "alpha".into());
            runmat_runtime::workspace::session::store_global_named("alpha", Value::Num(7.0));
            let mut vars = vec![Value::Num(7.0)];
            let _guard = set_workspace_state(
                HashMap::from([("alpha".into(), 0)]),
                HashSet::from(["alpha".into()]),
                &mut vars,
                aliases.clone(),
            );
            workspace_assign("alpha", Value::Num(8.0)).unwrap();
            assert_eq!(
                runmat_runtime::workspace::session::global_value("alpha"),
                Some(Value::Num(8.0))
            );
            workspace_remove("alpha").unwrap();
            assert_eq!(aliases.name_for_slot(0), None);
            workspace_assign("alpha", Value::Num(9.0)).unwrap();
            assert_eq!(
                runmat_runtime::workspace::session::global_value("alpha"),
                Some(Value::Num(8.0)),
                "ordinary clear removes the frame binding, not the global value"
            );
            assert_eq!(workspace_lookup("alpha"), Some(Value::Num(9.0)));
            assert_eq!(vars[0], Value::Num(7.0));
            reconcile_workspace_vars(&mut vars);
            assert_eq!(vars[0], Value::Num(9.0));
        }));
    }

    #[test]
    fn workspace_frames_do_not_retain_raw_execution_storage_pointers() {
        let source = include_str!("workspace.rs");
        let raw_vec_pointer = ["*mut", "Vec<Value>"].join(" ");
        let raw_value_pointer = ["*const", "Value"].join(" ");
        assert!(!source.contains(&raw_vec_pointer));
        assert!(!source.contains(&raw_value_pointer));
    }

    #[test]
    fn asynchronous_language_visible_writers_use_checked_slot_admission() {
        let writers = [
            ("fusion", include_str!("../accel/fusion.rs")),
            (
                "prepared outputs",
                include_str!("../interpreter/dispatch/prepared_outputs/commit.rs"),
            ),
            (
                "parfor",
                include_str!("../interpreter/dispatch/parallel/parfor.rs"),
            ),
            (
                "spmd",
                include_str!("../interpreter/dispatch/parallel/spmd.rs"),
            ),
            (
                "expanded nested calls",
                include_str!("../interpreter/dispatch/calls/dispatch/expanded.rs"),
            ),
            (
                "nested user calls",
                include_str!("../interpreter/dispatch/calls/dispatch/user.rs"),
            ),
            (
                "catch variables",
                include_str!("../interpreter/dispatch/exceptions.rs"),
            ),
        ];
        for (domain, source) in writers {
            assert!(
                source.contains("admit_workspace_slot_write"),
                "{domain} bypasses checked language-visible slot admission"
            );
        }
    }
}
