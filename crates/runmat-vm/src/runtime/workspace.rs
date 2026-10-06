use runmat_runtime::workspace::bindings::{FrameGlobalBinding, FrameGlobalBindings};
use runmat_thread_local::runmat_thread_local;
use runmat_value::Value;
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::rc::{Rc, Weak};

mod activation;
mod admission;
mod frame;
use activation::active_frame;
pub(crate) use activation::WorkspaceFrameActivation;
pub(crate) use admission::transactional_workspace_admission_with_global_bindings;
pub use admission::{
    workspace_admission, workspace_admission_with_global_bindings, PendingWorkspaceAdmission,
    WorkspaceExecutionCompletion,
};
pub(crate) use frame::{
    reconcile_workspace_frame, release_pending_workspace_service_frame,
    reserve_workspace_service_frame, workspace_frame_assign, workspace_frame_assign_from_executor,
    workspace_frame_clear, workspace_frame_declare_global, workspace_frame_for_service,
    workspace_frame_lookup, workspace_frame_name_is_global, workspace_frame_publishes,
    workspace_frame_remove, workspace_frame_snapshot, WorkspaceFrameId, WorkspaceFrameLifecycle,
};
use frame::{take_service_frame, PendingServiceFrame};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum WorkspaceOwner {
    Runtime(runmat_runtime::context::RuntimeContextLocalId),
    Ambient,
}

pub(crate) fn frame_activation(
    runtime: runmat_runtime::context::RuntimeContextLocalId,
    frame: WorkspaceFrameId,
) -> WorkspaceFrameActivation {
    WorkspaceFrameActivation::new(WorkspaceOwner::Runtime(runtime), frame)
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
    parent: Option<WorkspaceFrameId>,
    base: WorkspaceFrameId,
    lifecycle: Rc<Cell<WorkspaceFrameLifecycle>>,
    delegate: Rc<RefCell<Option<Rc<dyn runmat_runtime::context::RuntimeWorkspaceService>>>>,
    service: Option<Weak<dyn runmat_runtime::context::RuntimeWorkspaceService>>,
    session_variables: Option<runmat_runtime::workspace::session::SessionVariableAuthority>,
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
    pub(crate) global_aliases: FrameGlobalBindings,
}

#[derive(Debug, Clone)]
pub struct WorkspaceValueSnapshot {
    pub vars: Vec<Value>,
    pub names: HashMap<String, usize>,
    pub assigned: HashSet<String>,
    pub global_aliases: HashMap<usize, FrameGlobalBinding>,
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
    pub global_aliases: HashMap<usize, FrameGlobalBinding>,
}

#[derive(Debug, Clone, Default)]
pub struct WorkspaceAssignedReport {
    pub ids: HashSet<usize>,
    pub names: HashSet<String>,
    pub removed_ids: HashSet<usize>,
    pub removed_names: HashSet<String>,
}

runmat_thread_local! {
    static WORKSPACE_STACKS: RefCell<HashMap<WorkspaceOwner, Vec<WorkspaceFrame>>> = RefCell::new(HashMap::new());
    static LAST_WORKSPACE_STATES: RefCell<HashMap<WorkspaceFrameId, WorkspaceValueSnapshot>> = RefCell::new(HashMap::new());
    static LAST_WORKSPACE_ASSIGNED_REPORTS: RefCell<HashMap<WorkspaceFrameId, WorkspaceAssignedReport>> = RefCell::new(HashMap::new());
    static LAST_COMPLETED_WORKSPACE_FRAMES: RefCell<HashMap<WorkspaceOwner, WorkspaceFrameId>> = RefCell::new(HashMap::new());
    static PENDING_SERVICE_FRAMES: RefCell<HashMap<WorkspaceOwner, Vec<PendingServiceFrame>>> = RefCell::new(HashMap::new());
    static NEXT_WORKSPACE_FRAME_ID: Cell<usize> = const { Cell::new(1) };
}

fn with_stack<R>(owner: WorkspaceOwner, f: impl FnOnce(&[WorkspaceFrame]) -> R) -> R {
    WORKSPACE_STACKS.with(|stacks| {
        let stacks = stacks.borrow();
        let stack = stacks.get(&owner).map(Vec::as_slice).unwrap_or_default();
        let active = active_frame(owner)
            .and_then(|id| stack.iter().position(|frame| frame.id == id))
            .map(|index| &stack[..=index])
            .unwrap_or(stack);
        f(active)
    })
}

fn with_owner_stack_mut<R>(
    owner: WorkspaceOwner,
    f: impl FnOnce(&mut Vec<WorkspaceFrame>) -> R,
) -> R {
    WORKSPACE_STACKS.with(|stacks| {
        let mut stacks = stacks.borrow_mut();
        let result = f(stacks.entry(owner).or_default());
        if stacks.get(&owner).is_some_and(Vec::is_empty) {
            stacks.remove(&owner);
        }
        result
    })
}

fn with_active_frame_mut<R>(
    owner: WorkspaceOwner,
    operation: impl FnOnce(&mut WorkspaceFrame) -> R,
) -> Option<R> {
    WORKSPACE_STACKS.with(|stacks| {
        let mut stacks = stacks.borrow_mut();
        let stack = stacks.get_mut(&owner)?;
        let active = active_frame(owner).or_else(|| (stack.len() == 1).then(|| stack[0].id))?;
        let frame = stack.iter_mut().find(|frame| frame.id == active)?;
        Some(operation(frame))
    })
}

fn with_frame_by_id<R>(
    owner: WorkspaceOwner,
    id: WorkspaceFrameId,
    operation: impl FnOnce(&WorkspaceFrame) -> R,
) -> Option<R> {
    WORKSPACE_STACKS.with(|stacks| {
        let stacks = stacks.borrow();
        stacks
            .get(&owner)?
            .iter()
            .find(|frame| frame.id == id)
            .map(operation)
    })
}

fn with_frame_by_id_mut<R>(
    owner: WorkspaceOwner,
    id: WorkspaceFrameId,
    operation: impl FnOnce(&mut WorkspaceFrame) -> R,
) -> Option<R> {
    WORKSPACE_STACKS.with(|stacks| {
        let mut stacks = stacks.borrow_mut();
        stacks
            .get_mut(&owner)?
            .iter_mut()
            .find(|frame| frame.id == id)
            .map(operation)
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

fn target_frame_id(owner: WorkspaceOwner, target: WorkspaceTarget) -> Option<WorkspaceFrameId> {
    with_stack(owner, |stack| {
        let current = active_frame(owner).or_else(|| (stack.len() == 1).then(|| stack[0].id))?;
        let frame = stack.iter().find(|frame| frame.id == current)?;
        Some(match target {
            WorkspaceTarget::Current => current,
            WorkspaceTarget::Caller => frame.parent.unwrap_or(current),
            WorkspaceTarget::Base => frame.base,
        })
    })
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
    frame: WorkspaceFrameId,
}

impl Drop for WorkspaceStateGuard {
    fn drop(&mut self) {
        let Some(frame) = with_owner_stack_mut(self.owner, |stack| {
            stack
                .iter()
                .position(|frame| frame.id == self.frame)
                .map(|index| stack.remove(index))
        }) else {
            return;
        };
        frame.lifecycle.set(WorkspaceFrameLifecycle::Closed);
        let retired_delegate = frame.delegate.borrow_mut().take();
        if frame.policy.captures_completion() {
            let completed_frame = frame.id;
            let ws = frame.state;
            let removed_ids = ws.removed_slots_this_execution.keys().copied().collect();
            let removed_names = ws.removed_slots_this_execution.values().cloned().collect();
            LAST_WORKSPACE_ASSIGNED_REPORTS.with(|reports| {
                reports.borrow_mut().insert(
                    completed_frame,
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
                    completed_frame,
                    WorkspaceValueSnapshot {
                        vars: ws.values,
                        names: ws.names,
                        assigned: ws.assigned,
                        global_aliases: frame.global_aliases.snapshot(),
                    },
                );
            });
            LAST_COMPLETED_WORKSPACE_FRAMES
                .with(|completed| completed.borrow_mut().insert(self.owner, completed_frame));
        }
        drop(retired_delegate);
    }
}

pub(crate) fn take_pending_workspace_state() -> Option<PendingWorkspace> {
    frame::take_service_frame_admission(current_owner())
}

pub fn take_updated_workspace_state() -> Option<WorkspaceValueSnapshot> {
    let owner = current_owner();
    let frame = LAST_COMPLETED_WORKSPACE_FRAMES
        .with(|completed| completed.borrow().get(&owner).copied())?;
    let state = LAST_WORKSPACE_STATES.with(|states| states.borrow_mut().remove(&frame));
    clear_completed_frame_if_drained(owner, frame);
    state
}

pub fn take_updated_workspace_assigned_report() -> Option<WorkspaceAssignedReport> {
    let owner = current_owner();
    let frame = LAST_COMPLETED_WORKSPACE_FRAMES
        .with(|completed| completed.borrow().get(&owner).copied())?;
    let report =
        LAST_WORKSPACE_ASSIGNED_REPORTS.with(|reports| reports.borrow_mut().remove(&frame));
    clear_completed_frame_if_drained(owner, frame);
    report
}

pub(crate) fn take_workspace_completion(
    frame: WorkspaceFrameId,
) -> Option<WorkspaceExecutionCompletion> {
    let state = LAST_WORKSPACE_STATES.with(|states| states.borrow_mut().remove(&frame));
    let assigned =
        LAST_WORKSPACE_ASSIGNED_REPORTS.with(|reports| reports.borrow_mut().remove(&frame));
    LAST_COMPLETED_WORKSPACE_FRAMES.with(|completed| {
        completed
            .borrow_mut()
            .retain(|_, current| *current != frame);
    });
    match (state, assigned) {
        (Some(state), Some(assigned)) => Some(WorkspaceExecutionCompletion { state, assigned }),
        (None, None) => None,
        (state, assigned) => {
            debug_assert!(false, "workspace completion was only partially recorded");
            state.map(|state| WorkspaceExecutionCompletion {
                state,
                assigned: assigned.unwrap_or_default(),
            })
        }
    }
}

fn clear_completed_frame_if_drained(owner: WorkspaceOwner, frame: WorkspaceFrameId) {
    let has_state = LAST_WORKSPACE_STATES.with(|states| states.borrow().contains_key(&frame));
    let has_report =
        LAST_WORKSPACE_ASSIGNED_REPORTS.with(|reports| reports.borrow().contains_key(&frame));
    if !has_state && !has_report {
        LAST_COMPLETED_WORKSPACE_FRAMES.with(|completed| {
            let is_current = completed.borrow().get(&owner) == Some(&frame);
            if is_current {
                completed.borrow_mut().remove(&owner);
            }
        });
    }
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
    let frame_id = service_frame.id;
    let parent = service_frame.parent.get();
    let base = service_frame.base.get().unwrap_or(frame_id);
    let service = service_frame.service.borrow().clone();
    let session_variables = service_frame.session_variables.borrow().clone();
    with_owner_stack_mut(owner, |stack| {
        stack.push(WorkspaceFrame {
            id: service_frame.id,
            parent,
            base,
            lifecycle: service_frame.lifecycle,
            delegate: service_frame.delegate,
            service,
            session_variables,
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
    WorkspaceStateGuard {
        owner,
        frame: frame_id,
    }
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
    with_active_frame_mut(current_owner(), |frame| {
        frame.state.values.clone_from(&vars.to_vec());
    });
}

/// Refresh one execution-slot cache entry without cloning the full frame.
/// This does not change assignment lifecycle; declaration/import paths use it
/// before the corresponding language binding is admitted.
pub fn refresh_workspace_slot(index: usize, value: &Value) {
    with_active_frame_mut(current_owner(), |frame| {
        if frame.state.values.len() <= index {
            frame.state.values.resize(index + 1, Value::Num(0.0));
        }
        frame.state.values[index] = value.clone();
    });
}

pub fn reconcile_workspace_vars(vars: &mut Vec<Value>) {
    with_active_frame_mut(current_owner(), |frame| {
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
    let owner = current_owner();
    let frame = target_frame_id(owner, target)
        .ok_or_else(|| "load: workspace state unavailable".to_string())?;
    with_frame_by_id_mut(owner, frame, |frame| {
        set_workspace_variable_in_frame(frame, name, value)
    })
    .unwrap_or_else(|| Err("load: workspace state unavailable".to_string()))
}

pub fn workspace_clear() -> Result<(), String> {
    with_active_frame_mut(current_owner(), |frame| {
        clear_workspace_frame(frame);
        Ok(())
    })
    .unwrap_or_else(|| Err("clear: workspace state unavailable".to_string()))
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
    with_active_frame_mut(current_owner(), |frame| {
        remove_workspace_name(frame, name);
        Ok(())
    })
    .unwrap_or_else(|| Err("clear: workspace state unavailable".to_string()))
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
    let owner = current_owner();
    let frame =
        target_frame_id(owner, target).ok_or_else(|| "workspace state unavailable".to_string())?;
    with_frame_by_id(owner, frame, |frame| {
        Ok(WorkspaceTargetSnapshot {
            names: frame.state.names.clone(),
            assigned: frame.state.assigned.clone(),
            vars: frame.state.values.clone(),
            global_aliases: frame.global_aliases.snapshot(),
        })
    })
    .unwrap_or_else(|| Err("workspace state unavailable".to_string()))
}

pub fn replace_workspace_target_vars_and_state(
    target: WorkspaceTarget,
    vars: Vec<Value>,
    names: HashMap<String, usize>,
    assigned: HashSet<String>,
    global_aliases: HashMap<usize, FrameGlobalBinding>,
    report: WorkspaceAssignedReport,
) -> Result<(), runmat_runtime::RuntimeError> {
    let owner = current_owner();
    let target_frame = target_frame_id(owner, target)
        .ok_or_else(|| workspace_target_error("workspace state unavailable"))?;
    let (publishes, delegate, previous_globals) = with_frame_by_id(owner, target_frame, |frame| {
        (
            frame.policy.publishes_mutations(),
            frame.delegate.borrow().clone(),
            active_named_global_bindings(&frame.state.names, &frame.global_aliases.snapshot()),
        )
    })
    .ok_or_else(|| workspace_target_error("workspace state unavailable"))?;
    let next_globals = active_named_global_bindings(&names, &global_aliases);

    if publishes {
        if let Some(delegate) = delegate.as_ref() {
            delegate.ensure_mutation_authority()?;
        }
        let mut changed_names = report.names;
        changed_names.extend(report.removed_names);
        changed_names.extend(
            previous_globals
                .symmetric_difference(&next_globals)
                .cloned(),
        );
        let mut changed_names = changed_names.into_iter().collect::<Vec<_>>();
        changed_names.sort();
        for name in changed_names {
            if next_globals.contains(&name) {
                if let Some(delegate) = delegate.as_ref() {
                    delegate.bind_base_global(&name)?;
                }
            } else if assigned.contains(&name) {
                let slot = names.get(&name).copied().ok_or_else(|| {
                    workspace_target_error("assigned workspace name has no execution slot")
                })?;
                let value = vars.get(slot).cloned().ok_or_else(|| {
                    workspace_target_error("assigned workspace slot has no value")
                })?;
                runmat_runtime::execution::validate_storable_value(&value)?;
                if let Some(delegate) = delegate.as_ref() {
                    delegate.assign_base_local(&name, value)?;
                }
            } else {
                if let Some(delegate) = delegate.as_ref() {
                    delegate.unbind_base_global(&name)?;
                }
            }
        }
    }

    with_frame_by_id_mut(owner, target_frame, |frame| {
        let mut changed_slots = frame
            .state
            .slot_lifecycle
            .keys()
            .copied()
            .collect::<HashSet<_>>();
        changed_slots.extend(names.values().copied());
        frame.state.values = vars;
        frame.global_aliases.replace_with_snapshot(global_aliases);
        frame.state.names = names;
        frame.state.assigned = assigned;
        frame.state.slot_lifecycle =
            lifecycle_from_names(&frame.state.names, &frame.state.assigned);
        frame.state.pending_var_reconciliation.extend(changed_slots);
        Ok(())
    })
    .unwrap_or_else(|| Err(workspace_target_error("workspace state unavailable")))
}

fn active_named_global_bindings(
    names: &HashMap<String, usize>,
    global_aliases: &HashMap<usize, FrameGlobalBinding>,
) -> HashSet<String> {
    global_aliases
        .iter()
        .filter_map(|(slot, binding)| match binding {
            FrameGlobalBinding::Named(name) if names.get(name) == Some(slot) => Some(name.clone()),
            FrameGlobalBinding::LegacySlot => names
                .iter()
                .find_map(|(name, active_slot)| (active_slot == slot).then(|| name.clone())),
            FrameGlobalBinding::Named(_) => None,
        })
        .collect()
}

fn workspace_target_error(message: &'static str) -> runmat_runtime::RuntimeError {
    runmat_runtime::runtime_error::semantic_error("RunMat:Workspace:TargetState", message)
}

#[cfg(test)]
pub fn set_workspace_variable(
    name: &str,
    value: Value,
    vars: &mut Vec<Value>,
) -> Result<(), String> {
    with_active_frame_mut(current_owner(), |frame| {
        set_workspace_variable_in_frame(frame, name, value)
    })
    .unwrap_or_else(|| Err("load: workspace state unavailable".to_string()))?;
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
    set_workspace_variable_in_frame_with_reconciliation(frame, name, value, true, true)
}

fn set_workspace_variable_in_frame_with_reconciliation(
    frame: &mut WorkspaceFrame,
    name: &str,
    value: Value,
    reconcile_var: bool,
    store_global: bool,
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
    match (store_global, frame.global_aliases.binding_for_slot(idx)) {
        (true, Some(FrameGlobalBinding::Named(bound))) if bound == name => {
            if let Some(authority) = frame.session_variables.as_ref() {
                authority
                    .store_global(name, ws.values[idx].clone())
                    .map_err(|error| error.to_string())?;
            } else {
                #[cfg(test)]
                runmat_runtime::workspace::session::store_global_named(
                    name,
                    ws.values[idx].clone(),
                );
                #[cfg(not(test))]
                return Err("workspace frame has no session-variable authority".into());
            }
        }
        (true, Some(FrameGlobalBinding::LegacySlot)) => {
            if let Some(authority) = frame.session_variables.as_ref() {
                authority
                    .store_global_slot(idx, ws.values[idx].clone())
                    .map_err(|error| error.to_string())?;
            } else {
                #[cfg(test)]
                runmat_runtime::workspace::session::store_global_slot(idx, ws.values[idx].clone());
                #[cfg(not(test))]
                return Err("workspace frame has no session-variable authority".into());
            }
        }
        (true, Some(FrameGlobalBinding::Named(_))) | (true, None) | (false, _) => {}
    }
    ws.assigned.insert(name.to_string());
    ws.assigned_names_this_execution.insert(name.to_string());
    ws.assigned_ids_this_execution.insert(idx);
    mark_slot_assigned(ws, idx, name.to_string());
    Ok(())
}

pub fn ensure_workspace_slot_name(index: usize, name: &str) {
    with_active_frame_mut(current_owner(), |frame| {
        upsert_slot_lifecycle_name(&mut frame.state, index, name);
    });
}

pub fn mark_workspace_assigned(index: usize, value: &Value) {
    with_active_frame_mut(current_owner(), |frame| {
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
    activation::reset_for_tests();
    WORKSPACE_STACKS.with(|stacks| stacks.borrow_mut().clear());
    LAST_WORKSPACE_STATES.with(|states| states.borrow_mut().clear());
    LAST_WORKSPACE_ASSIGNED_REPORTS.with(|reports| reports.borrow_mut().clear());
    LAST_COMPLETED_WORKSPACE_FRAMES.with(|completed| completed.borrow_mut().clear());
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
        let first_frame = reserve_workspace_service_frame(first.local_identity(), None);
        let second_frame = reserve_workspace_service_frame(second.local_identity(), None);
        let mut first = Box::pin(
            frame_activation(first.local_identity(), first_frame.id)
                .scope(first.scope(session("first"))),
        );
        let mut second = Box::pin(
            frame_activation(second.local_identity(), second_frame.id)
                .scope(second.scope(session("second"))),
        );
        assert!(poll(first.as_mut()).is_pending());
        assert!(poll(second.as_mut()).is_pending());
        assert!(poll(first.as_mut()).is_ready());
        assert!(poll(second.as_mut()).is_ready());
    }

    #[test]
    fn active_frame_never_consumes_a_different_sole_reservation() {
        reset_thread_state_for_tests();
        let runtime = runmat_runtime::context::RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        let owner = runtime.local_identity();
        let missing = reserve_workspace_service_frame(owner, None);
        let remaining = reserve_workspace_service_frame(owner, None);
        release_pending_workspace_service_frame(owner, missing.id);
        let activation = frame_activation(owner, missing.id);

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            futures::executor::block_on(activation.scope(runtime.scope(async {
                let mut vars = vec![Value::Num(1.0)];
                let _guard = set_transient_workspace_state(
                    HashMap::from([("x".to_string(), 0)]),
                    HashSet::from(["x".to_string()]),
                    &mut vars,
                    FrameGlobalBindings::default(),
                );
            })));
        }));
        assert!(result.is_err());
        PENDING_SERVICE_FRAMES.with(|frames| {
            assert_eq!(
                frames
                    .borrow()
                    .get(&WorkspaceOwner::Runtime(owner))
                    .and_then(|pending| pending.first())
                    .map(|frame| frame.id),
                Some(remaining.id)
            );
        });
        reset_thread_state_for_tests();
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
        let reserved = reserve_workspace_service_frame(runtime_id, None);
        let activation = frame_activation(runtime_id, reserved.id);
        futures::executor::block_on(activation.scope(runtime.scope(async {
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
        })));
    }

    #[test]
    fn executor_admission_does_not_schedule_duplicate_register_reconciliation() {
        let runtime = runmat_runtime::context::RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        let runtime_id = runtime.local_identity();
        let reserved = reserve_workspace_service_frame(runtime_id, None);
        let activation = frame_activation(runtime_id, reserved.id);
        futures::executor::block_on(activation.scope(runtime.scope(async {
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
        })));
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
        let runtime_id = runtime.local_identity();
        let reserved = reserve_workspace_service_frame(runtime_id, None);
        let activation = frame_activation(runtime_id, reserved.id);
        futures::executor::block_on(activation.scope(runtime.scope(async {
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
        })));
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
