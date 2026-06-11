use runmat_builtins::Value;
use runmat_thread_local::runmat_thread_local;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

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
    data_ptr: *const Value,
    len: usize,
}

struct WorkspaceFrame {
    state: WorkspaceState,
    vars_ptr: *mut Vec<Value>,
    publish_on_drop: bool,
}

pub type WorkspaceSnapshot = (HashMap<String, usize>, HashSet<String>);

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
    pub vars_ptr: *mut Vec<Value>,
}

#[derive(Debug, Clone)]
pub struct WorkspaceAssignedReport {
    pub ids: HashSet<usize>,
    pub names: HashSet<String>,
    pub removed_ids: HashSet<usize>,
    pub removed_names: HashSet<String>,
}

runmat_thread_local! {
    static WORKSPACE_STACK: RefCell<Vec<WorkspaceFrame>> = const { RefCell::new(Vec::new()) };
    static PENDING_WORKSPACE: RefCell<Option<WorkspaceSnapshot>> = const { RefCell::new(None) };
    static LAST_WORKSPACE_STATE: RefCell<Option<WorkspaceValueSnapshot>> = const { RefCell::new(None) };
    static LAST_WORKSPACE_ASSIGNED_REPORT: RefCell<Option<WorkspaceAssignedReport>> = const { RefCell::new(None) };
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

pub struct WorkspaceStateGuard;

impl Drop for WorkspaceStateGuard {
    fn drop(&mut self) {
        WORKSPACE_STACK.with(|stack| {
            let mut stack = stack.borrow_mut();
            if let Some(frame) = stack.pop() {
                if !frame.publish_on_drop {
                    return;
                }
                let ws = frame.state;
                let removed_ids = ws.removed_slots_this_execution.keys().copied().collect();
                let removed_names = ws.removed_slots_this_execution.values().cloned().collect();
                LAST_WORKSPACE_ASSIGNED_REPORT.with(|slot| {
                    *slot.borrow_mut() = Some(WorkspaceAssignedReport {
                        ids: ws.assigned_ids_this_execution,
                        names: ws.assigned_names_this_execution,
                        removed_ids,
                        removed_names,
                    });
                });
                let vars = unsafe { (*frame.vars_ptr).clone() };
                LAST_WORKSPACE_STATE.with(|slot| {
                    *slot.borrow_mut() = Some(WorkspaceValueSnapshot {
                        vars,
                        names: ws.names,
                        assigned: ws.assigned,
                    });
                });
            }
        });
    }
}

pub struct PendingWorkspaceGuard;

impl Drop for PendingWorkspaceGuard {
    fn drop(&mut self) {
        PENDING_WORKSPACE.with(|slot| {
            slot.borrow_mut().take();
        });
    }
}

pub fn push_pending_workspace(
    names: HashMap<String, usize>,
    assigned: HashSet<String>,
) -> PendingWorkspaceGuard {
    PENDING_WORKSPACE.with(|slot| {
        *slot.borrow_mut() = Some((names, assigned));
    });
    PendingWorkspaceGuard
}

pub fn take_pending_workspace_state() -> Option<WorkspaceSnapshot> {
    PENDING_WORKSPACE.with(|slot| slot.borrow_mut().take())
}

pub fn take_updated_workspace_state() -> Option<WorkspaceValueSnapshot> {
    LAST_WORKSPACE_STATE.with(|slot| slot.borrow_mut().take())
}

pub fn take_updated_workspace_assigned_report() -> Option<WorkspaceAssignedReport> {
    LAST_WORKSPACE_ASSIGNED_REPORT.with(|slot| slot.borrow_mut().take())
}

pub fn set_workspace_state(
    names: HashMap<String, usize>,
    assigned: HashSet<String>,
    vars: &mut Vec<Value>,
) -> WorkspaceStateGuard {
    set_workspace_state_with_publish(names, assigned, vars, true)
}

pub fn set_transient_workspace_state(
    names: HashMap<String, usize>,
    assigned: HashSet<String>,
    vars: &mut Vec<Value>,
) -> WorkspaceStateGuard {
    set_workspace_state_with_publish(names, assigned, vars, false)
}

fn set_workspace_state_with_publish(
    names: HashMap<String, usize>,
    assigned: HashSet<String>,
    vars: &mut Vec<Value>,
    publish_on_drop: bool,
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
    let vars_ptr = vars as *mut Vec<Value>;
    WORKSPACE_STACK.with(|stack| {
        stack.borrow_mut().push(WorkspaceFrame {
            state: WorkspaceState {
                names,
                assigned,
                assigned_names_this_execution: HashSet::new(),
                assigned_ids_this_execution: HashSet::new(),
                removed_slots_this_execution: HashMap::new(),
                slot_lifecycle,
                data_ptr: vars.as_ptr(),
                len: vars.len(),
            },
            vars_ptr,
            publish_on_drop,
        });
    });
    WorkspaceStateGuard
}

pub fn refresh_workspace_state(vars: &[Value]) {
    WORKSPACE_STACK.with(|stack| {
        if let Some(frame) = stack.borrow_mut().last_mut() {
            frame.state.data_ptr = vars.as_ptr();
            frame.state.len = vars.len();
        }
    });
}

pub fn workspace_lookup(name: &str) -> Option<Value> {
    WORKSPACE_STACK.with(|stack| {
        let stack = stack.borrow();
        let frame = stack.last()?;
        let ws = &frame.state;
        let idx = ws.names.get(name)?;
        if !ws.assigned.contains(name) {
            return None;
        }
        if *idx >= ws.len {
            return None;
        }
        unsafe {
            let ptr = ws.data_ptr.add(*idx);
            Some((*ptr).clone())
        }
    })
}

pub fn workspace_slot_assigned(index: usize) -> Option<bool> {
    WORKSPACE_STACK.with(|stack| {
        let stack = stack.borrow();
        let frame = stack.last()?;
        let ws = &frame.state;
        ws.slot_lifecycle
            .get(&index)
            .map(SlotLifecycle::is_assigned)
    })
}

pub fn workspace_slot_name(index: usize) -> Option<String> {
    WORKSPACE_STACK.with(|stack| {
        let stack = stack.borrow();
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
    WORKSPACE_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        let index = target_frame_index(stack.len(), target)
            .ok_or_else(|| "load: workspace state unavailable".to_string())?;
        let frame = stack
            .get_mut(index)
            .ok_or_else(|| "load: workspace state unavailable".to_string())?;
        set_workspace_variable_in_frame(frame, name, value)
    })
}

pub fn workspace_clear() -> Result<(), String> {
    WORKSPACE_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        let frame = stack
            .last_mut()
            .ok_or_else(|| "clear: workspace state unavailable".to_string())?;
        let ws = &mut frame.state;
        let vars = unsafe { &mut *frame.vars_ptr };
        vars.clear();
        for (name, idx) in ws.names.clone() {
            mark_slot_unassigned(ws, idx, name);
        }
        ws.names.clear();
        ws.assigned.clear();
        ws.data_ptr = vars.as_ptr();
        ws.len = vars.len();
        Ok(())
    })
}

pub fn workspace_remove(name: &str) -> Result<(), String> {
    WORKSPACE_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        let frame = stack
            .last_mut()
            .ok_or_else(|| "clear: workspace state unavailable".to_string())?;
        let ws = &mut frame.state;
        let vars = unsafe { &mut *frame.vars_ptr };
        if let Some(idx) = ws.names.remove(name) {
            ws.assigned.remove(name);
            mark_slot_unassigned(ws, idx, name.to_string());
            ws.data_ptr = vars.as_ptr();
            ws.len = vars.len();
        }
        Ok(())
    })
}

pub fn workspace_snapshot() -> Vec<(String, Value)> {
    WORKSPACE_STACK.with(|stack| {
        let stack = stack.borrow();
        if let Some(frame) = stack.last() {
            let ws = &frame.state;
            let mut entries: Vec<(String, Value)> = ws
                .names
                .iter()
                .filter_map(|(name, idx)| {
                    if *idx >= ws.len {
                        return None;
                    }
                    if !ws.assigned.contains(name) {
                        return None;
                    }
                    unsafe {
                        let ptr = ws.data_ptr.add(*idx);
                        Some((name.clone(), (*ptr).clone()))
                    }
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
    WORKSPACE_STACK.with(|stack| {
        let stack = stack.borrow();
        let index = target_frame_index(stack.len(), target)
            .ok_or_else(|| "workspace state unavailable".to_string())?;
        let frame = stack
            .get(index)
            .ok_or_else(|| "workspace state unavailable".to_string())?;
        Ok(WorkspaceTargetSnapshot {
            names: frame.state.names.clone(),
            assigned: frame.state.assigned.clone(),
            vars_ptr: frame.vars_ptr,
        })
    })
}

pub fn replace_workspace_target_vars_and_state(
    target: WorkspaceTarget,
    vars: Vec<Value>,
    names: HashMap<String, usize>,
    assigned: HashSet<String>,
) -> Result<(), String> {
    WORKSPACE_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        let index = target_frame_index(stack.len(), target)
            .ok_or_else(|| "workspace state unavailable".to_string())?;
        let frame = stack
            .get_mut(index)
            .ok_or_else(|| "workspace state unavailable".to_string())?;
        let target_vars = unsafe { &mut *frame.vars_ptr };
        *target_vars = vars;
        frame.state.names = names;
        frame.state.assigned = assigned;
        frame.state.slot_lifecycle =
            lifecycle_from_names(&frame.state.names, &frame.state.assigned);
        frame.state.data_ptr = target_vars.as_ptr();
        frame.state.len = target_vars.len();
        Ok(())
    })
}

#[cfg(test)]
pub fn set_workspace_variable(
    name: &str,
    value: Value,
    vars: &mut Vec<Value>,
) -> Result<(), String> {
    WORKSPACE_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        match stack.last_mut() {
            Some(frame) => set_workspace_variable_in_frame(frame, name, value),
            None => Err("load: workspace state unavailable".to_string()),
        }
    })?;
    let _ = vars;
    Ok(())
}

fn set_workspace_variable_in_frame(
    frame: &mut WorkspaceFrame,
    name: &str,
    value: Value,
) -> Result<(), String> {
    let vars = unsafe { &mut *frame.vars_ptr };
    let ws = &mut frame.state;
    let idx = if let Some(idx) = ws.names.get(name).copied() {
        idx
    } else if let Some(idx) = find_unassigned_slot_for_name(ws, name) {
        ws.names.insert(name.to_string(), idx);
        idx
    } else {
        let idx = vars.len();
        ws.names.insert(name.to_string(), idx);
        idx
    };
    if idx >= vars.len() {
        vars.resize(idx + 1, Value::Num(0.0));
    }
    vars[idx] = value;
    ws.data_ptr = vars.as_ptr();
    ws.len = vars.len();
    ws.assigned.insert(name.to_string());
    ws.assigned_names_this_execution.insert(name.to_string());
    ws.assigned_ids_this_execution.insert(idx);
    mark_slot_assigned(ws, idx, name.to_string());
    Ok(())
}

pub fn ensure_workspace_slot_name(index: usize, name: &str) {
    WORKSPACE_STACK.with(|stack| {
        if let Some(frame) = stack.borrow_mut().last_mut() {
            upsert_slot_lifecycle_name(&mut frame.state, index, name);
        }
    });
}

pub fn mark_workspace_assigned(index: usize) {
    WORKSPACE_STACK.with(|stack| {
        if let Some(frame) = stack.borrow_mut().last_mut() {
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
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn take_report_after(f: impl FnOnce(&mut Vec<Value>)) -> WorkspaceAssignedReport {
        let _ = take_updated_workspace_assigned_report();
        let _ = take_updated_workspace_state();

        let mut vars = Vec::new();
        {
            let _guard = set_workspace_state(HashMap::new(), HashSet::new(), &mut vars);
            f(&mut vars);
        }

        take_updated_workspace_assigned_report().expect("workspace report should be recorded")
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
            let _guard = set_workspace_state(HashMap::new(), HashSet::new(), &mut vars);
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
}
