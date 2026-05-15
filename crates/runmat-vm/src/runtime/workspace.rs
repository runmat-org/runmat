use runmat_builtins::Value;
use runmat_thread_local::runmat_thread_local;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

struct WorkspaceState {
    names: HashMap<String, usize>,
    assigned: HashSet<String>,
    assigned_names_this_execution: HashSet<String>,
    assigned_ids_this_execution: HashSet<usize>,
    idx_to_name: HashMap<usize, String>,
    data_ptr: *const Value,
    len: usize,
}

pub type WorkspaceSnapshot = (HashMap<String, usize>, HashSet<String>);

#[derive(Debug, Clone)]
pub struct WorkspaceAssignedReport {
    pub ids: HashSet<usize>,
    pub names: HashSet<String>,
}

runmat_thread_local! {
    static WORKSPACE_STATE: RefCell<Option<WorkspaceState>> = const { RefCell::new(None) };
    static PENDING_WORKSPACE: RefCell<Option<WorkspaceSnapshot>> = const { RefCell::new(None) };
    static LAST_WORKSPACE_STATE: RefCell<Option<WorkspaceSnapshot>> = const { RefCell::new(None) };
    static LAST_WORKSPACE_ASSIGNED_REPORT: RefCell<Option<WorkspaceAssignedReport>> = const { RefCell::new(None) };
    static WORKSPACE_VARS: RefCell<Option<*mut Vec<Value>>> = const { RefCell::new(None) };
}

pub struct WorkspaceStateGuard;

impl Drop for WorkspaceStateGuard {
    fn drop(&mut self) {
        WORKSPACE_STATE.with(|state| {
            let mut state_mut = state.borrow_mut();
            if let Some(ws) = state_mut.take() {
                LAST_WORKSPACE_ASSIGNED_REPORT.with(|slot| {
                    *slot.borrow_mut() = Some(WorkspaceAssignedReport {
                        ids: ws.assigned_ids_this_execution,
                        names: ws.assigned_names_this_execution,
                    });
                });
                LAST_WORKSPACE_STATE.with(|slot| {
                    *slot.borrow_mut() = Some((ws.names, ws.assigned));
                });
            }
        });
        WORKSPACE_VARS.with(|slot| {
            slot.borrow_mut().take();
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

pub fn take_updated_workspace_state() -> Option<WorkspaceSnapshot> {
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
    let idx_to_name: HashMap<usize, String> = names.iter().map(|(k, &v)| (v, k.clone())).collect();
    WORKSPACE_STATE.with(|state| {
        *state.borrow_mut() = Some(WorkspaceState {
            names,
            assigned,
            assigned_names_this_execution: HashSet::new(),
            assigned_ids_this_execution: HashSet::new(),
            idx_to_name,
            data_ptr: vars.as_ptr(),
            len: vars.len(),
        });
    });
    let vars_ptr = vars as *mut Vec<Value>;
    WORKSPACE_VARS.with(|slot| {
        *slot.borrow_mut() = Some(vars_ptr);
    });
    WorkspaceStateGuard
}

pub fn refresh_workspace_state(vars: &[Value]) {
    WORKSPACE_STATE.with(|state| {
        if let Some(ws) = state.borrow_mut().as_mut() {
            ws.data_ptr = vars.as_ptr();
            ws.len = vars.len();
        }
    });
}

pub fn workspace_lookup(name: &str) -> Option<Value> {
    WORKSPACE_STATE.with(|state| {
        let state_ref = state.borrow();
        let ws = state_ref.as_ref()?;
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
    WORKSPACE_STATE.with(|state| {
        let state_ref = state.borrow();
        let ws = state_ref.as_ref()?;
        let name = ws.idx_to_name.get(&index)?;
        Some(ws.assigned.contains(name))
    })
}

pub fn workspace_state_available() -> bool {
    WORKSPACE_STATE.with(|state| state.borrow().is_some())
}

pub fn workspace_assign(name: &str, value: Value) -> Result<(), String> {
    let vars_ptr = WORKSPACE_VARS.with(|slot| *slot.borrow());
    let Some(vars_ptr) = vars_ptr else {
        return Err("load: workspace state unavailable".to_string());
    };
    let vars = unsafe { &mut *vars_ptr };
    set_workspace_variable(name, value, vars)
}

pub fn workspace_clear() -> Result<(), String> {
    let vars_ptr = WORKSPACE_VARS.with(|slot| *slot.borrow());
    let Some(vars_ptr) = vars_ptr else {
        return Err("clear: workspace state unavailable".to_string());
    };
    let vars = unsafe { &mut *vars_ptr };

    WORKSPACE_STATE.with(|state| {
        let mut state_mut = state.borrow_mut();
        let Some(ws) = state_mut.as_mut() else {
            return Err("clear: workspace state unavailable".to_string());
        };
        vars.clear();
        ws.names.clear();
        ws.assigned.clear();
        ws.assigned_names_this_execution.clear();
        ws.assigned_ids_this_execution.clear();
        ws.idx_to_name.clear();
        ws.data_ptr = vars.as_ptr();
        ws.len = vars.len();
        Ok(())
    })
}

pub fn workspace_remove(name: &str) -> Result<(), String> {
    let vars_ptr = WORKSPACE_VARS.with(|slot| *slot.borrow());
    let Some(vars_ptr) = vars_ptr else {
        return Err("clear: workspace state unavailable".to_string());
    };
    let vars = unsafe { &mut *vars_ptr };

    WORKSPACE_STATE.with(|state| {
        let mut state_mut = state.borrow_mut();
        let Some(ws) = state_mut.as_mut() else {
            return Err("clear: workspace state unavailable".to_string());
        };
        if let Some(idx) = ws.names.remove(name) {
            if idx < vars.len() {
                vars[idx] = Value::Num(0.0);
            }
            ws.assigned.remove(name);
            ws.assigned_names_this_execution.remove(name);
            ws.idx_to_name.remove(&idx);
            ws.assigned_ids_this_execution.remove(&idx);
            ws.data_ptr = vars.as_ptr();
            ws.len = vars.len();
        }
        Ok(())
    })
}

pub fn workspace_snapshot() -> Vec<(String, Value)> {
    WORKSPACE_STATE.with(|state| {
        if let Some(ws) = state.borrow().as_ref() {
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

pub fn set_workspace_variable(
    name: &str,
    value: Value,
    vars: &mut Vec<Value>,
) -> Result<(), String> {
    let mut result = Ok(());
    WORKSPACE_STATE.with(|state| {
        let mut state_mut = state.borrow_mut();
        match state_mut.as_mut() {
            Some(ws) => {
                let idx = if let Some(idx) = ws.names.get(name).copied() {
                    idx
                } else {
                    let idx = vars.len();
                    ws.names.insert(name.to_string(), idx);
                    ws.idx_to_name.insert(idx, name.to_string());
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
            }
            None => {
                result = Err("load: workspace state unavailable".to_string());
            }
        }
    });
    result
}

pub fn ensure_workspace_slot_name(index: usize, name: &str) {
    WORKSPACE_STATE.with(|state| {
        if let Some(ws) = state.borrow_mut().as_mut() {
            ws.names.entry(name.to_string()).or_insert(index);
            ws.idx_to_name
                .entry(index)
                .or_insert_with(|| name.to_string());
        }
    });
}

pub fn mark_workspace_assigned(index: usize) {
    WORKSPACE_STATE.with(|state| {
        if let Some(ws) = state.borrow_mut().as_mut() {
            if let Some(name) = ws.idx_to_name.get(&index).cloned() {
                ws.assigned.insert(name.clone());
                ws.assigned_names_this_execution.insert(name);
                ws.assigned_ids_this_execution.insert(index);
            }
        }
    });
}
