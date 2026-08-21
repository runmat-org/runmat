use crate::bytecode::instr::Instr;
use crate::bytecode::program::Bytecode;
use crate::runtime::workspace::refresh_workspace_state;
use runmat_value::Value;
use std::collections::HashMap;

pub fn workspace_global_names() -> Vec<String> {
    runmat_runtime::workspace::session::global_names()
}

pub fn get_global_value(name: &str) -> Option<Value> {
    runmat_runtime::workspace::session::global_value(name)
}

pub fn collect_thread_roots() -> Vec<Value> {
    runmat_runtime::workspace::session::roots()
}

pub fn update_global_store(
    stored_index: usize,
    stored_value: &Value,
    global_aliases: &HashMap<usize, String>,
) {
    runmat_runtime::workspace::session::update_global_slot(
        stored_index,
        global_aliases.get(&stored_index).map(String::as_str),
        stored_value,
    );
}

pub fn update_persistent_local_store(func_name: &str, stored_offset: usize, stored_value: &Value) {
    runmat_runtime::workspace::session::update_persistent_slot(
        func_name,
        stored_offset,
        stored_value,
    );
}

pub fn declare_global(indices: Vec<usize>, vars: &mut Vec<Value>) {
    for i in indices {
        let val_opt = runmat_runtime::workspace::session::global_slot_value(i);
        if let Some(v) = val_opt {
            if i >= vars.len() {
                vars.resize(i + 1, Value::Num(0.0));
                refresh_workspace_state(vars);
            }
            vars[i] = v;
            refresh_workspace_state(vars);
        }
    }
}

pub fn declare_global_named(
    indices: Vec<usize>,
    names: Vec<String>,
    vars: &mut Vec<Value>,
    global_aliases: &mut HashMap<usize, String>,
) {
    for (pos, i) in indices.into_iter().enumerate() {
        let name = names
            .get(pos)
            .cloned()
            .unwrap_or_else(|| format!("var_{i}"));
        let val_opt = runmat_runtime::workspace::session::global_value(&name);
        if let Some(v) = val_opt {
            if i >= vars.len() {
                vars.resize(i + 1, Value::Num(0.0));
                refresh_workspace_state(vars);
            }
            vars[i] = v;
            refresh_workspace_state(vars);
        }
        runmat_runtime::workspace::session::bind_global_slot(i, &name);
        global_aliases.insert(i, name);
    }
}

pub fn declare_persistent(func_name: &str, indices: Vec<usize>, vars: &mut Vec<Value>) {
    for i in indices {
        let val_opt = runmat_runtime::workspace::session::persistent_slot_value(func_name, i);
        if let Some(v) = val_opt {
            if i >= vars.len() {
                vars.resize(i + 1, Value::Num(0.0));
                refresh_workspace_state(vars);
            }
            vars[i] = v;
            refresh_workspace_state(vars);
        }
    }
}

pub fn declare_persistent_named(
    func_name: &str,
    indices: Vec<usize>,
    names: Vec<String>,
    vars: &mut Vec<Value>,
    persistent_aliases: &mut HashMap<usize, String>,
) {
    for (pos, i) in indices.into_iter().enumerate() {
        let name = names
            .get(pos)
            .cloned()
            .unwrap_or_else(|| format!("var_{i}"));
        let val_opt = runmat_runtime::workspace::session::persistent_named_value(func_name, &name)
            .or_else(|| runmat_runtime::workspace::session::persistent_slot_value(func_name, i));
        if let Some(v) = val_opt {
            if i >= vars.len() {
                vars.resize(i + 1, Value::Num(0.0));
                refresh_workspace_state(vars);
            }
            vars[i] = v;
            refresh_workspace_state(vars);
        }
        persistent_aliases.insert(i, name);
    }
}

pub fn persist_declared_for_bytecode(bytecode: &Bytecode, func_name: &str, vars: &[Value]) {
    for instr in &bytecode.instructions {
        match instr {
            Instr::DeclarePersistent(indices) => {
                for &i in indices {
                    if i < vars.len() {
                        runmat_runtime::workspace::session::store_persistent_slot(
                            func_name,
                            i,
                            vars[i].clone(),
                        );
                    }
                }
            }
            Instr::DeclarePersistentNamed(indices, names) => {
                for (pos, &i) in indices.iter().enumerate() {
                    if i < vars.len() {
                        let name = names
                            .get(pos)
                            .cloned()
                            .unwrap_or_else(|| format!("var_{i}"));
                        let val = vars[i].clone();
                        runmat_runtime::workspace::session::store_persistent_slot(
                            func_name,
                            i,
                            val.clone(),
                        );
                        runmat_runtime::workspace::session::store_persistent_named(
                            func_name, &name, val,
                        );
                    }
                }
            }
            _ => {}
        }
    }
}

pub(crate) fn reset_thread_state_for_tests() {
    runmat_runtime::workspace::session::reset_legacy_state_for_tests();
}
