use crate::bytecode::instr::Instr;
use crate::bytecode::program::Bytecode;
use crate::runtime::workspace::refresh_workspace_slot;
use runmat_runtime::workspace::bindings::{FrameGlobalBinding, FrameGlobalBindings};
use runmat_value::Value;
use std::collections::HashMap;

pub fn workspace_global_names() -> Vec<String> {
    runmat_runtime::workspace::session::global_names()
}

pub fn get_global_value(name: &str) -> Option<Value> {
    runmat_runtime::workspace::session::global_value(name)
}

pub fn current_bound_value(bindings: &FrameGlobalBindings, slot: usize) -> Option<Value> {
    match bindings.binding_for_slot(slot) {
        Some(FrameGlobalBinding::Named(name)) => get_global_value(&name),
        Some(FrameGlobalBinding::LegacySlot) => {
            runmat_runtime::workspace::session::global_slot_value(slot)
        }
        None => None,
    }
}

pub fn collect_thread_roots() -> Vec<Value> {
    runmat_runtime::workspace::session::roots()
}

pub fn update_global_store(
    stored_index: usize,
    stored_value: &Value,
    global_aliases: &FrameGlobalBindings,
) {
    match global_aliases.binding_for_slot(stored_index) {
        Some(FrameGlobalBinding::Named(alias)) => {
            runmat_runtime::workspace::session::store_global_named(&alias, stored_value.clone());
        }
        Some(FrameGlobalBinding::LegacySlot) => {
            runmat_runtime::workspace::session::store_global_slot(
                stored_index,
                stored_value.clone(),
            );
        }
        None => {}
    }
}

pub fn update_persistent_local_store(func_name: &str, stored_offset: usize, stored_value: &Value) {
    runmat_runtime::workspace::session::update_persistent_slot(
        func_name,
        stored_offset,
        stored_value,
    );
}

pub fn declare_global(
    indices: Vec<usize>,
    vars: &mut Vec<Value>,
    global_aliases: &FrameGlobalBindings,
) {
    for i in indices {
        global_aliases.declare_legacy_slot(i);
        let val_opt = runmat_runtime::workspace::session::global_slot_value(i);
        if let Some(v) = val_opt {
            if i >= vars.len() {
                vars.resize(i + 1, Value::Num(0.0));
            }
            vars[i] = v;
            refresh_workspace_slot(i, &vars[i]);
        }
    }
}

pub fn declare_global_named(
    indices: Vec<usize>,
    names: Vec<String>,
    vars: &mut Vec<Value>,
    global_aliases: &FrameGlobalBindings,
) -> Result<(), runmat_runtime::RuntimeError> {
    for (pos, i) in indices.into_iter().enumerate() {
        let name = names
            .get(pos)
            .cloned()
            .unwrap_or_else(|| format!("var_{i}"));
        let value = if let Some(runtime) = runmat_runtime::context::legacy::active() {
            if let Some(workspace) = runtime.service_ports().workspace() {
                workspace.declare_global_binding(i, &name)?;
            }
            runtime.session_variable_authority().declare_global(&name)?
        } else {
            runmat_runtime::workspace::session::declare_global_named(&name)
        };
        if i >= vars.len() {
            vars.resize(i + 1, Value::Num(0.0));
        }
        vars[i] = value;
        refresh_workspace_slot(i, &vars[i]);
        global_aliases.declare_named(i, name);
    }
    Ok(())
}

pub fn declare_persistent(func_name: &str, indices: Vec<usize>, vars: &mut Vec<Value>) {
    for i in indices {
        let val_opt = runmat_runtime::workspace::session::persistent_slot_value(func_name, i);
        if let Some(v) = val_opt {
            if i >= vars.len() {
                vars.resize(i + 1, Value::Num(0.0));
            }
            vars[i] = v;
            refresh_workspace_slot(i, &vars[i]);
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
            }
            vars[i] = v;
            refresh_workspace_slot(i, &vars[i]);
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

#[cfg(test)]
mod tests {
    use super::{update_global_store, FrameGlobalBindings};
    use runmat_value::Value;

    #[test]
    fn named_global_write_does_not_mutate_an_unrelated_legacy_slot() {
        let runtime = runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        futures::executor::block_on(runtime.scope(async {
            let bindings = FrameGlobalBindings::default();
            bindings.declare_named(4, "other".into());
            runmat_runtime::workspace::session::store_global_slot(4, Value::Num(11.0));
            update_global_store(4, &Value::Num(19.0), &bindings);
            assert_eq!(
                runmat_runtime::workspace::session::global_slot_value(4),
                Some(Value::Num(11.0))
            );
            assert_eq!(
                runmat_runtime::workspace::session::global_value("other"),
                Some(Value::Num(19.0))
            );
        }));
    }
}
