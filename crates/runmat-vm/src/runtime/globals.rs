use crate::bytecode::instr::Instr;
use crate::bytecode::program::Bytecode;
use crate::runtime::workspace::refresh_workspace_state;
use runmat_builtins::Value;
use runmat_thread_local::runmat_thread_local;
use std::cell::RefCell;
use std::collections::HashMap;

runmat_thread_local! {
    static GLOBALS: RefCell<HashMap<String, Value>> = RefCell::new(HashMap::new());
}

runmat_thread_local! {
    static PERSISTENTS: RefCell<HashMap<(String, usize), Value>> = RefCell::new(HashMap::new());
}

runmat_thread_local! {
    static PERSISTENTS_BY_NAME: RefCell<HashMap<(String, String), Value>> = RefCell::new(HashMap::new());
}

pub fn workspace_global_names() -> Vec<String> {
    let mut names = Vec::new();
    GLOBALS.with(|globals| {
        let map = globals.borrow();
        for key in map.keys() {
            if !key.starts_with("var_") {
                names.push(key.clone());
            }
        }
    });
    names.sort();
    names
}

pub fn collect_thread_roots() -> Vec<Value> {
    let mut thread_roots = Vec::new();
    GLOBALS.with(|g| {
        for v in g.borrow().values() {
            thread_roots.push(v.clone());
        }
    });
    PERSISTENTS.with(|p| {
        for v in p.borrow().values() {
            thread_roots.push(v.clone());
        }
    });
    PERSISTENTS_BY_NAME.with(|p| {
        for v in p.borrow().values() {
            thread_roots.push(v.clone());
        }
    });
    thread_roots
}

pub fn update_global_store(
    stored_index: usize,
    stored_value: &Value,
    global_aliases: &HashMap<usize, String>,
) {
    let key = format!("var_{stored_index}");
    GLOBALS.with(|g| {
        let mut m = g.borrow_mut();
        if m.contains_key(&key) {
            m.insert(key, stored_value.clone());
        }
    });
    if let Some(name) = global_aliases.get(&stored_index) {
        GLOBALS.with(|g| {
            g.borrow_mut().insert(name.clone(), stored_value.clone());
        });
    }
}

pub fn update_persistent_local_store(func_name: &str, stored_offset: usize, stored_value: &Value) {
    let key = (func_name.to_string(), stored_offset);
    PERSISTENTS.with(|p| {
        let mut m = p.borrow_mut();
        if m.contains_key(&key) {
            m.insert(key, stored_value.clone());
        }
    });
}

pub fn declare_global(indices: Vec<usize>, vars: &mut Vec<Value>) {
    for i in indices {
        let key = format!("var_{i}");
        let val_opt = GLOBALS.with(|g| g.borrow().get(&key).cloned());
        if let Some(v) = val_opt {
            if i >= vars.len() {
                vars.resize(i + 1, Value::Num(0.0));
                refresh_workspace_state(vars);
            }
            vars[i] = v;
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
        let val_opt = GLOBALS.with(|g| g.borrow().get(&name).cloned());
        if let Some(v) = val_opt {
            if i >= vars.len() {
                vars.resize(i + 1, Value::Num(0.0));
                refresh_workspace_state(vars);
            }
            vars[i] = v;
        }
        GLOBALS.with(|g| {
            let mut m = g.borrow_mut();
            if let Some(v) = m.get(&name).cloned() {
                m.insert(format!("var_{i}"), v);
            }
        });
        global_aliases.insert(i, name);
    }
}

pub fn declare_persistent(func_name: &str, indices: Vec<usize>, vars: &mut Vec<Value>) {
    for i in indices {
        let key = (func_name.to_string(), i);
        let val_opt = PERSISTENTS.with(|p| p.borrow().get(&key).cloned());
        if let Some(v) = val_opt {
            if i >= vars.len() {
                vars.resize(i + 1, Value::Num(0.0));
                refresh_workspace_state(vars);
            }
            vars[i] = v;
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
        let key = (func_name.to_string(), i);
        let val_opt = PERSISTENTS_BY_NAME
            .with(|p| {
                p.borrow()
                    .get(&(func_name.to_string(), name.clone()))
                    .cloned()
            })
            .or_else(|| PERSISTENTS.with(|p| p.borrow().get(&key).cloned()));
        if let Some(v) = val_opt {
            if i >= vars.len() {
                vars.resize(i + 1, Value::Num(0.0));
                refresh_workspace_state(vars);
            }
            vars[i] = v;
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
                        let key = (func_name.to_string(), i);
                        PERSISTENTS.with(|p| {
                            p.borrow_mut().insert(key, vars[i].clone());
                        });
                    }
                }
            }
            Instr::DeclarePersistentNamed(indices, names) => {
                for (pos, &i) in indices.iter().enumerate() {
                    if i < vars.len() {
                        let key = (func_name.to_string(), i);
                        let name_key = (
                            func_name.to_string(),
                            names
                                .get(pos)
                                .cloned()
                                .unwrap_or_else(|| format!("var_{i}")),
                        );
                        let val = vars[i].clone();
                        PERSISTENTS.with(|p| {
                            p.borrow_mut().insert(key, val.clone());
                        });
                        PERSISTENTS_BY_NAME.with(|p| {
                            p.borrow_mut().insert(name_key, val);
                        });
                    }
                }
            }
            _ => {}
        }
    }
}

pub fn clear_all_runtime_globals() {
    GLOBALS.with(|g| g.borrow_mut().clear());
    PERSISTENTS.with(|p| p.borrow_mut().clear());
    PERSISTENTS_BY_NAME.with(|p| p.borrow_mut().clear());
}
