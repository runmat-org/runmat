use crate::interpreter::vars::InterpreterVars;
use crate::runtime::workspace::refresh_workspace_slot_from_vars;
use runmat_runtime::workspace::bindings::FrameGlobalBindings;
use runmat_runtime::workspace::GlobalBindingTarget;
use runmat_value::Value;

pub fn workspace_global_names() -> Vec<String> {
    runmat_runtime::workspace::session::global_names()
}

pub fn get_global_value(name: &str) -> Option<Value> {
    runmat_runtime::workspace::session::global_value(name)
}

pub fn current_bound_value(bindings: &FrameGlobalBindings, slot: usize) -> Option<Value> {
    let binding = bindings.binding_handle_for_slot(slot)?;
    current_bound_value_for_handle(&binding)
}

pub fn current_bound_value_for_handle(binding: &GlobalBindingTarget) -> Option<Value> {
    match binding {
        GlobalBindingTarget::Named(name) => get_global_value(name.as_str()),
        GlobalBindingTarget::LegacySlot(slot) => {
            runmat_runtime::workspace::session::global_slot_value(*slot)
        }
    }
}

pub fn collect_thread_roots() -> Vec<Value> {
    runmat_runtime::workspace::session::roots()
}

pub fn declare_global(
    indices: &[usize],
    vars: &mut InterpreterVars,
    global_aliases: &FrameGlobalBindings,
) -> Result<(), runmat_runtime::RuntimeError> {
    for &i in indices {
        if let Some(runtime) = runmat_runtime::context::legacy::active() {
            let workspace = runtime.service_ports().workspace().ok_or_else(|| {
                runmat_runtime::runtime_error::semantic_error(
                    "RunMat:Workspace:FrameUnavailable",
                    "legacy global declaration has no active workspace frame",
                )
            })?;
            let existing = crate::runtime::workspace::workspace_slot_name_handle(i);
            let fallback;
            let name = match &existing {
                Some(name) => name.as_str(),
                None => {
                    fallback = format!("var_{i}");
                    &fallback
                }
            };
            workspace.declare_legacy_global_binding_with_context(&runtime, i, name)?;
            continue;
        }
        global_aliases.declare_legacy_slot(i);
        let val_opt = runmat_runtime::workspace::session::global_slot_value(i);
        if let Some(v) = val_opt {
            if i >= vars.len() {
                vars.resize_owned(i + 1, Value::Num(0.0));
            }
            let _retired = vars.replace_owned(i, v);
            refresh_workspace_slot_from_vars(i, vars);
        }
    }
    Ok(())
}

pub fn declare_global_named(
    indices: &[usize],
    names: &[String],
    vars: &mut InterpreterVars,
    global_aliases: &FrameGlobalBindings,
) -> Result<(), runmat_runtime::RuntimeError> {
    for (pos, &i) in indices.iter().enumerate() {
        let fallback;
        let name = match names.get(pos) {
            Some(name) => name.as_str(),
            None => {
                fallback = format!("var_{i}");
                &fallback
            }
        };
        if let Some(runtime) = runmat_runtime::context::legacy::active() {
            let workspace = runtime.service_ports().workspace().ok_or_else(|| {
                runmat_runtime::runtime_error::semantic_error(
                    "RunMat:Workspace:FrameUnavailable",
                    "named global declaration has no active workspace frame",
                )
            })?;
            workspace.declare_global_binding_with_context(&runtime, i, name)?;
            continue;
        }
        let value = runmat_runtime::workspace::session::declare_global_named(name)?;
        if i >= vars.len() {
            vars.resize_owned(i + 1, Value::Num(0.0));
        }
        let _retired = vars.replace_owned(i, value);
        refresh_workspace_slot_from_vars(i, vars);
        global_aliases.declare_named(i, name.to_owned());
    }
    Ok(())
}

pub(crate) fn reset_thread_state_for_tests() {
    runmat_runtime::workspace::session::reset_legacy_state_for_tests();
}
