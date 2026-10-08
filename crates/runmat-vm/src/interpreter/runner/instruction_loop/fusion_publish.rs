use super::*;

#[cfg(feature = "native-accel")]
pub(super) struct FusionPublicationFrame<'a> {
    pub vars: &'a mut Vec<Value>,
    pub context: &'a mut crate::bytecode::program::ExecutionContext,
    pub var_names: &'a HashMap<usize, String>,
    pub completion_assigned_slots: &'a mut Option<HashSet<usize>>,
    pub global_aliases: &'a runmat_runtime::workspace::bindings::FrameGlobalBindings,
    pub persistent_aliases: &'a crate::interpreter::PersistentAliases,
    pub current_function_name: &'a str,
    pub missing_input_slots: &'a mut HashSet<usize>,
}

#[cfg(feature = "native-accel")]
pub(super) fn publish_fusion_stores(
    stores: Vec<(runmat_accelerate::fusion::FusionStoreMaterialization, Value)>,
    frame: FusionPublicationFrame<'_>,
) -> Result<(), RuntimeError> {
    let FusionPublicationFrame {
        vars,
        context,
        var_names,
        completion_assigned_slots,
        global_aliases,
        persistent_aliases,
        current_function_name,
        missing_input_slots,
    } = frame;
    for (store, value) in stores {
        let index = store.binding.index;
        match store.binding.kind {
            runmat_accelerate::VarKind::Global => {
                crate::ops::stack::store_var_value(
                    value,
                    vars,
                    index,
                    var_names,
                    |_current, _incoming| {},
                    |stored_index, stored_value| {
                        after_global_store(
                            completion_assigned_slots,
                            global_aliases,
                            stored_index,
                            stored_value,
                        );
                    },
                )?;
                missing_input_slots.remove(&index);
                crate::interpreter::persistent::publish_store(
                    persistent_aliases,
                    vars,
                    &context.runtime.session_variable_authority(),
                    current_function_name,
                    index,
                )?;
            }
            runmat_accelerate::VarKind::Local => {
                if context.call_stack.last().is_some() {
                    crate::ops::stack::store_local_value(
                        value,
                        context,
                        vars,
                        index,
                        crate::ops::stack::StoreLocalHooks {
                            before_local_overwrite: |_current: &Value, _incoming: &Value| {},
                            before_var_overwrite: |_current: &Value, _incoming: &Value| {},
                            after_local_store: |stored_offset: usize, stored_value: &Value| {
                                after_local_store(global_aliases, stored_offset, stored_value);
                            },
                            after_fallback_store:
                                |_func_name: &str, stored_offset: usize, stored_value: &Value| {
                                    after_local_fallback_store(
                                        global_aliases,
                                        stored_offset,
                                        stored_value,
                                    );
                                },
                        },
                    )?;
                } else {
                    crate::ops::stack::store_var_value(
                        value,
                        vars,
                        index,
                        var_names,
                        |_current, _incoming| {},
                        |stored_index, stored_value| {
                            after_local_fallback_store(global_aliases, stored_index, stored_value);
                        },
                    )?;
                    missing_input_slots.remove(&index);
                    crate::interpreter::persistent::publish_store(
                        persistent_aliases,
                        vars,
                        &context.runtime.session_variable_authority(),
                        current_function_name,
                        index,
                    )?;
                }
            }
        }
    }
    Ok(())
}

pub(super) fn after_global_store(
    completion_assigned_slots: &mut Option<HashSet<usize>>,
    global_aliases: &runmat_runtime::workspace::bindings::FrameGlobalBindings,
    stored_index: usize,
    stored_value: &Value,
) {
    if let Some(assigned_slots) = completion_assigned_slots.as_mut() {
        assigned_slots.insert(stored_index);
    }
    runtime_globals::update_global_store(stored_index, stored_value, global_aliases);
}

pub(super) fn after_local_store(
    global_aliases: &runmat_runtime::workspace::bindings::FrameGlobalBindings,
    stored_offset: usize,
    stored_value: &Value,
) {
    runtime_globals::update_global_store(stored_offset, stored_value, global_aliases);
}

pub(super) fn after_local_fallback_store(
    global_aliases: &runmat_runtime::workspace::bindings::FrameGlobalBindings,
    stored_offset: usize,
    stored_value: &Value,
) {
    after_local_store(global_aliases, stored_offset, stored_value);
}
