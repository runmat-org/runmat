use super::super::*;

pub(super) async fn execute(
    meta: DispatchMeta<'_>,
    state: DispatchState<'_>,
    hooks: DispatchHooks<'_>,
) -> Result<Option<DispatchHandled>, RuntimeError> {
    let instr = meta.instr;
    let DispatchMeta { var_names, .. } = meta;
    let DispatchState {
        stack,
        vars,
        context,
        global_aliases,
        missing_input_slots,
        pc,
        ..
    } = state;
    let DispatchHooks {
        store_var_before_overwrite,
        store_var_after_store,
        store_local_before_local_overwrite,
        store_local_before_var_overwrite,
        store_local_after_store,
        store_local_after_fallback_store,
        ..
    } = hooks;
    match instr {
        Instr::EmitStackTop { label } => {
            emit_stack_top(stack, label, var_names).await?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::EmitVar { var_index, label } => {
            emit_var(vars, *var_index, label, var_names).await?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::LoadConst(value) => {
            load_const(stack, *value);
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::LoadInt(value) => {
            load_int(stack, value.clone());
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::LoadComplex(re, im) => {
            load_complex(stack, *re, *im);
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::LoadBool(value) => {
            load_bool(stack, *value);
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::LoadString(value) => {
            load_string(stack, value.clone());
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::LoadCharRow(value) => {
            load_char_row(stack, value.clone())?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::LoadLocal(offset) => {
            load_local(stack, context, vars, *offset)?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::LoadVar(index) => {
            if let Some(alias) = global_aliases.get(index) {
                if let Some(global_value) = crate::runtime::globals::get_global_value(alias) {
                    if *index >= vars.len() {
                        vars.resize(*index + 1, Value::Num(0.0));
                        refresh_workspace_state(vars);
                    }
                    vars[*index] = global_value;
                    refresh_workspace_state(vars);
                }
            }
            if missing_input_slots.contains(index) {
                return Err(crate::interpreter::errors::mex(
                    "NotEnoughInputs",
                    "Not enough input arguments.",
                ));
            }
            if let (Some(false), Some(slot_name), Some(var_name)) = (
                workspace_slot_assigned(*index),
                workspace_slot_name(*index),
                var_names.get(index),
            ) {
                if slot_name == *var_name {
                    if let Some(value) = builtin_constant_value(var_name) {
                        debug::trace_load_var(*pc, *index, &value);
                        stack.push(value);
                        return Ok(Some(DispatchHandled::Generic(
                            DispatchDecision::FallThrough,
                        )));
                    }
                    return Err(crate::interpreter::errors::mex(
                        "UndefinedVariable",
                        &format!("Undefined variable: {slot_name}"),
                    ));
                }
            }
            let value = vars[*index].clone();
            debug::trace_load_var(*pc, *index, &value);
            load_var(stack, vars, *index);
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::LoadVarForIndexAssignment(index) => {
            if let Some(alias) = global_aliases.get(index) {
                if let Some(global_value) = crate::runtime::globals::get_global_value(alias) {
                    if *index >= vars.len() {
                        vars.resize(*index + 1, Value::Num(0.0));
                        refresh_workspace_state(vars);
                    }
                    vars[*index] = global_value;
                    refresh_workspace_state(vars);
                }
            }
            if missing_input_slots.contains(index) {
                return Err(crate::interpreter::errors::mex(
                    "NotEnoughInputs",
                    "Not enough input arguments.",
                ));
            }
            if let (Some(false), Some(slot_name), Some(var_name)) = (
                workspace_slot_assigned(*index),
                workspace_slot_name(*index),
                var_names.get(index),
            ) {
                if slot_name == *var_name {
                    let empty = Tensor::new(Vec::new(), vec![0, 0]).map_err(|err| {
                        crate::interpreter::errors::mex(
                            "IndexAssignmentBaseInit",
                            &format!(
                                "failed to initialize undefined indexed-assignment base '{slot_name}': {err}"
                            ),
                        )
                    })?;
                    debug::trace_load_var(*pc, *index, &Value::Tensor(empty.clone()));
                    stack.push(Value::Tensor(empty));
                    return Ok(Some(DispatchHandled::Generic(
                        DispatchDecision::FallThrough,
                    )));
                }
            }
            let value = vars[*index].clone();
            debug::trace_load_var(*pc, *index, &value);
            load_var(stack, vars, *index);
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::StoreVar(index) => {
            let preview = stack
                .last()
                .cloned()
                .ok_or(crate::interpreter::errors::mex(
                    "StackUnderflow",
                    "stack underflow",
                ))?;
            debug::trace_store_var(*pc, *index, &preview);
            if *index < vars.len() {
                #[cfg(feature = "native-accel")]
                clear_overwritten_var_residency_excluding_live_values(
                    &vars[*index],
                    *index,
                    stack,
                    vars,
                    context,
                );
            }
            store_var(
                stack,
                vars,
                *index,
                var_names,
                store_var_before_overwrite,
                store_var_after_store,
            )?;
            missing_input_slots.remove(index);
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::StoreLocal(offset) => {
            if stack.last().is_some() {
                if let Some(current_frame) = context.call_stack.last() {
                    let local_index = current_frame.locals_start + *offset;
                    if local_index < context.locals.len() {
                        let current_value = context.locals[local_index].clone();
                        #[cfg(feature = "native-accel")]
                        clear_overwritten_local_residency_excluding_live_values(
                            &current_value,
                            local_index,
                            stack,
                            vars,
                            context,
                        );
                    }
                } else if *offset < vars.len() {
                    #[cfg(feature = "native-accel")]
                    clear_overwritten_var_residency_excluding_live_values(
                        &vars[*offset],
                        *offset,
                        stack,
                        vars,
                        context,
                    );
                }
            }
            if context.call_stack.last().is_none() {
                store_var(
                    stack,
                    vars,
                    *offset,
                    var_names,
                    store_local_before_var_overwrite,
                    |stored_index, stored_value| {
                        store_local_after_fallback_store("<main>", stored_index, stored_value);
                    },
                )?;
                missing_input_slots.remove(offset);
            } else {
                store_local(
                    stack,
                    context,
                    vars,
                    *offset,
                    store_local_before_local_overwrite,
                    store_local_before_var_overwrite,
                    store_local_after_store,
                    store_local_after_fallback_store,
                )?;
            }
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::Swap => {
            crate::ops::stack::swap(stack)?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::Pop => {
            if let Some(value) = stack.pop() {
                #[cfg(feature = "native-accel")]
                clear_popped_value_residency_excluding_live_values(&value, stack, vars, context);
            }
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        _ => Ok(None),
    }
}
