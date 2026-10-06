use super::super::*;

pub(super) async fn execute(
    meta: DispatchMeta<'_>,
    state: DispatchState<'_>,
    hooks: DispatchHooks<'_>,
) -> Result<Option<DispatchHandled>, RuntimeError> {
    let instr = meta.instr;
    let DispatchMeta {
        current_function_name,
        ..
    } = meta;
    let DispatchState {
        stack,
        vars,
        context,
        try_stack,
        imports,
        global_aliases,
        persistent_aliases,
        pc,
        ..
    } = state;
    let _ = hooks;
    match instr {
        Instr::AndAnd(target) => Ok(Some(DispatchHandled::Generic(apply_control_flow_action(
            crate::ops::control_flow::and_and(
                logical_truth_from_value(
                    &crate::interpreter::stack::pop_value(stack)?,
                    "short-circuit && condition",
                )
                .await?,
                *target,
            ),
            pc,
        )))),
        Instr::OrOr(target) => Ok(Some(DispatchHandled::Generic(apply_control_flow_action(
            crate::ops::control_flow::or_or(
                logical_truth_from_value(
                    &crate::interpreter::stack::pop_value(stack)?,
                    "short-circuit || condition",
                )
                .await?,
                *target,
            ),
            pc,
        )))),
        Instr::JumpIfFalse(target) => {
            let cond = crate::interpreter::stack::pop_value(stack)?;
            let truth = logical_truth_from_value(&cond, "if condition").await?;
            Ok(Some(DispatchHandled::Generic(apply_control_flow_action(
                crate::ops::control_flow::jump_if_false(truth, *target),
                pc,
            ))))
        }
        Instr::Jump(target) => Ok(Some(DispatchHandled::Generic(apply_control_flow_action(
            crate::ops::control_flow::jump(*target),
            pc,
        )))),
        Instr::EnterTry {
            scope,
            catch_pc,
            catch_var,
        } => {
            crate::ops::control_flow::enter_try(try_stack, *scope, *catch_pc, *catch_var);
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::LeaveTry(scope) => {
            crate::ops::control_flow::leave_try(try_stack, *scope);
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::ReturnValue => Ok(Some(DispatchHandled::ReturnValue(
            apply_control_flow_action(crate::ops::control_flow::return_value(stack)?, pc),
        ))),
        Instr::Return => Ok(Some(DispatchHandled::Return(apply_control_flow_action(
            crate::ops::control_flow::return_void(),
            pc,
        )))),
        Instr::EnterScope(local_count) => {
            crate::ops::control_flow::enter_scope(&mut context.locals, *local_count);
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::ExitScope(local_count) => {
            for _ in 0..*local_count {
                if let Some(value) = context.locals.pop() {
                    #[cfg(feature = "native-accel")]
                    clear_scope_value_residency_excluding_live_values(&value, stack, vars, context);
                    #[cfg(not(feature = "native-accel"))]
                    drop(value);
                }
            }
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::RegisterImport { path, wildcard } => {
            imports.push((path.clone(), *wildcard));
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::DeclareGlobal(indices) => {
            crate::runtime::globals::declare_global(indices.clone(), vars);
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::DeclareGlobalNamed(indices, names) => {
            crate::runtime::globals::declare_global_named(
                indices.clone(),
                names.clone(),
                vars,
                global_aliases,
            );
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::DeclarePersistent(indices) => {
            crate::runtime::globals::declare_persistent(
                current_function_name,
                indices.clone(),
                vars,
            );
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::DeclarePersistentNamed(indices, names) => {
            crate::runtime::globals::declare_persistent_named(
                current_function_name,
                indices.clone(),
                names.clone(),
                vars,
                persistent_aliases,
            );
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        _ => Ok(None),
    }
}
