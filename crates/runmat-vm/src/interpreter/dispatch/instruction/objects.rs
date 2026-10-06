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
    let DispatchState { stack, .. } = state;
    let _ = hooks;
    match instr {
        Instr::CallMethodOrMemberIndexMulti {
            identity,
            fallback_policy,
            arg_count,
            out_count,
        } => {
            handle_method_or_member_index_multi_call(
                stack,
                identity.clone(),
                *fallback_policy,
                *arg_count,
                *out_count,
                current_function_name,
            )
            .await?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CallMethodOrMemberIndexExpandMultiOutput {
            identity,
            fallback_policy,
            specs,
            out_count,
        } => {
            handle_method_or_member_index_expand_multi_call(
                stack,
                identity.clone(),
                *fallback_policy,
                specs,
                *out_count,
                current_function_name,
            )
            .await?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::LoadMethod(name) => {
            handle_load_method(stack, name.clone(), current_function_name)?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CreateClosure(func_name, capture_count) => {
            handle_create_closure(stack, func_name.clone(), *capture_count)?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CreateSemanticClosure(function, display_name, capture_count) => {
            handle_create_semantic_closure(stack, *function, display_name.clone(), *capture_count)?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CreateFunctionHandle(name) => {
            stack.push(Value::FunctionHandle(name.clone()));
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CreateExternalFunctionHandle(name) => {
            stack.push(Value::ExternalFunctionHandle(name.clone()));
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CreateMethodFunctionHandle(name) => {
            stack.push(Value::MethodFunctionHandle(name.clone()));
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CreateBoundFunctionHandle(function, name)
        | Instr::CreateExternalBoundFunctionHandle(function, name) => {
            stack.push(Value::BoundFunctionHandle {
                name: name.clone(),
                function: function.0,
            });
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::LoadStaticProperty(class_name, prop) => {
            handle_load_static_property(stack, class_name, prop)?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::LoadWorkspaceFirstStaticProperty {
            name,
            class_name,
            property,
        } => {
            if let Some(value) = crate::runtime::workspace::workspace_lookup(name) {
                stack.push(value);
            } else {
                handle_load_static_property(stack, class_name, property)?;
            }
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::RegisterClass {
            name,
            super_class,
            is_sealed,
            is_abstract,
            properties,
            methods,
            enumerations,
        } => {
            handle_register_class(
                name.clone(),
                super_class.clone(),
                *is_sealed,
                *is_abstract,
                properties.clone(),
                methods.clone(),
                enumerations.clone(),
            )?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        _ => Ok(None),
    }
}
