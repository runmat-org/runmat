//! Existing standard-substruct class protocol with rooted original references.
use super::*;
use crate::object::indexing::{ObjectIndexKind, ObjectIndexSelector, ObjectSubscriptPath};
use runmat_builtins::OptimizationVariableProperty as Property;

#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationVariable.subsref",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimvar::protocol"
)]
async fn subsref(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    crate::output_count::validate_bounded_outputs_in(
        &runmat_builtins::OPTIMVAR_SUBSREF_DESCRIPTOR,
        &runmat_builtins::OPTIMVAR_ERROR_INVALID_INPUT,
        "optimvar",
        runtime.as_ref(),
    )?;
    if args.len() != 2 {
        return Err(invalid(
            "subsref expects a variable and standard subscript descriptor",
        ));
    }
    let mut cooperation = super::cooperation(runtime.as_ref()).await?;
    let Value::Object(object) = &args[0] else {
        return Err(invalid("subsref requires an optimization variable"));
    };
    if !is_reference(object) {
        return Err(invalid("subsref requires an original variable reference"));
    }
    let path = crate::object::indexing::parse_standard_substruct(&args[1])?;
    let mut base = RootedValue::in_optional_context(args[0].clone(), cooperation.runtime())?;
    for step in path.steps() {
        base = if let Value::Object(object) = base.value() {
            if is_reference(object) {
                let view = View::new(object, &mut cooperation).await?;
                match step.kind() {
                    ObjectIndexKind::Paren => indexing::read(&view, step, &mut cooperation).await?,
                    ObjectIndexKind::Member => {
                        let ObjectIndexSelector::Member(name) = step.selector() else {
                            return Err(invalid("invalid variable member selector"));
                        };
                        let property = Property::parse(name)
                            .ok_or_else(|| invalid("unknown optimization-variable property"))?;
                        properties::get(&view, property, &mut cooperation).await?
                    }
                    ObjectIndexKind::Brace => {
                        return Err(invalid(
                            "optimization variables do not support brace indexing",
                        ))
                    }
                }
            } else {
                read_ordinary(base.value().clone(), step.clone(), cooperation.runtime()).await?
            }
        } else {
            read_ordinary(base.value().clone(), step.clone(), cooperation.runtime()).await?
        };
    }
    Ok(base)
}
async fn read_ordinary(
    value: Value,
    step: crate::object::indexing::ObjectSubscript,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<RootedValue> {
    let path = ObjectSubscriptPath::single_in_context(runtime, step)?;
    match runtime {
        Some(runtime) => {
            crate::object::protocol::read_subscript_path_in_context(runtime, value, path, None)
                .await
        }
        None => crate::object::protocol::read_subscript_path(value, path, None).await,
    }
}

#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationVariable.subsasgn",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimvar::protocol"
)]
async fn subsasgn(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    crate::output_count::validate_bounded_outputs_in(
        &runmat_builtins::OPTIMVAR_SUBSASGN_DESCRIPTOR,
        &runmat_builtins::OPTIMVAR_ERROR_INVALID_INPUT,
        "optimvar",
        runtime.as_ref(),
    )?;
    if args.len() != 3 {
        return Err(invalid(
            "subsasgn expects a variable, subscript descriptor and assigned value",
        ));
    }
    let mut cooperation = super::cooperation(runtime.as_ref()).await?;
    let Value::Object(object) = &args[0] else {
        return Err(invalid("subsasgn requires an optimization variable"));
    };
    if !is_reference(object) {
        return Err(invalid("subsasgn requires an original variable reference"));
    }
    let path = crate::object::indexing::parse_standard_substruct(&args[1])?;
    assign_path(&args[0], &path, &args[2], &mut cooperation).await
}
pub(crate) async fn assign_path(
    receiver: &Value,
    path: &ObjectSubscriptPath,
    rhs: &Value,
    cooperation: &mut super::super::common::ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    // Prepare the actual returned receiver before any definition mutation.
    let original = RootedValue::in_optional_context(receiver.clone(), cooperation.runtime())?;
    let mut base = original.clone();
    for (position, step) in path.steps().iter().enumerate() {
        let Value::Object(object) = base.value() else {
            return Err(invalid("variable assignment lost its reference"));
        };
        let view = View::new(object, cooperation).await?;
        match step.kind() {
            ObjectIndexKind::Paren => base = indexing::read(&view, step, cooperation).await?,
            ObjectIndexKind::Member => {
                let ObjectIndexSelector::Member(name) = step.selector() else {
                    return Err(invalid("invalid variable member selector"));
                };
                let property = Property::parse(name)
                    .ok_or_else(|| invalid("unknown optimization-variable property"))?;
                properties::ensure_writable(&view, property)?;
                if position + 1 < path.steps().len() {
                    properties::preflight(property, rhs)?;
                    let modified = {
                        let value = properties::get(&view, property, cooperation).await?;
                        let remaining =
                            ObjectSubscriptPath::new(path.steps()[position + 1..].to_vec())?;
                        let supplied = vec![rhs.clone()];
                        let runtime = cooperation.runtime().cloned();
                        value
                            .with_owned_async(|value| async move {
                                match runtime.as_ref() {
                                    Some(runtime) => {
                                        crate::object::protocol::assign_subscript_path_in_context(
                                            runtime, value, remaining, supplied, None,
                                        )
                                        .await
                                    }
                                    None => {
                                        crate::object::protocol::assign_subscript_path(
                                            value, remaining, supplied, None,
                                        )
                                        .await
                                    }
                                }
                            })
                            .await?
                    };
                    let _updated =
                        properties::set(&view, property, modified.value(), cooperation).await?;
                } else {
                    let _updated = properties::set(&view, property, rhs, cooperation).await?;
                }
                return Ok(original);
            }
            ObjectIndexKind::Brace => {
                return Err(invalid(
                    "optimization variables do not support brace assignment",
                ))
            }
        }
    }
    Err(invalid(
        "assign variable properties; expression element assignment requires an expression",
    ))
}
