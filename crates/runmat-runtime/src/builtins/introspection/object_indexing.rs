use crate::object::protocol::ProtocolDispatch;
use crate::sequence::ResolveValueSequence;
use runmat_builtins::*;
use runmat_macros::runtime_builtin;
use runmat_types::ObjectIndexingContext as IndexingContext;
use runmat_value::{NumericScalar, Value};

pub(crate) const NUM_ARGUMENTS_FROM_SUBSCRIPT_METHOD: runmat_types::StaticMethodName =
    crate::OBJECT_NUM_ARGUMENTS_FROM_SUBSCRIPT_METHOD;
pub(crate) const INDEXING_CONTEXT_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("matlab.indexing.IndexingContext");
pub(crate) const LEGACY_INDEXING_CONTEXT_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("matlab.mixin.util.IndexingContext");
const INDEXING_CONTEXT_STATEMENT: &str = "Statement";
const INDEXING_CONTEXT_EXPRESSION: &str = "Expression";
const INDEXING_CONTEXT_ASSIGNMENT: &str = "Assignment";
const ENUM_MEMBER_PROPERTY: &str = "__enum_member__";

#[derive(Clone)]
struct SubscriptLevel {
    kind: crate::object::indexing::ObjectIndexKind,
    subs: Value,
}

async fn dispatch_num_arguments_overload(
    runtime: Option<&crate::context::RuntimeContext>,
    target: Value,
    subscript: Value,
    indexing_context: Value,
) -> crate::BuiltinResult<Option<crate::execution::RootedValue>> {
    match crate::object::protocol::resolve_object_protocol(
        &target,
        crate::object::protocol::ObjectProtocol::NumArgumentsFromSubscript,
        &crate::object::protocol::ObjectAccessContext::default(),
    )? {
        None => Ok(None),
        Some(method) => {
            let arguments = vec![target, subscript, indexing_context];
            let sequence = match runtime {
                Some(runtime) => {
                    crate::object::protocol::invoke_resolved_object_method_in_context(
                        runtime, &method, arguments, 1,
                    )
                    .await?
                }
                None => {
                    crate::object::protocol::invoke_resolved_object_method(&method, arguments, 1)
                        .await?
                }
            };
            sequence
                .try_map_owned(require_single_sequence, |error| error.into_runtime_error())
                .map(Some)
        }
    }
}

#[cfg(test)]
pub(crate) async fn dispatch_subsasgn(
    obj: Value,
    subscript: Value,
    rhs: Value,
) -> crate::BuiltinResult<crate::execution::RootedValue> {
    dispatch_subsasgn_with_context(None, obj, subscript, rhs, ProtocolDispatch::Resolve).await
}

async fn dispatch_subsasgn_with_context(
    runtime: Option<&crate::context::RuntimeContext>,
    obj: Value,
    subscript: Value,
    rhs: Value,
    dispatch: ProtocolDispatch,
) -> crate::BuiltinResult<crate::execution::RootedValue> {
    let path = crate::object::indexing::parse_standard_substruct(&subscript)?;
    if matches!(dispatch, ProtocolDispatch::Default) {
        let access = builtin_indexing_access();
        return crate::object::protocol::assign_builtin_subscript_path(
            runtime,
            obj,
            path,
            vec![rhs],
            access,
        )
        .await;
    }
    match runtime {
        Some(runtime) => {
            crate::object::protocol::assign_subscript_path_in_context(
                runtime,
                obj,
                path,
                vec![rhs],
                None,
            )
            .await
        }
        None => crate::object::protocol::assign_subscript_path(obj, path, vec![rhs], None).await,
    }
}

#[runtime_builtin(
    name = "subsref",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::introspection::object_indexing"
)]
pub async fn subsref_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    obj: Value,
    subscript: Value,
) -> crate::BuiltinResult<crate::execution::RootedValue> {
    subsref_with_context(runtime.as_ref(), obj, subscript, ProtocolDispatch::Default).await
}

pub async fn subsref_builtin(
    obj: Value,
    subscript: Value,
) -> crate::BuiltinResult<crate::execution::RootedValue> {
    subsref_with_context(None, obj, subscript, ProtocolDispatch::Resolve).await
}

async fn subsref_with_context(
    runtime: Option<&crate::context::RuntimeContext>,
    obj: Value,
    subscript: Value,
    dispatch: ProtocolDispatch,
) -> crate::BuiltinResult<crate::execution::RootedValue> {
    let path = crate::object::indexing::parse_standard_substruct(&subscript)?;
    let requested = crate::output_count::current_output_count();
    let access = builtin_indexing_access();
    let request = match requested {
        None => crate::object::protocol::SubscriptReadRequest::require_single(access),
        Some(count) => crate::object::protocol::SubscriptReadRequest::select_outputs(count, access),
    };
    let usage = request.sequence_use;
    let sequence = match (dispatch, runtime) {
        (ProtocolDispatch::Default, runtime) => {
            crate::object::protocol::read_builtin_subscript_path(runtime, obj, path, request)
                .await?
        }
        (ProtocolDispatch::Resolve, Some(runtime)) => {
            crate::object::protocol::read_subscript_path_sequence_with_access_in_context(
                runtime, obj, path, request,
            )
            .await?
        }
        (ProtocolDispatch::Resolve, None) => {
            crate::object::protocol::read_subscript_path_sequence_with_access(obj, path, request)
                .await?
        }
    };
    sequence.try_map_owned(
        |sequence| {
            let values =
                sequence.resolve(usage, crate::sequence::SequenceResolutionContext::default())?;
            if requested.is_some() {
                Ok(Value::output_list(values))
            } else {
                Ok(crate::sequence::into_first_output(values)?.expect("single object output"))
            }
        },
        |error| error.into_runtime_error(),
    )
}

#[runtime_builtin(
    name = "subsasgn",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::introspection::object_indexing"
)]
pub async fn subsasgn_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    obj: Value,
    subscript: Value,
    rhs: Value,
) -> crate::BuiltinResult<crate::execution::RootedValue> {
    subsasgn_with_context(
        runtime.as_ref(),
        obj,
        subscript,
        rhs,
        ProtocolDispatch::Default,
    )
    .await
}

pub async fn subsasgn_builtin(
    obj: Value,
    subscript: Value,
    rhs: Value,
) -> crate::BuiltinResult<crate::execution::RootedValue> {
    subsasgn_with_context(None, obj, subscript, rhs, ProtocolDispatch::Resolve).await
}

async fn subsasgn_with_context(
    runtime: Option<&crate::context::RuntimeContext>,
    obj: Value,
    subscript: Value,
    rhs: Value,
    dispatch: ProtocolDispatch,
) -> crate::BuiltinResult<crate::execution::RootedValue> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::SUBSASGN_DESCRIPTOR,
        &runmat_builtins::SUBSASGN_ERROR_OUTPUTS,
        "subsasgn",
    )?;
    dispatch_subsasgn_with_context(runtime, obj, subscript, rhs, dispatch).await
}

fn builtin_indexing_access() -> crate::object::protocol::ObjectAccessContext {
    crate::object::protocol::ObjectAccessContext {
        caller_class: crate::class_access_context().as_deref().cloned(),
        ..Default::default()
    }
}

#[runtime_builtin(
    name = "numArgumentsFromSubscript",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::introspection::object_indexing"
)]
pub async fn num_arguments_from_subscript_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    target: Value,
    subscript: Value,
    indexing_context: Value,
) -> crate::BuiltinResult<crate::execution::RootedValue> {
    num_arguments_from_subscript_with_context(runtime.as_ref(), target, subscript, indexing_context)
        .await
}

pub async fn num_arguments_from_subscript_builtin(
    target: Value,
    subscript: Value,
    indexing_context: Value,
) -> crate::BuiltinResult<crate::execution::RootedValue> {
    num_arguments_from_subscript_with_context(None, target, subscript, indexing_context).await
}

async fn num_arguments_from_subscript_with_context(
    runtime: Option<&crate::context::RuntimeContext>,
    target: Value,
    subscript: Value,
    indexing_context: Value,
) -> crate::BuiltinResult<crate::execution::RootedValue> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::NUM_ARGUMENTS_FROM_SUBSCRIPT_DESCRIPTOR,
        &runmat_builtins::NUM_ARGUMENTS_FROM_SUBSCRIPT_ERROR_OUTPUTS,
        "numArgumentsFromSubscript",
    )?;
    ensure_indexing_context_classes_registered();
    let context = parse_indexing_context(&indexing_context)?;
    if crate::object_receiver_class_name(&target).is_some() {
        if let Some(value) = dispatch_num_arguments_overload(
            runtime,
            target.clone(),
            subscript.clone(),
            indexing_context.clone(),
        )
        .await?
        {
            return Ok(value);
        }
    }
    let levels = parse_subscript_levels(&subscript)?;
    crate::execution::RootedValue::new(Value::Num(exact_count_as_f64(default_num_arguments(
        &target, &levels, context,
    )?)?))
    .map_err(|error| error.into_runtime_error())
}

mod cardinality;
mod context;
mod selectors;
pub(crate) use cardinality::object_path_assignment_cardinality_with_access;
use cardinality::*;
pub use cardinality::{
    brace_assignment_cardinality, member_assignment_cardinality,
    object_path_assignment_cardinality, object_path_assignment_cardinality_in_context,
    object_path_cardinality, object_path_cardinality_in_context,
    PreparedObjectAssignmentCardinality,
};
pub(crate) use context::ensure_indexing_context_classes_registered;
use context::*;
use selectors::*;
#[cfg(test)]
mod tests;
