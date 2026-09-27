use crate::sequence::ResolveValueSequence;
use runmat_builtins::*;
use runmat_macros::runtime_builtin;
use runmat_types::ObjectIndexingContext as IndexingContext;
use runmat_value::{NumericScalar, Value};
use std::sync::OnceLock;

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

static INDEXING_CONTEXT_REGISTERED: OnceLock<()> = OnceLock::new();

#[derive(Clone)]
struct SubscriptLevel {
    kind: String,
    subs: Value,
}

async fn dispatch_num_arguments_overload(
    target: Value,
    subscript: Value,
    indexing_context: Value,
) -> crate::BuiltinResult<Option<Value>> {
    match crate::object::protocol::resolve_object_protocol(
        &target,
        crate::object::protocol::ObjectProtocol::NumArgumentsFromSubscript,
        &crate::object::protocol::ObjectAccessContext::default(),
    )? {
        crate::object::protocol::ProtocolResolution::DefaultIndexing => Ok(None),
        crate::object::protocol::ProtocolResolution::Method(method) => {
            let sequence = crate::object::protocol::invoke_resolved_object_method(
                &method,
                vec![target, subscript, indexing_context],
                1,
            )
            .await?;
            Ok(Some(require_single_sequence(sequence)?))
        }
    }
}

pub(crate) async fn dispatch_subsasgn(
    obj: Value,
    subscript: Value,
    rhs: Value,
) -> crate::BuiltinResult<Value> {
    let path = crate::object::indexing::parse_standard_substruct(&subscript)?;
    crate::object::protocol::assign_subscript_path(obj, path, vec![rhs], None).await
}

#[runtime_builtin(
    name = "subsref",
    binding_variant = "default",
    builtin_path = "crate::builtins::introspection::object_indexing"
)]
pub async fn subsref_builtin(obj: Value, subscript: Value) -> crate::BuiltinResult<Value> {
    let path = crate::object::indexing::parse_standard_substruct(&subscript)?;
    let requested = crate::output_count::current_output_count();
    let access = crate::object::protocol::ObjectAccessContext::default();
    let request = match requested {
        None => crate::object::protocol::SubscriptReadRequest::require_single(access),
        Some(count) => crate::object::protocol::SubscriptReadRequest::select_outputs(count, access),
    };
    let usage = request.sequence_use;
    let sequence =
        crate::object::protocol::read_subscript_path_sequence_with_access(obj, path, request)
            .await?;
    let mut values =
        sequence.resolve(usage, crate::sequence::SequenceResolutionContext::default())?;
    if requested.is_some() {
        // Existing shared builtin ABI carrier; S2 owns its eventual replacement.
        Ok(Value::OutputList(values))
    } else {
        Ok(values.remove(0))
    }
}

#[runtime_builtin(
    name = "subsasgn",
    binding_variant = "default",
    builtin_path = "crate::builtins::introspection::object_indexing"
)]
pub async fn subsasgn_builtin(
    obj: Value,
    subscript: Value,
    rhs: Value,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::SUBSASGN_DESCRIPTOR,
        &runmat_builtins::SUBSASGN_ERROR_OUTPUTS,
        "subsasgn",
    )?;
    dispatch_subsasgn(obj, subscript, rhs).await
}

#[runtime_builtin(
    name = "numArgumentsFromSubscript",
    binding_variant = "default",
    builtin_path = "crate::builtins::introspection::object_indexing"
)]
pub async fn num_arguments_from_subscript_builtin(
    target: Value,
    subscript: Value,
    indexing_context: Value,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::NUM_ARGUMENTS_FROM_SUBSCRIPT_DESCRIPTOR,
        &runmat_builtins::NUM_ARGUMENTS_FROM_SUBSCRIPT_ERROR_OUTPUTS,
        "numArgumentsFromSubscript",
    )?;
    ensure_indexing_context_classes_registered();
    let context = parse_indexing_context(&indexing_context)?;
    if crate::object_receiver_class_name(&target).is_some() {
        if let Some(value) = dispatch_num_arguments_overload(
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
    Ok(Value::Num(exact_count_as_f64(default_num_arguments(
        &target, &levels, context,
    )?)?))
}

mod context;
mod selectors;
mod cardinality;
use cardinality::*;
pub use cardinality::{
    brace_assignment_cardinality, member_assignment_cardinality,
    object_path_assignment_cardinality, object_path_cardinality,
    PreparedObjectAssignmentCardinality,
};
pub(crate) use context::ensure_indexing_context_classes_registered;
use context::*;
use selectors::*;
#[cfg(test)]
mod tests;
