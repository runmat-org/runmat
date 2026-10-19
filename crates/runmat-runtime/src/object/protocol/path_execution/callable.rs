//! Invocation of an already-selected callable receiver in a subscript path.

use runmat_value::Value;

use crate::context::RuntimeContext;
use crate::execution::RootedValueSequence;
use crate::object::indexing::{ObjectIndexComponent, ObjectIndexSelector, ObjectSubscript};
use crate::runtime_error::semantic_error;
use crate::RuntimeError;

pub(super) fn is_callable_value(value: &Value) -> bool {
    matches!(
        value,
        Value::FunctionHandle(_)
            | Value::ExternalFunctionHandle(_)
            | Value::MethodFunctionHandle(_)
            | Value::BoundFunctionHandle(_)
            | Value::Closure(_)
    )
}

pub(super) fn paren_arguments(step: ObjectSubscript) -> Result<Vec<Value>, RuntimeError> {
    let (ObjectIndexSelector::IndexValues { components }, _) = step.into_parts() else {
        return Err(semantic_error(
            "InvalidObjectSubscriptPath",
            "callable parentheses require value arguments",
        ));
    };
    // Reject the entire selector before materializing arguments or invoking
    // the selected function. A colon is an index selector, not a call argument.
    if components
        .iter()
        .any(|component| matches!(component, ObjectIndexComponent::Colon))
    {
        return Err(semantic_error(
            "UnsupportedFunctionHandleSelector",
            "function handle call does not support colon selector syntax",
        ));
    }
    Ok(components
        .into_iter()
        .map(|component| match component {
            ObjectIndexComponent::Value(value) => value,
            ObjectIndexComponent::Colon => {
                unreachable!("colon rejected before argument materialization")
            }
        })
        .collect())
}

pub(super) async fn invoke(
    runtime: Option<&RuntimeContext>,
    target: Value,
    arguments: Vec<Value>,
    requested_outputs: usize,
) -> Result<RootedValueSequence, RuntimeError> {
    let descriptor = crate::call::descriptor::CallableDescriptor::from_feval_value(
        target,
        arguments,
        requested_outputs,
        &RuntimeFunctionResolver,
    );
    match runtime {
        Some(runtime) => {
            crate::call::descriptor::execute_callable_descriptor_in_context(runtime, descriptor)
                .await
        }
        None => crate::call::descriptor::execute_callable_descriptor(descriptor).await,
    }
}

struct RuntimeFunctionResolver;

impl crate::call::descriptor::FunctionNameResolver for RuntimeFunctionResolver {
    fn resolve_function(&self, name: &str) -> Option<runmat_types::FunctionId> {
        crate::user_functions::resolve_semantic_function_by_name(name).map(runmat_types::FunctionId)
    }
}
