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
        Value::ScopedFunctionHandle(_)
            | Value::FunctionHandle(_)
            | Value::ExternalFunctionHandle(_)
            | Value::MethodFunctionHandle(_)
            | Value::BoundFunctionHandle(_)
            | Value::Closure(_)
    )
}

pub(super) fn paren_arguments(
    runtime: Option<&RuntimeContext>,
    step: ObjectSubscript,
) -> Result<crate::execution::RootedValues, RuntimeError> {
    let (ObjectIndexSelector::IndexValues { components }, _) = step.into_parts() else {
        return Err(semantic_error(
            "InvalidObjectSubscriptPath",
            "callable parentheses require value arguments",
        ));
    };
    component_arguments(runtime, components)
}

/// Project selectors directly into the canonical final argument backing.
/// The original selector owner and its roots survive final publication.
pub(super) fn component_arguments(
    runtime: Option<&RuntimeContext>,
    components: runmat_value::ExactArrayStorage<ObjectIndexComponent>,
) -> Result<crate::execution::RootedValues, RuntimeError> {
    if components.iter().any(|component| {
        matches!(
            component,
            ObjectIndexComponent::Colon | ObjectIndexComponent::End
        )
    }) {
        return Err(semantic_error(
            "UnsupportedFunctionHandleSelector",
            "function handle call does not support colon or contextual end selector syntax",
        ));
    }
    let values = components.iter().filter_map(|component| match component {
        ObjectIndexComponent::Value(value) => Some(value),
        ObjectIndexComponent::Colon | ObjectIndexComponent::End => None,
    });
    let _roots = match runtime {
        Some(runtime) => {
            crate::execution::RootedValueLease::for_values_iterator_in_context(values, runtime)?
        }
        None => crate::execution::RootedValueLease::for_values(values)
            .map_err(|e| e.into_runtime_error())?,
    };
    let count = components.len();
    let mut source = components.try_into_owned().map(|owner| owner.into_iter());
    let mut next = |index| {
        Ok(match &mut source {
            Ok(owner) => match owner.next().expect("exact selector count") {
                ObjectIndexComponent::Value(value) => value,
                ObjectIndexComponent::Colon | ObjectIndexComponent::End => {
                    unreachable!("syntax selectors rejected")
                }
            },
            Err(owner) => match &owner[index] {
                ObjectIndexComponent::Value(value) => value.clone(),
                ObjectIndexComponent::Colon | ObjectIndexComponent::End => {
                    unreachable!("syntax selectors rejected")
                }
            },
        })
    };
    match runtime {
        Some(runtime) => {
            crate::execution::RootedValues::try_initialize_in_context(count, runtime, &mut next)
        }
        None => {
            let mut values = runmat_value::exact_capacity_vec(count)
                .map_err(|e| crate::object::indexing::allocation_error(None, e.into()))?;
            for index in 0..count {
                values.push(next(index)?);
            }
            crate::execution::RootedValues::new(values).map_err(|e| e.into_runtime_error())
        }
    }
}

pub(super) async fn invoke(
    runtime: Option<&RuntimeContext>,
    target: Value,
    arguments: crate::call::BuiltinArguments,
    requested_outputs: usize,
    input_callsite: Option<crate::callsite::CallsiteInfo>,
) -> Result<RootedValueSequence, RuntimeError> {
    let target = crate::execution::RootedValue::in_optional_context(target, runtime)?;
    let selected = crate::call::descriptor::RepeatableCallable::select_rooted(
        runtime,
        target,
        &RuntimeFunctionResolver,
    )?
    .with_callsite(input_callsite);
    selected
        .invoke_retained(runtime, arguments, requested_outputs)
        .await
}

struct RuntimeFunctionResolver;

impl crate::call::descriptor::FunctionNameResolver for RuntimeFunctionResolver {
    fn resolve_function(&self, name: &str) -> Option<runmat_types::FunctionId> {
        crate::user_functions::resolve_semantic_function_by_name(name).map(runmat_types::FunctionId)
    }
}
