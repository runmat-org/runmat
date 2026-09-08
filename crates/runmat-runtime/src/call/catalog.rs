use runmat_types::{
    BuiltinId, CallInference, CallRequest, CallableFact, CallableIdentity, CapabilitySet,
    LiteralContext, OutputSelection, RequestedOutputCount, ValueFact,
};
use runmat_value::Value;

/// Query the canonical builtin contract for a runtime callable value.
///
/// This is a boundary adapter, not a name-based semantic registry. It resolves
/// runtime text/handle encodings, honors source-function shadowing, constructs
/// a typed builtin identity, and delegates all result semantics to the catalog.
pub(crate) fn infer_builtin_callback(
    callable: &Value,
    arguments: Vec<ValueFact>,
    requested_outputs: usize,
) -> Option<CallInference> {
    let identity = resolved_builtin_catalog_identity(callable)?;
    Some(infer_builtin_identity_callback(
        identity,
        arguments,
        requested_outputs,
    ))
}

pub(crate) fn infer_builtin_identity_callback(
    identity: runmat_builtins::BuiltinCatalogIdentity,
    arguments: Vec<ValueFact>,
    requested_outputs: usize,
) -> CallInference {
    let fact = CallableFact {
        identity: Some(CallableIdentity::Builtin(BuiltinId(identity.name.into()))),
        capabilities: CapabilitySet::default(),
        parameters: Vec::new(),
        parameters_complete: false,
        outputs: Vec::new(),
        outputs_complete: false,
        variadic_inputs: true,
        variadic_outputs: true,
        captures: Vec::new(),
        captures_complete: false,
    };
    runmat_builtins::infer_callable_call(
        &fact,
        &CallRequest {
            arguments,
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(requested_outputs)),
        },
    )
}

pub(crate) fn resolved_builtin_catalog_identity(
    callable: &Value,
) -> Option<runmat_builtins::BuiltinCatalogIdentity> {
    let name = callable_name(callable)?;
    if crate::user_functions::resolve_semantic_function_by_name(&name).is_some() {
        return None;
    }
    runmat_builtins::builtin_catalog_entry_by_name(&name).map(|entry| entry.identity)
}

fn callable_name(callable: &Value) -> Option<String> {
    let text = match callable {
        Value::FunctionHandle(name) | Value::ExternalFunctionHandle(name) => name.clone(),
        Value::Closure(closure) if closure.bound_function.is_none() => {
            closure.function_name.clone()
        }
        Value::String(text) => text.clone(),
        Value::StringArray(array) if array.data.len() == 1 => array.data.first()?.clone(),
        Value::CharArray(array) if array.rows == 1 => array.data.iter().collect(),
        Value::MethodFunctionHandle(_) | Value::BoundFunctionHandle { .. } | Value::Closure(_) => {
            return None
        }
        _ => return None,
    };
    let trimmed = text.trim();
    let name = trimmed.strip_prefix('@').unwrap_or(trimmed).trim();
    (!name.is_empty()).then(|| name.to_string())
}

pub(crate) fn scalarized_fact(value: &Value) -> ValueFact {
    let fact = crate::value_fact::value_fact(value);
    ValueFact::scalar(fact.kind)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{NumericClass, NumericDomain, NumericFact, ValueKindFact};
    use std::sync::Arc;

    fn single() -> ValueFact {
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }))
    }

    #[test]
    fn builtin_handle_delegates_to_catalog_inference() {
        let inferred = infer_builtin_callback(
            &Value::FunctionHandle("plus".into()),
            vec![single(), single()],
            1,
        )
        .expect("builtin callback");
        assert_eq!(
            inferred.outputs[0].numeric().map(|numeric| numeric.class),
            Some(NumericClass::Single)
        );
    }

    #[test]
    fn source_function_shadowing_prevents_builtin_classification() {
        let _resolver =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "plus").then_some(7)
            })));
        assert!(infer_builtin_callback(
            &Value::FunctionHandle("plus".into()),
            vec![single(), single()],
            1,
        )
        .is_none());
    }
}
