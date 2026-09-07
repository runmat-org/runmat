use runmat_types::{
    infer_call, CallInference, CallRequest, CallableFact, CallableIdentity, DynamicReason,
};

/// Infer one invocation through a typed callable fact.
///
/// Builtin identities delegate to their canonical catalog entry so higher-order
/// builtins observe the same argument-dependent result as a direct call. Other
/// callable identities retain the output contract carried by the value fact;
/// names that are still dynamically resolved are deliberately not treated as
/// builtin identities because a source function may shadow them.
pub fn infer_callable_call(callable: &CallableFact, request: &CallRequest) -> CallInference {
    if let Some(CallableIdentity::Builtin(identity)) = &callable.identity {
        if let Some(entry) = crate::builtin_catalog_entry_by_name(&identity.0) {
            return super::infer_catalog_call(entry, request);
        }
    }

    infer_call(
        &callable.call_contract(DynamicReason::UnresolvedCallable),
        request,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{
        BuiltinId, CapabilitySet, LiteralContext, NumericClass, NumericDomain, NumericFact,
        OutputSelection, RequestedOutputCount, ValueFact, ValueKindFact,
    };

    fn callable(identity: Option<CallableIdentity>, outputs: Vec<ValueFact>) -> CallableFact {
        CallableFact {
            identity,
            capabilities: CapabilitySet::default(),
            parameters: Vec::new(),
            parameters_complete: false,
            outputs_complete: !outputs.is_empty(),
            outputs,
            variadic_inputs: true,
            variadic_outputs: false,
            captures: Vec::new(),
            captures_complete: true,
        }
    }

    fn request(arguments: Vec<ValueFact>) -> CallRequest {
        CallRequest {
            arguments,
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        }
    }

    #[test]
    fn typed_builtin_identity_uses_argument_dependent_catalog_contract() {
        let single = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }));
        let inferred = infer_callable_call(
            &callable(
                Some(CallableIdentity::Builtin(BuiltinId("plus".into()))),
                Vec::new(),
            ),
            &request(vec![single.clone(), single]),
        );
        assert!(inferred.diagnostics.is_empty());
        assert_eq!(
            inferred.outputs[0].numeric().map(|numeric| numeric.class),
            Some(NumericClass::Single)
        );
    }

    #[test]
    fn dynamic_names_do_not_bypass_the_callable_contract() {
        let declared = ValueFact::scalar(ValueKindFact::Character);
        let inferred = infer_callable_call(
            &callable(
                Some(CallableIdentity::DynamicName(runmat_types::SymbolName(
                    "plus".into(),
                ))),
                vec![declared.clone()],
            ),
            &request(Vec::new()),
        );
        assert_eq!(inferred.outputs, vec![declared]);
    }
}
