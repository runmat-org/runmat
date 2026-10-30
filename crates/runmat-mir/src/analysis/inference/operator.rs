use runmat_types::{
    CallRequest, DynamicReason, FactInference, LiteralContext, LiteralValue, OperatorKind,
    OutputSelection, RequestedOutputCount, ValueFact, ValueKindFact,
};

/// Infers a binary language operator through the same typed class-method
/// catalog used by ordinary calls. Primitive operator identities resolve
/// their canonical builtin contracts; uncataloged primitives retain the
/// generic value rules. No builtin-specific inference belongs in MIR.
pub(crate) fn infer_binary_operator(
    operator: OperatorKind,
    left: ValueFact,
    right: ValueFact,
    literals: [LiteralValue; 2],
) -> FactInference {
    let receiver_class = match (&left.kind, &right.kind) {
        (ValueKindFact::Object(object), _) => object.runtime_class.clone(),
        (_, ValueKindFact::Object(object)) => object.runtime_class.clone(),
        _ => None,
    };
    let has_object_receiver = matches!(&left.kind, ValueKindFact::Object(_))
        || matches!(&right.kind, ValueKindFact::Object(_));
    if !has_object_receiver {
        if let Some(entry) = operator.overload_name().and_then(|method| {
            runmat_builtins::builtin_catalog_entry_by_name(method.display_name())
        }) {
            return infer_entry_operator(entry, vec![left, right], literals.to_vec());
        }
        return runmat_types::infer_binary(operator, &left, &right);
    }
    infer_cataloged_operator(
        operator,
        receiver_class.as_ref(),
        vec![left, right],
        literals.to_vec(),
    )
}

/// Infers a unary language operator through typed object-method metadata when
/// the operand is an object, otherwise through the primitive value rules.
pub(crate) fn infer_unary_operator(
    operator: OperatorKind,
    operand: ValueFact,
    literal: LiteralValue,
) -> FactInference {
    let ValueKindFact::Object(receiver) = &operand.kind else {
        return runmat_types::infer_unary(operator, &operand);
    };
    let receiver_class = receiver.runtime_class.clone();
    infer_cataloged_operator(
        operator,
        receiver_class.as_ref(),
        vec![operand],
        vec![literal],
    )
}

fn infer_cataloged_operator(
    operator: OperatorKind,
    receiver_class: Option<&runmat_types::ClassIdentity>,
    arguments: Vec<ValueFact>,
    literals: Vec<LiteralValue>,
) -> FactInference {
    let Some(method) = operator.overload_name() else {
        return dynamic_dispatch();
    };
    let Some(class) = receiver_class else {
        // A local or externally supplied class can still define the operator.
        // Absence of a builtin runtime identity is not proof of an invalid
        // primitive numeric operation.
        return dynamic_dispatch();
    };
    let method = method.owned();
    let Some(entry) = runmat_builtins::builtin_catalog_entry_for_class_method(class, &method)
    else {
        // The runtime can resolve registered or external class methods which
        // are not represented by the builtin catalog. Keep that path dynamic
        // instead of emitting a false primitive-numeric diagnostic.
        return dynamic_dispatch();
    };
    infer_entry_operator(entry, arguments, literals)
}

fn infer_entry_operator(
    entry: &runmat_builtins::BuiltinCatalogEntry,
    arguments: Vec<ValueFact>,
    literals: Vec<LiteralValue>,
) -> FactInference {
    let inferred = runmat_builtins::infer_catalog_call(
        entry,
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        },
    );
    FactInference {
        fact: inferred
            .outputs
            .into_iter()
            .next()
            .unwrap_or_else(|| ValueFact::unknown(DynamicReason::DynamicDispatch)),
        diagnostics: inferred.diagnostics,
    }
}

fn dynamic_dispatch() -> FactInference {
    FactInference::exact(ValueFact::unknown(DynamicReason::DynamicDispatch))
}

#[cfg(test)]
mod tests;
