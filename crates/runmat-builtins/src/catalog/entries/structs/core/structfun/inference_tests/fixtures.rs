use crate::{builtin_catalog_entry_by_name, infer_catalog_call};
use runmat_types::{
    BuiltinId, CallRequest, CallableFact, CallableIdentity, CapabilitySet, LiteralContext,
    NumericClass, NumericDomain, NumericFact, OutputSelection, RequestedOutputCount, StructFact,
    ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

pub(super) fn numeric(class: NumericClass) -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class,
        domain: NumericDomain::Real,
    }))
}

pub(super) fn callable(name: &str) -> ValueFact {
    ValueFact::scalar(ValueKindFact::Callable(CallableFact {
        identity: Some(CallableIdentity::Builtin(BuiltinId(name.into()))),
        capabilities: CapabilitySet::default(),
        parameters: Vec::new(),
        parameters_complete: false,
        outputs: Vec::new(),
        outputs_complete: false,
        variadic_inputs: true,
        variadic_outputs: true,
        captures: Vec::new(),
        captures_complete: true,
    }))
}

pub(super) fn structure(fields: &[(&str, ValueFact)]) -> ValueFact {
    structure_with_completeness(fields, true)
}

pub(super) fn structure_with_completeness(
    fields: &[(&str, ValueFact)],
    fields_complete: bool,
) -> ValueFact {
    ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(
        fields
            .iter()
            .map(|(name, value)| ((*name).into(), value.clone()))
            .collect::<BTreeMap<_, _>>(),
        fields_complete,
    )))
}

pub(super) fn infer(
    arguments: Vec<ValueFact>,
    literals: LiteralContext,
    count: RequestedOutputCount,
) -> runmat_types::CallInference {
    infer_catalog_call(
        builtin_catalog_entry_by_name("structfun").unwrap(),
        &CallRequest {
            arguments,
            literals,
            outputs: OutputSelection::new(count),
        },
    )
}
