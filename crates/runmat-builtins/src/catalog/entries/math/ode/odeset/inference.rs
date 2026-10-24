use crate::catalog::inference::finish_fixed;
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, EffectKind, StructFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    // Names and merge values may be dynamic. Prove only the fixed scalar
    // structure schema, never a numeric value/class or a solver-ready option.
    let fields = super::super::OdeOptionField::ALL
        .into_iter()
        .map(|field| {
            (
                field.public_name().to_owned(),
                ValueFact::unknown(DynamicReason::RuntimeValue),
            )
        })
        .collect();
    let mut result = finish_fixed(
        entry,
        request,
        ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(fields, true))),
        Vec::new(),
    );
    // Only zero-input, zero-output calls print. Keep the entry's safe console
    // upper bound when nargout is unresolved and that form remains possible.
    if !request.arguments.is_empty()
        || request
            .outputs
            .requested
            .known_count()
            .is_some_and(|count| count > 0)
    {
        result.effects.0.remove(&EffectKind::FilesystemWrite);
        result.effects.0.remove(&EffectKind::UserInterface);
        result.effects.0.remove(&EffectKind::HostCallback);
    }
    result
}
