use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let output = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    if request
        .arguments
        .first()
        .is_some_and(definitely_console_target)
        || request
            .literals
            .literal_args
            .first()
            .is_some_and(definitely_standard_stream_id)
    {
        result.effects.0.remove(&EffectKind::FilesystemWrite);
        result
            .capabilities
            .0
            .remove(&CapabilityRequirement::Filesystem);
    }
    if request.arguments.is_empty() {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-FPRINTF-ARITY",
            "fprintf requires a format specification",
            0,
        ));
    }
    // The runtime's admitted calling convention locates a format value after
    // optional stream/fid operands, so unknown values cannot prove a position.
    if request.arguments.iter().all(definitely_not_format) {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-FPRINTF-FORMAT",
            "fprintf requires a text or gated numeric-code format",
            0,
        ));
    }
    result
}

fn definitely_standard_stream_id(value: &LiteralValue) -> bool {
    match value {
        LiteralValue::Number(value) => *value == 1.0 || *value == 2.0,
        LiteralValue::Real { text, .. } => text
            .parse::<f64>()
            .is_ok_and(|value| value == 1.0 || value == 2.0),
        LiteralValue::Integer { .. } => matches!(value.integer_value_i128(), Some(1 | 2)),
        _ => false,
    }
}

fn definitely_console_target(value: &ValueFact) -> bool {
    matches!(value.kind, ValueKindFact::Character)
        || matches!(value.kind, ValueKindFact::String) && value.shape.element_count() == Some(1)
}

fn definitely_not_format(value: &ValueFact) -> bool {
    matches!(
        value.kind,
        ValueKindFact::Cell(_)
            | ValueKindFact::Struct(_)
            | ValueKindFact::Object(_)
            | ValueKindFact::Callable(_)
            | ValueKindFact::Void
            | ValueKindFact::Never
    )
}
