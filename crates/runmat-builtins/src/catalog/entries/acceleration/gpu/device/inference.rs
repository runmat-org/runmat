use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, LiteralValue, NumericClass, NumericDomain, NumericFact,
    ResidencyFact, StorageFact, StructFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

fn numeric(class: NumericClass) -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class,
        domain: NumericDomain::Real,
    }))
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut fields = BTreeMap::new();
    fields.insert("device_id".into(), numeric(NumericClass::UInt32));
    fields.insert("index".into(), numeric(NumericClass::UInt32));
    fields.insert("name".into(), ValueFact::scalar(ValueKindFact::String));
    fields.insert("vendor".into(), ValueFact::scalar(ValueKindFact::String));
    fields.insert("precision".into(), ValueFact::scalar(ValueKindFact::String));
    fields.insert(
        "supports_double".into(),
        ValueFact::scalar(ValueKindFact::Logical),
    );
    // `backend` and `memory_bytes` are conditional on provider-reported metadata.
    let mut output = ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(fields, false)));
    output.storage = StorageFact::Dense;
    output.residency = ResidencyFact::Host;
    let mut diagnostics = Vec::new();
    if request.arguments.len() > 1 {
        diagnostics.push(argument_error(
            "RM.GPU_DEVICE.TOO_MANY_INPUTS",
            "gpuDevice accepts at most one selector",
            1,
        ));
    } else if let Some(selector) = request.arguments.first() {
        validate_selector(
            selector,
            request.literals.literal_args.first(),
            &mut diagnostics,
        );
    }
    finish_fixed(entry, request, output, diagnostics)
}

fn validate_selector(
    selector: &ValueFact,
    literal: Option<&LiteralValue>,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
) {
    if matches!(literal, Some(LiteralValue::Character(text) | LiteralValue::String(text) | LiteralValue::Keyword(text)) if text.trim().eq_ignore_ascii_case("reset"))
    {
        diagnostics.push(argument_error(
            "RM.GPU_DEVICE.RESET_NOT_SUPPORTED",
            "gpuDevice reset is unsupported",
            0,
        ));
        return;
    }
    let count = selector.shape.element_count();
    if count == Some(0)
        && matches!(
            selector.kind,
            ValueKindFact::Numeric(_) | ValueKindFact::Logical
        )
        && selector.storage != StorageFact::Sparse
        && !matches!(selector.residency, ResidencyFact::Device { .. })
    {
        diagnostics.push(argument_error(
            "RM.GPU_DEVICE.RESET_NOT_SUPPORTED",
            "gpuDevice reset by empty array is unsupported",
            0,
        ));
        return;
    }
    match &selector.kind {
        ValueKindFact::Unknown => {}
        ValueKindFact::Numeric(fact)
            if fact.domain == NumericDomain::Real
                && selector.storage != StorageFact::Sparse
                && !matches!(selector.residency, ResidencyFact::Device { .. }) => {}
        ValueKindFact::Logical if !matches!(selector.residency, ResidencyFact::Device { .. }) => {}
        _ => {
            diagnostics.push(argument_error(
                "RM.GPU_DEVICE.UNSUPPORTED_ARGUMENT",
                "gpuDevice selector is not a host real numeric or logical scalar",
                0,
            ));
            return;
        }
    }
    if count.is_some_and(|count| count > 1) {
        diagnostics.push(argument_error(
            "RM.GPU_DEVICE.INVALID_INDEX",
            "gpuDevice requires a scalar selector",
            0,
        ));
        return;
    }
    let invalid = match literal {
        Some(LiteralValue::Bool(false)) => true,
        Some(LiteralValue::Number(value)) => !valid_index(*value),
        Some(LiteralValue::Real { text, .. }) => text
            .parse::<f64>()
            .ok()
            .is_some_and(|value| !valid_index(value)),
        Some(LiteralValue::Integer { text, .. }) => text
            .parse::<i128>()
            .ok()
            .is_some_and(|value| value < 1 || value > i128::from(u32::MAX)),
        _ => false,
    };
    if invalid {
        diagnostics.push(argument_error(
            "RM.GPU_DEVICE.INVALID_INDEX",
            "gpuDevice index must be a positive uint32-range integer",
            0,
        ));
        return;
    }
    let unavailable = match literal {
        Some(LiteralValue::Number(value)) if valid_index(*value) => value.round() != 1.0,
        Some(LiteralValue::Real { text, .. }) => text
            .parse::<f64>()
            .ok()
            .is_some_and(|value| valid_index(value) && value.round() != 1.0),
        Some(LiteralValue::Integer { text, .. }) => text
            .parse::<i128>()
            .ok()
            .is_some_and(|value| value > 1 && value <= i128::from(u32::MAX)),
        _ => false,
    };
    if unavailable {
        diagnostics.push(argument_error(
            "RM.GPU_DEVICE.DEVICE_NOT_AVAILABLE",
            "only the active provider at index 1 is available",
            0,
        ));
    }
}

fn valid_index(value: f64) -> bool {
    let rounded = value.round();
    value.is_finite()
        && rounded >= 1.0
        && rounded <= u32::MAX as f64
        && (rounded - value).abs() <= 1e-9
}
