use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, LiteralContext, LiteralValue, NumericClass, NumericDomain,
    NumericFact, ObjectFact, ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

use super::selector::{
    gpu_device_index_from_float, gpu_device_index_from_integer, GPU_DEVICE_ACTIVE_INDEX,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut fields = BTreeMap::new();
    for property in super::GpuDeviceProperty::ALL {
        fields.insert(property.name().into(), property.fact());
    }
    let mut output = ValueFact::scalar(ValueKindFact::Object(ObjectFact {
        class: None,
        runtime_class: Some(super::GPU_DEVICE_CLASS.owned()),
        properties: fields,
        properties_complete: false,
        handle_semantics: Some(true),
    }));
    output.storage = StorageFact::Opaque;
    output.residency = ResidencyFact::Host;
    if request.arguments.first().is_some_and(|selector| {
        selector.shape.element_count() == Some(0)
            && matches!(
                selector.kind,
                ValueKindFact::Numeric(_) | ValueKindFact::Logical
            )
    }) {
        output = ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(vec![Some(0), Some(0)]),
            StorageFact::Dense,
        );
        output.residency = ResidencyFact::Host;
    }
    let mut diagnostics = Vec::new();
    if request.arguments.len() > 1 {
        // This is a semantic arity diagnosis, not a promise about runtime error
        // precedence: compatibility gating of a logical first selector runs first.
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
            "RM.GPU_DEVICE.UNSUPPORTED_ARGUMENT",
            "Use reset(device), not a string selector",
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
    let known = match literal {
        Some(value @ LiteralValue::Integer { .. }) => value
            .integer_value_i128()
            .map(gpu_device_index_from_integer),
        Some(value @ (LiteralValue::Number(_) | LiteralValue::Real { .. })) => {
            LiteralContext::numeric_from_literal(value).map(gpu_device_index_from_float)
        }
        Some(LiteralValue::Bool(value)) => Some(if *value {
            Ok(GPU_DEVICE_ACTIVE_INDEX)
        } else {
            Err(super::selector::GpuDeviceIndexError::Invalid)
        }),
        _ => None,
    };
    if let Some(Err(_)) = known {
        diagnostics.push(argument_error(
            "RM.GPU_DEVICE.INVALID_INDEX",
            "gpuDevice index must be a positive uint32-range integer",
            0,
        ));
    }
}
