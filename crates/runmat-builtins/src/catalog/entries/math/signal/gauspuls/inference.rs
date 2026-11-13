use crate::catalog::inference::{argument_error, literal_text};
use crate::BuiltinCatalogEntry;
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let mode = request.literals.literal_args.first().and_then(literal_text);
    let cutoff = mode
        .as_ref()
        .is_some_and(|mode| mode.eq_ignore_ascii_case("cutoff"));
    let possibly_cutoff = request.arguments.first().is_some_and(|value| {
        matches!(
            value.kind,
            ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
        )
    });
    if let Some(mode) = mode {
        if !cutoff {
            diagnostics.push(argument_error(
                "RM-CATALOG-GAUSPULS-MODE",
                format!("unsupported gauspuls mode '{mode}'"),
                0,
            ));
        }
    }
    let max = if cutoff || possibly_cutoff { 5 } else { 4 };
    if request.arguments.is_empty() || request.arguments.len() > max {
        diagnostics.push(argument_error("RM-CATALOG-GAUSPULS-ARITY","gauspuls expects sample times and up to three controls, or cutoff mode and up to four controls",request.arguments.len().saturating_sub(1)));
    }
    let primary = if cutoff {
        double_scalar()
    } else {
        sample_output(request, &mut diagnostics)
    };
    for (index, value) in request.arguments.iter().enumerate().skip(1) {
        if value.shape.element_count() == Some(0) {
            continue;
        }
        if value.shape.element_count().is_some_and(|count| count != 1)
            || !matches!(
                value.kind,
                ValueKindFact::Numeric(NumericFact {
                    domain: NumericDomain::Real,
                    ..
                }) | ValueKindFact::Logical
                    | ValueKindFact::Unknown
            )
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-GAUSPULS-PARAMETER",
                "gauspuls controls must be real scalar values or empty default placeholders",
                index,
            ));
        }
        if let Some(number) = request.literals.numeric_at(index) {
            if !number.is_finite()
                || if index <= 2 {
                    number <= 0.0
                } else {
                    number >= 0.0
                }
            {
                diagnostics.push(argument_error("RM-CATALOG-GAUSPULS-PARAMETER","frequency/bandwidth must be positive; reference/cutoff levels must be negative",index));
            }
        }
    }
    let count = request.outputs.requested.known_count().unwrap_or(1);
    let outputs = (0..count)
        .map(|slot| {
            if cutoff && slot >= 1 || !possibly_cutoff && slot >= 3 {
                double_scalar()
            } else if possibly_cutoff && !cutoff {
                ValueFact::unknown(DynamicReason::RuntimeValue)
            } else {
                primary.clone()
            }
        })
        .collect();
    let mut contract = CallContract::fixed(outputs);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.dynamic_outputs = request.outputs.requested.known_count().is_none();
    result.diagnostics.extend(diagnostics);
    result
}
fn double_scalar() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }))
}
fn sample_output(request: &CallRequest, diagnostics: &mut Vec<InferenceDiagnostic>) -> ValueFact {
    let Some(input) = request.arguments.first() else {
        return ValueFact::unknown(DynamicReason::RuntimeValue);
    };
    let class = match input.kind {
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }) => NumericClass::Single,
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Real,
            ..
        })
        | ValueKindFact::Logical => NumericClass::Double,
        ValueKindFact::Unknown | ValueKindFact::String | ValueKindFact::Character => {
            return ValueFact::unknown(DynamicReason::RuntimeValue)
        }
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-GAUSPULS-INPUT",
                "gauspuls expects real numeric sample times",
                0,
            ));
            return ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
    };
    let storage = if input.shape.element_count() == Some(1) && class == NumericClass::Double {
        StorageFact::Scalar
    } else if input.shape.element_count().is_some() {
        StorageFact::Dense
    } else {
        StorageFact::Unknown
    };
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        input.shape.clone(),
        storage,
    )
}
