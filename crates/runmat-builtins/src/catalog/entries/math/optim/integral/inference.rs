use super::super::scalar_facts;
use super::IntegralOptionName;
use crate::catalog::inference::{argument_error, literal_text};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DynamicReason, NumericClass,
    NumericDomain, NumericFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn scalar_bound(
    request: &CallRequest,
    index: usize,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
) {
    let Some(fact) = request.arguments.get(index) else {
        return;
    };
    let admissible = matches!(
        fact.kind,
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Real,
            ..
        }) | ValueKindFact::Logical
            | ValueKindFact::Unknown
    ) && fact.storage != StorageFact::Sparse
        && !fact.shape.element_count().is_some_and(|count| count != 1);
    let finite_literal = request
        .literals
        .literal_args
        .get(index)
        .and_then(runmat_types::LiteralContext::numeric_from_literal)
        .is_none_or(f64::is_finite);
    if !admissible || !finite_literal {
        diagnostics.push(argument_error(
            "RM-CATALOG-INTEGRAL-BOUND",
            "integral requires finite real scalar bounds",
            index,
        ));
    }
}

fn controls(request: &CallRequest, diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>) {
    let rest = request.arguments.len().saturating_sub(3);
    if rest == 1 {
        let option = &request.arguments[3];
        if !matches!(
            option.kind,
            ValueKindFact::Struct(_) | ValueKindFact::Unknown
        ) || option.shape.element_count().is_some_and(|count| count != 1)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-INTEGRAL-OPTIONS",
                "integral expects a scalar options struct or name/value pairs",
                3,
            ));
        }
        return;
    }
    if !rest.is_multiple_of(2) {
        diagnostics.push(argument_error(
            "RM-CATALOG-INTEGRAL-OPTIONS",
            "integral requires complete name/value pairs",
            request.arguments.len().saturating_sub(1),
        ));
    }
    for index in (3..request.arguments.len()).step_by(2) {
        let name = &request.arguments[index];
        let valid_kind = matches!(
            name.kind,
            ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
        );
        let valid_shape = match name.kind {
            ValueKindFact::Character => !matches!(
                &name.shape,
                ShapeFact::Shaped { dims }
                    if matches!(dims.first(), Some(runmat_types::DimensionFact::Known(rows)) if *rows != 1)
            ),
            ValueKindFact::String => !name.shape.element_count().is_some_and(|count| count != 1),
            _ => true,
        };
        if !valid_kind || !valid_shape {
            diagnostics.push(argument_error(
                "RM-CATALOG-INTEGRAL-OPTION-NAME",
                "integral option names must be scalar text",
                index,
            ));
        }
        if let Some(text) = request
            .literals
            .literal_args
            .get(index)
            .and_then(literal_text)
        {
            if IntegralOptionName::parse(text).is_none() {
                diagnostics.push(argument_error(
                    "RM-CATALOG-INTEGRAL-OPTION-NAME",
                    "unsupported integral option",
                    index,
                ));
            }
        }
    }
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() < 3 {
        diagnostics.push(argument_error(
            "RM-CATALOG-INTEGRAL-ARITY",
            "integral requires a callback and two bounds",
            request.arguments.len().saturating_sub(1),
        ));
    }
    if request
        .outputs
        .requested
        .known_count()
        .is_some_and(|count| count > 1)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-INTEGRAL-OUTPUTS",
            "integral returns one numerical output",
            1,
        ));
    }
    scalar_facts::callback(request, &mut diagnostics);
    scalar_bound(request, 1, &mut diagnostics);
    scalar_bound(request, 2, &mut diagnostics);
    controls(request, &mut diagnostics);
    let value = if diagnostics.is_empty() {
        scalar_facts::host_scalar(NumericClass::Double)
    } else {
        ValueFact::unknown(DynamicReason::UnsupportedRepresentation)
    };
    let mut contract = CallContract::fixed(vec![value]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    if let Some(callback) = scalar_facts::callback_invocation(
        request,
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })),
    ) {
        contract.effects.0.extend(callback.effects.0);
        contract.capabilities.0.extend(callback.capabilities.0);
        diagnostics.extend(callback.diagnostics);
    }
    let mut inference = infer_call(&contract, request);
    inference.diagnostics.extend(diagnostics);
    inference
}
