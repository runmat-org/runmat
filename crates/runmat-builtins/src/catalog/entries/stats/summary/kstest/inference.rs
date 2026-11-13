//! Scalar result facts and conservative known-invalid admission for kstest.
use super::options::{KstestNamedCdf, KstestOptionName, KstestTail};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DynamicReason, InferenceDiagnostic,
    LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact, ValueFact,
    ValueKindFact,
};

fn literal_text(value: &LiteralValue) -> Option<&str> {
    match value {
        LiteralValue::Character(text)
        | LiteralValue::String(text)
        | LiteralValue::Keyword(text) => Some(text),
        _ => None,
    }
}

fn diagnose_sample(request: &CallRequest, diagnostics: &mut Vec<InferenceDiagnostic>) {
    let Some(sample) = request.arguments.first() else {
        diagnostics.push(InferenceDiagnostic::error(
            "ARITY",
            "kstest requires a sample",
        ));
        return;
    };
    if !matches!(
        &sample.kind,
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Real,
            ..
        }) | ValueKindFact::Logical
            | ValueKindFact::Unknown
    ) {
        diagnostics.push(InferenceDiagnostic::error(
            "INPUT",
            "kstest sample must be real numeric or logical",
        ));
    }
    if let Some(dims) = sample.shape.known_dims() {
        if dims
            .iter()
            .filter(|extent| extent.is_some_and(|value| value > 1))
            .count()
            > 1
        {
            diagnostics.push(InferenceDiagnostic::error(
                "SHAPE",
                "kstest sample must be a vector",
            ));
        }
        if dims.iter().all(|extent| extent.is_some()) && dims.contains(&Some(0)) {
            diagnostics.push(InferenceDiagnostic::error(
                "EMPTY",
                "kstest requires an observation",
            ));
        }
    }
}

fn diagnose_options(request: &CallRequest, diagnostics: &mut Vec<InferenceDiagnostic>) {
    if request.arguments.is_empty() || !(request.arguments.len() - 1).is_multiple_of(2) {
        diagnostics.push(InferenceDiagnostic::error(
            "OPTION",
            "kstest options require name-value pairs",
        ));
        return;
    }
    for index in (1..request.arguments.len()).step_by(2) {
        let Some(name) = request
            .literals
            .literal_args
            .get(index)
            .and_then(literal_text)
        else {
            if request.arguments.get(index).is_some_and(|fact| {
                matches!(
                    &fact.kind,
                    ValueKindFact::Numeric(_) | ValueKindFact::Logical
                )
            }) {
                diagnostics.push(InferenceDiagnostic::error(
                    "OPTION",
                    "kstest option name must be text",
                ));
            }
            continue;
        };
        let Some(option) = KstestOptionName::from_public(name) else {
            diagnostics.push(InferenceDiagnostic::error(
                "OPTION",
                "unsupported kstest option",
            ));
            continue;
        };
        let value = request.literals.literal_args.get(index + 1);
        match option {
            KstestOptionName::Alpha => {
                if let Some(value) = value {
                    if let Some(number) = LiteralContext::numeric_from_literal(value) {
                        if !(number.is_finite() && number > 0.0 && number < 1.0) {
                            diagnostics.push(InferenceDiagnostic::error(
                                "ALPHA",
                                "kstest Alpha must be in (0,1)",
                            ));
                        }
                    } else if !matches!(value, LiteralValue::Unknown) {
                        diagnostics.push(InferenceDiagnostic::error(
                            "ALPHA",
                            "kstest Alpha must be numeric",
                        ));
                    }
                }
                if request
                    .arguments
                    .get(index + 1)
                    .is_some_and(|fact| fact.shape.element_count().is_some_and(|count| count != 1))
                {
                    diagnostics.push(InferenceDiagnostic::error(
                        "ALPHA",
                        "kstest Alpha must be scalar",
                    ));
                }
                if request.arguments.get(index + 1).is_some_and(|fact| {
                    matches!(&fact.kind, ValueKindFact::Character | ValueKindFact::String)
                }) {
                    diagnostics.push(InferenceDiagnostic::error(
                        "ALPHA",
                        "kstest Alpha must be numeric",
                    ));
                }
            }
            KstestOptionName::Tail => {
                if let Some(value) = value {
                    if let Some(text) = literal_text(value) {
                        if KstestTail::from_public(text).is_none() {
                            diagnostics.push(InferenceDiagnostic::error(
                                "TAIL",
                                "unsupported kstest Tail",
                            ));
                        }
                    } else if !matches!(value, LiteralValue::Unknown) {
                        diagnostics.push(InferenceDiagnostic::error(
                            "TAIL",
                            "kstest Tail must be text",
                        ));
                    }
                }
                if request.arguments.get(index + 1).is_some_and(|fact| {
                    matches!(
                        &fact.kind,
                        ValueKindFact::Numeric(_) | ValueKindFact::Logical
                    )
                }) {
                    diagnostics.push(InferenceDiagnostic::error(
                        "TAIL",
                        "kstest Tail must be text",
                    ));
                }
            }
            KstestOptionName::Cdf => {
                if let Some(value) = value {
                    if let Some(text) = literal_text(value) {
                        if KstestNamedCdf::from_public(text).is_none() {
                            diagnostics.push(InferenceDiagnostic::error(
                                "CDF",
                                "unsupported named kstest CDF",
                            ));
                        }
                    }
                }
                if let Some(fact) = request.arguments.get(index + 1) {
                    if matches!(
                        &fact.kind,
                        ValueKindFact::Numeric(NumericFact {
                            domain: NumericDomain::Complex,
                            ..
                        }) | ValueKindFact::Logical
                    ) {
                        diagnostics.push(InferenceDiagnostic::error(
                            "CDF",
                            "kstest CDF table must be real numeric",
                        ));
                    }
                    if matches!(&fact.kind, ValueKindFact::Numeric(_)) {
                        if let Some(dims) = fact.shape.known_dims() {
                            if dims.len() != 2
                                || dims
                                    .get(1)
                                    .is_some_and(|extent| extent.is_some_and(|v| v != 2))
                                || dims
                                    .first()
                                    .is_some_and(|extent| extent.is_some_and(|v| v < 2))
                            {
                                diagnostics.push(InferenceDiagnostic::error(
                                    "CDF",
                                    "kstest CDF table must be n-by-2 with n >= 2",
                                ));
                            }
                        }
                    }
                }
            }
        }
    }
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    diagnose_sample(request, &mut diagnostics);
    diagnose_options(request, &mut diagnostics);
    let outputs = if diagnostics.is_empty() {
        vec![
            ValueFact::scalar(ValueKindFact::Logical),
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            })),
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            })),
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            })),
        ]
    } else {
        vec![ValueFact::unknown(DynamicReason::RuntimeValue); 4]
    };
    let mut contract = CallContract::fixed(outputs);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut inferred = infer_call(&contract, request);
    inferred.diagnostics.extend(diagnostics);
    inferred
}
