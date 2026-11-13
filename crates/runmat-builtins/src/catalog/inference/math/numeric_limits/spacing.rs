//! `eps` combines the scalar numeric-limit spellings with an array operation.

use crate::{BuiltinCatalogEntry, FloatingLimitKind, NumericLimitRule};
use runmat_types::{
    CallInference, CallRequest, DynamicReason, NumericClass, NumericDomain, NumericFact,
    ResidencyFact, StorageFact, ValueFact, ValueKindFact,
};

use super::super::super::{argument_error, finish_fixed, literal_text};

const RULE: NumericLimitRule = NumericLimitRule::Floating(FloatingLimitKind::SpacingAtOne);

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let output = match request.arguments.as_slice() {
        [] => super::output::default(RULE),
        [input] => match &input.kind {
            ValueKindFact::Numeric(numeric) => {
                if !matches!(numeric.class, NumericClass::Single | NumericClass::Double) {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-EPS-CLASS",
                        "eps requires a single or double input",
                        0,
                    ));
                    ValueFact::unknown(DynamicReason::UnsupportedRepresentation)
                } else {
                    let storage = match &input.residency {
                        ResidencyFact::Device { .. } => StorageFact::Opaque,
                        ResidencyFact::Host
                            if input.storage == StorageFact::Scalar
                                && numeric.class == NumericClass::Double =>
                        {
                            StorageFact::Scalar
                        }
                        ResidencyFact::Host
                            if matches!(
                                input.storage,
                                StorageFact::Scalar | StorageFact::Dense | StorageFact::Sparse
                            ) =>
                        {
                            StorageFact::Dense
                        }
                        _ => StorageFact::Unknown,
                    };
                    let mut result = ValueFact::proven(
                        ValueKindFact::Numeric(NumericFact {
                            class: numeric.class,
                            domain: NumericDomain::Real,
                        }),
                        input.shape.clone(),
                        storage,
                    );
                    result.residency = input.residency.clone();
                    if !matches!(result.residency, ResidencyFact::Host) {
                        result.layout = runmat_types::LayoutFact::Unknown;
                        result.contiguity = runmat_types::ContiguityFact::Unknown;
                    }
                    result
                }
            }
            ValueKindFact::String | ValueKindFact::Character => {
                class_name(request, &mut diagnostics)
            }
            ValueKindFact::Unknown => ValueFact::unknown(DynamicReason::RuntimeValue),
            _ => {
                diagnostics.push(argument_error(
                    "RM-CATALOG-EPS-CLASS",
                    "eps requires a floating array or a floating class name",
                    0,
                ));
                ValueFact::unknown(DynamicReason::UnsupportedRepresentation)
            }
        },
        [keyword, prototype] => {
            let keyword_literal = request.literals.literal_args.first().and_then(literal_text);
            match keyword_literal.as_deref() {
                Some(value) if value.eq_ignore_ascii_case("like") => {
                    super::output::like(RULE, prototype, "eps", &mut diagnostics)
                }
                Some(_) => {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-EPS-LIKE",
                        "eps accepts only the `like` option",
                        0,
                    ));
                    ValueFact::unknown(DynamicReason::UnsupportedRepresentation)
                }
                None if matches!(
                    keyword.kind,
                    ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
                ) =>
                {
                    ValueFact::unknown(DynamicReason::RuntimeValue)
                }
                None => {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-EPS-LIKE",
                        "eps requires `like` before its prototype",
                        0,
                    ));
                    ValueFact::unknown(DynamicReason::UnsupportedRepresentation)
                }
            }
        }
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-EPS-ARITY",
                "eps accepts zero inputs, one floating value or class, or `like` and a prototype",
                request.arguments.len().saturating_sub(1),
            ));
            ValueFact::unknown(DynamicReason::RuntimeValue)
        }
    };
    finish_fixed(entry, request, output, diagnostics)
}

fn class_name(
    request: &CallRequest,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
) -> ValueFact {
    match request.literals.literal_args.first().and_then(literal_text) {
        Some(name) => match NumericClass::from_class_name(name.trim()) {
            Some(class) if matches!(class, NumericClass::Single | NumericClass::Double) => {
                super::output::scalar(class, NumericDomain::Real)
            }
            _ => {
                diagnostics.push(argument_error(
                    "RM-CATALOG-EPS-CLASS",
                    format!("eps does not support class `{name}`"),
                    0,
                ));
                ValueFact::unknown(DynamicReason::UnsupportedRepresentation)
            }
        },
        None => ValueFact::unknown(DynamicReason::RuntimeValue),
    }
}
