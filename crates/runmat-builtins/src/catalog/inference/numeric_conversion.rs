use super::{argument_error, finish_fixed, literal_text};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_numeric_conversion, CallInference, CallRequest, DynamicReason, InferenceDiagnostic,
    NumericClass, NumericDomain, NumericFact, ResidencyFact, ValueFact, ValueKindFact,
};

pub(super) fn infer_numeric_conversion_call(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    target: NumericClass,
) -> CallInference {
    let mut diagnostics = Vec::new();
    let Some(input) = request.arguments.first() else {
        diagnostics.push(argument_error(
            "RM-CATALOG-NUMERIC-CONVERSION-ARITY",
            format!("{} requires exactly one input", entry.identity.name),
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    };
    if request.arguments.len() > 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-NUMERIC-CONVERSION-ARITY",
            format!("{} accepts exactly one input", entry.identity.name),
            1,
        ));
    }
    let mut inference = infer_numeric_conversion(input, target);
    diagnostics.append(&mut inference.diagnostics);
    finish_fixed(entry, request, inference.fact, diagnostics)
}

pub(super) fn infer_numeric_conversion_with_like_call(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    target: NumericClass,
) -> CallInference {
    let mut diagnostics = Vec::new();
    let Some(input) = request.arguments.first() else {
        diagnostics.push(argument_error(
            "RM-CATALOG-NUMERIC-CONVERSION-ARITY",
            format!("{} requires an input value", entry.identity.name),
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    };

    let mut inference = infer_numeric_conversion(input, target);
    diagnostics.append(&mut inference.diagnostics);
    match request.arguments.as_slice() {
        [_] => {
            if matches!(inference.fact.residency, ResidencyFact::Device { .. }) {
                inference.fact.residency = ResidencyFact::Unknown;
            }
        }
        [_, keyword, prototype] => {
            let keyword_literal = request.literals.literal_args.get(1).and_then(literal_text);
            match keyword_literal.as_deref() {
                Some(value) if value.eq_ignore_ascii_case("like") => {
                    apply_like_prototype_residency(
                        &mut inference.fact,
                        prototype,
                        &mut diagnostics,
                    );
                }
                Some(_) => {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-NUMERIC-CONVERSION-LIKE",
                        format!("{} accepts only the \"like\" option", entry.identity.name),
                        1,
                    ));
                    inference.fact.residency = ResidencyFact::Unknown;
                }
                None if matches!(
                    keyword.kind,
                    ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
                ) =>
                {
                    inference.fact.residency = ResidencyFact::Unknown;
                }
                None => {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-NUMERIC-CONVERSION-LIKE",
                        format!(
                            "{} requires \"like\" as its second input",
                            entry.identity.name
                        ),
                        1,
                    ));
                    inference.fact.residency = ResidencyFact::Unknown;
                }
            }
        }
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-NUMERIC-CONVERSION-ARITY",
                format!(
                    "{} accepts either one input or an input followed by \"like\" and a prototype",
                    entry.identity.name
                ),
                request.arguments.len().saturating_sub(1),
            ));
            inference.fact.residency = ResidencyFact::Unknown;
        }
    }

    finish_fixed(entry, request, inference.fact, diagnostics)
}

fn apply_like_prototype_residency(
    output: &mut ValueFact,
    prototype: &ValueFact,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    match prototype.kind {
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Real,
            ..
        })
        | ValueKindFact::Logical => {
            output.residency = match &prototype.residency {
                ResidencyFact::Host => ResidencyFact::Host,
                ResidencyFact::Device { provider } => ResidencyFact::Device {
                    provider: provider.clone(),
                },
                ResidencyFact::Unknown => ResidencyFact::Unknown,
                ResidencyFact::Remote { .. } => {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-NUMERIC-CONVERSION-PROTOTYPE",
                        "a remote value cannot be used as a numeric conversion prototype",
                        2,
                    ));
                    ResidencyFact::Unknown
                }
            };
        }
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Complex,
            ..
        }) => {
            diagnostics.push(argument_error(
                "RM-CATALOG-NUMERIC-CONVERSION-PROTOTYPE",
                "complex numeric conversion prototypes are not supported",
                2,
            ));
            output.residency = ResidencyFact::Unknown;
        }
        ValueKindFact::Unknown => output.residency = ResidencyFact::Unknown,
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-NUMERIC-CONVERSION-PROTOTYPE",
                "numeric conversion prototypes must be real numeric or logical values",
                2,
            ));
            output.residency = ResidencyFact::Unknown;
        }
    }
}
