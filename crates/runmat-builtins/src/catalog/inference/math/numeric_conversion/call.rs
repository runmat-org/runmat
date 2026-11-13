use super::super::super::{argument_error, finish_fixed, literal_text};
use super::residency::{apply_default_conversion_residency, apply_like_prototype_residency};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_numeric_conversion, CallInference, CallRequest, DynamicReason, NumericClass,
    NumericDomain, ResidencyFact, ValueFact, ValueKindFact,
};

pub(in crate::catalog::inference) fn infer_numeric_conversion_call(
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
    if target.integer_class().is_some()
        && input.storage == runmat_types::StorageFact::Sparse
        && input
            .numeric()
            .is_some_and(|numeric| numeric.domain == NumericDomain::Complex)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-INTEGER-CONVERSION-COMPLEX-SPARSE",
            format!(
                "{} does not support complex sparse inputs",
                entry.identity.name
            ),
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
            diagnostics,
        );
    }
    let mut inference = infer_numeric_conversion(input, target);
    diagnostics.append(&mut inference.diagnostics);
    finish_fixed(entry, request, inference.fact, diagnostics)
}

pub(in crate::catalog::inference) fn infer_numeric_conversion_with_like_call(
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
            apply_default_conversion_residency(&mut inference.fact);
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

pub(in crate::catalog::inference) fn infer_single_fusion_fact(
    request: &CallRequest,
    target: NumericClass,
) -> Option<ValueFact> {
    let [input] = request.arguments.as_slice() else {
        return None;
    };
    let admitted = matches!(
        input.kind,
        ValueKindFact::Numeric(_)
            | ValueKindFact::Logical
            | ValueKindFact::Character
            | ValueKindFact::Symbolic
            | ValueKindFact::Unknown
    ) || matches!(input.kind, ValueKindFact::String)
        && target == NumericClass::Double;
    if !admitted {
        return None;
    }
    let mut inference = infer_numeric_conversion(input, target);
    if !inference.diagnostics.is_empty() {
        return None;
    }
    apply_default_conversion_residency(&mut inference.fact);
    Some(inference.fact)
}
