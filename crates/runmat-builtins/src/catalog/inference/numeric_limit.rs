use super::{argument_error, finish_fixed, literal_text};
use crate::{BuiltinCatalogEntry, NumericLimitRule};
use runmat_types::{
    CallInference, CallRequest, DynamicReason, InferenceDiagnostic, NumericClass, NumericDomain,
    NumericFact, StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer_numeric_limit(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    rule: NumericLimitRule,
) -> CallInference {
    let mut diagnostics = Vec::new();
    let default_class = match rule {
        NumericLimitRule::Integer(_) => NumericClass::Int32,
        NumericLimitRule::Floating(_) => NumericClass::Double,
    };
    let mut output = numeric_limit_scalar(default_class, NumericDomain::Real);

    match request.arguments.as_slice() {
        [] => {}
        [class] => match request.literals.literal_args.first().and_then(literal_text) {
            Some(name) => match NumericClass::from_class_name(&name) {
                Some(class) if numeric_limit_accepts_class(rule, class) => {
                    output = numeric_limit_scalar(class, NumericDomain::Real);
                }
                _ => {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-NUMERIC-LIMIT-CLASS",
                        format!("{} does not support class `{name}`", entry.identity.name),
                        0,
                    ));
                    output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
                }
            },
            None if matches!(
                class.kind,
                ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
            ) =>
            {
                output = ValueFact::unknown(DynamicReason::RuntimeValue);
            }
            None => {
                diagnostics.push(argument_error(
                    "RM-CATALOG-NUMERIC-LIMIT-CLASS",
                    format!("{} requires a numeric class name", entry.identity.name),
                    0,
                ));
                output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
            }
        },
        [keyword, prototype] => {
            let keyword_literal = request.literals.literal_args.first().and_then(literal_text);
            match keyword_literal.as_deref() {
                Some(value) if value.eq_ignore_ascii_case("like") => {
                    output = infer_numeric_limit_like(rule, prototype, entry, &mut diagnostics);
                }
                Some(_) => {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-NUMERIC-LIMIT-LIKE",
                        format!("{} accepts only the `like` option", entry.identity.name),
                        0,
                    ));
                    output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
                }
                None if matches!(
                    keyword.kind,
                    ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
                ) =>
                {
                    output = ValueFact::unknown(DynamicReason::RuntimeValue);
                }
                None => {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-NUMERIC-LIMIT-LIKE",
                        format!(
                            "{} requires `like` before its prototype",
                            entry.identity.name
                        ),
                        0,
                    ));
                    output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
                }
            }
        }
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-NUMERIC-LIMIT-ARITY",
                format!(
                    "{} accepts no input, a class name, or `like` and a prototype",
                    entry.identity.name
                ),
                request.arguments.len().saturating_sub(1),
            ));
            output = ValueFact::unknown(DynamicReason::RuntimeValue);
        }
    }

    finish_fixed(entry, request, output, diagnostics)
}

fn numeric_limit_accepts_class(rule: NumericLimitRule, class: NumericClass) -> bool {
    match rule {
        NumericLimitRule::Integer(_) => {
            !matches!(class, NumericClass::Double | NumericClass::Single)
        }
        NumericLimitRule::Floating(_) => {
            matches!(class, NumericClass::Double | NumericClass::Single)
        }
    }
}

fn numeric_limit_scalar(class: NumericClass, domain: NumericDomain) -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact { class, domain }))
}

fn infer_numeric_limit_like(
    rule: NumericLimitRule,
    prototype: &ValueFact,
    entry: &BuiltinCatalogEntry,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> ValueFact {
    let ValueKindFact::Numeric(numeric) = prototype.kind else {
        if !matches!(prototype.kind, ValueKindFact::Unknown) {
            diagnostics.push(argument_error(
                "RM-CATALOG-NUMERIC-LIMIT-PROTOTYPE",
                format!("{} requires a numeric prototype", entry.identity.name),
                1,
            ));
        }
        return ValueFact::unknown(DynamicReason::RuntimeValue);
    };
    if !numeric_limit_accepts_class(rule, numeric.class)
        || matches!(rule, NumericLimitRule::Integer(_))
            && matches!(prototype.storage, StorageFact::Sparse)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-NUMERIC-LIMIT-PROTOTYPE",
            format!(
                "{} does not support this prototype representation",
                entry.identity.name
            ),
            1,
        ));
        return ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
    }

    let mut output = numeric_limit_scalar(numeric.class, numeric.domain);
    output.storage = if matches!(prototype.storage, StorageFact::Sparse) {
        StorageFact::Sparse
    } else {
        StorageFact::Scalar
    };
    output.residency = prototype.residency.clone();
    output
}
