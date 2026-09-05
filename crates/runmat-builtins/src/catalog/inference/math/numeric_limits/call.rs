use crate::{BuiltinCatalogEntry, NumericLimitRule};
use runmat_types::{
    CallInference, CallRequest, DynamicReason, NumericClass, ValueFact, ValueKindFact,
};

use super::super::super::{argument_error, finish_fixed, literal_text};

pub(in crate::catalog::inference) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    rule: NumericLimitRule,
) -> CallInference {
    let mut diagnostics = Vec::new();
    let mut output = super::output::default(rule);
    let builtin = entry.identity.name;

    match request.arguments.as_slice() {
        [] => {}
        [class] => match request.literals.literal_args.first().and_then(literal_text) {
            Some(name) => match NumericClass::from_class_name(&name) {
                Some(class) if super::output::accepts(rule, class) => {
                    output = super::output::scalar(class, runmat_types::NumericDomain::Real);
                }
                _ => {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-NUMERIC-LIMIT-CLASS",
                        format!("{builtin} does not support class `{name}`"),
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
                output = ValueFact::unknown(DynamicReason::RuntimeValue)
            }
            None => {
                diagnostics.push(argument_error(
                    "RM-CATALOG-NUMERIC-LIMIT-CLASS",
                    format!("{builtin} requires a numeric class name"),
                    0,
                ));
                output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
            }
        },
        [keyword, prototype] => {
            let keyword_literal = request.literals.literal_args.first().and_then(literal_text);
            match keyword_literal.as_deref() {
                Some(value) if value.eq_ignore_ascii_case("like") => {
                    output = super::output::like(rule, prototype, builtin, &mut diagnostics);
                }
                Some(_) => {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-NUMERIC-LIMIT-LIKE",
                        format!("{builtin} accepts only the `like` option"),
                        0,
                    ));
                    output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
                }
                None if matches!(
                    keyword.kind,
                    ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
                ) =>
                {
                    output = ValueFact::unknown(DynamicReason::RuntimeValue)
                }
                None => {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-NUMERIC-LIMIT-LIKE",
                        format!("{builtin} requires `like` before its prototype"),
                        0,
                    ));
                    output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
                }
            }
        }
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-NUMERIC-LIMIT-ARITY",
                format!("{builtin} accepts no input, a class name, or `like` and a prototype"),
                request.arguments.len().saturating_sub(1),
            ));
            output = ValueFact::unknown(DynamicReason::RuntimeValue);
        }
    }

    finish_fixed(entry, request, output, diagnostics)
}
