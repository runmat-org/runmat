//! Source-known projection of regexprep's runtime option grammar.

use crate::catalog::inference::argument_error;
use runmat_types::{
    CallRequest, InferenceDiagnostic, LiteralContext, LiteralValue, NumericClass, StorageFact,
    ValueKindFact,
};

enum Toggle {
    Consumed,
    NotConsumed,
    Dynamic,
}

pub(super) fn diagnose(request: &CallRequest, diagnostics: &mut Vec<InferenceDiagnostic>) {
    let mut index = 3;
    while index < request.arguments.len() {
        if let Some(valid) = request
            .literals
            .literal_args
            .get(index)
            .and_then(known_occurrence)
        {
            if !valid || request.arguments[index].shape.is_definitely_nonscalar() {
                diagnostics.push(argument_error(
                    "RM-CATALOG-REGEXPREP-OCCURRENCE",
                    "regexprep occurrence must be a positive integer scalar",
                    index,
                ));
                break;
            }
            index += 1;
            continue;
        }
        let Some(option) = request.literals.literal_string_at(index) else {
            break;
        };
        index += 1;
        match option.as_str() {
            "ignorecase" | "matchcase" | "once" | "noemptymatch" | "stringanchors" => {}
            "lineanchors" | "dotall" | "dotexceptnewline" => {
                if index < request.arguments.len() {
                    match project_toggle(request, index) {
                        Toggle::Consumed => index += 1,
                        Toggle::NotConsumed => {}
                        Toggle::Dynamic => break,
                    }
                }
            }
            "emptymatch" => {
                if index < request.arguments.len() {
                    match request.literals.literal_string_at(index).as_deref() {
                        Some("allow" | "remove") => index += 1,
                        Some(_) => {}
                        None => break,
                    }
                }
            }
            "preservecase" => {
                if index < request.arguments.len() {
                    match project_toggle(request, index) {
                        Toggle::Consumed => index += 1,
                        Toggle::NotConsumed => {
                            diagnostics.push(argument_error(
                                "RM-CATALOG-REGEXPREP-OPTION",
                                "regexprep expects logical or on/off after preservecase",
                                index,
                            ));
                            break;
                        }
                        Toggle::Dynamic => break,
                    }
                }
            }
            "warnings" | "useparallel" => index += usize::from(index < request.arguments.len()),
            _ => {
                diagnostics.push(argument_error(
                    "RM-CATALOG-REGEXPREP-OPTION",
                    "regexprep option name is not recognised",
                    index - 1,
                ));
                break;
            }
        }
    }
}

// Match the runtime Value carrier, not merely a numeric literal's mathematical
// value: typed Single is a Tensor, never a parse_toggle scalar Num.
fn project_toggle(request: &CallRequest, index: usize) -> Toggle {
    let Some(literal) = request.literals.literal_args.get(index) else {
        return Toggle::Dynamic;
    };
    let fact = &request.arguments[index];
    match literal {
        LiteralValue::Bool(_)
            if matches!(&fact.kind, ValueKindFact::Logical)
                && fact.storage == StorageFact::Scalar =>
        {
            Toggle::Consumed
        }
        LiteralValue::Number(_)
            if matches!(&fact.kind, ValueKindFact::Numeric(n) if n.class == NumericClass::Double)
                && fact.storage == StorageFact::Scalar =>
        {
            Toggle::Consumed
        }
        LiteralValue::Integer { .. }
            if matches!(&fact.kind, ValueKindFact::Numeric(n) if n.class.integer_class().is_some())
                && fact.storage == StorageFact::Scalar
                && literal.integer_value_i128().is_some() =>
        {
            Toggle::Consumed
        }
        LiteralValue::Real {
            class: NumericClass::Single,
            ..
        } => Toggle::NotConsumed,
        LiteralValue::Character(_) | LiteralValue::String(_) | LiteralValue::Keyword(_)
            if matches!(&fact.kind, ValueKindFact::Character | ValueKindFact::String) =>
        {
            match request.literals.literal_string_at(index).as_deref() {
                Some("on" | "true" | "yes" | "1" | "off" | "false" | "no" | "0") => {
                    Toggle::Consumed
                }
                Some(_) => Toggle::NotConsumed,
                None => Toggle::Dynamic,
            }
        }
        _ => Toggle::Dynamic,
    }
}

// Only target-independent rejection is asserted. Positive integer values may
// exceed wasm32 usize yet be valid on a 64-bit host, so remain undiagnosed.
fn known_occurrence(value: &LiteralValue) -> Option<bool> {
    if let Some(exact) = value.integer_value_i128() {
        return Some(exact > 0);
    }
    let numeric = match value {
        LiteralValue::Number(_) | LiteralValue::Real { .. } => {
            LiteralContext::numeric_from_literal(value)?
        }
        _ => return None,
    };
    Some(
        numeric.is_finite() && numeric > 0.0 && numeric.fract() == 0.0 && numeric < u64::MAX as f64,
    )
}
