use super::super::statset::runtime_empty_fact;
use crate::catalog::inference::{finish_fixed, gathered_value_fact, literal_text};
use crate::{BuiltinCatalogEntry, StatOptionField};
use runmat_types::{
    CallInference, CallRequest, DynamicReason, InferenceDiagnostic, NumericClass, NumericDomain,
    NumericFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn empty_numeric_fact() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(0), Some(0)]),
        StorageFact::Dense,
    )
}

fn selected_fact(request: &CallRequest) -> ValueFact {
    let Some(options) = request.arguments.first() else {
        return ValueFact::unknown(DynamicReason::RuntimeValue);
    };
    let Some(name) = request.literals.literal_args.get(1).and_then(literal_text) else {
        return ValueFact::unknown(DynamicReason::RuntimeValue);
    };
    let Some(field) = StatOptionField::resolve(&name) else {
        return empty_numeric_fact();
    };
    let ValueKindFact::Struct(structure) = &options.kind else {
        return ValueFact::unknown(DynamicReason::RuntimeValue);
    };
    let selected = structure
        .fields
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case(field.name()))
        .map(|(_, value)| value);
    let selected = match selected {
        Some(value) => gathered_value_fact(value),
        None if structure.fields_complete => empty_numeric_fact(),
        None => ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    if let Some(default) = request.arguments.get(2) {
        match runtime_empty_fact(&selected) {
            Some(true) => gathered_value_fact(default),
            Some(false) => selected,
            None => ValueFact::unknown(DynamicReason::RuntimeValue),
        }
    } else {
        selected
    }
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(2..=3).contains(&request.arguments.len()) {
        diagnostics.push(InferenceDiagnostic::error(
            "RM.STATGET.INVALID_ARGUMENT",
            "statget requires options, a field, and at most one default",
        ));
    }
    if let Some(options) = request.arguments.first() {
        if !matches!(
            options.kind,
            ValueKindFact::Struct(_) | ValueKindFact::Unknown
        ) {
            diagnostics.push(InferenceDiagnostic::error(
                "RM.STATGET.INVALID_OPTION",
                "statget options must be a structure",
            ));
        }
    }
    if let Some(field) = request.arguments.get(1) {
        if !matches!(
            field.kind,
            ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
        ) {
            diagnostics.push(InferenceDiagnostic::error(
                "RM.STATGET.INVALID_OPTION",
                "statget field must be text",
            ));
        }
    }
    finish_fixed(entry, request, selected_fact(request), diagnostics)
}
