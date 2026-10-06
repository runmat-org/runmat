//! Conservative static facts for session-owned timer objects and methods.

use crate::catalog::inference::{argument_error, default_double_scalar, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::*;
use std::collections::BTreeMap;

use super::property::TimerProperty;

fn timer_fact() -> ValueFact {
    let mut fact = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(StaticClassIdentity::new("timer").owned()),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(true),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    fact.alias = AliasFact::Identity;
    fact.mutation = MutationFact::HandleSemantics;
    fact
}

fn validate_pairs(request: &CallRequest, filter: bool) -> Vec<InferenceDiagnostic> {
    let mut diagnostics = Vec::new();
    if request.arguments.len() % 2 != 0 {
        diagnostics.push(argument_error(
            "RM-CATALOG-TIMER-PAIRS",
            "timer properties require name/value pairs",
            request.arguments.len().saturating_sub(1),
        ));
    }
    for index in (0..request.arguments.len()).step_by(2) {
        let name = &request.arguments[index];
        if !matches!(
            name.kind,
            ValueKindFact::Unknown | ValueKindFact::String | ValueKindFact::Character
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-TIMER-PROPERTY-NAME",
                "timer property name must be scalar text",
                index,
            ));
        }
        if let Some(LiteralValue::String(text) | LiteralValue::Character(text)) =
            request.literals.literal_args.get(index)
        {
            match TimerProperty::parse(text) {
                None => diagnostics.push(argument_error(
                    "RM-CATALOG-TIMER-PROPERTY-NAME",
                    "unknown timer property",
                    index,
                )),
                Some(property) if !filter && property.is_readonly() => {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-TIMER-READONLY",
                        "timer property is read-only",
                        index,
                    ))
                }
                _ => {}
            }
        }
    }
    diagnostics
}

pub(super) fn constructor(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    finish_fixed(entry, request, timer_fact(), validate_pairs(request, false))
}

pub(super) fn find(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    // Zero, one, or many matches change the runtime representation.
    finish_fixed(
        entry,
        request,
        ValueFact::unknown(DynamicReason::RuntimeValue),
        validate_pairs(request, true),
    )
}

fn receiver(request: &CallRequest, diagnostics: &mut Vec<InferenceDiagnostic>) {
    if let Some(value) = request.arguments.first() {
        let admissible = match &value.kind {
            ValueKindFact::Unknown | ValueKindFact::Cell(_) => true,
            ValueKindFact::Object(object) => object
                .runtime_class
                .as_ref()
                .is_none_or(|class| class == &StaticClassIdentity::new("timer").owned()),
            _ => false,
        };
        if !admissible {
            diagnostics.push(argument_error(
                "RM-CATALOG-TIMER-RECEIVER",
                "expected a timer handle or cell of timer handles",
                0,
            ));
        }
    }
}

pub(super) fn method(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_method(request, entry, false)
}

pub(super) fn startat(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_method(request, entry, true)
}

fn infer_method(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    optional_delay: bool,
) -> CallInference {
    let mut diagnostics = Vec::new();
    let expected = if optional_delay { 1..=2 } else { 1..=1 };
    if !expected.contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-TIMER-METHOD-ARITY",
            "timer method has an invalid number of arguments",
            request.arguments.len().saturating_sub(1),
        ));
    }
    receiver(request, &mut diagnostics);
    finish_fixed(entry, request, default_double_scalar(), diagnostics)
}

pub(super) fn setter(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-TIMER-SETTER-ARITY",
            "timer property setter requires receiver and value",
            request.arguments.len().saturating_sub(1),
        ));
    }
    receiver(request, &mut diagnostics);
    finish_fixed(entry, request, timer_fact(), diagnostics)
}
