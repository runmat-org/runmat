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
    if !request.arguments.len().is_multiple_of(2) {
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
    let mut output = timer_fact();
    if let ValueKindFact::Object(object) = &mut output.kind {
        for index in (0..request.arguments.len()).step_by(2) {
            let Some(value) = request.arguments.get(index + 1) else {
                break;
            };
            let Some(LiteralValue::String(name) | LiteralValue::Character(name)) =
                request.literals.literal_args.get(index)
            else {
                continue;
            };
            if let Some(property) =
                TimerProperty::parse(name).filter(|property| !property.is_readonly())
            {
                object
                    .properties
                    .insert(property.name().into(), value.clone());
            }
        }
    }
    finish_fixed(entry, request, output, validate_pairs(request, false))
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

pub(super) fn start(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_method(request, entry, false, CallbackPhase::Start)
}

pub(super) fn startat(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_method(request, entry, true, CallbackPhase::Start)
}

pub(super) fn stop(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_method(request, entry, false, CallbackPhase::Stop)
}

pub(super) fn wait(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_method(request, entry, false, CallbackPhase::Wait)
}

pub(super) fn delete(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_method(request, entry, false, CallbackPhase::Stop)
}

#[derive(Clone, Copy)]
enum CallbackPhase {
    Start,
    Stop,
    Wait,
}

impl CallbackPhase {
    fn properties(self) -> &'static [TimerProperty] {
        match self {
            Self::Start | Self::Wait => &[
                TimerProperty::StartFcn,
                TimerProperty::TimerFcn,
                TimerProperty::ErrorFcn,
                TimerProperty::StopFcn,
            ],
            Self::Stop => &[TimerProperty::StopFcn],
        }
    }
}

fn infer_method(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    optional_delay: bool,
    callbacks: CallbackPhase,
) -> CallInference {
    let mut diagnostics = Vec::new();
    let valid_arity = if optional_delay {
        matches!(request.arguments.len(), 1 | 2 | 4 | 7)
    } else {
        request.arguments.len() == 1
    };
    if !valid_arity {
        diagnostics.push(argument_error(
            "RM-CATALOG-TIMER-METHOD-ARITY",
            "timer method has an invalid number of arguments",
            request.arguments.len().saturating_sub(1),
        ));
    }
    receiver(request, &mut diagnostics);
    let mut result = finish_fixed(entry, request, default_double_scalar(), diagnostics);
    if let Some(receiver) = request.arguments.first() {
        infer_callback_effects(receiver, callbacks, &mut result);
    } else {
        result.effects.0.insert(EffectKind::Unknown);
    }
    result
}

fn infer_callback_effects(receiver: &ValueFact, phase: CallbackPhase, result: &mut CallInference) {
    let ValueKindFact::Object(object) = &receiver.kind else {
        result.effects.0.insert(EffectKind::Unknown);
        return;
    };
    for property in phase.properties() {
        let Some(callback) = object.properties.get(property.name()) else {
            if !object.properties_complete {
                result.effects.0.insert(EffectKind::Unknown);
            }
            continue;
        };
        let callback = match &callback.kind {
            ValueKindFact::Cell(cell) => {
                // Extra cell arguments can change a callback's catalog route.
                // Without evaluating that route here, keep the entire call
                // dynamically effectful rather than dropping those facts.
                if cell.elements.len() > 1 || !cell.elements_complete {
                    result.effects.0.insert(EffectKind::Unknown);
                }
                cell.elements
                    .first()
                    .or_else(|| (!cell.elements_complete).then_some(cell.element.as_ref()))
            }
            _ => Some(callback),
        };
        let Some(callback) = callback else { continue };
        match &callback.kind {
            ValueKindFact::Callable(callable) => {
                result
                    .capabilities
                    .0
                    .extend(callable.capabilities.0.iter().copied());
                if let Some(CallableIdentity::Builtin(identity)) = &callable.identity {
                    if let Some(entry) = crate::builtin_catalog_entry_by_name(&identity.0) {
                        // Project the declared contract once. Recursive
                        // argument-dependent inference can re-enter this
                        // timer (or a mutually recursive timer) indefinitely.
                        result.effects.0.extend(entry.contract.effect_set().0);
                        result
                            .capabilities
                            .0
                            .extend(entry.contract.capability_set().0);
                        result.effects.0.insert(EffectKind::Unknown);
                    } else {
                        result.effects.0.insert(EffectKind::Unknown);
                    }
                } else {
                    result.effects.0.insert(EffectKind::Unknown);
                }
            }
            ValueKindFact::String | ValueKindFact::Character => {
                result.effects.0.insert(EffectKind::WorkspaceRead);
                result.effects.0.insert(EffectKind::WorkspaceWrite);
                result.effects.0.insert(EffectKind::Unknown);
            }
            _ => {
                result.effects.0.insert(EffectKind::Unknown);
            }
        }
    }
}

macro_rules! setter_inference {
    ($name:ident, $property:ident) => {
        pub(super) fn $name(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
            infer_setter(request, entry, TimerProperty::$property)
        }
    };
}

setter_inference!(set_timer_fcn, TimerFcn);
setter_inference!(set_start_fcn, StartFcn);
setter_inference!(set_stop_fcn, StopFcn);
setter_inference!(set_error_fcn, ErrorFcn);
setter_inference!(set_period, Period);
setter_inference!(set_start_delay, StartDelay);
setter_inference!(set_tasks_to_execute, TasksToExecute);
setter_inference!(set_busy_mode, BusyMode);
setter_inference!(set_execution_mode, ExecutionMode);
setter_inference!(set_name, Name);
setter_inference!(set_tag, Tag);
setter_inference!(set_object_visibility, ObjectVisibility);
setter_inference!(set_user_data, UserData);

fn infer_setter(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    property: TimerProperty,
) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-TIMER-SETTER-ARITY",
            "timer property setter requires receiver and value",
            request.arguments.len().saturating_sub(1),
        ));
    }
    receiver(request, &mut diagnostics);
    let mut output = request
        .arguments
        .first()
        .cloned()
        .unwrap_or_else(timer_fact);
    if !matches!(output.kind, ValueKindFact::Object(_)) {
        output = timer_fact();
    }
    if let (Some(value), ValueKindFact::Object(object)) =
        (request.arguments.get(1), &mut output.kind)
    {
        object
            .properties
            .insert(property.name().into(), value.clone());
    }
    finish_fixed(entry, request, output, diagnostics)
}
