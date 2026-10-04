use crate::catalog::inference::{argument_error, finish_fixed_outputs};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, CapabilityRequirement, EffectKind, ExecutionFact, LiteralValue,
    ValueKindFact,
};

use super::DeleteOption;

#[derive(Clone, Copy, PartialEq, Eq)]
enum KnownRoute {
    Files,
    Handles,
    Pool,
    Unknown,
    Invalid,
}

fn route(kind: &ValueKindFact) -> KnownRoute {
    match kind {
        ValueKindFact::String | ValueKindFact::Character => KnownRoute::Files,
        // A cell may contain paths or handles. Do not refine its capability to
        // either service without position-preserving element proof.
        ValueKindFact::Cell(_) => KnownRoute::Unknown,
        ValueKindFact::Object(object) if object.handle_semantics == Some(true) => {
            KnownRoute::Handles
        }
        ValueKindFact::Object(_) => KnownRoute::Unknown,
        ValueKindFact::Execution(ExecutionFact::Pool) => KnownRoute::Pool,
        ValueKindFact::Unknown | ValueKindFact::Composite(_) => KnownRoute::Unknown,
        _ => KnownRoute::Invalid,
    }
}

fn known_invalid_flag(literal: &LiteralValue) -> bool {
    match literal {
        LiteralValue::Bool(_) => false,
        LiteralValue::Number(value) => *value != 0.0 && *value != 1.0,
        LiteralValue::Integer { .. } => literal
            .integer_value_i128()
            .is_some_and(|value| value != 0 && value != 1),
        LiteralValue::Real { text, .. } => text
            .parse::<f64>()
            .ok()
            .is_some_and(|value| value != 0.0 && value != 1.0),
        _ => false,
    }
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.is_empty() {
        diagnostics.push(argument_error(
            "RM-CATALOG-DELETE-ARITY",
            "delete requires at least one target",
            0,
        ));
    }

    let mut operand_end = request.arguments.len();
    if operand_end >= 2
        && request
            .literals
            .literal_string_at(operand_end - 1)
            .is_some_and(|name| DeleteOption::from_name(&name).is_some())
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-DELETE-OPTION-VALUE",
            "ResolveSymbolicLinks requires an exact scalar logical or numeric zero/one",
            operand_end - 1,
        ));
    }
    if operand_end >= 3
        && request
            .literals
            .literal_string_at(operand_end - 2)
            .is_some_and(|name| DeleteOption::from_name(&name).is_some())
    {
        let name_index = operand_end - 2;
        let value_index = operand_end - 1;
        if !matches!(
            request.arguments[value_index].kind,
            ValueKindFact::Logical | ValueKindFact::Numeric(_) | ValueKindFact::Unknown
        ) || request.arguments[value_index]
            .shape
            .element_count()
            .is_some_and(|count| count != 1)
            || request
                .literals
                .literal_args
                .get(value_index)
                .is_some_and(known_invalid_flag)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-DELETE-OPTION-VALUE",
                "ResolveSymbolicLinks requires an exact scalar logical or numeric zero/one",
                value_index,
            ));
        }
        if request.arguments[name_index]
            .shape
            .element_count()
            .is_some_and(|count| count == 0)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-DELETE-OPTION-NAME",
                "ResolveSymbolicLinks must be a scalar text option name",
                name_index,
            ));
        }
        operand_end -= 2;
    }

    let routes: Vec<_> = request
        .arguments
        .iter()
        .take(operand_end)
        .map(|argument| route(&argument.kind))
        .collect();
    for (index, route) in routes.iter().enumerate() {
        if *route == KnownRoute::Invalid {
            diagnostics.push(argument_error(
                "RM-CATALOG-DELETE-TARGET",
                "delete accepts filenames, handle objects, listeners, or a pool",
                index,
            ));
        }
        if request
            .literals
            .literal_string_at(index)
            .is_some_and(|path| path.is_empty())
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-DELETE-EMPTY",
                "delete filename cannot be empty",
                index,
            ));
        }
    }
    if routes.contains(&KnownRoute::Pool)
        && (routes.len() != 1 || operand_end != request.arguments.len())
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-DELETE-POOL",
            "delete accepts a pool only as its sole input",
            0,
        ));
    }
    if routes.contains(&KnownRoute::Files) && routes.contains(&KnownRoute::Handles) {
        diagnostics.push(argument_error(
            "RM-CATALOG-DELETE-MIXED",
            "delete cannot mix proven filename and handle inputs",
            0,
        ));
    }

    let mut inferred = finish_fixed_outputs(entry, request, Vec::new(), diagnostics);
    if routes.len() == 1 && routes[0] == KnownRoute::Pool {
        inferred
            .capabilities
            .0
            .retain(|capability| *capability == CapabilityRequirement::ParallelRuntime);
        inferred.effects.0.remove(&EffectKind::FilesystemRead);
        inferred.effects.0.remove(&EffectKind::FilesystemWrite);
        inferred.effects.0.remove(&EffectKind::HostCallback);
    } else if !routes.is_empty() && routes.iter().all(|route| *route == KnownRoute::Files) {
        inferred
            .capabilities
            .0
            .retain(|capability| *capability == CapabilityRequirement::Filesystem);
        inferred.effects.0.remove(&EffectKind::HostCallback);
        inferred.effects.0.remove(&EffectKind::Unknown);
    } else if !routes.is_empty() && routes.iter().all(|route| *route == KnownRoute::Handles) {
        inferred
            .capabilities
            .0
            .retain(|capability| *capability == CapabilityRequirement::HostRuntime);
        inferred.effects.0.remove(&EffectKind::FilesystemRead);
        inferred.effects.0.remove(&EffectKind::FilesystemWrite);
    }
    inferred
}
