use crate::catalog::inference::{
    argument_error, default_double_scalar, finish_fixed, known_invalid_text_scalar, literal_text,
};
use crate::warning_semantics::{WarningCommand, WarningTarget};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, CellFact, DynamicReason, ResidencyFact, StructFact, ValueFact,
    ValueKindFact,
};
use std::collections::BTreeMap;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let output = if request.arguments.is_empty() {
        state_cell()
    } else if matches!(
        request.arguments[0].kind,
        ValueKindFact::Struct(_) | ValueKindFact::Cell(_)
    ) {
        if request.arguments.len() != 1 {
            diagnostics.push(argument_error(
                "RM-CATALOG-WARNING-STATE-ARITY",
                "warning state restoration accepts exactly one argument",
                1,
            ));
        }
        default_double_scalar()
    } else if let Some(command) = request.literals.literal_args.first().and_then(literal_text) {
        let target = request.literals.literal_args.get(1).and_then(literal_text);
        match WarningCommand::parse(&command) {
            Some(WarningCommand::SetMode(mode)) => {
                if request.arguments.len() > 2 {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-WARNING-MODE-ARITY",
                        "warning mode accepts at most one target",
                        2,
                    ));
                }
                if target
                    .as_deref()
                    .map(WarningTarget::parse)
                    .is_some_and(|target| {
                        matches!(target, WarningTarget::Backtrace | WarningTarget::Verbose)
                            && !matches!(
                                mode,
                                crate::warning_semantics::WarningMode::On
                                    | crate::warning_semantics::WarningMode::Off
                            )
                    })
                {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-WARNING-SPECIAL-MODE",
                        "warning backtrace and verbose targets accept only on or off",
                        1,
                    ));
                }
                state_struct()
            }
            Some(WarningCommand::Reset) => {
                if request.arguments.len() != 1 {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-WARNING-RESET-ARITY",
                        "warning reset accepts no target",
                        1,
                    ));
                }
                default_double_scalar()
            }
            Some(WarningCommand::Status) => {
                if request.arguments.len() != 1 {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-WARNING-STATUS-ARITY",
                        "warning status accepts no target",
                        1,
                    ));
                }
                state_cell()
            }
            Some(WarningCommand::Default | WarningCommand::Query) => {
                if request.arguments.len() > 2 {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-WARNING-COMMAND-ARITY",
                        "warning command accepts at most one target",
                        2,
                    ));
                }
                match target.as_deref().map(WarningTarget::parse) {
                    Some(WarningTarget::All) => state_cell(),
                    Some(_) => state_struct(),
                    None if request.arguments.len() == 1 => state_cell(),
                    None => ValueFact::unknown(DynamicReason::RuntimeValue),
                }
            }
            Some(WarningCommand::Backtrace) => {
                if request.arguments.len() > 2 {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-WARNING-BACKTRACE-ARITY",
                        "warning backtrace accepts at most one state",
                        2,
                    ));
                }
                if request.arguments.len() == 1 {
                    ValueFact::scalar(ValueKindFact::String)
                } else {
                    default_double_scalar()
                }
            }
            None => default_double_scalar(),
        }
    } else if matches!(request.arguments[0].kind, ValueKindFact::Exception(_)) {
        if request.arguments.len() != 1 {
            diagnostics.push(argument_error(
                "RM-CATALOG-WARNING-EXCEPTION-ARITY",
                "warning exception form accepts exactly one argument",
                1,
            ));
        }
        default_double_scalar()
    } else {
        ValueFact::unknown(DynamicReason::RuntimeValue)
    };
    if let Some(first) = request.arguments.first() {
        if !matches!(
            first.kind,
            ValueKindFact::Struct(_) | ValueKindFact::Cell(_) | ValueKindFact::Exception(_)
        ) && invalid_text_argument(first)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-WARNING-TEXT",
                "warning requires scalar text or a state/exception payload",
                0,
            ));
        }
    }
    if request.arguments.len() > 1
        && request
            .literals
            .literal_args
            .first()
            .and_then(literal_text)
            .and_then(|text| WarningCommand::parse(&text))
            .is_some_and(|command| {
                matches!(
                    command,
                    WarningCommand::SetMode(_)
                        | WarningCommand::Default
                        | WarningCommand::Query
                        | WarningCommand::Backtrace
                )
            })
        && invalid_text_argument(&request.arguments[1])
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-WARNING-TARGET-TEXT",
            "warning command requires a scalar text target or state",
            1,
        ));
    }
    finish_fixed(entry, request, output, diagnostics)
}

fn invalid_text_argument(value: &ValueFact) -> bool {
    match value.kind {
        ValueKindFact::Unknown | ValueKindFact::String | ValueKindFact::Character => {
            known_invalid_text_scalar(value)
        }
        _ => true,
    }
}

fn state_struct() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(
        BTreeMap::new(),
        false,
    )))
}

fn state_cell() -> ValueFact {
    let mut fact = ValueFact::unknown(DynamicReason::RuntimeValue);
    fact.kind = ValueKindFact::Cell(CellFact {
        element: Box::new(state_struct()),
        elements: Vec::new(),
        elements_complete: false,
    });
    fact.residency = ResidencyFact::Host;
    fact
}
