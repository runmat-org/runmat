use crate::catalog::inference::{
    argument_error, default_double_scalar, finish_fixed, known_invalid_text_scalar, literal_text,
};
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
        match command.trim().to_ascii_lowercase().as_str() {
            "on" | "off" | "once" | "error" => {
                if request.arguments.len() > 2 {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-WARNING-MODE-ARITY",
                        "warning mode accepts at most one target",
                        2,
                    ));
                }
                state_struct()
            }
            "reset" => {
                if request.arguments.len() != 1 {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-WARNING-RESET-ARITY",
                        "warning reset accepts no target",
                        1,
                    ));
                }
                default_double_scalar()
            }
            "status" => {
                if request.arguments.len() != 1 {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-WARNING-STATUS-ARITY",
                        "warning status accepts no target",
                        1,
                    ));
                }
                state_cell()
            }
            "default" | "query" => {
                if request.arguments.len() > 2 {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-WARNING-COMMAND-ARITY",
                        "warning command accepts at most one target",
                        2,
                    ));
                }
                match target {
                    Some(target) if target.trim().eq_ignore_ascii_case("all") => state_cell(),
                    Some(_) => state_struct(),
                    None if request.arguments.len() == 1 => state_cell(),
                    None => ValueFact::unknown(DynamicReason::RuntimeValue),
                }
            }
            "backtrace" => {
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
            _ => default_double_scalar(),
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
        if known_invalid_text_scalar(first)
            && !matches!(
                first.kind,
                ValueKindFact::Struct(_) | ValueKindFact::Cell(_)
            )
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-WARNING-TEXT",
                "warning requires scalar text or a state/exception payload",
                0,
            ));
        }
    }
    finish_fixed(entry, request, output, diagnostics)
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
