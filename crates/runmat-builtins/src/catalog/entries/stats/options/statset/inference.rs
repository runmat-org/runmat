use super::option_value::{invalid_option_value, normalized_value, runtime_empty_fact};
use super::StatOptionField;
use crate::catalog::inference::{finish_fixed, literal_text};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, CellFact, DynamicReason, InferenceDiagnostic, NumericClass,
    NumericDomain, NumericFact, ShapeFact, StorageFact, StructFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

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

fn empty_option_fact(field: StatOptionField) -> ValueFact {
    if field == StatOptionField::Streams {
        ValueFact::proven(
            ValueKindFact::Cell(CellFact {
                element: Box::new(ValueFact::unknown(DynamicReason::RuntimeValue)),
                elements: vec![],
                elements_complete: true,
            }),
            ShapeFact::from(vec![Some(0), Some(0)]),
            StorageFact::Dense,
        )
    } else {
        empty_numeric_fact()
    }
}

pub(crate) fn empty_options_fact() -> ValueFact {
    let fields = StatOptionField::ALL
        .into_iter()
        .map(|field| (field.name().to_string(), empty_option_fact(field)))
        .collect::<BTreeMap<_, _>>();
    ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(fields, true)))
}

fn unknown_options_fact() -> ValueFact {
    let fields = StatOptionField::ALL
        .into_iter()
        .map(|field| {
            (
                field.name().to_string(),
                ValueFact::unknown(DynamicReason::RuntimeValue),
            )
        })
        .collect::<BTreeMap<_, _>>();
    ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(fields, true)))
}

fn check_source_structure(input: &ValueFact, diagnostics: &mut Vec<InferenceDiagnostic>) {
    let ValueKindFact::Struct(structure) = &input.kind else {
        return;
    };
    for (name, value) in &structure.fields {
        match StatOptionField::resolve(name) {
            Some(field) if invalid_option_value(field, value) => {
                diagnostics.push(InferenceDiagnostic::error(
                    "RM.STATSET.INVALID_OPTION",
                    format!("statset option {} rejects that value class", field.name()),
                ));
            }
            None => diagnostics.push(InferenceDiagnostic::error(
                "RM.STATSET.INVALID_OPTION",
                format!("statset has no unique option named {name}"),
            )),
            _ => {}
        }
    }
}

fn copied_options(input: &ValueFact) -> ValueFact {
    let ValueKindFact::Struct(source) = &input.kind else {
        return unknown_options_fact();
    };
    let mut output = unknown_options_fact();
    let ValueKindFact::Struct(target) = &mut output.kind else {
        unreachable!()
    };
    if source.fields_complete {
        for (name, value) in &source.fields {
            if let Some(field) = StatOptionField::resolve(name) {
                target
                    .fields
                    .insert(field.name().to_string(), normalized_value(field, value));
            }
        }
        // A complete known source can omit fields; runtime initializes them empty.
        for field in StatOptionField::ALL {
            if !source
                .fields
                .keys()
                .any(|name| StatOptionField::resolve(name) == Some(field))
            {
                target
                    .fields
                    .insert(field.name().to_string(), empty_option_fact(field));
            }
        }
    }
    output
}

fn nonempty(value: &ValueFact) -> Option<bool> {
    runtime_empty_fact(value).map(|empty| !empty)
}

fn merge_options(output: &mut ValueFact, incoming: &ValueFact) {
    let ValueKindFact::Struct(source) = &incoming.kind else {
        *output = unknown_options_fact();
        return;
    };
    let ValueKindFact::Struct(target) = &mut output.kind else {
        unreachable!()
    };
    if !source.fields_complete {
        *output = unknown_options_fact();
        return;
    }
    for (name, value) in &source.fields {
        if let Some(field) = StatOptionField::resolve(name) {
            let replacement = normalized_value(field, value);
            match nonempty(value) {
                Some(true) => {
                    target.fields.insert(field.name().to_string(), replacement);
                }
                Some(false) => {}
                None => {
                    target.fields.insert(
                        field.name().to_string(),
                        ValueFact::unknown(DynamicReason::RuntimeValue),
                    );
                }
            }
        }
    }
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let args = &request.arguments;
    let mut output = if args.is_empty() {
        empty_options_fact()
    } else if matches!(args[0].kind, ValueKindFact::Struct(_)) {
        copied_options(&args[0])
    } else {
        match request.literals.literal_args.first().and_then(literal_text) {
            Some(name) if StatOptionField::resolve(&name).is_some() => empty_options_fact(),
            Some(_) => unknown_options_fact(),
            None => unknown_options_fact(),
        }
    };

    // A runtime name may be either a field (followed by its value) or a
    // function-default selector. Their grammars differ, so no arity verdict
    // is sound until the first token is known.
    if !args.is_empty()
        && !matches!(args[0].kind, ValueKindFact::Struct(_))
        && request
            .literals
            .literal_args
            .first()
            .and_then(literal_text)
            .is_none()
    {
        if !matches!(
            args[0].kind,
            ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
        ) {
            diagnostics.push(InferenceDiagnostic::error(
                "RM.STATSET.INVALID_ARGUMENT",
                "statset requires a function name, option name, or options structure",
            ));
        }
        return finish_fixed(entry, request, output, diagnostics);
    }

    let mut pair_start = if args.is_empty() {
        0
    } else if matches!(args[0].kind, ValueKindFact::Struct(_)) {
        1
    } else if request
        .literals
        .literal_args
        .first()
        .and_then(literal_text)
        .is_some_and(|name| StatOptionField::resolve(&name).is_some())
    {
        0
    } else {
        1
    };
    if pair_start == 1
        && args.len() > 1
        && matches!(args[0].kind, ValueKindFact::Struct(_))
        && matches!(args[1].kind, ValueKindFact::Struct(_))
    {
        check_source_structure(&args[1], &mut diagnostics);
        merge_options(&mut output, &args[1]);
        pair_start = 2;
    }
    if args
        .first()
        .is_some_and(|first| matches!(first.kind, ValueKindFact::Struct(_)))
    {
        check_source_structure(&args[0], &mut diagnostics);
    }
    let remaining = args.len().saturating_sub(pair_start);
    if remaining % 2 != 0 {
        diagnostics.push(InferenceDiagnostic::error(
            "RM.STATSET.INVALID_ARGUMENT",
            "statset requires complete name-value pairs",
        ));
    }
    for index in (pair_start..args.len().saturating_sub(1)).step_by(2) {
        match request
            .literals
            .literal_args
            .get(index)
            .and_then(literal_text)
        {
            Some(name) => match StatOptionField::resolve(&name) {
                Some(field) => {
                    if invalid_option_value(field, &args[index + 1]) {
                        diagnostics.push(InferenceDiagnostic::error(
                            "RM.STATSET.INVALID_OPTION",
                            format!("statset option {} rejects that value class", field.name()),
                        ));
                    }
                    let ValueKindFact::Struct(structure) = &mut output.kind else {
                        unreachable!()
                    };
                    structure.fields.insert(
                        field.name().to_string(),
                        normalized_value(field, &args[index + 1]),
                    );
                }
                None => diagnostics.push(InferenceDiagnostic::error(
                    "RM.STATSET.INVALID_OPTION",
                    format!("statset has no unique option named {name}"),
                )),
            },
            None => output = unknown_options_fact(),
        }
    }
    if args.first().is_some_and(|first| {
        !matches!(
            first.kind,
            ValueKindFact::Struct(_)
                | ValueKindFact::String
                | ValueKindFact::Character
                | ValueKindFact::Unknown
        )
    }) {
        diagnostics.push(InferenceDiagnostic::error(
            "RM.STATSET.INVALID_ARGUMENT",
            "statset requires a function name, option name, or options structure",
        ));
    }
    if let ValueKindFact::Struct(structure) = &mut output.kind {
        structure.elements = vec![structure.fields.clone()];
        structure.elements_complete = true;
    }
    finish_fixed(entry, request, output, diagnostics)
}
