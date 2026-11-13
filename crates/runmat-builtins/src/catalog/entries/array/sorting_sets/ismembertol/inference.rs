use super::super::facts::*;
use crate::catalog::inference::{argument_error, finish_fixed_outputs};
use crate::BuiltinCatalogEntry;
use runmat_types::*;

fn boolean(
    request: &CallRequest,
    index: usize,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> Option<bool> {
    let value = &request.arguments[index];
    if value.shape.is_definitely_nonscalar()
        || !matches!(
            value.kind,
            ValueKindFact::Unknown
                | ValueKindFact::Logical
                | ValueKindFact::Numeric(NumericFact {
                    domain: NumericDomain::Real,
                    ..
                })
        )
    {
        diagnostics.push(argument_error(
            "RM-ISMEMBERTOL-BOOLEAN",
            "boolean option must be scalar logical or numeric zero/one",
            index,
        ));
    }
    if let Some(value) = request.literals.literal_bool_at(index) {
        return Some(value);
    }
    match request.literals.numeric_at(index) {
        Some(0.0) => Some(false),
        Some(1.0) => Some(true),
        Some(_) => {
            diagnostics.push(argument_error(
                "RM-ISMEMBERTOL-BOOLEAN",
                "boolean option must be zero or one",
                index,
            ));
            None
        }
        None => None,
    }
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    count(request, 2, &mut diagnostics);
    let a = argument(request, 0);
    let b = argument(request, 1);
    for (index, value) in [&a, &b].into_iter().enumerate() {
        accepted_data(value, false, false, &mut diagnostics, index);
        if matches!(
            value.kind,
            ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Complex,
                ..
            })
        ) {
            diagnostics.push(argument_error(
                "RM-ISMEMBERTOL-REAL",
                "data must be real",
                index,
            ));
        }
    }
    let mut by_rows = Some(false);
    let mut all = Some(false);
    let mut index = 2;
    if let Some(value) = request.arguments.get(index) {
        if !matches!(value.kind, ValueKindFact::String | ValueKindFact::Character) {
            if value.shape.is_definitely_nonscalar()
                || !matches!(
                    value.kind,
                    ValueKindFact::Unknown
                        | ValueKindFact::Logical
                        | ValueKindFact::Numeric(NumericFact {
                            domain: NumericDomain::Real,
                            ..
                        })
                )
                || request
                    .literals
                    .numeric_at(index)
                    .is_some_and(|v| !v.is_finite() || v <= 0.0)
            {
                diagnostics.push(argument_error(
                    "RM-ISMEMBERTOL-TOLERANCE",
                    "tolerance must be a finite positive real scalar",
                    index,
                ));
            }
            index += 1;
        }
    }
    if !(request.arguments.len().saturating_sub(index)).is_multiple_of(2) {
        diagnostics.push(argument_error(
            "RM-ISMEMBERTOL-PAIRS",
            "name-value options must appear in pairs",
            index,
        ));
    }
    while index + 1 < request.arguments.len() {
        match text_option(request, index, &mut diagnostics).as_deref() {
            Some("byrows") => by_rows = boolean(request, index + 1, &mut diagnostics),
            Some("outputallindices") => all = boolean(request, index + 1, &mut diagnostics),
            Some("datascale") => {
                let value = &request.arguments[index + 1];
                if !matches!(
                    value.kind,
                    ValueKindFact::Unknown
                        | ValueKindFact::Logical
                        | ValueKindFact::Numeric(NumericFact {
                            domain: NumericDomain::Real,
                            ..
                        })
                ) || request
                    .literals
                    .numeric_at(index + 1)
                    .is_some_and(|v| v.is_nan() || v < 0.0)
                {
                    diagnostics.push(argument_error(
                        "RM-ISMEMBERTOL-SCALE",
                        "DataScale must be nonnegative real numeric data",
                        index + 1,
                    ));
                }
            }
            Some(_) => diagnostics.push(argument_error(
                "RM-ISMEMBERTOL-OPTION",
                "unsupported tolerance-membership option",
                index,
            )),
            None => {
                by_rows = None;
                all = None;
            }
        }
        index += 2;
    }
    let shape = match by_rows {
        Some(true) => {
            row_compatibility(&a, &b, &mut diagnostics);
            column(dimension(&a.shape, 0))
        }
        Some(false) => a.shape.clone(),
        None => ShapeFact::Unknown,
    };
    let resident = first_resident(&a, &b);
    let residency = match (all, request.outputs.requested.known_count()) {
        (Some(true), Some(2)) => ResidencyFact::Host,
        (Some(true), None) | (None, Some(2) | None) => ResidencyFact::Unknown,
        _ => resident,
    };
    let locations = match all {
        Some(true) => ValueFact::proven(
            ValueKindFact::Cell(CellFact {
                element: Box::new(super::super::facts::index(
                    column(None),
                    ResidencyFact::Host,
                )),
                elements: Vec::new(),
                elements_complete: false,
            }),
            shape.clone(),
            StorageFact::Opaque,
        ),
        Some(false) => super::super::facts::index(shape.clone(), residency.clone()),
        None => unknown(),
    };
    finish_fixed_outputs(
        entry,
        request,
        vec![logical(shape, residency), locations],
        diagnostics,
    )
}
