use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, NumericClass, NumericDomain, NumericFact,
    ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn double(shape: ShapeFact) -> ValueFact {
    let storage = if shape.element_count() == Some(1) {
        StorageFact::Scalar
    } else {
        StorageFact::Dense
    };
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
        storage,
    )
}

fn dynamic_host(shape: ShapeFact) -> ValueFact {
    let mut value = ValueFact::unknown(DynamicReason::RuntimeValue);
    value.shape = shape;
    value.residency = ResidencyFact::Host;
    value
}

fn shape_for_date(input: Option<&ValueFact>) -> ShapeFact {
    match input {
        Some(value)
            if matches!(
                value.kind,
                ValueKindFact::Numeric(_) | ValueKindFact::Object(_)
            ) =>
        {
            value.shape.clone()
        }
        Some(value) if value.is_scalar() && matches!(value.kind, ValueKindFact::String) => {
            ShapeFact::Scalar
        }
        _ => ShapeFact::Unknown,
    }
}

fn matching_or_scalar_shape(left: Option<&ValueFact>, right: Option<&ValueFact>) -> ShapeFact {
    match (left, right) {
        (Some(a), Some(b)) if a.shape.element_count() == Some(1) => shape_for_date(Some(b)),
        (Some(a), Some(b)) if b.shape.element_count() == Some(1) => shape_for_date(Some(a)),
        (Some(a), Some(b)) if a.shape == b.shape => shape_for_date(Some(a)),
        _ => ShapeFact::Unknown,
    }
}

fn invalid_date(value: &ValueFact) -> bool {
    !matches!(
        value.kind,
        ValueKindFact::Unknown
            | ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Real,
                ..
            })
            | ValueKindFact::String
            | ValueKindFact::Character
            | ValueKindFact::Object(_)
    )
}

fn date_diagnostics(request: &CallRequest, count: usize) -> Vec<runmat_types::InferenceDiagnostic> {
    let mut diagnostics = Vec::new();
    for (index, value) in request.arguments.iter().take(count).enumerate() {
        if invalid_date(value) {
            diagnostics.push(argument_error(
                "RM-DATETIME-BUSINESS-DATE",
                "expected a real numeric, text, or datetime date",
                index,
            ));
        }
    }
    diagnostics
}

fn arity(
    request: &CallRequest,
    min: usize,
    max: usize,
    name: &str,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
) {
    if request.arguments.len() < min || request.arguments.len() > max {
        diagnostics.push(argument_error(
            "RM-DATETIME-BUSINESS-ARITY",
            format!("{name} expects {min} to {max} arguments"),
            0,
        ));
    }
}

pub(super) fn isbusday(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = date_diagnostics(request, 1);
    arity(request, 1, 3, "isbusday", &mut diagnostics);
    if let Some(mask) = request.arguments.get(2) {
        if mask.shape.element_count().is_some_and(|count| count != 7) {
            diagnostics.push(argument_error(
                "RM-ISBUSDAY-WEEKEND",
                "Weekend must contain seven Sunday-first flags",
                2,
            ));
        }
    }
    let shape = shape_for_date(request.arguments.first());
    finish_fixed(
        entry,
        request,
        ValueFact::proven(
            ValueKindFact::Logical,
            shape.clone(),
            if shape.element_count() == Some(1) {
                StorageFact::Scalar
            } else {
                StorageFact::Dense
            },
        ),
        diagnostics,
    )
}

pub(super) fn busdays(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = date_diagnostics(request, 2);
    arity(request, 2, 4, "busdays", &mut diagnostics);
    if request.literals.numeric_at(2).is_some_and(|mode| {
        !mode.is_finite() || mode.fract() != 0.0 || !(1.0..=6.0).contains(&mode)
    }) {
        diagnostics.push(argument_error(
            "RM-BUSDAYS-PERIOD",
            "bdmode must select daily through annual frequency",
            2,
        ));
    }
    let numeric_inputs = request.arguments.iter().take(2).all(|value| {
        matches!(
            value.kind,
            ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Real,
                ..
            })
        )
    });
    // A datetime Holiday selects datetime output even when both date endpoints
    // are serial numbers. Only prove double when the optional Holiday is known
    // not to be a datetime object.
    let numeric_output = numeric_inputs
        && request.arguments.get(3).is_none_or(|holiday| {
            matches!(
                holiday.kind,
                ValueKindFact::Numeric(_) | ValueKindFact::String | ValueKindFact::Character
            )
        });
    finish_fixed(
        entry,
        request,
        if numeric_output {
            double(ShapeFact::Unknown)
        } else {
            dynamic_host(ShapeFact::Unknown)
        },
        diagnostics,
    )
}

pub(super) fn days252bus(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = date_diagnostics(request, 2);
    arity(request, 2, 3, "days252bus", &mut diagnostics);
    finish_fixed(
        entry,
        request,
        double(matching_or_scalar_shape(
            request.arguments.first(),
            request.arguments.get(1),
        )),
        diagnostics,
    )
}

pub(super) fn daysdif(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = date_diagnostics(request, 2);
    arity(request, 2, 3, "daysdif", &mut diagnostics);
    if request
        .arguments
        .get(2)
        .is_some_and(|basis| basis.shape.element_count().is_some_and(|n| n != 1))
        || request
            .literals
            .numeric_at(2)
            .is_some_and(|basis| !matches!(basis, 0.0 | 1.0))
    {
        diagnostics.push(argument_error(
            "RM-DAYSDIF-BASIS",
            "only scalar Basis 0 and 1 are currently supported",
            2,
        ));
    }
    finish_fixed(
        entry,
        request,
        double(matching_or_scalar_shape(
            request.arguments.first(),
            request.arguments.get(1),
        )),
        diagnostics,
    )
}

fn month_date(request: &CallRequest, entry: &BuiltinCatalogEntry, name: &str) -> CallInference {
    let mut diagnostics = Vec::new();
    arity(request, 2, 5, name, &mut diagnostics);
    for (index, value) in request.arguments.iter().take(2).enumerate() {
        if !matches!(
            value.kind,
            ValueKindFact::Unknown
                | ValueKindFact::Numeric(NumericFact {
                    domain: NumericDomain::Real,
                    ..
                })
        ) {
            diagnostics.push(argument_error(
                "RM-DATETIME-BUSINESS-COMPONENT",
                "Year and Month must be real numeric",
                index,
            ));
        }
    }
    if request.arguments.get(3).is_some_and(|weekend| {
        weekend
            .shape
            .element_count()
            .is_some_and(|n| n != 0 && n != 7)
    }) {
        diagnostics.push(argument_error(
            "RM-DATETIME-BUSINESS-WEEKEND",
            "Weekend must be empty or contain seven Sunday-first flags",
            3,
        ));
    }
    let output_type = request
        .literals
        .literal_string_at(4)
        .map(|text| text.to_ascii_lowercase());
    if output_type
        .as_deref()
        .is_some_and(|text| !matches!(text, "datenum" | "datetime"))
    {
        diagnostics.push(argument_error(
            "RM-DATETIME-BUSINESS-OUTPUT",
            "outputType must be datenum or datetime",
            4,
        ));
    }
    let shape = matching_or_scalar_shape(request.arguments.first(), request.arguments.get(1));
    let datetime_output = output_type.as_deref() == Some("datetime")
        || (output_type.is_none()
            && request
                .arguments
                .get(2)
                .is_some_and(|value| matches!(value.kind, ValueKindFact::Object(_))));
    let unknown_output = output_type.is_none() && request.arguments.len() >= 5
        || (request.arguments.len() >= 3
            && request
                .arguments
                .get(2)
                .is_some_and(|value| value.kind == ValueKindFact::Unknown));
    finish_fixed(
        entry,
        request,
        if datetime_output || unknown_output {
            dynamic_host(shape)
        } else {
            double(shape)
        },
        diagnostics,
    )
}

pub(super) fn fbusdate(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    month_date(request, entry, "fbusdate")
}

pub(super) fn lbusdate(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    month_date(request, entry, "lbusdate")
}

pub(super) fn holidays(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = date_diagnostics(request, 2);
    arity(request, 0, 3, "holidays", &mut diagnostics);
    let numeric_range = (2..=3).contains(&request.arguments.len())
        && request.arguments.iter().all(|value| {
            matches!(
                value.kind,
                ValueKindFact::Numeric(NumericFact {
                    domain: NumericDomain::Real,
                    ..
                })
            )
        });
    finish_fixed(
        entry,
        request,
        if numeric_range {
            double(ShapeFact::Unknown)
        } else {
            dynamic_host(ShapeFact::Unknown)
        },
        diagnostics,
    )
}
