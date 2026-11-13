use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, CellFact, DimensionFact, DynamicReason, NumericClass,
    NumericDomain, NumericFact, ObjectFact, ResidencyFact, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};
use std::collections::BTreeMap;

fn shape(rows: usize, cols: usize) -> ShapeFact {
    ShapeFact::Shaped {
        dims: vec![DimensionFact::Known(rows), DimensionFact::Known(cols)],
    }
}

fn double(shape: ShapeFact) -> ValueFact {
    let storage = match shape.element_count() {
        Some(1) => StorageFact::Scalar,
        Some(_) => StorageFact::Dense,
        None => StorageFact::Unknown,
    };
    let mut value = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
        storage,
    );
    value.residency = ResidencyFact::Host;
    value
}

fn is_datetime(value: &ValueFact) -> bool {
    matches!(&value.kind, ValueKindFact::Object(object)
        if object.runtime_class.as_ref() == Some(&runmat_types::standard::DATETIME.owned()))
}

fn component_shape(value: &ValueFact) -> ShapeFact {
    match &value.kind {
        ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::String => {
            value.shape.clone()
        }
        ValueKindFact::Object(_) if is_datetime(value) => value.shape.clone(),
        ValueKindFact::Character => value
            .shape
            .known_dims()
            .and_then(|dims| dims.first().copied().flatten())
            .map(|rows| shape(rows, 1))
            .unwrap_or(ShapeFact::Unknown),
        _ => ShapeFact::Unknown,
    }
}

fn names(shape: ShapeFact) -> ValueFact {
    let mut value = ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(ValueFact::proven(
                ValueKindFact::Character,
                ShapeFact::Shaped {
                    dims: vec![DimensionFact::Known(1), DimensionFact::Unknown],
                },
                StorageFact::Dense,
            )),
            elements: Vec::new(),
            elements_complete: false,
        }),
        shape,
        StorageFact::Opaque,
    );
    value.residency = ResidencyFact::Host;
    value
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum CalendarBuiltinKind {
    Year,
    Month,
    Day,
    Hour,
    Minute,
    Now,
    Today,
    Clock,
}

macro_rules! typed_inference {
    ($($handler:ident => $kind:ident),+ $(,)?) => {$(
        pub(super) fn $handler(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
            infer(request, entry, CalendarBuiltinKind::$kind)
        }
    )+};
}
typed_inference!(
    year => Year, month => Month, day => Day, hour => Hour,
    minute => Minute, now => Now, today => Today, clock => Clock,
);

fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    kind: CalendarBuiltinKind,
) -> CallInference {
    let name = entry.identity.name;
    let current = matches!(
        kind,
        CalendarBuiltinKind::Now | CalendarBuiltinKind::Today | CalendarBuiltinKind::Clock
    );
    let mut diagnostics = Vec::new();
    if (kind == CalendarBuiltinKind::Today && request.arguments.len() > 1)
        || (current && kind != CalendarBuiltinKind::Today && !request.arguments.is_empty())
        || (!current && !(1..=2).contains(&request.arguments.len()))
    {
        diagnostics.push(argument_error(
            "RM-DATETIME-COMPONENT-ARITY",
            if kind == CalendarBuiltinKind::Today {
                "today expects at most one outputType".to_string()
            } else if current {
                format!("{name} takes no inputs")
            } else {
                format!("{name} expects one input and at most one mode or legacy format")
            },
            0,
        ));
    }
    let result = match kind {
        CalendarBuiltinKind::Now => double(ShapeFact::Scalar),
        CalendarBuiltinKind::Clock => double(shape(1, 6)),
        CalendarBuiltinKind::Today => {
            let output_type = if request.arguments.is_empty() {
                Some(crate::TodayOutputType::Datenum)
            } else {
                let literal = request.literals.literal_string_at(0);
                let parsed = literal.as_deref().and_then(crate::TodayOutputType::parse);
                let invalid = request.arguments.first().is_some_and(|value| {
                    !matches!(
                        value.kind,
                        ValueKindFact::Unknown | ValueKindFact::String | ValueKindFact::Character
                    ) || crate::catalog::inference::known_invalid_text_scalar(value)
                });
                if invalid || (literal.is_some() && parsed.is_none()) {
                    diagnostics.push(argument_error(
                        "RM-TODAY-OUTPUT-TYPE",
                        "today outputType must be 'datenum' or 'datetime' scalar text",
                        0,
                    ));
                }
                parsed
            };
            if output_type == Some(crate::TodayOutputType::Datenum) {
                return finish_fixed(entry, request, double(ShapeFact::Scalar), diagnostics);
            }
            if output_type.is_none() {
                let mut result = ValueFact::unknown(DynamicReason::RuntimeValue);
                result.shape = ShapeFact::Scalar;
                result.residency = ResidencyFact::Host;
                return finish_fixed(entry, request, result, diagnostics);
            }
            let mut result = ValueFact::proven(
                ValueKindFact::Object(ObjectFact {
                    class: None,
                    runtime_class: Some(runmat_types::standard::DATETIME.owned()),
                    properties: BTreeMap::new(),
                    properties_complete: false,
                    handle_semantics: Some(false),
                }),
                shape(1, 1),
                StorageFact::Opaque,
            );
            result.residency = ResidencyFact::Host;
            result
        }
        CalendarBuiltinKind::Year
        | CalendarBuiltinKind::Month
        | CalendarBuiltinKind::Day
        | CalendarBuiltinKind::Hour
        | CalendarBuiltinKind::Minute => {
            let shape = request
                .arguments
                .first()
                .map(component_shape)
                .unwrap_or(ShapeFact::Unknown);
            let modern = request.arguments.first().is_some_and(is_datetime);
            let may_be_modern = request.arguments.first().is_none_or(|value| {
                matches!(
                    value.kind,
                    ValueKindFact::Unknown | ValueKindFact::Object(_)
                )
            });
            let name_mode = request
                .literals
                .literal_string_at(1)
                .map(|mode| mode.trim().to_ascii_lowercase());
            if matches!(kind, CalendarBuiltinKind::Month | CalendarBuiltinKind::Day)
                && request.arguments.len() == 2
                && may_be_modern
                && (!modern || name_mode.is_none())
            {
                // Runtime mode selection can return either doubles or cells.
                let mut result = ValueFact::unknown(DynamicReason::RuntimeValue);
                result.shape = shape;
                result.residency = ResidencyFact::Host;
                result
            } else if modern
                && matches!(kind, CalendarBuiltinKind::Month | CalendarBuiltinKind::Day)
                && name_mode
                    .as_deref()
                    .is_some_and(|mode| matches!(mode, "name" | "shortname"))
            {
                names(shape)
            } else {
                double(shape)
            }
        }
    };
    finish_fixed(entry, request, result, diagnostics)
}
