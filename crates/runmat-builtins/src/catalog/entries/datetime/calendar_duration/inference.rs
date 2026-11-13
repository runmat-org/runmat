use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, NumericClass, NumericDomain, NumericFact,
    ObjectFact, ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

fn calendar_class(value: &ValueFact) -> Option<bool> {
    match &value.kind {
        ValueKindFact::Object(object) => object
            .runtime_class
            .as_ref()
            .map(|class| class == &runmat_types::standard::CALENDAR_DURATION.owned()),
        _ => Some(false),
    }
}

fn calendar(shape: ShapeFact) -> ValueFact {
    let mut value = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(runmat_types::standard::CALENDAR_DURATION.owned()),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        shape,
        StorageFact::Opaque,
    );
    value.residency = ResidencyFact::Host;
    value
}

fn double(shape: ShapeFact) -> ValueFact {
    let storage = if shape.element_count() == Some(1) {
        StorageFact::Scalar
    } else {
        StorageFact::Dense
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

fn valid_numeric_or_calendar(value: &ValueFact) -> bool {
    match &value.kind {
        ValueKindFact::Unknown | ValueKindFact::Logical => true,
        ValueKindFact::Numeric(number) => number.domain == NumericDomain::Real,
        ValueKindFact::Object(_) => calendar_class(value).is_none_or(|is_calendar| is_calendar),
        _ => false,
    }
}

pub(super) fn unit(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CALENDAR-UNIT-ARITY",
            format!("{} expects one argument", entry.identity.name),
            0,
        ));
    }
    if let Some(input) = request.arguments.first() {
        if !valid_numeric_or_calendar(input) {
            diagnostics.push(argument_error(
                "RM-CALENDAR-UNIT-INPUT",
                "expected a real numeric, logical, or calendarDuration array",
                0,
            ));
        }
    }
    let output = match request.arguments.first() {
        Some(input) if calendar_class(input) == Some(true) => double(input.shape.clone()),
        Some(input)
            if matches!(
                input.kind,
                ValueKindFact::Numeric(_) | ValueKindFact::Logical
            ) =>
        {
            calendar(input.shape.clone())
        }
        Some(input) => {
            let mut value = ValueFact::unknown(DynamicReason::RuntimeValue);
            value.shape = input.shape.clone();
            value.residency = ResidencyFact::Host;
            value
        }
        None => ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    finish_fixed(entry, request, output, diagnostics)
}

fn component_shape(request: &CallRequest) -> ShapeFact {
    let mut shape = ShapeFact::Scalar;
    for input in &request.arguments {
        if input.shape.element_count() != Some(1) {
            if shape == ShapeFact::Scalar {
                shape = input.shape.clone();
            } else if shape != input.shape {
                return ShapeFact::Unknown;
            }
        }
    }
    shape
}

pub(super) fn constructor(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let count = request.arguments.len();
    if !matches!(count, 0 | 1 | 3..=6) {
        diagnostics.push(argument_error(
            "RM-CALENDAR-DURATION-ARITY",
            "calendarDuration expects no arguments, one input matrix, or three to six components",
            0,
        ));
    }
    for (index, value) in request.arguments.iter().enumerate() {
        if count != 1
            && !matches!(
                value.kind,
                ValueKindFact::Unknown
                    | ValueKindFact::Logical
                    | ValueKindFact::Numeric(NumericFact {
                        domain: NumericDomain::Real,
                        ..
                    })
            )
            || count == 1 && !valid_numeric_or_calendar(value)
        {
            diagnostics.push(argument_error(
                "RM-CALENDAR-DURATION-INPUT",
                "expected real numeric calendar components or a calendarDuration value",
                index,
            ));
        }
    }
    let shape = match count {
        0 => ShapeFact::Scalar,
        1 => {
            let input = &request.arguments[0];
            if calendar_class(input) == Some(true) {
                input.shape.clone()
            } else if let Some(dims) = input.shape.known_dims() {
                if dims.len() == 2 && matches!(dims[1], Some(3 | 6)) {
                    ShapeFact::Shaped {
                        dims: vec![
                            dims[0].map_or(
                                runmat_types::DimensionFact::Unknown,
                                runmat_types::DimensionFact::Known,
                            ),
                            runmat_types::DimensionFact::Known(1),
                        ],
                    }
                } else {
                    input.shape.clone()
                }
            } else {
                ShapeFact::Unknown
            }
        }
        _ => component_shape(request),
    };
    finish_fixed(entry, request, calendar(shape), diagnostics)
}

pub(super) fn predicate(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let diagnostics = if request.arguments.len() == 1 {
        Vec::new()
    } else {
        vec![argument_error(
            "RM-ISCALENDARDURATION-ARITY",
            "iscalendarduration expects one input",
            0,
        )]
    };
    let mut output = ValueFact::proven(
        ValueKindFact::Logical,
        ShapeFact::Scalar,
        StorageFact::Scalar,
    );
    output.residency = ResidencyFact::Host;
    finish_fixed(entry, request, output, diagnostics)
}
