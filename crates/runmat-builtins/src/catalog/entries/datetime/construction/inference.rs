use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, DimensionFact, DynamicReason, ObjectFact, ResidencyFact, ShapeFact,
    StorageFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

fn datetime(shape: ShapeFact) -> ValueFact {
    let mut value = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(runmat_types::standard::DATETIME.owned()),
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

fn date_vector_shape(input: &ValueFact) -> Option<ShapeFact> {
    let dims = input.shape.known_dims()?;
    if dims.len() != 2 || !matches!(dims[1], Some(3 | 6)) {
        return None;
    }
    Some(ShapeFact::Shaped {
        dims: vec![
            dims[0].map_or(DimensionFact::Unknown, DimensionFact::Known),
            DimensionFact::Known(1),
        ],
    })
}

fn single_input_shape(input: &ValueFact) -> ShapeFact {
    match &input.kind {
        ValueKindFact::Object(object)
            if object.runtime_class.as_ref() == Some(&runmat_types::standard::DATETIME.owned()) =>
        {
            input.shape.clone()
        }
        ValueKindFact::String => input.shape.clone(),
        ValueKindFact::Cell(_) => input.shape.clone(),
        ValueKindFact::Character => input
            .shape
            .known_dims()
            .and_then(|dims| dims.first().copied().flatten())
            .map(|rows| ShapeFact::Shaped {
                dims: vec![DimensionFact::Known(rows), DimensionFact::Known(1)],
            })
            .unwrap_or(ShapeFact::Unknown),
        ValueKindFact::Numeric(_) | ValueKindFact::Logical => {
            if let Some(vector) = date_vector_shape(input) {
                vector
            } else if input
                .shape
                .known_dims()
                .is_some_and(|dims| dims.len() == 2 && dims[1].is_none())
            {
                // A runtime second dimension of three or six changes the
                // result from a serial-shaped array into one date per row.
                ShapeFact::Unknown
            } else {
                input.shape.clone()
            }
        }
        _ => ShapeFact::Unknown,
    }
}

fn possible_character_vector(value: &ValueFact) -> bool {
    if !matches!(
        value.kind,
        ValueKindFact::Unknown | ValueKindFact::Character
    ) {
        return false;
    }
    value.shape.known_dims().is_none_or(|dims| {
        dims.first().copied().flatten().is_none_or(|rows| rows == 1)
            && dims
                .iter()
                .skip(2)
                .all(|dim| dim.is_none_or(|size| size == 1))
    })
}

fn possible_cellstr(value: &ValueFact) -> bool {
    let ValueKindFact::Cell(cell) = &value.kind else {
        return false;
    };
    possible_character_vector(&cell.element) && cell.elements.iter().all(possible_character_vector)
}

fn component_shape(arguments: &[ValueFact]) -> (ShapeFact, bool) {
    let mut result: Option<&ShapeFact> = None;
    let mut mismatch = false;
    let mut uncertain = false;
    for argument in arguments {
        if argument.shape.element_count() == Some(1) {
            continue;
        }
        if argument.shape.element_count().is_none() {
            uncertain = true;
        }
        if let Some(current) = result {
            if argument
                .shape
                .known_dims()
                .is_some_and(|dims| dims.iter().all(Option::is_some))
                && current
                    .known_dims()
                    .is_some_and(|dims| dims.iter().all(Option::is_some))
                && !argument.shape.is_proven_equivalent(current)
            {
                mismatch = true;
            }
        } else {
            result = Some(&argument.shape);
        }
    }
    (
        if uncertain {
            ShapeFact::Unknown
        } else {
            result.cloned().unwrap_or(ShapeFact::Scalar)
        },
        mismatch,
    )
}

fn possible_scalar_text(value: &ValueFact) -> bool {
    match &value.kind {
        ValueKindFact::Unknown => true,
        ValueKindFact::String => {
            value.shape.element_count() != Some(0)
                && value.shape.element_count().is_none_or(|count| count == 1)
        }
        ValueKindFact::Character => value
            .shape
            .known_dims()
            .is_none_or(|dims| dims.first().copied().flatten().is_none_or(|rows| rows == 1)),
        _ => false,
    }
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let mut positional_end = request.arguments.len();
    let mut explicit_conversion = false;
    let mut input_format = false;
    let mut format_seen = false;
    let mut unresolved_options = false;
    while positional_end >= 2 {
        let Some(option) = request.literals.literal_string_at(positional_end - 2) else {
            if matches!(
                &request.arguments[positional_end - 2].kind,
                ValueKindFact::String | ValueKindFact::Character
            ) {
                unresolved_options = true;
            }
            break;
        };
        if !matches!(option.trim(), "format" | "inputformat" | "convertfrom") {
            diagnostics.push(argument_error(
                "RM-DATETIME-OPTION-NAME",
                format!("unsupported datetime option '{option}'"),
                positional_end - 2,
            ));
            unresolved_options = true;
            break;
        }
        let value_index = positional_end - 1;
        if !possible_scalar_text(&request.arguments[value_index]) {
            diagnostics.push(argument_error(
                "RM-DATETIME-OPTION-VALUE",
                format!("{option} requires a text scalar"),
                value_index,
            ));
        }
        if option.trim() == "convertfrom" {
            if explicit_conversion {
                diagnostics.push(argument_error(
                    "RM-DATETIME-DUPLICATE-OPTION",
                    "ConvertFrom cannot be specified more than once",
                    positional_end - 2,
                ));
            }
            explicit_conversion = true;
            if let Some(value) = request.literals.literal_string_at(value_index) {
                if value.trim() != "datenum" {
                    diagnostics.push(argument_error(
                        "RM-DATETIME-CONVERSION",
                        "only ConvertFrom='datenum' is implemented",
                        value_index,
                    ));
                }
            }
        }
        if option.trim() == "inputformat" {
            if input_format {
                diagnostics.push(argument_error(
                    "RM-DATETIME-DUPLICATE-OPTION",
                    "InputFormat cannot be specified more than once",
                    positional_end - 2,
                ));
            }
            input_format = true;
        }
        if option.trim() == "format" {
            if format_seen {
                diagnostics.push(argument_error(
                    "RM-DATETIME-DUPLICATE-OPTION",
                    "Format cannot be specified more than once",
                    positional_end - 2,
                ));
            }
            format_seen = true;
        }
        positional_end -= 2;
    }

    let positional = &request.arguments[..positional_end];
    if input_format
        && (explicit_conversion
            || positional.len() != 1
            || !matches!(
                &positional[0].kind,
                ValueKindFact::Unknown
                    | ValueKindFact::String
                    | ValueKindFact::Character
                    | ValueKindFact::Cell(_)
            ))
    {
        diagnostics.push(argument_error(
            "RM-DATETIME-INPUT-FORMAT",
            "InputFormat requires exactly one text input",
            0,
        ));
    }
    let shape = if unresolved_options {
        ShapeFact::Unknown
    } else if explicit_conversion {
        if positional.len() != 1 {
            diagnostics.push(argument_error(
                "RM-DATETIME-CONVERSION-ARITY",
                "ConvertFrom='datenum' requires one numeric input",
                0,
            ));
            ShapeFact::Unknown
        } else {
            if !matches!(
                &positional[0].kind,
                ValueKindFact::Unknown | ValueKindFact::Logical
            ) && !matches!(
                &positional[0].kind,
                ValueKindFact::Numeric(number)
                    if number.domain == runmat_types::NumericDomain::Real
            ) {
                diagnostics.push(argument_error(
                    "RM-DATETIME-CONVERSION-INPUT",
                    "ConvertFrom='datenum' requires real numeric input",
                    0,
                ));
            }
            positional[0].shape.clone()
        }
    } else {
        match positional.len() {
            0 => ShapeFact::Scalar,
            1 => {
                let input = &positional[0];
                match &input.kind {
                    ValueKindFact::Unknown
                    | ValueKindFact::String
                    | ValueKindFact::Character
                    | ValueKindFact::Logical => {}
                    ValueKindFact::Cell(_) if possible_cellstr(input) => {}
                    ValueKindFact::Numeric(numeric)
                        if numeric.domain == runmat_types::NumericDomain::Real => {}
                    ValueKindFact::Object(object)
                        if object.runtime_class.as_ref()
                            == Some(&runmat_types::standard::DATETIME.owned())
                            || (object.runtime_class.is_none() && object.class.is_none()) => {}
                    _ => diagnostics.push(argument_error(
                        "RM-DATETIME-INPUT",
                        "expected date text, a datetime object, or real numeric date values",
                        0,
                    )),
                }
                single_input_shape(input)
            }
            3..=7 => {
                for (index, value) in positional.iter().enumerate() {
                    if !matches!(&value.kind, ValueKindFact::Unknown | ValueKindFact::Logical)
                        && !matches!(
                            &value.kind,
                            ValueKindFact::Numeric(number)
                                if number.domain == runmat_types::NumericDomain::Real
                        )
                    {
                        diagnostics.push(argument_error(
                            "RM-DATETIME-COMPONENT",
                            "calendar components must be real numeric values",
                            index,
                        ));
                    }
                }
                let (shape, mismatch) = component_shape(positional);
                if mismatch {
                    diagnostics.push(argument_error(
                        "RM-DATETIME-COMPONENT-SIZE",
                        "non-scalar calendar components must have matching sizes",
                        0,
                    ));
                }
                shape
            }
            _ => {
                diagnostics.push(argument_error(
                    "RM-DATETIME-ARITY",
                    "expected date text, a date-vector matrix, or three to seven components",
                    0,
                ));
                ShapeFact::Unknown
            }
        }
    };

    let output = if shape == ShapeFact::Unknown {
        let mut value = datetime(ShapeFact::Unknown);
        value.certainty = runmat_types::CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
        value
    } else {
        datetime(shape)
    };
    finish_fixed(entry, request, output, diagnostics)
}
