use super::{
    ImwriteAdmissionFailure, ImwriteEncoderFacts, ImwriteFormat, ImwriteOption, ImwriteWriteMode,
};
use crate::catalog::inference::{argument_error, finish_fixed_outputs};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, LiteralValue, NumericClass, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};

fn image_class_possible(value: &ValueFact) -> bool {
    matches!(
        value.kind,
        ValueKindFact::Unknown
            | ValueKindFact::Logical
            | ValueKindFact::Numeric(runmat_types::NumericFact {
                class: NumericClass::Double
                    | NumericClass::Single
                    | NumericClass::UInt8
                    | NumericClass::UInt16,
                ..
            })
    )
}

fn image_geometry_possible(shape: &ShapeFact) -> bool {
    if shape.rank().is_some_and(|rank| rank > 3) {
        return false;
    }
    !shape.known_dims().is_some_and(|dims| {
        dims.get(2)
            .and_then(|channel| *channel)
            .is_some_and(|channels| !matches!(channels, 1 | 3 | 4))
    })
}

fn colormap_possible(value: &ValueFact) -> bool {
    if matches!(value.storage, StorageFact::Sparse | StorageFact::Opaque) {
        return false;
    }
    if !matches!(
        value.kind,
        ValueKindFact::Unknown
            | ValueKindFact::Numeric(runmat_types::NumericFact {
                class: NumericClass::Double,
                ..
            })
    ) {
        return false;
    }
    !value.shape.known_dims().is_some_and(|dims| {
        dims.len() != 2
            || dims
                .first()
                .and_then(|row| *row)
                .is_some_and(|rows| rows == 0)
            || dims
                .get(1)
                .and_then(|column| *column)
                .is_some_and(|columns| columns != 3)
    })
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() < 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-IMWRITE-ARITY",
            "imwrite requires image data and a destination",
            0,
        ));
    }
    if let Some(image) = request.arguments.first() {
        if !image_class_possible(image) || !image_geometry_possible(&image.shape) {
            diagnostics.push(argument_error(
                "RM-CATALOG-IMWRITE-IMAGE",
                "image must be a supported numeric or logical 2-D or 3-D array",
                0,
            ));
        }
    }
    let second = request.arguments.get(1);
    let definitely_indexed =
        second.is_some_and(|value| !super::super::text::could_be_scalar_text(value));
    let filename_index = if definitely_indexed { 2 } else { 1 };
    if definitely_indexed {
        if let Some(map) = second {
            if !colormap_possible(map) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-IMWRITE-MAP",
                    "indexed colormap must be double Nx3",
                    1,
                ));
            }
        }
    }
    if let Some(filename) = request.arguments.get(filename_index) {
        if !super::super::text::could_be_scalar_text(filename) {
            diagnostics.push(argument_error(
                "RM-CATALOG-IMWRITE-FILENAME",
                "destination must be scalar text",
                filename_index,
            ));
        }
        if request
            .literals
            .literal_string_at(filename_index)
            .is_some_and(|text| text.trim().is_empty())
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-IMWRITE-FILENAME",
                "destination must not be empty",
                filename_index,
            ));
        }
    } else if definitely_indexed {
        diagnostics.push(argument_error(
            "RM-CATALOG-IMWRITE-ARITY",
            "indexed images require X, map, and filename",
            2,
        ));
    }
    infer_known_tail(
        request,
        filename_index,
        definitely_indexed,
        &mut diagnostics,
    );
    if request
        .outputs
        .requested
        .known_count()
        .is_some_and(|count| count > 0)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-IMWRITE-OUTPUTS",
            "imwrite does not return outputs",
            0,
        ));
    }
    finish_fixed_outputs(entry, request, vec![], diagnostics)
}

fn infer_known_tail(
    request: &CallRequest,
    filename_index: usize,
    indexed: bool,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
) {
    // If argument two may be a map or a filename, no later operand role is
    // proven. Do not diagnose a possible indexed call as a malformed direct call.
    if filename_index == 1
        && !request.arguments.get(1).is_some_and(|value| {
            matches!(value.kind, ValueKindFact::String | ValueKindFact::Character)
                && super::super::text::could_be_scalar_text(value)
        })
    {
        return;
    }
    let mut format = request
        .literals
        .literal_string_at(filename_index)
        .and_then(|path| {
            std::path::Path::new(&path)
                .extension()
                .and_then(|extension| extension.to_str())
                .and_then(ImwriteFormat::from_hint)
        });
    let mut alpha = false;
    let mut append = false;
    let mut index = filename_index + 1;
    let mut complete = true;
    if let Some(value) = request.arguments.get(index) {
        if !super::super::text::could_be_scalar_text(value) {
            diagnostics.push(argument_error(
                "RM-CATALOG-IMWRITE-TEXT",
                "format or option name must be scalar text",
                index,
            ));
            return;
        }
        match request.literals.literal_string_at(index) {
            Some(first) if ImwriteOption::from_name(&first).is_none() => {
                format = ImwriteFormat::from_hint(&first);
                if first.trim().trim_start_matches('.').is_empty() {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-IMWRITE-FORMAT",
                        "explicit image format must not be empty",
                        index,
                    ));
                }
                index += 1;
            }
            None => complete = false,
            _ => {}
        }
    }
    if !complete {
        return;
    }
    while index < request.arguments.len() {
        let value = &request.arguments[index];
        if !super::super::text::could_be_scalar_text(value) {
            diagnostics.push(argument_error(
                "RM-CATALOG-IMWRITE-TEXT",
                "option name must be scalar text",
                index,
            ));
            complete = false;
            break;
        }
        let Some(name) = request.literals.literal_string_at(index) else {
            complete = false;
            break;
        };
        let Some(option) = ImwriteOption::from_name(&name) else {
            diagnostics.push(argument_error(
                "RM-CATALOG-IMWRITE-OPTION",
                "writer option is unsupported",
                index,
            ));
            complete = false;
            break;
        };
        if option == ImwriteOption::UnsupportedAdvanced {
            diagnostics.push(argument_error(
                "RM-CATALOG-IMWRITE-OPTION",
                "writer option is recognized but unsupported",
                index,
            ));
        }
        let Some(_value) = request.arguments.get(index + 1) else {
            diagnostics.push(argument_error(
                "RM-CATALOG-IMWRITE-OPTION",
                "writer option requires a value",
                index,
            ));
            complete = false;
            break;
        };
        if option == ImwriteOption::Alpha {
            alpha = true;
        }
        if matches!(
            option,
            ImwriteOption::Quality | ImwriteOption::DelayTime | ImwriteOption::LoopCount
        ) {
            let known = match request.literals.literal_args.get(index + 1) {
                Some(LiteralValue::Number(number)) => Some(*number),
                Some(LiteralValue::Real { text, .. }) => text.parse::<f64>().ok(),
                _ => None,
            };
            if let Some(number) = known {
                if let Err(message) = option.validate_numeric(number) {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-IMWRITE-OPTION",
                        message,
                        index + 1,
                    ));
                }
            }
        }
        if option == ImwriteOption::WriteMode {
            if let Some(mode) = request.literals.literal_string_at(index + 1) {
                match ImwriteWriteMode::from_name(&mode) {
                    Some(ImwriteWriteMode::Append) => append = true,
                    Some(ImwriteWriteMode::Overwrite) => append = false,
                    None => diagnostics.push(argument_error(
                        "RM-CATALOG-IMWRITE-OPTION",
                        "WriteMode must be 'overwrite' or 'append'",
                        index + 1,
                    )),
                }
            }
        }
        index += 2;
    }
    let Some(format) = format else { return };
    let Some(image) = request.arguments.first() else {
        return;
    };
    let channels = if alpha {
        Some(4)
    } else if indexed {
        Some(3)
    } else {
        image.shape.known_dims().and_then(|dims| match dims.len() {
            0..=2 => Some(1),
            3 => dims[2],
            _ => None,
        })
    };
    let sixteen_bit = matches!(
        image.kind,
        ValueKindFact::Numeric(runmat_types::NumericFact {
            class: NumericClass::UInt16,
            ..
        })
    ) && !indexed;
    if let Some(channels) = channels {
        if let Err(failure) = (ImwriteEncoderFacts {
            format,
            channels,
            sixteen_bit,
            indexed_source: indexed,
            alpha_applied: alpha,
            append,
        })
        .validate()
        {
            // A later dynamic option could still supply Alpha or WriteMode.
            if !complete && !sixteen_bit {
                return;
            }
            let (code, message) = match failure {
                ImwriteAdmissionFailure::InvalidImage(message) => {
                    ("RM-CATALOG-IMWRITE-IMAGE", message)
                }
                ImwriteAdmissionFailure::InvalidOption(message) => {
                    ("RM-CATALOG-IMWRITE-OPTION", message)
                }
            };
            diagnostics.push(argument_error(code, message, 0));
        }
    }
}
