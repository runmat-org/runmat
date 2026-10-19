use super::super::{
    ImwriteAdmissionFailure, ImwriteEncoderFacts, ImwriteFormat, ImwriteOption, ImwriteWriteMode,
};
use crate::catalog::inference::argument_error;
use runmat_types::{
    CallRequest, LiteralValue, NumericClass, NumericDomain, ShapeFact, ValueKindFact,
};

fn known_image_rows_cols(shape: &ShapeFact) -> Option<(Option<usize>, Option<usize>)> {
    let dims = shape.known_dims()?;
    match dims.as_slice() {
        [] => Some((Some(1), Some(1))),
        [columns] => Some((Some(1), *columns)),
        [rows, columns, ..] => Some((*rows, *columns)),
    }
}

fn known_alpha_rows_cols(shape: &ShapeFact) -> Option<(Option<usize>, Option<usize>)> {
    let dims = shape.known_dims()?;
    match dims.as_slice() {
        [] => Some((Some(1), Some(1))),
        [rows, columns] => Some((*rows, *columns)),
        _ => None,
    }
}

pub(super) fn infer_known_tail(
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
                && super::super::super::text::could_be_scalar_text(value)
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
        if !super::super::super::text::could_be_scalar_text(value) {
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
        if !super::super::super::text::could_be_scalar_text(value) {
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
        let Some(value) = request.arguments.get(index + 1) else {
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
            let class_possible = matches!(
                value.kind,
                ValueKindFact::Unknown
                    | ValueKindFact::Logical
                    | ValueKindFact::Numeric(runmat_types::NumericFact {
                        class: NumericClass::Double
                            | NumericClass::Single
                            | NumericClass::UInt8
                            | NumericClass::UInt16,
                        domain: NumericDomain::Real,
                    })
            );
            if !class_possible {
                diagnostics.push(argument_error(
                    "RM-CATALOG-IMWRITE-OPTION",
                    "Alpha class must be double, single, uint8, or uint16",
                    index + 1,
                ));
            }
            if value
                .shape
                .rank()
                .is_some_and(|rank| rank != 0 && rank != 2)
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-IMWRITE-OPTION",
                    "Alpha must be an MxN array matching the image dimensions",
                    index + 1,
                ));
            }
            if let (Some((image_rows, image_cols)), Some((alpha_rows, alpha_cols))) = (
                request
                    .arguments
                    .first()
                    .and_then(|image| known_image_rows_cols(&image.shape)),
                known_alpha_rows_cols(&value.shape),
            ) {
                if image_rows.zip(alpha_rows).is_some_and(|(a, b)| a != b)
                    || image_cols.zip(alpha_cols).is_some_and(|(a, b)| a != b)
                {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-IMWRITE-OPTION",
                        "Alpha must be an MxN array matching the image dimensions",
                        index + 1,
                    ));
                }
            }
        }
        if matches!(
            option,
            ImwriteOption::Quality | ImwriteOption::DelayTime | ImwriteOption::LoopCount
        ) {
            if !matches!(
                value.kind,
                ValueKindFact::Unknown
                    | ValueKindFact::Logical
                    | ValueKindFact::Numeric(runmat_types::NumericFact {
                        domain: NumericDomain::Real,
                        ..
                    })
            ) || value
                .shape
                .known_dims()
                .is_some_and(|dims| dims.iter().any(|extent| extent.is_some_and(|n| n != 1)))
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-IMWRITE-OPTION",
                    "writer control must be a numeric scalar",
                    index + 1,
                ));
            }
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
            if !super::super::super::text::could_be_scalar_text(value) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-IMWRITE-OPTION",
                    "WriteMode must be a string scalar or char vector",
                    index + 1,
                ));
            }
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
