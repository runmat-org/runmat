use super::super::facts::*;
use super::{FileDatastoreOption, FileDatastoreReadMode};
use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::*;

fn text_list(value: &ValueFact) -> bool {
    match &value.kind {
        ValueKindFact::Unknown | ValueKindFact::String => true,
        ValueKindFact::Character => {
            value.shape.element_count() == Some(0)
                || !matches!(&value.shape,
            ShapeFact::Shaped { dims } if matches!(dims.first(), Some(DimensionFact::Known(rows)) if *rows != 1))
        }
        ValueKindFact::Cell(cell) => {
            text_list(&cell.element) && cell.elements.iter().all(text_list)
        }
        _ => false,
    }
}

fn boolean(request: &CallRequest, index: usize, diagnostics: &mut Vec<InferenceDiagnostic>) {
    let value = &request.arguments[index];
    let bad_shape = value.shape.element_count().is_some_and(|count| count != 1);
    let invalid = match value.kind {
        ValueKindFact::Unknown => false,
        ValueKindFact::Logical => bad_shape,
        ValueKindFact::Numeric(numeric) => {
            numeric.domain != NumericDomain::Real
                || bad_shape
                || (!matches!(numeric.class, NumericClass::Double)
                    && numeric.class.integer_class().is_none())
                || request
                    .literals
                    .numeric_at(index)
                    .is_some_and(|v| !v.is_finite())
            // Integer scalars are Value::Int, but a one-element dense
            // integer tensor cannot be distinguished here. Runtime owns
            // that representation-specific final admission.
        }
        ValueKindFact::String | ValueKindFact::Character => {
            text(request, index, diagnostics);
            request
                .literals
                .literal_string_at(index)
                .is_some_and(|value| {
                    !["true", "on", "yes", "false", "off", "no"]
                        .iter()
                        .any(|alias| value.eq_ignore_ascii_case(alias))
                })
        }
        _ => true,
    };
    if invalid {
        diagnostics.push(argument_error(
            "RM-FILE-DATASTORE-BOOLEAN",
            "IncludeSubfolders must be scalar logical or double zero/one; RunMat mode also accepts finite scalar doubles, native integer scalars and text boolean aliases",
            index,
        ));
    }
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    option_pairs(request, &mut diagnostics);
    let mut reader = string();
    let mut extensions = empty_names();
    let include = ValueFact::scalar(ValueKindFact::Logical);
    let mut mode = string();
    for index in (1..request.arguments.len().saturating_sub(1)).step_by(2) {
        text(request, index, &mut diagnostics);
        let Some(name) = request.literals.literal_string_at(index) else {
            // An unknown option may overwrite any retained option payload.
            reader = ValueFact::unknown(DynamicReason::RuntimeValue);
            extensions = ValueFact::unknown(DynamicReason::RuntimeValue);
            mode = ValueFact::unknown(DynamicReason::RuntimeValue);
            continue;
        };
        match FileDatastoreOption::parse(&name) {
            Some(FileDatastoreOption::ReadFcn) => reader = retained(request, index + 1),
            Some(FileDatastoreOption::FileExtensions) => {
                if !text_list(&request.arguments[index + 1]) {
                    diagnostics.push(argument_error(
                        "RM-FILE-DATASTORE-EXTENSIONS",
                        "FileExtensions must be text, a string array, or cellstr",
                        index + 1,
                    ));
                }
                extensions = retained(request, index + 1);
            }
            Some(FileDatastoreOption::IncludeSubfolders) => {
                boolean(request, index + 1, &mut diagnostics);
            }
            Some(FileDatastoreOption::ReadMode) => {
                text(request, index + 1, &mut diagnostics);
                if request
                    .literals
                    .literal_string_at(index + 1)
                    .is_some_and(|value| FileDatastoreReadMode::parse(&value).is_none())
                {
                    diagnostics.push(argument_error(
                        "RM-FILE-DATASTORE-READ-MODE",
                        "ReadMode must be file or partialfile",
                        index + 1,
                    ));
                }
                mode = string();
            }
            None => diagnostics.push(argument_error(
                "RM-FILE-DATASTORE-OPTION",
                "unsupported fileDatastore option",
                index,
            )),
        }
    }
    finish_fixed(
        entry,
        request,
        object(
            StaticClassIdentity::new("fileDatastore"),
            &[
                ("Files", retained(request, 0)),
                ("ReadFcn", reader),
                ("FileExtensions", extensions),
                ("IncludeSubfolders", include),
                ("ReadMode", mode),
            ],
        ),
        diagnostics,
    )
}
