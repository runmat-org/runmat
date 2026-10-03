#[path = "inference/tail.rs"]
mod tail;

use crate::catalog::inference::{argument_error, finish_fixed_outputs};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, NumericClass, NumericDomain, ShapeFact, StorageFact, ValueFact,
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
                domain: NumericDomain::Real,
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
                domain: NumericDomain::Real,
            })
    ) {
        return false;
    }
    if value.shape.rank().is_some_and(|rank| rank != 2) {
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
    tail::infer_known_tail(
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
