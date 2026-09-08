use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, CellFact, DimensionFact, DynamicReason, ShapeFact, StorageFact,
    ValueFact, ValueKindFact,
};

pub(in crate::catalog) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    finish_fixed(
        entry,
        request,
        output(request.arguments.first()),
        diagnostics(request),
    )
}

fn output(input: Option<&ValueFact>) -> ValueFact {
    let Some(input) = input else {
        return ValueFact::unknown(DynamicReason::RuntimeValue);
    };
    let shape = match input.kind {
        ValueKindFact::Character => character_output_shape(&input.shape),
        ValueKindFact::String
        | ValueKindFact::Symbolic
        | ValueKindFact::Cell(_)
        | ValueKindFact::Unknown => input.shape.clone(),
        _ => ShapeFact::Unknown,
    };
    ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(character_row()),
            elements: Vec::new(),
            elements_complete: shape.element_count() == Some(0),
        }),
        shape,
        StorageFact::Dense,
    )
}

fn character_output_shape(input: &ShapeFact) -> ShapeFact {
    match input {
        ShapeFact::Scalar => ShapeFact::Scalar,
        ShapeFact::Shaped { dims } => ShapeFact::Shaped {
            dims: vec![
                dims.first().cloned().unwrap_or(DimensionFact::Known(0)),
                DimensionFact::Known(1),
            ],
        },
        ShapeFact::Ranked { .. } | ShapeFact::Unknown => ShapeFact::from(vec![None, Some(1)]),
    }
}

fn character_row() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(1), None]),
        StorageFact::Dense,
    )
}

fn diagnostics(request: &CallRequest) -> Vec<runmat_types::InferenceDiagnostic> {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-CELLSTR-ARITY",
            "cellstr requires exactly one input",
            request.arguments.len().saturating_sub(1),
        ));
        return diagnostics;
    }
    let input = &request.arguments[0];
    match &input.kind {
        ValueKindFact::Character
        | ValueKindFact::String
        | ValueKindFact::Symbolic
        | ValueKindFact::Unknown => {}
        ValueKindFact::Cell(cell) => {
            if !text_element(&cell.element)
                || cell.elements.iter().any(|value| !text_element(value))
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-CELLSTR-CONTENTS",
                    "cellstr cell inputs require character vectors or string scalars",
                    0,
                ));
            }
        }
        _ => diagnostics.push(argument_error(
            "RM-CATALOG-CELLSTR-INPUT",
            "cellstr expects a character array, string array, or supported RunMat text value",
            0,
        )),
    }
    diagnostics
}

fn text_element(value: &ValueFact) -> bool {
    match value.kind {
        ValueKindFact::String => matches!(value.shape.element_count(), None | Some(1)),
        ValueKindFact::Unknown => true,
        ValueKindFact::Character => character_vector(&value.shape),
        _ => false,
    }
}

fn character_vector(shape: &ShapeFact) -> bool {
    match shape {
        ShapeFact::Scalar | ShapeFact::Unknown | ShapeFact::Ranked { .. } => true,
        ShapeFact::Shaped { dims } => matches!(
            dims.as_slice(),
            [] | [DimensionFact::Known(1), ..]
                | [DimensionFact::Unknown, ..]
                | [DimensionFact::Known(0), DimensionFact::Known(0)]
        ),
    }
}
