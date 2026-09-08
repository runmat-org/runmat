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
    let diagnostics = diagnostics(request);
    let output = output(request).unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue));
    finish_fixed(entry, request, output, diagnostics)
}

fn output(request: &CallRequest) -> Option<ValueFact> {
    let input = request.arguments.first()?;
    if request.arguments.len() < 2 {
        return None;
    }
    let shape = partition_grid_shape(request).unwrap_or(ShapeFact::Unknown);
    let element = ValueFact::proven(input.kind.clone(), ShapeFact::Unknown, StorageFact::Dense);
    Some(ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(element),
            elements: Vec::new(),
            elements_complete: shape.element_count() == Some(0),
        }),
        shape,
        StorageFact::Dense,
    ))
}

fn partition_grid_shape(request: &CallRequest) -> Option<ShapeFact> {
    let mut counts = Vec::new();
    for index in 1..request.arguments.len() {
        counts.push(Some(request.literals.numeric_vector_at(index)?.len()));
    }
    let input_rank = request.arguments.first()?.shape.rank()?;
    counts.resize(counts.len().max(input_rank), Some(1));
    if counts.len() == 1 {
        counts.push(Some(1));
    }
    Some(ShapeFact::from(counts))
}

fn diagnostics(request: &CallRequest) -> Vec<runmat_types::InferenceDiagnostic> {
    let mut diagnostics = Vec::new();
    if request.arguments.len() < 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-MAT2CELL-ARITY",
            "mat2cell expects an array and at least one partition vector",
            request.arguments.len().saturating_sub(1),
        ));
        return diagnostics;
    }
    if matches!(
        request.arguments[0].kind,
        ValueKindFact::Cell(_) | ValueKindFact::Struct(_) | ValueKindFact::Object(_)
    ) {
        diagnostics.push(argument_error(
            "RM-CATALOG-MAT2CELL-INPUT",
            "mat2cell expects a numeric, logical, string, or character array",
            0,
        ));
    }
    for (index, argument) in request.arguments.iter().enumerate().skip(1) {
        if !matches!(
            argument.kind,
            ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Unknown
        ) || !shape_may_be_vector(&argument.shape)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-MAT2CELL-PARTITION",
                "mat2cell partitions must be numeric scalar or vector values",
                index,
            ));
        }
    }
    diagnostics
}

fn shape_may_be_vector(shape: &ShapeFact) -> bool {
    match shape {
        ShapeFact::Scalar | ShapeFact::Ranked { rank: 1 } | ShapeFact::Unknown => true,
        ShapeFact::Shaped { dims } => {
            dims.iter()
                .filter(|dim| !matches!(dim, DimensionFact::Known(0 | 1)))
                .count()
                <= 1
        }
        ShapeFact::Ranked { .. } => false,
    }
}
