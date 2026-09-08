use crate::catalog::inference::finish_fixed;
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, CellFact, DimensionFact, DynamicReason, ShapeFact, StorageFact,
    ValueFact, ValueKindFact,
};

use super::{diagnostics, dimensions};

pub(in crate::catalog) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    finish_fixed(
        entry,
        request,
        output(request),
        diagnostics::for_request(request),
    )
}

fn output(request: &CallRequest) -> ValueFact {
    let Some(input) = request.arguments.first() else {
        return ValueFact::unknown(DynamicReason::RuntimeValue);
    };
    let outer_shape = dimensions::grouped(request)
        .and_then(|dims| partition_shapes(&input.shape, &dims).map(|shapes| shapes.0))
        .unwrap_or_else(|| {
            if request.arguments.len() == 1 {
                input.shape.clone()
            } else {
                ShapeFact::Unknown
            }
        });
    let element_shape = dimensions::grouped(request)
        .and_then(|dims| partition_shapes(&input.shape, &dims).map(|shapes| shapes.1))
        .unwrap_or(ShapeFact::Scalar);
    let element = element_fact(input, element_shape);
    ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(element),
            elements: Vec::new(),
            elements_complete: outer_shape.element_count() == Some(0),
        }),
        outer_shape,
        StorageFact::Dense,
    )
}

fn element_fact(input: &ValueFact, shape: ShapeFact) -> ValueFact {
    let kind = match &input.kind {
        ValueKindFact::Cell(cell) => ValueKindFact::Cell(cell.clone()),
        kind => kind.clone(),
    };
    ValueFact::proven(kind, shape, StorageFact::Dense)
}

fn partition_shapes(shape: &ShapeFact, dims: &[usize]) -> Option<(ShapeFact, ShapeFact)> {
    let source = match shape {
        ShapeFact::Scalar => vec![DimensionFact::Known(1), DimensionFact::Known(1)],
        ShapeFact::Shaped { dims } => dims.clone(),
        ShapeFact::Ranked { rank } => vec![DimensionFact::Unknown; *rank],
        ShapeFact::Unknown => return None,
    };
    if dims.is_empty() {
        return Some((ShapeFact::Shaped { dims: source }, ShapeFact::Scalar));
    }
    if dims.iter().any(|dim| *dim == 0 || *dim > source.len()) {
        return None;
    }
    let mut outer = source.clone();
    let mut selected = dims.to_vec();
    selected.sort_unstable();
    let mut slice = vec![DimensionFact::Known(1); source.len()];
    for (&requested, &destination) in dims.iter().zip(selected.iter()) {
        outer[requested - 1] = DimensionFact::Known(1);
        slice[destination - 1] = source[requested - 1].clone();
    }
    Some((
        ShapeFact::Shaped { dims: outer },
        ShapeFact::Shaped { dims: slice },
    ))
}
