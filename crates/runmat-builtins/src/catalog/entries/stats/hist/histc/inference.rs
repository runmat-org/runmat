use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(2..=3).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-HISTC-ARITY",
            "histc expects x, binranges, and optional dim",
            request.arguments.len().saturating_sub(1),
        ));
    }
    for (index, value) in request.arguments.iter().take(2).enumerate() {
        if !matches!(
            value.kind,
            ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Unknown
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-HISTC-INPUT",
                "histc expects numeric or logical data and edges",
                index,
            ));
        }
    }
    if request
        .outputs
        .requested
        .known_count()
        .is_some_and(|count| count > 2)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-HISTC-OUTPUTS",
            "histc supports at most two outputs",
            0,
        ));
    }
    let explicit = request.arguments.len() >= 3;
    let dim = request.literals.numeric_at(2).and_then(|d| {
        if d.is_finite()
            && d.fract() == 0.0
            && (1.0..=super::HISTC_MAX_DIMENSION as f64).contains(&d)
        {
            Some(d as usize - 1)
        } else if !d.is_finite() || !(1.0..=super::HISTC_MAX_DIMENSION as f64).contains(&d.round())
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-HISTC-DIM",
                "histc dim must be an integer from 1 through 32",
                2,
            ));
            None
        } else {
            // Runtime dimension admission permits a small fractional tolerance.
            // Without a shared floating admission policy, retain an unknown axis
            // rather than issuing an incorrect rejection or guessing its shape.
            None
        }
    });
    if let Some(value) = request.arguments.get(2) {
        if value.shape.element_count().is_some_and(|n| n != 1)
            || !matches!(
                value.kind,
                ValueKindFact::Numeric(NumericFact {
                    domain: NumericDomain::Real,
                    ..
                }) | ValueKindFact::Logical
                    | ValueKindFact::Unknown
            )
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-HISTC-DIM",
                "histc dim must be a real scalar",
                2,
            ));
        }
    }
    let input = request
        .arguments
        .first()
        .and_then(|v| known_shape(&v.shape));
    let edges = request.arguments.get(1).and_then(|v| known_shape(&v.shape));
    let edge_count = edges.as_ref().and_then(|shape| checked_count(shape));
    let edge_rows = edges.as_ref().and_then(|shape| {
        if checked_count(shape) == Some(0) || vector(shape) {
            checked_count(shape)
        } else {
            shape.first().copied()
        }
    });
    let counts = match (&input, &edges, edge_rows) {
        (Some(x), Some(e), Some(edge_rows)) if !explicit || dim.is_some() => {
            if !explicit && vector(x) {
                shape(e)
            } else {
                let axis = dim.unwrap_or_else(|| x.iter().position(|d| *d != 1).unwrap_or(0));
                let mut result = x.clone();
                result.resize(result.len().max(axis + 1), 1);
                result[axis] = edge_rows;
                shape(&result)
            }
        }
        _ => ShapeFact::Unknown,
    };
    let indices = match edge_count {
        Some(0) => shape(&[0, 0]),
        Some(_) => request
            .arguments
            .first()
            .map(|v| v.shape.clone())
            .unwrap_or(ShapeFact::Unknown),
        None => ShapeFact::Unknown,
    };
    let count = request.outputs.requested.known_count().unwrap_or(1);
    let outputs = (0..count)
        .map(|slot| {
            numeric(if slot == 0 {
                counts.clone()
            } else if slot == 1 {
                indices.clone()
            } else {
                ShapeFact::Unknown
            })
        })
        .collect();
    let mut contract = CallContract::fixed(outputs);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.dynamic_outputs = request.outputs.requested.known_count().is_none();
    result.diagnostics.extend(diagnostics);
    result
}
fn known_shape(shape: &ShapeFact) -> Option<Vec<usize>> {
    match shape {
        ShapeFact::Scalar => Some(vec![1, 1]),
        ShapeFact::Shaped { dims } => dims
            .iter()
            .map(|d| {
                if let DimensionFact::Known(n) = d {
                    Some(*n)
                } else {
                    None
                }
            })
            .collect(),
        _ => None,
    }
}
fn vector(shape: &[usize]) -> bool {
    shape.iter().filter(|d| **d != 1).count() <= 1
}
fn checked_count(shape: &[usize]) -> Option<usize> {
    shape
        .iter()
        .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
}
fn shape(dims: &[usize]) -> ShapeFact {
    ShapeFact::from(dims.iter().copied().map(Some).collect::<Vec<_>>())
}
fn numeric(shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    )
}
