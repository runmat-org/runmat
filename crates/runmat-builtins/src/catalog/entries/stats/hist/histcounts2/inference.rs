use super::Histcounts2Option;
use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DimensionFact, LiteralValue,
    NumericClass, NumericDomain, NumericFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() < 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-HISTCOUNTS2-ARITY",
            "histcounts2 requires paired X and Y observations",
            request.arguments.len(),
        ));
    }
    for (index, value) in request.arguments.iter().take(2).enumerate() {
        if value.storage == StorageFact::Sparse
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
                "RM-CATALOG-HISTCOUNTS2-DATA",
                "histcounts2 requires real numeric or logical observations",
                index,
            ));
        }
    }
    if let [x, y, ..] = request.arguments.as_slice() {
        if x.shape
            .known_dims()
            .is_some_and(|dims| dims.iter().all(Option::is_some))
            && y.shape
                .known_dims()
                .is_some_and(|dims| dims.iter().all(Option::is_some))
            && !x.shape.is_proven_equivalent(&y.shape)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-HISTCOUNTS2-SHAPE",
                "histcounts2 requires X and Y to have the same shape",
                1,
            ));
        }
    }
    let mut x_bins = None;
    let mut y_bins = None;
    let mut index = 2;
    if index < request.arguments.len()
        && request.literals.literal_string_at(index).is_none()
        && matches!(
            request.arguments[index].kind,
            ValueKindFact::Numeric(_) | ValueKindFact::Logical
        )
    {
        if index + 1 < request.arguments.len()
            && request.literals.literal_string_at(index + 1).is_none()
            && matches!(
                request.arguments[index + 1].kind,
                ValueKindFact::Numeric(_) | ValueKindFact::Logical
            )
        {
            let x_shape = &request.arguments[index].shape;
            let y_shape = &request.arguments[index + 1].shape;
            if edge_bins(x_shape).is_some() && edge_bins(y_shape).is_some() {
                x_bins = edge_bins(x_shape);
                y_bins = edge_bins(y_shape);
            } else if x_shape.element_count() == Some(1) && y_shape.element_count() == Some(1) {
                x_bins = positive_count(request.literals.literal_args.get(index));
                y_bins = positive_count(request.literals.literal_args.get(index + 1));
            }
            index += 2;
        } else {
            if let Some(LiteralValue::Vector(values)) = request.literals.literal_args.get(index) {
                match values.as_slice() {
                    [x, y] => {
                        x_bins = positive_count(Some(x));
                        y_bins = positive_count(Some(y));
                    }
                    [both] => x_bins = positive_count(Some(both)),
                    _ => {}
                }
            } else {
                x_bins = positive_count(request.literals.literal_args.get(index));
            }
            if y_bins.is_none() {
                y_bins = x_bins;
            }
            index += 1;
        }
    }
    while index + 1 < request.arguments.len() {
        let argument = index + 1;
        match request
            .literals
            .literal_string_at(index)
            .as_deref()
            .and_then(Histcounts2Option::parse)
        {
            Some(Histcounts2Option::XBinEdges) => {
                x_bins = edge_bins(&request.arguments[argument].shape)
            }
            Some(Histcounts2Option::YBinEdges) => {
                y_bins = edge_bins(&request.arguments[argument].shape)
            }
            Some(Histcounts2Option::NumBins) => {
                if let Some(LiteralValue::Vector(values)) =
                    request.literals.literal_args.get(argument)
                {
                    if let [x, y] = values.as_slice() {
                        x_bins = positive_count(Some(x));
                        y_bins = positive_count(Some(y));
                    } else {
                        x_bins = None;
                        y_bins = None;
                    }
                } else {
                    let both = positive_count(request.literals.literal_args.get(argument));
                    x_bins = both;
                    y_bins = both;
                }
            }
            Some(Histcounts2Option::XBinWidth | Histcounts2Option::XBinMethod) => x_bins = None,
            Some(Histcounts2Option::YBinWidth | Histcounts2Option::YBinMethod) => y_bins = None,
            Some(Histcounts2Option::BinWidth | Histcounts2Option::BinMethod) => {
                x_bins = None;
                y_bins = None;
            }
            _ => {}
        }
        index += 2;
    }
    if index < request.arguments.len() {
        diagnostics.push(argument_error(
            "RM-CATALOG-HISTCOUNTS2-OPTIONS",
            "histcounts2 options require name/value pairs",
            index,
        ));
    }
    let requested = request.outputs.requested.known_count().unwrap_or(5);
    if requested > 5 {
        diagnostics.push(argument_error(
            "RM-CATALOG-HISTCOUNTS2-OUTPUTS",
            "histcounts2 supports at most five outputs",
            requested,
        ));
    }
    let dimensions = match (x_bins, y_bins) {
        (Some(x), Some(y)) => ShapeFact::Shaped {
            dims: vec![DimensionFact::Known(x), DimensionFact::Known(y)],
        },
        _ => ShapeFact::Unknown,
    };
    let outputs = (0..requested.min(5))
        .map(|slot| match slot {
            0 => double(dimensions.clone()),
            1 => double(edge_shape(x_bins)),
            2 => double(edge_shape(y_bins)),
            3 => double(
                request
                    .arguments
                    .first()
                    .map_or(ShapeFact::Unknown, |v| v.shape.clone()),
            ),
            _ => double(
                request
                    .arguments
                    .get(1)
                    .map_or(ShapeFact::Unknown, |v| v.shape.clone()),
            ),
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

fn edge_bins(shape: &ShapeFact) -> Option<usize> {
    let dims = shape.known_dims()?;
    let mut length = 1usize;
    let mut non_singletons = 0;
    for dim in dims {
        let dim = dim?;
        non_singletons += usize::from(dim > 1);
        length = length.checked_mul(dim)?;
    }
    (non_singletons == 1 && length > 1).then_some(length - 1)
}

fn positive_count(value: Option<&LiteralValue>) -> Option<usize> {
    if let Some(LiteralValue::Integer { text, .. }) = value {
        return text.parse::<usize>().ok().filter(|n| *n > 0);
    }
    let number = runmat_types::LiteralContext::numeric_from_literal(value?)?;
    (number.is_finite() && number > 0.0 && number.fract() == 0.0 && number < usize::MAX as f64)
        .then_some(number as usize)
}

fn edge_shape(bins: Option<usize>) -> ShapeFact {
    bins.and_then(|n| n.checked_add(1))
        .map_or(ShapeFact::Unknown, |n| ShapeFact::Shaped {
            dims: vec![DimensionFact::Known(1), DimensionFact::Known(n)],
        })
}

fn double(shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    )
}
