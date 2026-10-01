use super::HistcountsOption;
use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DimensionFact, LiteralValue,
    NumericClass, NumericDomain, NumericFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.is_empty() {
        diagnostics.push(argument_error(
            "RM-CATALOG-HISTCOUNTS-ARITY",
            "histcounts requires observations",
            0,
        ));
    }
    if let Some(data) = request.arguments.first() {
        if data.storage == StorageFact::Sparse
            || !matches!(
                data.kind,
                ValueKindFact::Numeric(NumericFact {
                    domain: NumericDomain::Real,
                    ..
                }) | ValueKindFact::Logical
                    | ValueKindFact::Unknown
            )
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-HISTCOUNTS-DATA",
                "histcounts requires real numeric or logical observations",
                0,
            ));
        }
    }

    // A known edge-vector length fixes the successful result shape without
    // inspecting values. A bin count also fixes it; a default or method does
    // not, because constant data can collapse the default to one bin.
    let mut bins = None;
    let mut options_start = 1;
    if request.arguments.len() > 1
        && request.literals.literal_string_at(1).is_none()
        && matches!(
            request.arguments[1].kind,
            ValueKindFact::Numeric(_) | ValueKindFact::Logical
        )
    {
        let control = &request.arguments[1];
        bins = edge_bins(&control.shape)
            .or_else(|| known_positive_bin_count(request.literals.literal_args.get(1)));
        options_start = 2;
    }
    let mut index = options_start;
    while index + 1 < request.arguments.len() {
        match request
            .literals
            .literal_string_at(index)
            .as_deref()
            .and_then(HistcountsOption::parse)
        {
            Some(HistcountsOption::BinEdges) => {
                bins = edge_bins(&request.arguments[index + 1].shape);
            }
            Some(HistcountsOption::NumBins) => {
                bins = known_positive_bin_count(request.literals.literal_args.get(index + 1));
            }
            Some(HistcountsOption::BinWidth | HistcountsOption::BinMethod) => bins = None,
            _ => {}
        }
        index += 2;
    }
    if index < request.arguments.len() {
        diagnostics.push(argument_error(
            "RM-CATALOG-HISTCOUNTS-OPTIONS",
            "histcounts options require name/value pairs",
            index,
        ));
    }
    let count_shape = bins.map_or(ShapeFact::Unknown, row);
    let edge_shape = bins
        .and_then(|n| n.checked_add(1))
        .map_or(ShapeFact::Unknown, row);
    let count = request.outputs.requested.known_count().unwrap_or(3);
    if count > 3 {
        diagnostics.push(argument_error(
            "RM-CATALOG-HISTCOUNTS-OUTPUTS",
            "histcounts supports at most three outputs",
            count,
        ));
    }
    let outputs = (0..count.min(3))
        .map(|slot| match slot {
            0 => double(count_shape.clone()),
            1 => double(edge_shape.clone()),
            _ => double(
                request
                    .arguments
                    .first()
                    .map_or(ShapeFact::Unknown, |value| value.shape.clone()),
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

fn known_positive_bin_count(literal: Option<&LiteralValue>) -> Option<usize> {
    match literal? {
        LiteralValue::Integer { text, .. } => text.parse::<usize>().ok().filter(|n| *n > 0),
        LiteralValue::Number(number)
            if number.is_finite()
                && *number > 0.0
                && number.fract() == 0.0
                && *number < usize::MAX as f64 =>
        {
            Some(*number as usize)
        }
        LiteralValue::Real { text, .. } => text
            .parse::<f64>()
            .ok()
            .and_then(|n| known_positive_bin_count(Some(&LiteralValue::Number(n)))),
        _ => None,
    }
}

fn row(length: usize) -> ShapeFact {
    ShapeFact::Shaped {
        dims: vec![DimensionFact::Known(1), DimensionFact::Known(length)],
    }
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
