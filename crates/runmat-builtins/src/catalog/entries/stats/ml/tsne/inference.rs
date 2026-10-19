use super::{TsneAlgorithm, TsneDistance, TsneOption};
use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::*;

const MAX_EMBEDDING_DIMS: usize = 128;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.is_empty() {
        diagnostics.push(argument_error(
            "RM-CATALOG-TSNE-ARITY",
            "tsne requires X",
            0,
        ));
    }
    let output_rows = observation_rows(request.arguments.first(), &mut diagnostics);
    let dimensions = options(request, &mut diagnostics);
    let embedding = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![output_rows, dimensions]),
        StorageFact::Dense,
    );
    let loss = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let mut contract = CallContract::fixed(vec![embedding, loss.clone()]);
    // The current runtime pads requests beyond the documented two outputs
    // with double zeros. Preserve that carrier until a reviewed policy change.
    contract.maximum_outputs = None;
    contract.variadic_output = Some(Box::new(loss));
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}

fn observation_rows(
    value: Option<&ValueFact>,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> Option<usize> {
    let value = value?;
    if !matches!(
        value.kind,
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Real,
            ..
        }) | ValueKindFact::Logical
            | ValueKindFact::Unknown
    ) {
        diagnostics.push(argument_error(
            "RM-CATALOG-TSNE-CLASS",
            "X must be real numeric or logical observations",
            0,
        ));
    }
    let raw_rows = match &value.shape {
        ShapeFact::Scalar => Some(1),
        ShapeFact::Shaped { dims } if dims.len() == 1 => known(&dims[0]),
        ShapeFact::Shaped { dims } if dims.len() == 2 => known(&dims[0]),
        ShapeFact::Shaped { .. } => {
            diagnostics.push(argument_error(
                "RM-CATALOG-TSNE-RANK",
                "X must be a vector or 2-D matrix",
                0,
            ));
            None
        }
        ShapeFact::Ranked { rank: 1 | 2 } | ShapeFact::Unknown => None,
        ShapeFact::Ranked { .. } => {
            diagnostics.push(argument_error(
                "RM-CATALOG-TSNE-RANK",
                "X must be a vector or 2-D matrix",
                0,
            ));
            None
        }
    };
    if raw_rows == Some(0) {
        diagnostics.push(argument_error(
            "RM-CATALOG-TSNE-EMPTY",
            "tsne requires at least one complete observation",
            0,
        ));
    }
    // NaN omission depends on data contents. A single successful row is the
    // one case where its complete output cardinality follows from geometry.
    raw_rows.filter(|rows| *rows == 1)
}

fn options(request: &CallRequest, diagnostics: &mut Vec<InferenceDiagnostic>) -> Option<usize> {
    let mut dimensions = Some(2);
    if !request.arguments.len().saturating_sub(1).is_multiple_of(2) {
        diagnostics.push(argument_error(
            "RM-CATALOG-TSNE-OPTIONS",
            "name-value options must be paired",
            1,
        ));
    }
    for index in (1..request.arguments.len().saturating_sub(1)).step_by(2) {
        let name = match request.literals.literal_args.get(index) {
            Some(LiteralValue::Unknown) | None => {
                // This selector may be NumDimensions. A later known selector
                // can still override it, in the same order as runtime options.
                dimensions = None;
                continue;
            }
            Some(value) => match literal_text(value) {
                Some(name) => name,
                None => {
                    dimensions = None;
                    diagnostics.push(argument_error(
                        "RM-CATALOG-TSNE-OPTION-TYPE",
                        "tsne option names must be text",
                        index,
                    ));
                    continue;
                }
            },
        };
        let Some(option) = TsneOption::from_keyword(name) else {
            diagnostics.push(argument_error(
                "RM-CATALOG-TSNE-OPTION",
                "unsupported tsne option",
                index,
            ));
            continue;
        };
        let value = request.literals.literal_args.get(index + 1);
        match option {
            TsneOption::NumDimensions => match value {
                Some(LiteralValue::Unknown) | None => dimensions = None,
                Some(value) => match positive_count(value) {
                    Some(count) if count <= MAX_EMBEDDING_DIMS => dimensions = Some(count),
                    _ => {
                        dimensions = None;
                        diagnostics.push(argument_error(
                            "RM-CATALOG-TSNE-DIMENSIONS",
                            "NumDimensions must be a positive bounded integer",
                            index + 1,
                        ));
                    }
                },
            },
            TsneOption::Algorithm => {
                if value
                    .and_then(literal_text)
                    .is_some_and(|value| TsneAlgorithm::from_keyword(value).is_none())
                {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-TSNE-ALGORITHM",
                        "unsupported Algorithm",
                        index + 1,
                    ));
                }
            }
            TsneOption::Distance => {
                if value
                    .and_then(literal_text)
                    .is_some_and(|value| TsneDistance::from_keyword(value).is_none())
                {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-TSNE-DISTANCE",
                        "unsupported Distance",
                        index + 1,
                    ));
                }
            }
            TsneOption::CacheSize
            | TsneOption::Exaggeration
            | TsneOption::NumPcaComponents
            | TsneOption::Perplexity
            | TsneOption::Standardize
            | TsneOption::InitialY
            | TsneOption::LearnRate
            | TsneOption::NumPrint
            | TsneOption::Options
            | TsneOption::Theta
            | TsneOption::Verbose => {}
        }
    }
    dimensions
}

fn known(dimension: &DimensionFact) -> Option<usize> {
    match dimension {
        DimensionFact::Known(value) => Some(*value),
        _ => None,
    }
}

fn literal_text(value: &LiteralValue) -> Option<&str> {
    match value {
        LiteralValue::Character(value)
        | LiteralValue::String(value)
        | LiteralValue::Keyword(value) => Some(value),
        _ => None,
    }
}

fn positive_count(value: &LiteralValue) -> Option<usize> {
    match value {
        LiteralValue::Integer { text, .. } => text.parse::<usize>().ok().filter(|value| *value > 0),
        LiteralValue::Real { text, .. } => text
            .parse::<f64>()
            .ok()
            .and_then(|value| positive_count(&LiteralValue::Number(value))),
        LiteralValue::Number(value)
            if value.is_finite()
                && *value >= 1.0
                && value.fract() == 0.0
                && *value < usize::MAX as f64 =>
        {
            Some(*value as usize)
        }
        _ => None,
    }
}
