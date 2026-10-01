use super::{KmeansDistance, KmeansEmptyAction, KmeansOption, KmeansStart};
use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::*;

const FLOAT_COUNT_TOLERANCE: f64 = 1.0e-12;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() < 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-KMEANS-ARITY",
            "kmeans requires X and k",
            request.arguments.len(),
        ));
    }
    let (rows, cols) = observations(request.arguments.first(), &mut diagnostics);
    let controls = controls(request, &mut diagnostics);
    let k = request
        .literals
        .literal_args
        .get(1)
        .and_then(positive_count)
        .or_else(|| {
            request.arguments.get(1).and_then(|fact| {
                definitely_empty(&fact.shape)
                    .then_some(controls.start_clusters)
                    .flatten()
            })
        });
    // Shape alone cannot establish how many rows remain after NaN omission.
    // EmptyAction='drop' also admits cluster counts above that number.
    if request.literals.literal_args.get(1).is_some_and(|literal| {
        matches!(
            literal,
            LiteralValue::Number(_) | LiteralValue::Real { .. } | LiteralValue::Integer { .. }
        ) && positive_count(literal).is_none()
    }) {
        diagnostics.push(argument_error(
            "RM-CATALOG-KMEANS-K",
            "k must be a positive integer scalar",
            1,
        ));
    }
    if request
        .arguments
        .get(1)
        .is_some_and(|fact| definitely_empty(&fact.shape))
        && controls.start_clusters.is_none()
        && controls.start_is_known_non_numeric
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-KMEANS-EMPTY-K",
            "k may be empty only with numeric Start centroids",
            1,
        ));
    }
    let mut outputs = vec![
        numeric_shape(vec![rows, Some(1)]),
        numeric_shape(vec![k, cols]),
        numeric_shape(vec![k, Some(1)]),
        numeric_shape(vec![rows, k]),
    ];
    if request
        .arguments
        .first()
        .is_some_and(|fact| !matches!(fact.residency, ResidencyFact::Host))
    {
        for output in &mut outputs {
            output.residency = ResidencyFact::Unknown;
            output.certainty = CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
        }
    }
    let mut contract = CallContract::fixed(outputs);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}

fn numeric_shape(dims: Vec<Option<usize>>) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(dims),
        StorageFact::Dense,
    )
}

fn observations(
    value: Option<&ValueFact>,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> (Option<usize>, Option<usize>) {
    let Some(value) = value else {
        return (None, None);
    };
    if !matches!(
        value.kind,
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Real,
            ..
        }) | ValueKindFact::Logical
            | ValueKindFact::Unknown
    ) {
        diagnostics.push(argument_error(
            "RM-CATALOG-KMEANS-CLASS",
            "X must be a real numeric observation array",
            0,
        ));
    }
    match &value.shape {
        ShapeFact::Scalar => (Some(1), Some(1)),
        ShapeFact::Shaped { dims } if dims.len() == 1 => (known(&dims[0]), Some(1)),
        ShapeFact::Shaped { dims } if dims.len() == 2 => {
            let (rows, cols) = (known(&dims[0]), known(&dims[1]));
            if rows == Some(1) || cols == Some(1) {
                (
                    rows.zip(cols)
                        .and_then(|(rows, cols)| rows.checked_mul(cols)),
                    Some(1),
                )
            } else {
                (rows, cols)
            }
        }
        ShapeFact::Shaped { .. } => {
            diagnostics.push(argument_error(
                "RM-CATALOG-KMEANS-RANK",
                "X must be a numeric vector or 2-D matrix",
                0,
            ));
            (None, None)
        }
        ShapeFact::Ranked { rank: 1 } => (None, Some(1)),
        ShapeFact::Ranked { rank: 2 } | ShapeFact::Unknown => (None, None),
        ShapeFact::Ranked { .. } => {
            diagnostics.push(argument_error(
                "RM-CATALOG-KMEANS-RANK",
                "X must be a numeric vector or 2-D matrix",
                0,
            ));
            (None, None)
        }
    }
}

fn known(dimension: &DimensionFact) -> Option<usize> {
    match dimension {
        DimensionFact::Known(value) => Some(*value),
        _ => None,
    }
}

fn definitely_empty(shape: &ShapeFact) -> bool {
    matches!(shape, ShapeFact::Shaped { dims } if dims.iter().any(|dim| matches!(dim, DimensionFact::Known(0))))
}

#[derive(Default)]
struct KnownControls {
    start_clusters: Option<usize>,
    start_is_known_non_numeric: bool,
}

fn controls(request: &CallRequest, diagnostics: &mut Vec<InferenceDiagnostic>) -> KnownControls {
    let mut known = KnownControls::default();
    if !request.arguments.len().saturating_sub(2).is_multiple_of(2) {
        diagnostics.push(argument_error(
            "RM-CATALOG-KMEANS-OPTIONS",
            "options must be name-value pairs",
            2,
        ));
    }
    for index in (2..request.arguments.len().saturating_sub(1)).step_by(2) {
        let Some(name) = request
            .literals
            .literal_args
            .get(index)
            .and_then(literal_text)
        else {
            continue;
        };
        let Some(option) = KmeansOption::from_keyword(name) else {
            diagnostics.push(argument_error(
                "RM-CATALOG-KMEANS-OPTION",
                "unsupported kmeans option",
                index,
            ));
            continue;
        };
        let value = request.arguments.get(index + 1);
        let literal = request.literals.literal_args.get(index + 1);
        match option {
            KmeansOption::Start => {
                if let Some(text) = literal.and_then(literal_text) {
                    known.start_is_known_non_numeric = true;
                    if KmeansStart::from_keyword(text).is_none() {
                        diagnostics.push(argument_error(
                            "RM-CATALOG-KMEANS-START",
                            "unsupported Start method",
                            index + 1,
                        ));
                    }
                } else if let Some(fact) = value {
                    known.start_clusters = match &fact.shape {
                        ShapeFact::Shaped { dims } if dims.len() == 2 || dims.len() == 3 => {
                            dims.first().and_then(known_dimension)
                        }
                        _ => None,
                    };
                }
            }
            KmeansOption::Distance => {
                if let Some(text) = literal.and_then(literal_text) {
                    if KmeansDistance::from_keyword(text).is_none() {
                        diagnostics.push(argument_error(
                            "RM-CATALOG-KMEANS-DISTANCE",
                            "unsupported clustering distance",
                            index + 1,
                        ));
                    }
                }
            }
            KmeansOption::EmptyAction => {
                if let Some(text) = literal.and_then(literal_text) {
                    if KmeansEmptyAction::from_keyword(text).is_none() {
                        diagnostics.push(argument_error(
                            "RM-CATALOG-KMEANS-EMPTY-ACTION",
                            "unsupported EmptyAction",
                            index + 1,
                        ));
                    }
                }
            }
            KmeansOption::MaxIter | KmeansOption::Replicates => {
                if literal.is_some_and(|value| !matches!(value, LiteralValue::Unknown))
                    && literal.and_then(positive_count).is_none()
                {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-KMEANS-COUNT",
                        "iteration and replicate counts must be positive integers",
                        index + 1,
                    ));
                }
            }
            KmeansOption::Display | KmeansOption::OnlinePhase | KmeansOption::Options => {}
        }
    }
    known
}

fn known_dimension(dimension: &DimensionFact) -> Option<usize> {
    known(dimension)
}

fn literal_text(value: &LiteralValue) -> Option<&str> {
    match value {
        LiteralValue::String(value)
        | LiteralValue::Character(value)
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
                && value.fract().abs() <= FLOAT_COUNT_TOLERANCE
                && *value < usize::MAX as f64 =>
        {
            Some(*value as usize)
        }
        _ => None,
    }
}
