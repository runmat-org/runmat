use super::super::pairwise_distance::DistanceMetricKind;
use super::{KnnMetric, KnnOption, KnnSearchMethod};
use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::*;

#[derive(Clone, Copy)]
struct KnownOptions {
    k: Option<usize>,
    include_ties: Option<bool>,
    metric: Option<KnnMetric>,
    method: Option<KnnSearchMethod>,
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() < 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-KNNSEARCH-ARITY",
            "knnsearch requires X and Y",
            request.arguments.len(),
        ));
    }
    let [x_rows, x_cols] = observations(request.arguments.first(), 0, &mut diagnostics);
    let [y_rows, y_cols] = observations(request.arguments.get(1), 1, &mut diagnostics);
    if x_cols.is_some() && y_cols.is_some() && x_cols != y_cols {
        diagnostics.push(argument_error(
            "RM-CATALOG-KNNSEARCH-COLUMNS",
            "X and Y must have the same number of columns",
            1,
        ));
    }
    let options = options(request, &mut diagnostics);
    if options.k.zip(x_rows).is_some_and(|(k, rows)| k > rows) {
        diagnostics.push(argument_error(
            "RM-CATALOG-KNNSEARCH-K",
            "K must not exceed the number of observations in X",
            2,
        ));
    }
    if matches!(options.method, Some(KnnSearchMethod::KdTree))
        && options.metric.is_some_and(|metric| !metric.allows_kdtree())
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-KNNSEARCH-METHOD",
            "kd-tree method requires Euclidean, cityblock, Chebychev, or Minkowski distance",
            2,
        ));
    }
    let shape = match options.include_ties {
        Some(true) => ShapeFact::from(vec![y_rows, Some(1)]),
        Some(false) => ShapeFact::from(vec![y_rows, options.k]),
        None => ShapeFact::Unknown,
    };
    let numeric = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::Unknown,
        StorageFact::Dense,
    );
    let mut output = match options.include_ties {
        Some(true) => ValueFact::proven(
            ValueKindFact::Cell(CellFact {
                element: Box::new(numeric.clone()),
                elements: Vec::new(),
                elements_complete: false,
            }),
            shape,
            StorageFact::Dense,
        ),
        Some(false) => {
            let mut fact = numeric.clone();
            fact.shape = shape;
            fact
        }
        None => ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    if request
        .arguments
        .iter()
        .take(2)
        .any(|value| !matches!(value.residency, ResidencyFact::Host))
    {
        output.residency = ResidencyFact::Unknown;
        output.certainty = CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
    }
    let mut contract = CallContract::fixed(vec![output.clone(), output]);
    contract.maximum_outputs = None;
    contract.variadic_output = Some(Box::new(ValueFact::scalar(ValueKindFact::Numeric(
        NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        },
    ))));
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}

fn observations(
    value: Option<&ValueFact>,
    index: usize,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> [Option<usize>; 2] {
    let Some(value) = value else {
        return [None, None];
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
            "RM-CATALOG-KNNSEARCH-CLASS",
            "observations must be real numeric or logical arrays",
            index,
        ));
    }
    match &value.shape {
        ShapeFact::Scalar => [Some(1), Some(1)],
        ShapeFact::Shaped { dims } if dims.len() == 1 => [
            match dims[0] {
                DimensionFact::Known(value) => Some(value),
                _ => None,
            },
            Some(1),
        ],
        ShapeFact::Shaped { dims } if dims.len() == 2 => [
            match dims[0] {
                DimensionFact::Known(value) => Some(value),
                _ => None,
            },
            match dims[1] {
                DimensionFact::Known(value) => Some(value),
                _ => None,
            },
        ],
        ShapeFact::Shaped { .. } => {
            diagnostics.push(argument_error(
                "RM-CATALOG-KNNSEARCH-RANK",
                "observations must be two-dimensional",
                index,
            ));
            [None, None]
        }
        ShapeFact::Ranked { rank: 1 } => [None, Some(1)],
        ShapeFact::Ranked { .. } | ShapeFact::Unknown => [None, None],
    }
}

fn options(request: &CallRequest, diagnostics: &mut Vec<InferenceDiagnostic>) -> KnownOptions {
    let mut known = KnownOptions {
        k: Some(1),
        include_ties: Some(false),
        metric: Some(KnnMetric {
            kind: DistanceMetricKind::Euclidean,
            fast_alias: false,
        }),
        method: None,
    };
    if !(request.arguments.len().saturating_sub(2)).is_multiple_of(2) {
        diagnostics.push(argument_error(
            "RM-CATALOG-KNNSEARCH-OPTIONS",
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
            known.k = None;
            known.include_ties = None;
            known.metric = None;
            known.method = None;
            continue;
        };
        let Some(option) = KnnOption::from_keyword(name) else {
            diagnostics.push(argument_error(
                "RM-CATALOG-KNNSEARCH-OPTION",
                "unsupported knnsearch option",
                index,
            ));
            continue;
        };
        let literal = request.literals.literal_args.get(index + 1);
        match option {
            KnnOption::K => {
                known.k = literal.and_then(positive_count);
                if literal.is_some_and(|value| !matches!(value, LiteralValue::Unknown))
                    && known.k.is_none()
                {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-KNNSEARCH-K",
                        "K must be a positive exact integer scalar",
                        index + 1,
                    ));
                }
            }
            KnnOption::IncludeTies => {
                known.include_ties = literal.and_then(logical_toggle);
                if literal.is_some_and(|value| !matches!(value, LiteralValue::Unknown))
                    && known.include_ties.is_none()
                {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-KNNSEARCH-TIES",
                        "IncludeTies must be a logical scalar",
                        index + 1,
                    ));
                }
            }
            KnnOption::Distance => {
                known.metric = literal
                    .and_then(literal_text)
                    .and_then(KnnMetric::from_keyword);
                if literal.and_then(literal_text).is_some() && known.metric.is_none() {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-KNNSEARCH-METRIC",
                        "unsupported named distance metric",
                        index + 1,
                    ));
                }
            }
            KnnOption::NsMethod => {
                known.method = literal
                    .and_then(literal_text)
                    .and_then(KnnSearchMethod::from_keyword);
                if literal.and_then(literal_text).is_some() && known.method.is_none() {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-KNNSEARCH-METHOD",
                        "unsupported NSMethod",
                        index + 1,
                    ));
                }
            }
            KnnOption::P
            | KnnOption::Cov
            | KnnOption::Scale
            | KnnOption::SortIndices
            | KnnOption::CacheSize
            | KnnOption::BucketSize => {}
        }
    }
    known
}

fn literal_text(literal: &LiteralValue) -> Option<&str> {
    match literal {
        LiteralValue::String(text)
        | LiteralValue::Character(text)
        | LiteralValue::Keyword(text) => Some(text),
        _ => None,
    }
}

fn positive_count(value: &LiteralValue) -> Option<usize> {
    match value {
        LiteralValue::Integer { text, .. } => text.parse::<usize>().ok().filter(|value| *value > 0),
        LiteralValue::Number(value)
            if value.is_finite()
                && *value >= 1.0
                && value.fract().abs() <= 1.0e-12
                && *value < usize::MAX as f64 =>
        {
            Some(*value as usize)
        }
        LiteralValue::Real { text, .. } => text
            .parse::<f64>()
            .ok()
            .and_then(|value| positive_count(&LiteralValue::Number(value))),
        _ => None,
    }
}

fn logical_toggle(value: &LiteralValue) -> Option<bool> {
    match value {
        LiteralValue::Bool(value) => Some(*value),
        LiteralValue::Integer { text, .. } => match text.parse::<i128>().ok()? {
            0 => Some(false),
            1 => Some(true),
            _ => None,
        },
        LiteralValue::Number(value) if *value == 0.0 => Some(false),
        LiteralValue::Number(value) if *value == 1.0 => Some(true),
        _ => None,
    }
}
