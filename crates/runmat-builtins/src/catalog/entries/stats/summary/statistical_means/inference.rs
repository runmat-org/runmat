//! Static facts for geometric, harmonic and root-mean-square reductions.
//! Text is decoded only at the source-language option boundary; the rest of
//! inference uses a closed selector and missing-value mode.

use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DynamicReason, InferenceDiagnostic,
    LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact, ShapeFact, StorageFact,
    ValueFact, ValueKindFact,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum MeanKind {
    Geometric,
    Harmonic,
    RootMeanSquare,
    RootMeanSquaredError,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Selector {
    Default,
    All,
    Dimensions(Vec<usize>),
    Dynamic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MissingMode {
    Include,
    Omit,
}

pub(super) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    kind: MeanKind,
) -> CallInference {
    let mut diagnostics = Vec::new();
    let data_count = if kind == MeanKind::RootMeanSquaredError {
        2
    } else {
        1
    };
    if request.arguments.len() < data_count {
        diagnostics.push(error("ARITY", "missing required numerical input"));
    }
    let first = request.arguments.first();
    let second = if data_count == 2 {
        request.arguments.get(1)
    } else {
        None
    };
    for (index, fact) in [first, second].into_iter().flatten().enumerate() {
        let admissible = matches!(
            fact.kind,
            ValueKindFact::Numeric(_) | ValueKindFact::Unknown
        ) || (kind == MeanKind::RootMeanSquare
            && matches!(fact.kind, ValueKindFact::Logical | ValueKindFact::Character));
        if !admissible {
            diagnostics.push(error(
                "INPUT",
                format!("numerical input {index} has an unsupported class"),
            ));
        }
        if matches!(kind, MeanKind::Geometric | MeanKind::Harmonic)
            && matches!(
                fact.kind,
                ValueKindFact::Numeric(NumericFact {
                    domain: NumericDomain::Complex,
                    ..
                })
            )
        {
            diagnostics.push(error("COMPLEX", "this mean requires real input"));
        }
    }

    let (selector, _missing_mode) = parse_options(request, data_count, kind, &mut diagnostics);
    let input_shape = match (first, second) {
        (Some(left), Some(right)) => broadcast_shape(&left.shape, &right.shape, &mut diagnostics),
        (Some(left), None) => left.shape.clone(),
        _ => ShapeFact::Unknown,
    };
    let output_shape = reduced_shape(&input_shape, &selector);
    let output_class = result_class(first, second);
    let mut output = match output_class {
        Some(class) => ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class,
                domain: NumericDomain::Real,
            }),
            output_shape,
            StorageFact::Unknown,
        ),
        None => {
            let mut fact = ValueFact::unknown(DynamicReason::RuntimeValue);
            fact.shape = output_shape;
            fact
        }
    };
    output.storage = match output.shape.element_count() {
        Some(1) => StorageFact::Scalar,
        Some(_) => StorageFact::Dense,
        None => StorageFact::Unknown,
    };
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}

fn result_class(left: Option<&ValueFact>, right: Option<&ValueFact>) -> Option<NumericClass> {
    fn class(fact: &ValueFact) -> Option<NumericClass> {
        match fact.kind {
            ValueKindFact::Numeric(NumericFact { class, .. }) => Some(class),
            ValueKindFact::Logical | ValueKindFact::Character => Some(NumericClass::Double),
            _ => None,
        }
    }
    let left = class(left?)?;
    let right = match right {
        Some(fact) => Some(class(fact)?),
        None => None,
    };
    Some(
        if left == NumericClass::Single
            && right.unwrap_or(NumericClass::Single) == NumericClass::Single
        {
            NumericClass::Single
        } else {
            NumericClass::Double
        },
    )
}

fn parse_options(
    request: &CallRequest,
    start: usize,
    kind: MeanKind,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> (Selector, MissingMode) {
    let mut selector = None;
    let mut missing = None;
    let mut index = start;
    while index < request.arguments.len() {
        let literal = request
            .literals
            .literal_args
            .get(index)
            .unwrap_or(&LiteralValue::Unknown);
        match literal {
            LiteralValue::Character(text)
            | LiteralValue::String(text)
            | LiteralValue::Keyword(text) => {
                let keyword = text.to_ascii_lowercase();
                match keyword.as_str() {
                    "all" if selector.is_none() => selector = Some(Selector::All),
                    "all" => {
                        diagnostics.push(error("OPTIONS", "only one dimension selector is allowed"))
                    }
                    "includenan" | "includemissing" if missing.is_none() => {
                        missing = Some(MissingMode::Include)
                    }
                    "omitnan" | "omitmissing" if missing.is_none() => {
                        missing = Some(MissingMode::Omit)
                    }
                    "includenan" | "includemissing" | "omitnan" | "omitmissing" => {
                        diagnostics.push(error("OPTIONS", "only one missing-value flag is allowed"))
                    }
                    "weights" if kind == MeanKind::RootMeanSquaredError => {
                        if request.arguments.get(index + 1).is_none() {
                            diagnostics.push(error("WEIGHTS", "Weights requires a value"));
                        }
                        if matches!(selector, Some(Selector::All))
                            || matches!(&selector, Some(Selector::Dimensions(dims)) if dims.len() > 1)
                        {
                            diagnostics
                                .push(error("WEIGHTS", "Weights does not support all or vecdim"));
                        }
                        index += 1;
                    }
                    _ => diagnostics.push(error("OPTIONS", "unsupported reduction option")),
                }
            }
            LiteralValue::Number(_) | LiteralValue::Real { .. } | LiteralValue::Integer { .. } => {
                if selector.is_some() {
                    diagnostics.push(error("OPTIONS", "only one dimension selector is allowed"));
                } else {
                    selector = Some(parse_dimensions(std::slice::from_ref(literal), diagnostics));
                }
            }
            LiteralValue::Vector(values) => {
                if selector.is_some() {
                    diagnostics.push(error("OPTIONS", "only one dimension selector is allowed"));
                } else {
                    selector = Some(parse_dimensions(values, diagnostics));
                }
            }
            LiteralValue::Unknown => selector = selector.or(Some(Selector::Dynamic)),
            _ => diagnostics.push(error("OPTIONS", "unsupported reduction control")),
        }
        index += 1;
    }
    (
        selector.unwrap_or(Selector::Default),
        missing.unwrap_or(MissingMode::Include),
    )
}

fn parse_dimensions(
    values: &[LiteralValue],
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> Selector {
    let mut dimensions = Vec::with_capacity(values.len());
    for literal in values {
        let dimension = match literal {
            LiteralValue::Integer { text, .. } => text.parse::<usize>().ok(),
            other => LiteralContext::numeric_from_literal(other).and_then(|value| {
                (value.is_finite()
                    && value >= 1.0
                    && value.fract() == 0.0
                    && value < usize::MAX as f64)
                    .then_some(value as usize)
            }),
        };
        match dimension.filter(|dimension| *dimension > 0) {
            Some(dimension) if !dimensions.contains(&dimension) => dimensions.push(dimension),
            Some(_) => diagnostics.push(error("DIMENSION", "dimensions must be unique")),
            None => diagnostics.push(error("DIMENSION", "dimensions must be positive integers")),
        }
    }
    if dimensions.is_empty() {
        Selector::Dynamic
    } else {
        Selector::Dimensions(dimensions)
    }
}

fn broadcast_shape(
    left: &ShapeFact,
    right: &ShapeFact,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> ShapeFact {
    let (Some(mut left), Some(mut right)) = (left.known_dims(), right.known_dims()) else {
        return ShapeFact::Unknown;
    };
    let rank = left.len().max(right.len()).max(2);
    left.resize(rank, Some(1));
    right.resize(rank, Some(1));
    let mut output = Vec::with_capacity(rank);
    for (a, b) in left.into_iter().zip(right) {
        output.push(match (a, b) {
            (Some(a), Some(b)) if a == b => Some(a),
            (Some(1), other) | (other, Some(1)) => other,
            (Some(_), Some(_)) => {
                diagnostics.push(error(
                    "BROADCAST",
                    "forecast and actual shapes are incompatible",
                ));
                None
            }
            (Some(a), None) if a > 1 => None,
            (None, Some(b)) if b > 1 => None,
            _ => None,
        });
    }
    ShapeFact::from(output)
}

fn reduced_shape(input: &ShapeFact, selector: &Selector) -> ShapeFact {
    if matches!(selector, Selector::All) {
        return ShapeFact::Scalar;
    }
    if matches!(selector, Selector::Dynamic) {
        return if input.element_count() == Some(1) {
            ShapeFact::Scalar
        } else {
            ShapeFact::Unknown
        };
    }
    let Some(mut dims) = input.known_dims() else {
        return ShapeFact::Unknown;
    };
    let axes = match selector {
        Selector::Dimensions(dimensions) => dimensions.clone(),
        Selector::Default => {
            let mut axis = 1;
            for (index, extent) in dims.iter().enumerate() {
                match extent {
                    Some(1) => {}
                    Some(_) => {
                        axis = index + 1;
                        break;
                    }
                    None => return ShapeFact::Unknown,
                }
            }
            vec![axis]
        }
        Selector::All | Selector::Dynamic => unreachable!("handled above"),
    };
    for axis in axes {
        if let Some(extent) = dims.get_mut(axis - 1) {
            *extent = Some(1);
        }
    }
    let result = ShapeFact::from(dims);
    if result.element_count() == Some(1) {
        ShapeFact::Scalar
    } else {
        result
    }
}

fn error(code: &'static str, message: impl Into<String>) -> InferenceDiagnostic {
    InferenceDiagnostic::error(code, message)
}
